from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .attention_gate import (
    attention_ctx_ratio,
    build_attention_stat_features,
    extract_attention_head_masses,
    infer_context_token_span,
)
from .data import HotpotExample
from .gate_kl import build_attn_ratio_and_logit_topk_features, build_gate_features
from .models import get_model_input_device, synchronize_model, prepare_inputs

def _pick_next_token(logits, temperature: float = 0.0, top_p: float = 1.0,
                     generator: "torch.Generator | None" = None):
    """Greedy by default (temperature<=0 -> argmax, byte-identical to torch.argmax).
    temperature>0 -> nucleus(top_p) sampling with an optional seeded generator.
    logits: [batch, vocab]; returns LongTensor [batch] (same shape/semantics as torch.argmax(logits, dim=-1))."""
    if temperature is None or temperature <= 0.0:
        return torch.argmax(logits, dim=-1)
    logits = logits.float() / temperature
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum - torch.softmax(sorted_logits, dim=-1) > top_p  # keep first token always
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


CANONICAL_RULE_FUSION_MODES = {
    "rule_attn_ctx_ratio",
    "rule_attn_ctx_minus_shr",
    "rule_attn_top_head_ratio",
}
LEGACY_RULE_FUSION_ALIASES = {
    "attn_ctx_ratio": "rule_attn_ctx_ratio",
    "attn_ctx_minus_shr": "rule_attn_ctx_minus_shr",
    "attn_top_head": "rule_attn_top_head_ratio",
}
CANONICAL_LEARNED_GATE_TYPES = {
    "logit_topk",
    "attn_ratio_and_logit_topk",
    "attn_linear_weighted",
    "attn_stats_mlp_unweighted",
    "attn_ratio_mlp_weighted",
}
LEGACY_LEARNED_GATE_ALIASES = {
    "attn_linear": "attn_linear_weighted",
    "attn_mlp": "attn_stats_mlp_unweighted",
    "attn_ratio_mlp": "attn_ratio_mlp_weighted",
}


def normalize_rule_fusion_mode(name: str) -> str:
    return LEGACY_RULE_FUSION_ALIASES.get(name, name)


def normalize_learned_gate_type(name: str) -> str:
    return LEGACY_LEARNED_GATE_ALIASES.get(name, name)


@dataclass
class FusionResult:
    text: str
    generated_token_ids: list[int]
    terminated_early: bool
    timing: Optional["GenerationTiming"] = None
    lambda_stats: Optional[dict[str, float]] = None


@dataclass
class GenerationTiming:
    prefill_s: float
    decode_s: float
    total_s: float


@dataclass
class FusionTraceStep:
    position: int
    token_id: int
    token: str
    slm_entropy: Optional[float]
    lm_entropy: Optional[float]
    lambda_w: Optional[float]


def _as_token_id_set(value: Any) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, int):
        return {int(value)}
    if isinstance(value, (list, tuple, set)):
        ids: set[int] = set()
        for item in value:
            try:
                ids.add(int(item))
            except (TypeError, ValueError):
                continue
        return ids
    return set()


def get_eos_token_ids(tokenizer: PreTrainedTokenizerBase, model: Optional[PreTrainedModel] = None) -> set[int]:
    eos_ids = _as_token_id_set(getattr(tokenizer, "eos_token_id", None))
    if model is not None:
        generation_config = getattr(model, "generation_config", None)
        eos_ids |= _as_token_id_set(getattr(generation_config, "eos_token_id", None))
        config = getattr(model, "config", None)
        eos_ids |= _as_token_id_set(getattr(config, "eos_token_id", None))
    return eos_ids


def _tokenizer_vocab_fingerprint(tokenizer: PreTrainedTokenizerBase) -> tuple[int, int, tuple[tuple[str, int], ...]]:
    vocab = tokenizer.get_vocab()
    special_ids = tuple(
        sorted(
            (str(tok), int(idx))
            for tok, idx in getattr(tokenizer, "special_tokens_map", {}).items()
            if isinstance(idx, int)
        )
    )
    return len(vocab), max(vocab.values()) if vocab else -1, special_ids


def validate_fusion_tokenizers(
    slm_tokenizer: PreTrainedTokenizerBase,
    lm_tokenizer: PreTrainedTokenizerBase,
    *,
    slm_name: str = "SLM",
    lm_name: str = "LM",
) -> None:
    """
    Logit fusion chooses one raw token id and feeds that same id to both models.
    That is only valid when both tokenizers assign the same ids to the same tokens.
    """
    if slm_tokenizer is lm_tokenizer:
        return
    if _tokenizer_vocab_fingerprint(slm_tokenizer) != _tokenizer_vocab_fingerprint(lm_tokenizer):
        raise ValueError(
            "SLM+LM logit fusion requires token-id compatible tokenizers. "
            f"{slm_name} tokenizer appears incompatible with {lm_name} tokenizer. "
            "Use models from the same tokenizer family, such as Llama-3 SLM with Llama-3 LM, "
            "or add an explicit cross-tokenizer alignment layer before mixing logits."
        )
    slm_vocab = slm_tokenizer.get_vocab()
    lm_vocab = lm_tokenizer.get_vocab()
    if slm_vocab != lm_vocab:
        raise ValueError(
            "SLM+LM logit fusion requires identical token-to-id vocabularies. "
            f"{slm_name} and {lm_name} have matching sizes but different token mappings."
        )


class FixedLambdaFusionDecoder:
    """
    Step-by-step greedy decoder that fuses logits from SLM (with docs) and LM (query-only).

    Supports single-device loads and Transformers/Accelerate sharded `device_map` loads.
    """

    def __init__(
        self,
        slm_model: PreTrainedModel,
        slm_tokenizer: PreTrainedTokenizerBase,
        lm_model: PreTrainedModel,
        lm_tokenizer: PreTrainedTokenizerBase,
        *,
        slm_base_model: Optional[PreTrainedModel] = None,
        slm_base_tokenizer: Optional[PreTrainedTokenizerBase] = None,
        lambda_weight: float = 0.5,
        fusion_mode: str = "weighted_sum",
        entropy_scale: float = 1.0,
        entropy_mode: str = "threshold",
        top_k_entropy: int = 10,
        entropy_threshold: float = 0.9,
        max_new_tokens: int = 128,
        max_length: int = 4096,
        prompt_builder: Optional[Callable[[Any, bool, PreTrainedTokenizerBase], str]] = None,
        stop_strings: Optional[list[str]] = None,
        stop_on_repeat: Optional[str] = None,
        stop_after_answer: Optional[str] = None,
        answer_token_budget: int = 16,
        learned_gate: Optional[torch.nn.Module] = None,
        learned_gate_top_k: int = 10,
        fixed_mu_target: Optional[float] = None,
        fixed_mu_lambda_min: float = -50.0,
        fixed_mu_lambda_max: float = 50.0,
        fixed_mu_steps: int = 40,
        fixed_mu_score_type: str = "slm_logit",
        fixed_mu_rank_k: float = 10.0,
        learned_gate_type: str = "logit_topk",
        attn_rule_scale: float = 8.0,
        attn_rule_bias: float = 0.0,
        attn_rule_shared_scale: float = 1.0,
        attn_rule_top_head_frac: float = 0.1,
        decode_temperature: float = 0.0,
        decode_top_p: float = 1.0,
        decode_seed: int = 0,
        decode_sync_interval: int = 1,
        slm_kv_quant_bits: int = 0,
        slm_kv_backend: str = "quanto",
        slm_kv_group_size: int = 64,
        slm_kv_residual_length: int = 128,
        slm_kv_axis_key: int = 0,
        slm_kv_axis_value: int = 0,
    ) -> None:
        self.slm_model = slm_model
        self.slm_tokenizer = slm_tokenizer
        self.lm_model = lm_model
        self.lm_tokenizer = lm_tokenizer
        self.slm_base_model = slm_base_model or slm_model
        self.slm_base_tokenizer = slm_base_tokenizer or slm_tokenizer
        self.lambda_weight = lambda_weight
        self.fusion_mode = normalize_rule_fusion_mode(fusion_mode)
        self.entropy_scale = entropy_scale
        self.entropy_mode = entropy_mode
        self.top_k_entropy = top_k_entropy
        self.entropy_threshold = entropy_threshold
        self.max_new_tokens = max_new_tokens
        self.max_length = max_length
        self.prompt_builder = prompt_builder
        self.stop_strings = stop_strings or []
        self.stop_on_repeat = stop_on_repeat or None
        self.stop_after_answer = stop_after_answer or None
        self.answer_token_budget = max(int(answer_token_budget), 1)
        self.learned_gate = learned_gate
        self.learned_gate_top_k = learned_gate_top_k
        self.fixed_mu_target = fixed_mu_target
        self.fixed_mu_lambda_min = fixed_mu_lambda_min
        self.fixed_mu_lambda_max = fixed_mu_lambda_max
        self.fixed_mu_steps = fixed_mu_steps
        self.fixed_mu_score_type = fixed_mu_score_type
        self.fixed_mu_rank_k = fixed_mu_rank_k
        self.learned_gate_type = normalize_learned_gate_type(learned_gate_type)
        self.attn_rule_scale = attn_rule_scale
        self.attn_rule_bias = attn_rule_bias
        self.attn_rule_shared_scale = attn_rule_shared_scale
        self.attn_rule_top_head_frac = attn_rule_top_head_frac
        self.decode_sync_interval = max(int(decode_sync_interval), 1)
        self.decode_temperature = decode_temperature
        self.decode_top_p = decode_top_p
        self.decode_seed = decode_seed
        # Optional KV-cache quantization for ONLY the SLM/context branch (default off).
        # When >0 the SLM/context prefill builds a HF QuantizedCache so the long-context
        # KV is stored at slm_kv_quant_bits-bit; the LM/query branch is never touched.
        self.slm_kv_quant_bits = int(slm_kv_quant_bits or 0)
        self.slm_kv_backend = str(slm_kv_backend or "quanto")
        self.slm_kv_group_size = int(slm_kv_group_size)
        self.slm_kv_residual_length = int(slm_kv_residual_length)
        self.slm_kv_axis_key = int(slm_kv_axis_key)
        self.slm_kv_axis_value = int(slm_kv_axis_value)
        if self.slm_kv_quant_bits and self.slm_kv_backend.lower() == "quanto" and self.slm_kv_quant_bits not in (2, 4):
            raise ValueError(
                f"slm_kv_backend='quanto' supports only 2- or 4-bit KV (got {self.slm_kv_quant_bits}); "
                "use slm_kv_backend='hqq' for 8-bit."
            )
        self._decode_gen = None
        validate_fusion_tokenizers(slm_tokenizer, lm_tokenizer)
        self._attention_rule_modes = set(CANONICAL_RULE_FUSION_MODES)
        self._needs_attention = (self.fusion_mode in self._attention_rule_modes) or (
            self.learned_gate is not None
            and self.learned_gate_type in {
                "attn_ratio_and_logit_topk",
                "attn_linear_weighted",
                "attn_stats_mlp_unweighted",
                "attn_ratio_mlp_weighted",
            }
        )
        if self.learned_gate is not None:
            self.learned_gate.eval()
        if self.fusion_mode in {"contrastive", "contrastive_prob"} and self.learned_gate is not None and self.learned_gate_type != "logit_topk":
            raise ValueError("Contrastive fusion currently supports only learned_gate_type='logit_topk'.")

    def build_slm_kv_quant_cache(self):
        """Build a fresh HF ``QuantizedCache`` for the SLM/context branch prefill, or
        ``None`` when KV quantization is disabled (``slm_kv_quant_bits``==0).

        Mirrors ``run_kv_quant_teacher_experiments._build_quantized_cache``. Only the
        SLM/context branch uses this; the LM/query branch is left at full precision.
        """
        if not self.slm_kv_quant_bits:
            return None
        from transformers.cache_utils import QuantizedCache

        try:
            return QuantizedCache(
                backend=self.slm_kv_backend.lower(),
                config=self.slm_model.config,
                nbits=int(self.slm_kv_quant_bits),
                axis_key=int(self.slm_kv_axis_key),
                axis_value=int(self.slm_kv_axis_value),
                q_group_size=int(self.slm_kv_group_size),
                residual_length=int(self.slm_kv_residual_length),
            )
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                f"SLM KV-quant backend={self.slm_kv_backend!r} is not installed in this venv "
                f"({exc}). Install it, or pass --slm-kv-backend hqq (hqq is available and "
                "supports 4/8-bit)."
            ) from exc

    def _learned_lambda(
        self,
        slm_logits: torch.Tensor,
        lm_logits: torch.Tensor,
        *,
        slm_base_logits: Optional[torch.Tensor] = None,
        slm_attentions=None,
        context_span: Optional[tuple[int, int]] = None,
        kv_len: Optional[int] = None,
    ) -> torch.Tensor:
        if self.learned_gate is None:
            raise RuntimeError("learned_gate is not initialized.")
        gate_device = next(self.learned_gate.parameters()).device
        with torch.no_grad():
            if self.learned_gate_type == "logit_topk":
                gate_slm_logits = slm_logits[0]
                if self.fusion_mode in {"contrastive", "contrastive_prob"}:
                    if slm_base_logits is None:
                        raise RuntimeError("slm_base_logits is required for contrastive learned gating.")
                    gate_slm_logits = gate_slm_logits - slm_base_logits[0]
                feats = build_gate_features(gate_slm_logits, lm_logits[0], gate_top_k=self.learned_gate_top_k).unsqueeze(0)
                lambda_w = self.learned_gate(feats.to(gate_device)).view(1)
            elif self.learned_gate_type == "attn_ratio_and_logit_topk":
                if slm_attentions is None or kv_len is None:
                    raise RuntimeError("Attention features are required for learned_gate_type='attn_ratio_and_logit_topk'.")
                ctx_heads, shr_heads = extract_attention_head_masses(
                    slm_attentions,
                    context_span=context_span,
                    kv_len=int(kv_len),
                )
                ctx_ratio = attention_ctx_ratio(ctx_heads, shr_heads)
                feats = build_attn_ratio_and_logit_topk_features(
                    slm_logits[0],
                    lm_logits[0],
                    gate_top_k=self.learned_gate_top_k,
                    ctx_attn_ratio=ctx_ratio,
                ).unsqueeze(0)
                lambda_w = self.learned_gate(feats.to(gate_device)).view(1)
            elif self.learned_gate_type == "attn_linear_weighted":
                if slm_attentions is None or kv_len is None:
                    raise RuntimeError("Attention features are required for learned_gate_type='attn_linear_weighted'.")
                ctx_heads, shr_heads = extract_attention_head_masses(
                    slm_attentions,
                    context_span=context_span,
                    kv_len=int(kv_len),
                )
                lambda_w = self.learned_gate(
                    ctx_heads.unsqueeze(0).to(gate_device),
                    shr_heads.unsqueeze(0).to(gate_device),
                ).view(1)
            elif self.learned_gate_type == "attn_stats_mlp_unweighted":
                if slm_attentions is None or kv_len is None:
                    raise RuntimeError("Attention features are required for learned_gate_type='attn_stats_mlp_unweighted'.")
                ctx_heads, shr_heads = extract_attention_head_masses(
                    slm_attentions,
                    context_span=context_span,
                    kv_len=int(kv_len),
                )
                feats = build_attention_stat_features(ctx_heads, shr_heads).unsqueeze(0)
                lambda_w = self.learned_gate(feats.to(gate_device)).view(1)
            elif self.learned_gate_type == "attn_ratio_mlp_weighted":
                if slm_attentions is None or kv_len is None:
                    raise RuntimeError("Attention features are required for learned_gate_type='attn_ratio_mlp_weighted'.")
                ctx_heads, shr_heads = extract_attention_head_masses(
                    slm_attentions,
                    context_span=context_span,
                    kv_len=int(kv_len),
                )
                lambda_w = self.learned_gate(
                    ctx_heads.unsqueeze(0).to(gate_device),
                    shr_heads.unsqueeze(0).to(gate_device),
                ).view(1)
            else:
                raise ValueError(f"Unknown learned_gate_type: {self.learned_gate_type}")
        return torch.clamp(lambda_w.to(lm_logits.device), 0.0, 1.0)

    def _attention_rule_lambda(self, slm_attentions, context_span: Optional[tuple[int, int]], kv_len: int, device) -> torch.Tensor:
        ctx_heads, shr_heads = extract_attention_head_masses(
            slm_attentions,
            context_span=context_span,
            kv_len=int(kv_len),
        )
        if self.fusion_mode == "rule_attn_ctx_ratio":
            ratio = attention_ctx_ratio(ctx_heads, shr_heads)
            z = self.attn_rule_scale * (ratio - 0.5) + self.attn_rule_bias
        elif self.fusion_mode == "rule_attn_ctx_minus_shr":
            ctx = ctx_heads.mean()
            shr = shr_heads.mean()
            z = self.attn_rule_scale * (ctx - self.attn_rule_shared_scale * shr) + self.attn_rule_bias
        elif self.fusion_mode == "rule_attn_top_head_ratio":
            ratio_lh = (ctx_heads / (ctx_heads + shr_heads + 1e-6)).reshape(-1)
            k = max(1, int(float(self.attn_rule_top_head_frac) * ratio_lh.numel()))
            top_vals, _ = torch.topk(ratio_lh, k=k)
            top_ratio = top_vals.mean()
            z = self.attn_rule_scale * (top_ratio - 0.5) + self.attn_rule_bias
        else:
            raise ValueError(f"Unsupported attention rule fusion mode: {self.fusion_mode}")
        lam = torch.sigmoid(z).to(dtype=torch.float32)
        return torch.clamp(lam, 0.0, 1.0).to(device=device).view(1)

    @staticmethod
    def _mu_of_lambda(log_p_l: torch.Tensor, s: torch.Tensor, lam: float) -> float:
        # q_lambda(v) ∝ p_L(v) exp(lambda * s(v))  => log q up to constant: log_p_l + lambda * s
        logits = log_p_l + float(lam) * s
        q = torch.softmax(logits, dim=-1)
        return float((q * s).sum().item())

    def _fixed_mu_support_score(self, slm_logits: torch.Tensor, lm_logits: torch.Tensor) -> torch.Tensor:
        s_type = self.fixed_mu_score_type
        if s_type == "slm_logit":
            return slm_logits[0]
        if s_type == "slm_prob":
            return torch.softmax(slm_logits[0], dim=-1)
        if s_type == "prob_diff":
            return torch.softmax(slm_logits[0], dim=-1) - torch.softmax(lm_logits[0], dim=-1)
        if s_type == "logit_diff":
            return slm_logits[0] - lm_logits[0]
        if s_type == "rank_exp":
            # rank 0 for the highest-logit token.
            sorted_idx = torch.argsort(slm_logits[0], dim=-1, descending=True)
            ranks = torch.empty_like(sorted_idx, dtype=slm_logits.dtype)
            ranks[sorted_idx] = torch.arange(sorted_idx.shape[0], device=sorted_idx.device, dtype=slm_logits.dtype)
            k = max(float(self.fixed_mu_rank_k), 1e-6)
            return torch.exp(-ranks / k)
        raise ValueError(f"Unknown fixed_mu_score_type: {s_type}")

    def _solve_lambda_fixed_mu(self, slm_logits: torch.Tensor, lm_logits: torch.Tensor) -> float:
        if self.fixed_mu_target is None:
            raise ValueError("fixed_mu_target must be set when fusion_mode='fixed_mu'.")
        log_p_l = torch.log_softmax(lm_logits[0], dim=-1)
        s = self._fixed_mu_support_score(slm_logits, lm_logits)
        lo = float(self.fixed_mu_lambda_min)
        hi = float(self.fixed_mu_lambda_max)
        steps = max(1, int(self.fixed_mu_steps))
        target = float(self.fixed_mu_target)

        mu_lo = self._mu_of_lambda(log_p_l, s, lo)
        mu_hi = self._mu_of_lambda(log_p_l, s, hi)
        if target <= mu_lo:
            return lo
        if target >= mu_hi:
            return hi

        for _ in range(steps):
            mid = 0.5 * (lo + hi)
            mu_mid = self._mu_of_lambda(log_p_l, s, mid)
            if mu_mid < target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def _prompt(self, example: Any, include_docs: bool, tokenizer: PreTrainedTokenizerBase) -> str:
        if self.prompt_builder:
            return self.prompt_builder(example, include_docs, tokenizer)
        return example.prompt(include_docs=include_docs)

    def _apply_stop_strings(self, text: str) -> tuple[str, bool]:
        if not self.stop_strings:
            return text, False
        earliest = None
        for stop in self.stop_strings:
            if not stop:
                continue
            idx = text.find(stop)
            if idx != -1 and (earliest is None or idx < earliest):
                earliest = idx
        if earliest is None:
            return text, False
        return text[:earliest], True

    def _apply_stop_on_repeat(self, text: str) -> tuple[str, bool]:
        """Opt-in: once ``stop_on_repeat`` appears a SECOND time, truncate before the 2nd
        occurrence. Kills the manual-greedy 'Final Answer: X Final Answer: X ...' loop.
        Disabled (no-op) when ``stop_on_repeat`` is unset/empty."""
        marker = self.stop_on_repeat
        if not marker or text.count(marker) < 2:
            return text, False
        second = text.find(marker, text.find(marker) + 1)
        return text[:second], True

    def _get_device(self, model: PreTrainedModel) -> torch.device:
        return get_model_input_device(model)

    @staticmethod
    def _contrastive_prob_fusion(
        slm_logits: torch.Tensor,
        slm_base_logits: torch.Tensor,
        lm_logits: torch.Tensor,
        lambda_w: torch.Tensor | float,
    ) -> torch.Tensor:
        lm_probs = torch.softmax(lm_logits, dim=-1)
        slm_probs = torch.softmax(slm_logits, dim=-1)
        slm_base_probs = torch.softmax(slm_base_logits, dim=-1)
        fused_probs = lm_probs + lambda_w * (slm_probs - slm_base_probs)
        fused_probs = torch.clamp(fused_probs, min=1e-12)
        fused_probs = fused_probs / fused_probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        return torch.log(fused_probs)

    @staticmethod
    def _prob_mix_fusion(slm_logits, lm_logits, lambda_w):
        """Scale-robust PROBABILITY-MIXTURE fusion: lambda*softmax(SLM) + (1-lambda)*softmax(LM).
        Each model contributes a bounded distribution, so a large-magnitude LM (e.g. 72B, whose raw
        logits have bigger spread than a 7B) cannot dominate the small SLM the way raw-logit
        weighted_sum lets it (the 72B-fusion magnitude bug). Returns log(mixture) so the downstream
        argmax/sampling is unchanged. NOTE: this is a DIFFERENT fusion from weighted_sum (nonlinear),
        so it is a separate opt-in mode and must be re-validated; it does not replace weighted_sum."""
        p = lambda_w * torch.softmax(slm_logits, dim=-1) + (1 - lambda_w) * torch.softmax(lm_logits, dim=-1)
        return torch.log(p.clamp_min(1e-12))

    def decode(self, example: Any) -> FusionResult:
        needs_slm_base = self.fusion_mode in {"contrastive", "contrastive_prob"}
        slm_prompt = self._prompt(example, include_docs=True, tokenizer=self.slm_tokenizer)
        lm_prompt = self._prompt(example, include_docs=False, tokenizer=self.lm_tokenizer)
        slm_base_prompt = (
            self._prompt(example, include_docs=False, tokenizer=self.slm_base_tokenizer) if needs_slm_base else None
        )

        slm_inputs = prepare_inputs(self.slm_tokenizer, slm_prompt, max_length=self.max_length, return_tensors="pt")
        lm_inputs = prepare_inputs(self.lm_tokenizer, lm_prompt, max_length=self.max_length, return_tensors="pt")
        slm_base_inputs = (
            prepare_inputs(self.slm_base_tokenizer, slm_base_prompt, max_length=self.max_length, return_tensors="pt")
            if slm_base_prompt is not None
            else None
        )

        slm_ids = slm_inputs["input_ids"].to(self._get_device(self.slm_model))
        slm_mask = slm_inputs["attention_mask"].to(self._get_device(self.slm_model))
        lm_ids = lm_inputs["input_ids"].to(self._get_device(self.lm_model))
        lm_mask = lm_inputs["attention_mask"].to(self._get_device(self.lm_model))
        slm_base_ids = (
            slm_base_inputs["input_ids"].to(self._get_device(self.slm_base_model)) if slm_base_inputs is not None else None
        )
        slm_base_mask = (
            slm_base_inputs["attention_mask"].to(self._get_device(self.slm_base_model))
            if slm_base_inputs is not None
            else None
        )
        context_span = infer_context_token_span(slm_prompt, self.slm_tokenizer, int(slm_ids.shape[1]))

        generated_ids: list[int] = []
        text_parts: list[str] = []
        current_text = ""  # pre-bound so _consume_one's `nonlocal current_text` has an enclosing binding
        eos_ids = get_eos_token_ids(self.lm_tokenizer, self.lm_model)
        terminated = False
        lambda_values: list[float] = []

        slm_device = slm_ids.device
        lm_device = lm_ids.device
        slm_base_device = slm_base_ids.device if slm_base_ids is not None else None
        can_parallel = (
            not needs_slm_base
            and torch.cuda.is_available()
            and slm_device.type == "cuda"
            and lm_device.type == "cuda"
            and slm_device != lm_device
        )

        def _sync() -> None:
            synchronize_model(self.slm_model)
            if needs_slm_base and self.slm_base_model is not self.slm_model:
                synchronize_model(self.slm_base_model)
            if self.lm_model is not self.slm_model and self.lm_model is not self.slm_base_model:
                synchronize_model(self.lm_model)

        # Prefill to build caches and get the first logits.
        _sync()
        prefill_start = time.perf_counter()
        # Build the (optional) quantized KV cache for ONLY the SLM/context branch.
        # When None (default), the calls below are byte-identical to the original code.
        slm_kv_cache = self.build_slm_kv_quant_cache()
        with torch.no_grad():
            if self._needs_attention and int(slm_ids.shape[1]) > 1:
                slm_prefix_ids = slm_ids[:, :-1]
                slm_prefix_mask = slm_mask[:, :-1]
                slm_last_ids = slm_ids[:, -1:]
                slm_prefix_kwargs = dict(
                    input_ids=slm_prefix_ids,
                    attention_mask=slm_prefix_mask,
                    use_cache=True,
                    output_attentions=False,
                    logits_to_keep=1,
                )
                if slm_kv_cache is not None:
                    slm_prefix_kwargs["past_key_values"] = slm_kv_cache
                slm_prefix_outputs = self.slm_model(**slm_prefix_kwargs)
                slm_outputs = self.slm_model(
                    input_ids=slm_last_ids,
                    attention_mask=slm_mask,
                    past_key_values=slm_prefix_outputs.past_key_values,
                    use_cache=True,
                    output_attentions=True,
                    logits_to_keep=1,
                )
            else:
                slm_plain_kwargs = dict(
                    input_ids=slm_ids,
                    attention_mask=slm_mask,
                    use_cache=True,
                    output_attentions=self._needs_attention,
                    logits_to_keep=1,
                )
                if slm_kv_cache is not None:
                    slm_plain_kwargs["past_key_values"] = slm_kv_cache
                slm_outputs = self.slm_model(**slm_plain_kwargs)
            slm_base_outputs = (
                self.slm_base_model(
                    input_ids=slm_base_ids,
                    attention_mask=slm_base_mask,
                    use_cache=True,
                    output_attentions=False,
                    logits_to_keep=1,
                )
                if slm_base_ids is not None and slm_base_mask is not None
                else None
            )
            lm_outputs = self.lm_model(input_ids=lm_ids, attention_mask=lm_mask, use_cache=True, logits_to_keep=1)
        _sync()
        prefill_time = time.perf_counter() - prefill_start
        slm_past = slm_outputs.past_key_values
        slm_base_past = slm_base_outputs.past_key_values if slm_base_outputs is not None else None
        lm_past = lm_outputs.past_key_values

        slm_stream = torch.cuda.Stream(device=slm_device) if can_parallel else None
        lm_stream = torch.cuda.Stream(device=lm_device) if can_parallel else None

        decode_gen = (
            torch.Generator(device=lm_device).manual_seed(int(self.decode_seed))
            if self.decode_temperature > 0
            else None
        )
        decode_start = time.perf_counter()
        answer_seen_at: Optional[int] = None  # token index when stop_after_answer marker completed

        def _consume_one(token_id: int) -> bool:
            """Append one generated token + run all stop checks. Returns True if generation should stop.
            Identical logic to the original inline per-token block; shared by the K=1 path and the
            K>1 batched-flush path so the two cannot diverge. eos is folded in (output-equivalent to
            the original post-append eos check)."""
            nonlocal current_text, answer_seen_at, terminated, text_parts
            generated_ids.append(token_id)
            text_parts.append(self.lm_tokenizer.decode([token_id], skip_special_tokens=False))
            current_text = "".join(text_parts)
            current_text, _stopped = self._apply_stop_strings(current_text)
            if _stopped:
                terminated = True; text_parts = [current_text]; return True
            current_text, _stopped_repeat = self._apply_stop_on_repeat(current_text)
            if _stopped_repeat:
                terminated = True; text_parts = [current_text]; return True
            if self.stop_after_answer:
                if answer_seen_at is None and self.stop_after_answer in current_text:
                    answer_seen_at = len(generated_ids)
                if answer_seen_at is not None:
                    mi = current_text.find(self.stop_after_answer)
                    nl = current_text.find("\n", mi + len(self.stop_after_answer))
                    if nl != -1:
                        current_text = current_text[:nl]; terminated = True; text_parts = [current_text]; return True
                    if len(generated_ids) - answer_seen_at >= self.answer_token_budget:
                        terminated = True; text_parts = [current_text]; return True
            if token_id in eos_ids:
                terminated = True; return True
            return False

        # decode_sync_interval>1 defers the GPU->CPU .item() sync + the per-token consume to every K tokens,
        # so the GPU runs forwards back-to-back instead of stalling on a sync each step. K=1 = exact original.
        _K = self.decode_sync_interval
        _buffered: list = []
        for _ in range(self.max_new_tokens):

            slm_logits = slm_outputs.logits[:, -1, :]
            slm_base_logits = slm_base_outputs.logits[:, -1, :] if slm_base_outputs is not None else None
            lm_logits = lm_outputs.logits[:, -1, :]
            # Align vocab sizes by truncating to the shared prefix.
            vocab_size = min(
                [slm_logits.shape[-1], lm_logits.shape[-1]]
                + ([slm_base_logits.shape[-1]] if slm_base_logits is not None else [])
            )
            slm_logits = slm_logits[..., :vocab_size]
            lm_logits = lm_logits[..., :vocab_size]
            if slm_base_logits is not None:
                slm_base_logits = slm_base_logits[..., :vocab_size]
            if slm_logits.device != lm_logits.device:
                slm_logits = slm_logits.to(lm_logits.device)
            if slm_base_logits is not None and slm_base_logits.device != lm_logits.device:
                slm_base_logits = slm_base_logits.to(lm_logits.device)
            if self.learned_gate is not None:
                lambda_w = self._learned_lambda(
                    slm_logits,
                    lm_logits,
                    slm_base_logits=slm_base_logits,
                    slm_attentions=slm_outputs.attentions if self._needs_attention else None,
                    context_span=context_span,
                    kv_len=int(slm_ids.shape[1]),
                )
                if self.fusion_mode == "contrastive_prob":
                    if slm_base_logits is None:
                        raise RuntimeError("slm_base_logits is required for contrastive_prob fusion.")
                    fused_logits = self._contrastive_prob_fusion(slm_logits, slm_base_logits, lm_logits, lambda_w)
                elif self.fusion_mode == "contrastive":
                    if slm_base_logits is None:
                        raise RuntimeError("slm_base_logits is required for contrastive fusion.")
                    fused_logits = lm_logits + lambda_w * (slm_logits - slm_base_logits)
                elif self.fusion_mode == "prob_mix":
                    fused_logits = self._prob_mix_fusion(slm_logits, lm_logits, lambda_w)
                else:
                    fused_logits = lambda_w * slm_logits + (1 - lambda_w) * lm_logits
            elif self.fusion_mode in self._attention_rule_modes:
                lambda_w = self._attention_rule_lambda(
                    slm_outputs.attentions,
                    context_span=context_span,
                    kv_len=int(slm_ids.shape[1]),
                    device=lm_logits.device,
                )
                fused_logits = lambda_w * slm_logits + (1 - lambda_w) * lm_logits
            elif self.fusion_mode == "fixed_mu":
                lam = self._solve_lambda_fixed_mu(slm_logits, lm_logits)
                lambda_values.append(float(lam))
                lambda_w = torch.tensor([lam], device=lm_logits.device, dtype=lm_logits.dtype)
                fused_logits = torch.log_softmax(lm_logits, dim=-1) + lam * slm_logits
            elif self.fusion_mode == "max":
                fused_logits = torch.maximum(slm_logits, lm_logits)
            elif self.fusion_mode == "entropy":
                if self.entropy_mode == "topk":
                    slm_entropy = self._entropy_topk_raw(slm_logits, self.top_k_entropy)
                    lm_entropy = self._entropy_topk_raw(lm_logits, self.top_k_entropy)
                elif self.entropy_mode == "topk_norm":
                    slm_entropy = self._entropy_topk_norm(slm_logits, self.top_k_entropy)
                    lm_entropy = self._entropy_topk_norm(lm_logits, self.top_k_entropy)
                else:
                    slm_probs = torch.softmax(slm_logits, dim=-1)
                    lm_probs = torch.softmax(lm_logits, dim=-1)
                    slm_entropy = self._entropy_from_threshold(slm_probs, self.entropy_threshold)
                    lm_entropy = self._entropy_from_threshold(lm_probs, self.entropy_threshold)
                gate = torch.sigmoid(lm_entropy - slm_entropy)  # >0.5 -> favor SLM when it is more confident
                lambda_w = self.entropy_scale * (gate - 0.5) + 0.5
                lambda_w = torch.clamp(lambda_w, 0.0, 1.0)
                fused_logits = lambda_w * slm_logits + (1 - lambda_w) * lm_logits
            elif self.fusion_mode == "contrastive":
                if slm_base_logits is None:
                    raise RuntimeError("slm_base_logits is required for contrastive fusion.")
                fused_logits = lm_logits + self.lambda_weight * (slm_logits - slm_base_logits)
            elif self.fusion_mode == "contrastive_prob":
                if slm_base_logits is None:
                    raise RuntimeError("slm_base_logits is required for contrastive_prob fusion.")
                fused_logits = self._contrastive_prob_fusion(slm_logits, slm_base_logits, lm_logits, self.lambda_weight)
            else:
                fused_logits = self.lambda_weight * slm_logits + (1 - self.lambda_weight) * lm_logits
            next_token = _pick_next_token(fused_logits, self.decode_temperature, self.decode_top_p, decode_gen)
            # CONSUME — K=1: process now (exact original behavior, byte-identical, break-before-advance on stop).
            # K>1: buffer the GPU token tensor; the consume + .item() sync is batched at the flush below.
            if _K == 1:
                if _consume_one(int(next_token[0].item())):
                    break
            else:
                _buffered.append(next_token)

            next_token_tensor = next_token.unsqueeze(0)
            slm_ids = torch.cat([slm_ids, next_token_tensor.to(slm_ids.device)], dim=1)
            lm_ids = torch.cat([lm_ids, next_token_tensor.to(lm_ids.device)], dim=1)
            if slm_base_ids is not None:
                slm_base_ids = torch.cat([slm_base_ids, next_token_tensor.to(slm_base_ids.device)], dim=1)
            ones_s = torch.ones_like(next_token_tensor, device=slm_mask.device, dtype=slm_mask.dtype)
            ones_l = torch.ones_like(next_token_tensor, device=lm_mask.device, dtype=lm_mask.dtype)
            slm_mask = torch.cat([slm_mask, ones_s], dim=1)
            lm_mask = torch.cat([lm_mask, ones_l], dim=1)
            if slm_base_mask is not None:
                ones_sb = torch.ones_like(next_token_tensor, device=slm_base_mask.device, dtype=slm_base_mask.dtype)
                slm_base_mask = torch.cat([slm_base_mask, ones_sb], dim=1)

            if can_parallel:
                with torch.no_grad():
                    with torch.cuda.stream(slm_stream):
                        slm_outputs = self.slm_model(
                            input_ids=next_token_tensor.to(slm_device),
                            attention_mask=slm_mask,
                            past_key_values=slm_past,
                            use_cache=True,
                            output_attentions=self._needs_attention,
                        )
                    with torch.cuda.stream(lm_stream):
                        lm_outputs = self.lm_model(
                            input_ids=next_token_tensor.to(lm_device),
                            attention_mask=lm_mask,
                            past_key_values=lm_past,
                            use_cache=True,
                        )
                torch.cuda.synchronize(slm_device)
                torch.cuda.synchronize(lm_device)
            else:
                with torch.no_grad():
                    slm_outputs = self.slm_model(
                        input_ids=next_token_tensor.to(slm_device),
                        attention_mask=slm_mask,
                        past_key_values=slm_past,
                        use_cache=True,
                        output_attentions=self._needs_attention,
                    )
                    if needs_slm_base:
                        if slm_base_device is None or slm_base_mask is None or slm_base_past is None:
                            raise RuntimeError("SLM base state is required for contrastive decoding.")
                        slm_base_outputs = self.slm_base_model(
                            input_ids=next_token_tensor.to(slm_base_device),
                            attention_mask=slm_base_mask,
                            past_key_values=slm_base_past,
                            use_cache=True,
                            output_attentions=False,
                        )
                    lm_outputs = self.lm_model(
                        input_ids=next_token_tensor.to(lm_device),
                        attention_mask=lm_mask,
                        past_key_values=lm_past,
                        use_cache=True,
                    )

            slm_past = slm_outputs.past_key_values
            slm_base_past = slm_base_outputs.past_key_values if slm_base_outputs is not None else None
            lm_past = lm_outputs.past_key_values

            # K>1: batched flush — one .item() sync for the whole buffer, then consume in order.
            # A stop mid-buffer truncates exactly at that token (later buffered tokens are discarded);
            # output is byte-identical to K=1, only the sync/CPU work is amortized across K steps.
            if _K > 1 and len(_buffered) >= _K:
                _stop = False
                for _nt in _buffered:
                    if _consume_one(int(_nt[0].item())):
                        _stop = True
                        break
                _buffered = []
                if _stop:
                    break

        # K>1: consume any tail tokens left when the loop hit max_new_tokens without a flush.
        if _K > 1 and _buffered:
            for _nt in _buffered:
                if _consume_one(int(_nt[0].item())):
                    break

        _sync()
        decode_time = time.perf_counter() - decode_start
        timing = GenerationTiming(
            prefill_s=prefill_time,
            decode_s=decode_time,
            total_s=prefill_time + decode_time,
        )
        lambda_stats = None
        if lambda_values:
            lambda_stats = {
                "avg": float(sum(lambda_values) / len(lambda_values)),
                "min": float(min(lambda_values)),
                "max": float(max(lambda_values)),
                "steps": float(len(lambda_values)),
            }
        # ALWAYS reconstruct the final text by full-decoding the accumulated ids — correct for EVERY tokenizer.
        # (SentencePiece/Llama lose leading spaces under per-token decode+join, e.g. "FinalAnswer:"; byte-level BPE
        # like Qwen are unaffected, so full-decode == join for them → no-op. The old class-name gate silently failed
        # in the fusion path even for LlamaTokenizerFast, hence unconditional.) Then re-apply ALL stop rules on the
        # clean text so the space-loss can no longer hide the "Final Answer:" marker from stop/extraction.
        text = self.lm_tokenizer.decode(generated_ids, skip_special_tokens=True) if generated_ids else "".join(text_parts)
        text, _ = self._apply_stop_strings(text)
        text, _ = self._apply_stop_on_repeat(text)
        if self.stop_after_answer:
            mi = text.find(self.stop_after_answer)
            if mi != -1:
                nl = text.find("\n", mi + len(self.stop_after_answer))
                if nl != -1:
                    text = text[:nl]
        return FusionResult(
            text=text,
            generated_token_ids=generated_ids,
            terminated_early=terminated,
            timing=timing,
            lambda_stats=lambda_stats,
        )

    def decode_with_trace(self, example: Any) -> tuple[FusionResult, list[FusionTraceStep]]:
        if self.fusion_mode in {"contrastive", "contrastive_prob"}:
            return self.decode(example), []
        slm_prompt = self._prompt(example, include_docs=True, tokenizer=self.slm_tokenizer)
        lm_prompt = self._prompt(example, include_docs=False, tokenizer=self.lm_tokenizer)

        slm_inputs = prepare_inputs(self.slm_tokenizer, slm_prompt, max_length=self.max_length, return_tensors="pt")
        lm_inputs = prepare_inputs(self.lm_tokenizer, lm_prompt, max_length=self.max_length, return_tensors="pt")

        slm_ids = slm_inputs["input_ids"].to(self._get_device(self.slm_model))
        slm_mask = slm_inputs["attention_mask"].to(self._get_device(self.slm_model))
        lm_ids = lm_inputs["input_ids"].to(self._get_device(self.lm_model))
        lm_mask = lm_inputs["attention_mask"].to(self._get_device(self.lm_model))
        context_span = infer_context_token_span(slm_prompt, self.slm_tokenizer, int(slm_ids.shape[1]))

        generated_ids: list[int] = []
        text_parts: list[str] = []
        trace: list[FusionTraceStep] = []
        eos_ids = get_eos_token_ids(self.lm_tokenizer, self.lm_model)
        terminated = False

        slm_device = slm_ids.device
        lm_device = lm_ids.device
        can_parallel = (
            torch.cuda.is_available()
            and slm_device.type == "cuda"
            and lm_device.type == "cuda"
            and slm_device != lm_device
        )

        with torch.no_grad():
            if self._needs_attention and int(slm_ids.shape[1]) > 1:
                slm_prefix_ids = slm_ids[:, :-1]
                slm_prefix_mask = slm_mask[:, :-1]
                slm_last_ids = slm_ids[:, -1:]
                slm_prefix_outputs = self.slm_model(
                    input_ids=slm_prefix_ids,
                    attention_mask=slm_prefix_mask,
                    use_cache=True,
                    output_attentions=False,
                )
                slm_outputs = self.slm_model(
                    input_ids=slm_last_ids,
                    attention_mask=slm_mask,
                    past_key_values=slm_prefix_outputs.past_key_values,
                    use_cache=True,
                    output_attentions=True,
                )
            else:
                slm_outputs = self.slm_model(
                    input_ids=slm_ids,
                    attention_mask=slm_mask,
                    use_cache=True,
                    output_attentions=self._needs_attention,
                )
            lm_outputs = self.lm_model(input_ids=lm_ids, attention_mask=lm_mask, use_cache=True)
        slm_past = slm_outputs.past_key_values
        lm_past = lm_outputs.past_key_values

        slm_stream = torch.cuda.Stream(device=slm_device) if can_parallel else None
        lm_stream = torch.cuda.Stream(device=lm_device) if can_parallel else None

        decode_gen = (
            torch.Generator(device=lm_device).manual_seed(int(self.decode_seed))
            if self.decode_temperature > 0
            else None
        )
        for position in range(self.max_new_tokens):
            slm_logits = slm_outputs.logits[:, -1, :]
            lm_logits = lm_outputs.logits[:, -1, :]
            vocab_size = min(slm_logits.shape[-1], lm_logits.shape[-1])
            slm_logits = slm_logits[..., :vocab_size]
            lm_logits = lm_logits[..., :vocab_size]
            if slm_logits.device != lm_logits.device:
                slm_logits = slm_logits.to(lm_logits.device)

            slm_entropy = None
            lm_entropy = None
            lambda_w = None

            if self.learned_gate is not None:
                lambda_w = self._learned_lambda(
                    slm_logits,
                    lm_logits,
                    slm_attentions=slm_outputs.attentions if self._needs_attention else None,
                    context_span=context_span,
                    kv_len=int(slm_ids.shape[1]),
                )
                fused_logits = lambda_w * slm_logits + (1 - lambda_w) * lm_logits
            elif self.fusion_mode in self._attention_rule_modes:
                lambda_w = self._attention_rule_lambda(
                    slm_outputs.attentions,
                    context_span=context_span,
                    kv_len=int(slm_ids.shape[1]),
                    device=lm_logits.device,
                )
                fused_logits = lambda_w * slm_logits + (1 - lambda_w) * lm_logits
            elif self.fusion_mode == "fixed_mu":
                lam = self._solve_lambda_fixed_mu(slm_logits, lm_logits)
                lambda_w = torch.tensor([lam], device=lm_logits.device, dtype=lm_logits.dtype)
                fused_logits = torch.log_softmax(lm_logits, dim=-1) + lam * slm_logits
            elif self.fusion_mode == "max":
                fused_logits = torch.maximum(slm_logits, lm_logits)
            elif self.fusion_mode == "entropy":
                if self.entropy_mode == "topk":
                    slm_entropy = self._entropy_topk_raw(slm_logits, self.top_k_entropy)
                    lm_entropy = self._entropy_topk_raw(lm_logits, self.top_k_entropy)
                elif self.entropy_mode == "topk_norm":
                    slm_entropy = self._entropy_topk_norm(slm_logits, self.top_k_entropy)
                    lm_entropy = self._entropy_topk_norm(lm_logits, self.top_k_entropy)
                else:
                    slm_probs = torch.softmax(slm_logits, dim=-1)
                    lm_probs = torch.softmax(lm_logits, dim=-1)
                    slm_entropy = self._entropy_from_threshold(slm_probs, self.entropy_threshold)
                    lm_entropy = self._entropy_from_threshold(lm_probs, self.entropy_threshold)
                gate = torch.sigmoid(lm_entropy - slm_entropy)
                lambda_w = self.entropy_scale * (gate - 0.5) + 0.5
                lambda_w = torch.clamp(lambda_w, 0.0, 1.0)
                fused_logits = lambda_w * slm_logits + (1 - lambda_w) * lm_logits
            else:
                fused_logits = self.lambda_weight * slm_logits + (1 - self.lambda_weight) * lm_logits

            next_token = _pick_next_token(fused_logits, self.decode_temperature, self.decode_top_p, decode_gen)
            token_id = int(next_token[0].item())
            generated_ids.append(token_id)
            text_parts.append(self.lm_tokenizer.decode([token_id], skip_special_tokens=False))
            current_text = "".join(text_parts)
            current_text, stopped = self._apply_stop_strings(current_text)
            if stopped:
                terminated = True
                text_parts = [current_text]
                break
            current_text, stopped_repeat = self._apply_stop_on_repeat(current_text)
            if stopped_repeat:
                terminated = True
                text_parts = [current_text]
                break
            trace.append(
                FusionTraceStep(
                    position=position,
                    token_id=token_id,
                    token=self.lm_tokenizer.convert_ids_to_tokens(token_id),
                    slm_entropy=float(slm_entropy.item()) if slm_entropy is not None else None,
                    lm_entropy=float(lm_entropy.item()) if lm_entropy is not None else None,
                    lambda_w=float(lambda_w.item()) if lambda_w is not None else None,
                )
            )

            next_token_tensor = next_token.unsqueeze(0)
            slm_ids = torch.cat([slm_ids, next_token_tensor.to(slm_ids.device)], dim=1)
            lm_ids = torch.cat([lm_ids, next_token_tensor.to(lm_ids.device)], dim=1)
            ones_s = torch.ones_like(next_token_tensor, device=slm_mask.device, dtype=slm_mask.dtype)
            ones_l = torch.ones_like(next_token_tensor, device=lm_mask.device, dtype=lm_mask.dtype)
            slm_mask = torch.cat([slm_mask, ones_s], dim=1)
            lm_mask = torch.cat([lm_mask, ones_l], dim=1)

            if token_id in eos_ids:
                terminated = True
                break

            if can_parallel:
                with torch.no_grad():
                    with torch.cuda.stream(slm_stream):
                        slm_outputs = self.slm_model(
                            input_ids=next_token_tensor.to(slm_device),
                            attention_mask=slm_mask,
                            past_key_values=slm_past,
                            use_cache=True,
                            output_attentions=self._needs_attention,
                        )
                    with torch.cuda.stream(lm_stream):
                        lm_outputs = self.lm_model(
                            input_ids=next_token_tensor.to(lm_device),
                            attention_mask=lm_mask,
                            past_key_values=lm_past,
                            use_cache=True,
                        )
                torch.cuda.synchronize(slm_device)
                torch.cuda.synchronize(lm_device)
            else:
                with torch.no_grad():
                    slm_outputs = self.slm_model(
                        input_ids=next_token_tensor.to(slm_device),
                        attention_mask=slm_mask,
                        past_key_values=slm_past,
                        use_cache=True,
                        output_attentions=self._needs_attention,
                    )
                    lm_outputs = self.lm_model(
                        input_ids=next_token_tensor.to(lm_device),
                        attention_mask=lm_mask,
                        past_key_values=lm_past,
                        use_cache=True,
                    )

            slm_past = slm_outputs.past_key_values
            lm_past = lm_outputs.past_key_values

        # ALWAYS reconstruct the final text by full-decoding the accumulated ids — correct for EVERY tokenizer.
        # (SentencePiece/Llama lose leading spaces under per-token decode+join, e.g. "FinalAnswer:"; byte-level BPE
        # like Qwen are unaffected, so full-decode == join for them → no-op. The old class-name gate silently failed
        # in the fusion path even for LlamaTokenizerFast, hence unconditional.) Then re-apply ALL stop rules on the
        # clean text so the space-loss can no longer hide the "Final Answer:" marker from stop/extraction.
        text = self.lm_tokenizer.decode(generated_ids, skip_special_tokens=True) if generated_ids else "".join(text_parts)
        text, _ = self._apply_stop_strings(text)
        text, _ = self._apply_stop_on_repeat(text)
        if self.stop_after_answer:
            mi = text.find(self.stop_after_answer)
            if mi != -1:
                nl = text.find("\n", mi + len(self.stop_after_answer))
                if nl != -1:
                    text = text[:nl]
        return FusionResult(text=text, generated_token_ids=generated_ids, terminated_early=terminated), trace

    @staticmethod
    def _entropy_from_threshold(probs: torch.Tensor, threshold: float) -> torch.Tensor:
        if probs.numel() == 0:
            return torch.zeros(probs.shape[:-1], device=probs.device)
        threshold_tensor = torch.tensor(threshold, device=probs.device, dtype=probs.dtype)
        threshold_tensor = torch.clamp(threshold_tensor, 0.0, 1.0)
        probs_sorted, _ = torch.sort(probs, dim=-1, descending=True)
        cumulative = probs_sorted.cumsum(dim=-1)
        cutoff = (cumulative >= threshold_tensor).float().argmax(dim=-1)
        idx = torch.arange(probs_sorted.shape[-1], device=probs.device).unsqueeze(0)
        while idx.dim() < probs_sorted.dim():
            idx = idx.unsqueeze(0)
        mask = idx <= cutoff.unsqueeze(-1)
        selected = probs_sorted * mask
        denom = selected.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        norm = selected / denom
        entropy = -(norm * norm.clamp(min=1e-12).log()).sum(dim=-1)
        return entropy

    @staticmethod
    def _entropy_topk_raw(logits: torch.Tensor, k: int) -> torch.Tensor:
        logprobs = torch.log_softmax(logits, dim=-1)
        vals, _ = torch.topk(logprobs, k=k, dim=-1)
        probs = torch.exp(vals)
        return -(probs * vals).sum(dim=-1)

    @staticmethod
    def _entropy_topk_norm(logits: torch.Tensor, k: int) -> torch.Tensor:
        logprobs = torch.log_softmax(logits, dim=-1)
        vals, _ = torch.topk(logprobs, k=k, dim=-1)
        probs = torch.exp(vals)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        return -(probs * probs.clamp(min=1e-12).log()).sum(dim=-1)


def generate_with_single_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    *,
    max_new_tokens: int = 128,
    min_new_tokens: int = 0,
    max_length: int = 150000,
    use_generate: bool = True,
    return_timing: bool = False,
    stop_strings: Optional[list[str]] = None,
    stop_on_repeat: Optional[str] = None,
    attention_mask_override: Optional[torch.Tensor] = None,
    incremental_attention_mask_builder: Optional[Callable[[int, torch.device, torch.dtype], torch.Tensor]] = None,
    input_ids_override: Optional[torch.Tensor] = None,
    position_ids_override: Optional[torch.Tensor] = None,
    incremental_position_ids_builder: Optional[Callable[[int, torch.device], torch.Tensor]] = None,
    memory_trace: Optional[dict] = None,
    decode_temperature: float = 0.0,
    decode_top_p: float = 1.0,
    decode_seed: int = 0,
    kv_cache_factory: Optional[Callable[[], object]] = None,
    kvpress_press: Optional[Callable[[object], object]] = None,
    prebuilt_prefill: Optional[tuple] = None,
) -> str | tuple[str, GenerationTiming]:
    """Simple greedy generation baseline for either the SLM or LM alone.

    kv_cache_factory: optional callable returning a fresh KV cache (e.g. minference SnapKVCache)
    injected into the manual prefill so KV-compression engages in MANUAL decode (minference's
    compression is otherwise wired only into model.generate()). Default None = standard cache,
    byte-identical to before.
    kvpress_press: optional KVPress press object (official SnapKV/H2O/StreamingLLM/etc.); when given
    its context manager wraps the manual PREFILL forward so the press hooks compress the KV cache,
    then manual greedy decode continues with the compressed cache (fairness-compliant: same manual
    decode as ours). Default None = no compression, byte-identical to before."""
    input_device = get_model_input_device(model)
    if input_ids_override is not None:
        input_ids = input_ids_override.to(input_device)
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids_override must be rank-2 [batch, seq], got shape={tuple(input_ids.shape)}")
        base_attention_mask = torch.ones_like(input_ids, device=input_device, dtype=torch.long)
    else:
        inputs = prepare_inputs(tokenizer, prompt, max_length=max_length, return_tensors="pt")
        input_ids = inputs["input_ids"].to(input_device)
        base_attention_mask = inputs["attention_mask"].to(input_device)
    attention_mask = (
        attention_mask_override.to(input_device) if attention_mask_override is not None else base_attention_mask
    )
    position_ids = position_ids_override.to(input_device) if position_ids_override is not None else None
    if position_ids is not None and position_ids.shape != input_ids.shape:
        raise ValueError(
            f"position_ids_override must match input_ids shape={tuple(input_ids.shape)}, "
            f"got shape={tuple(position_ids.shape)}"
        )
    mask_dtype = next(model.parameters()).dtype
    if attention_mask_override is not None and attention_mask.dim() != 2 and use_generate:
        print("[Info] Falling back to manual decoding because block attention uses a custom 4D mask.")
        use_generate = False
    if (position_ids_override is not None or incremental_position_ids_builder is not None) and use_generate:
        print("[Info] Falling back to manual decoding because custom position ids require the manual path.")
        use_generate = False
    if use_generate:
        if memory_trace is not None:
            from .memory import reset_cuda_peak_memory, snapshot_cuda_memory

            reset_cuda_peak_memory()
        start = time.perf_counter()
        generate_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if position_ids is not None:
            generate_kwargs["position_ids"] = position_ids
        generated = model.generate(**generate_kwargs)
        total_time = time.perf_counter() - start
        if memory_trace is not None:
            memory_trace["generate"] = snapshot_cuda_memory()
        new_tokens = generated[0, input_ids.shape[1] :]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        if stop_strings:
            for stop in stop_strings:
                if stop and stop in text:
                    text = text[: text.find(stop)]
                    break
        if return_timing:
            timing = GenerationTiming(prefill_s=0.0, decode_s=total_time, total_s=total_time)
            return text, timing
        return text

    eos_ids = get_eos_token_ids(tokenizer, model)
    gen_ids = []
    text_parts: list[str] = []
    def _sync() -> None:
        synchronize_model(model)

    _sync()
    if memory_trace is not None:
        from .memory import reset_cuda_peak_memory, snapshot_cuda_memory

        reset_cuda_peak_memory()
    _injected_cache = kv_cache_factory() if kv_cache_factory is not None else None
    import contextlib as _ctxlib
    _press_ctx = kvpress_press(model) if kvpress_press is not None else _ctxlib.nullcontext()
    prefill_start = time.perf_counter()
    if prebuilt_prefill is not None:
        # Skip the prefill forward: use an externally-built KV cache + its last-token logits (e.g. CacheBlend's
        # selective-recompute cache). The DECODE below is then byte-for-byte the canonical teacher decode of that
        # cache — so cacheblend-r decoded here == teacher-14B decode of the same prefill KV (up to GPU noise).
        past, logits = prebuilt_prefill
        prefill_time = 0.0
    else:
        with torch.no_grad(), _press_ctx:  # KVPress press (if any) compresses the KV during this prefill forward
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=_injected_cache,  # None = default DynamicCache; minference SnapKVCache engages compression in manual decode
                use_cache=True,
                logits_to_keep=1,
            )
        _sync()
        prefill_time = time.perf_counter() - prefill_start
        if memory_trace is not None:
            memory_trace["prefill"] = snapshot_cuda_memory()
        past = outputs.past_key_values
        logits = outputs.logits[:, -1, :]

    # ★ KVPress compresses (evicts tokens from) the KV cache DURING the prefill forward above. Decode then
    # needs TWO corrections vs an uncompressed cache (else degenerate/incoherent output — 90% of
    # babilong-16k snapkv-manual):
    #   (1) attention_mask LENGTH must match the COMPRESSED cache (n_kept), not the full prompt — otherwise
    #       the mask/cache mismatch corrupts attention (pure "Final\nFinal..." loops).
    #   (2) position_ids (RoPE) for the new tokens must continue from the ORIGINAL prompt length, NOT the
    #       compressed-slot index. KVPress keeps tokens with their ORIGINAL RoPE (baked in at prefill); a new
    #       token at compressed-slot n_kept must still carry RoPE position = orig_prompt_len, or q·k phases
    #       mismatch → incoherent text ("Final Answer: The Peruvura, a a Answer:"). cache_position auto-tracks
    #       the real DynamicCache length, so only position_ids must be supplied explicitly.
    _kvp_orig_len = int(base_attention_mask.shape[1])
    _kvp_compressed_len = None
    if kvpress_press is not None and incremental_attention_mask_builder is None and incremental_position_ids_builder is None:
        try:
            _kvp_compressed_len = int(past.get_seq_length())
        except Exception:
            _kvp_compressed_len = None
        if _kvp_compressed_len and _kvp_compressed_len > 0 and _kvp_compressed_len != _kvp_orig_len:
            print(f"[KVPress] cache compressed {_kvp_orig_len}→{_kvp_compressed_len} tokens; "
                  f"rebuilding decode attention_mask + continuing RoPE from original pos {_kvp_orig_len}")
            base_attention_mask = torch.ones((1, _kvp_compressed_len), device=input_device, dtype=base_attention_mask.dtype)

    if memory_trace is not None:
        reset_cuda_peak_memory()
    gen = None
    if decode_temperature > 0 and decode_seed:
        gen = torch.Generator(device=logits.device).manual_seed(int(decode_seed))
    decode_start = time.perf_counter()
    _eos_list = list(eos_ids) if eos_ids else []
    for _ in range(max_new_tokens):
        # ★ min_new_tokens: for length-controlled summarization eval (gov_report rougeLsum is recall-
        # biased → length-dominated; equal generation length makes the content comparison fair). Mask EOS
        # in the logits until min_new_tokens are produced so the model keeps generating instead of stopping
        # short. A pure decode control (no metric change) — same budget for teacher/ours/snapkv alike.
        if min_new_tokens and len(gen_ids) < min_new_tokens and _eos_list:
            logits[:, _eos_list] = float("-inf")
        next_token = _pick_next_token(logits, decode_temperature, decode_top_p, gen)
        token_id = int(next_token[0].item())
        gen_ids.append(token_id)
        text_parts.append(tokenizer.decode([token_id], skip_special_tokens=False))
        if stop_strings:
            current_text = "".join(text_parts)
            for stop in stop_strings:
                if stop and stop in current_text:
                    cutoff = current_text.find(stop)
                    current_text = current_text[:cutoff]
                    text_parts = [current_text]
                    _sync()
                    decode_time = time.perf_counter() - decode_start
                    if memory_trace is not None:
                        memory_trace["decode"] = snapshot_cuda_memory()
                    text = "".join(text_parts)
                    if return_timing:
                        timing = GenerationTiming(
                            prefill_s=prefill_time,
                            decode_s=decode_time,
                            total_s=prefill_time + decode_time,
                        )
                        return text, timing
                    return text
        if stop_on_repeat:
            current_text = "".join(text_parts)
            if current_text.count(stop_on_repeat) >= 2:
                second = current_text.find(stop_on_repeat, current_text.find(stop_on_repeat) + 1)
                current_text = current_text[:second]
                text_parts = [current_text]
                _sync()
                decode_time = time.perf_counter() - decode_start
                if memory_trace is not None:
                    memory_trace["decode"] = snapshot_cuda_memory()
                text = "".join(text_parts)
                if return_timing:
                    timing = GenerationTiming(
                        prefill_s=prefill_time,
                        decode_s=decode_time,
                        total_s=prefill_time + decode_time,
                    )
                    return text, timing
                return text
        if token_id in eos_ids:
            break
        next_ids = next_token.unsqueeze(0)  # shape [1, 1]
        if incremental_attention_mask_builder is not None:
            attention_mask = incremental_attention_mask_builder(
                int(input_ids.shape[1] + len(gen_ids)),
                input_device,
                mask_dtype,
            )
        else:
            base_attention_mask = torch.cat(
                [
                    base_attention_mask,
                    torch.ones((1, 1), device=base_attention_mask.device, dtype=base_attention_mask.dtype),
                ],
                dim=1,
            )
            attention_mask = base_attention_mask
        next_position_ids = None
        if incremental_position_ids_builder is not None:
            next_position_ids = incremental_position_ids_builder(
                int(input_ids.shape[1] + len(gen_ids)),
                input_device,
            )
        elif _kvp_compressed_len and _kvp_compressed_len != _kvp_orig_len:
            # KVPress: the token now being fed sits at original sequence position orig_len + (step-1),
            # while the cache slot is compressed_len + (step-1). Supply the ORIGINAL-position RoPE id;
            # cache_position auto-tracks the real (compressed) cache length.
            _step = len(gen_ids) - 1
            next_position_ids = torch.tensor([[_kvp_orig_len + _step]], device=input_device, dtype=torch.long)
        with torch.no_grad():
            outputs = model(
                input_ids=next_ids,
                attention_mask=attention_mask,
                position_ids=next_position_ids,
                use_cache=True,
                past_key_values=past,
                logits_to_keep=1,
            )
        past = outputs.past_key_values
        logits = outputs.logits[:, -1, :]

    _sync()
    decode_time = time.perf_counter() - decode_start
    if memory_trace is not None:
        memory_trace["decode"] = snapshot_cuda_memory()
    # SentencePiece tokenizers (Llama) drop leading spaces under per-token decode+join -> full-sequence decode.
    if text_parts and "llama" in type(tokenizer).__name__.lower():
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    else:
        text = "".join(text_parts) if text_parts else tokenizer.decode(gen_ids, skip_special_tokens=True)
    if return_timing:
        timing = GenerationTiming(
            prefill_s=prefill_time,
            decode_s=decode_time,
            total_s=prefill_time + decode_time,
        )
        return text, timing
    return text
