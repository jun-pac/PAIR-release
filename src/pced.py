from __future__ import annotations

import time
import inspect
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .data import HotpotExample
from .eval import compute_em_f1, extract_final_answer
from .fusion import generate_with_single_model
from .models import get_model_input_device, synchronize_model, prepare_inputs


class PCEDContrastiveDecoder:
    """
    PCED (Parallel Context-of-Experts Decoding), approximating the paper recipe.

    For each document, run an LM with (doc + query) plus an amateur LM with (query only).
    Experts share the same generated history, but keep independent KV/cache states.

    Formula:
        hat_s_k = (1 + beta0) * s_k - beta0 * s_0 + gamma * log r_k
    Token choice: argmax_v max_k hat_s_k(v)

    The original paper uses retrieval + reranker score fusion. Our benchmark loaders expose
    one retrieval score list, so this implementation normalizes that single signal into r_k.
    """

    def __init__(
        self,
        lm_model,
        tokenizer,
        *,
        beta0: float = 0.75,
        beta_mode: str = "dynamic",
        beta_warmup: float = 0.0,
        beta_reduce: str = "max",
        gamma: float = 2.5,
        relevance_mode: str = "sparse",
        max_new_tokens: int = 128,
        max_length: int = 4096,
        prompt_builder: Optional[Callable[..., str]] = None,
        r_clip_eps: float = 1e-8,
        stop_strings: Optional[List[str]] = None,
    ) -> None:
        self.lm_model = lm_model
        self.tokenizer = tokenizer
        self.beta0 = beta0
        self.beta_mode = beta_mode
        self.beta_warmup = beta_warmup
        self.beta_reduce = beta_reduce
        self.gamma = gamma
        self.relevance_mode = relevance_mode
        self.max_new_tokens = max_new_tokens
        self.max_length = max_length
        self.prompt_builder = prompt_builder
        self.r_clip_eps = r_clip_eps
        self.stop_strings = stop_strings or []

    def _prompt(self, example: HotpotExample, include_docs: bool) -> str:
        if self.prompt_builder:
            try:
                sig = inspect.signature(self.prompt_builder)
                if len(sig.parameters) >= 3:
                    return self.prompt_builder(example, include_docs, self.tokenizer)
            except (TypeError, ValueError):
                pass
            return self.prompt_builder(example, include_docs)
        return example.prompt(include_docs=include_docs)

    def _prepare_states(self, example: HotpotExample) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Build per-context tokenized inputs. Returns parallel lists of input_ids and attention masks.
        Order: one entry per document expert, followed by amateur (query-only).
        """
        ids_list: List[torch.Tensor] = []
        masks_list: List[torch.Tensor] = []

        for doc in example.documents:
            single = HotpotExample(example.example_id, example.question, [doc], example.answer)
            prompt = self._prompt(single, include_docs=True)
            inputs = prepare_inputs(
                self.tokenizer, prompt, max_length=self.max_length, return_tensors="pt"
            )
            input_device = get_model_input_device(self.lm_model)
            ids_list.append(inputs["input_ids"].to(input_device))
            masks_list.append(inputs["attention_mask"].to(input_device))

        # Amateur (query only)
        amateur_prompt = self._prompt(example, include_docs=False)
        am_inputs = prepare_inputs(
            self.tokenizer, amateur_prompt, max_length=self.max_length, return_tensors="pt"
        )
        input_device = get_model_input_device(self.lm_model)
        ids_list.append(am_inputs["input_ids"].to(input_device))
        masks_list.append(am_inputs["attention_mask"].to(input_device))
        return ids_list, masks_list

    def _normalize_relevance(
        self,
        retrieval_scores: Optional[Sequence[float]],
        num_experts: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if retrieval_scores is None:
            return torch.full((num_experts,), 1.0 - self.r_clip_eps, device=device, dtype=dtype)
        score_values = list(retrieval_scores[:num_experts])
        if len(score_values) < num_experts:
            score_values.extend([0.0] * (num_experts - len(score_values)))
        scores = torch.tensor(score_values, device=device, dtype=torch.float32)
        eps = float(self.r_clip_eps)
        mode = self.relevance_mode
        if mode == "dense":
            rel = (scores + 1.0) / 2.0
        elif mode == "sparse":
            rel = (2.0 / torch.pi) * torch.atan(torch.clamp(scores, min=0.0))
        elif mode == "minmax":
            s_min = torch.min(scores)
            s_max = torch.max(scores)
            rel = (scores - s_min) / (s_max - s_min + eps)
        elif mode == "softmax":
            rel = torch.softmax(scores, dim=0)
        elif mode == "none":
            rel = scores
        else:
            raise ValueError(f"Unsupported PCED relevance_mode: {mode!r}")
        return torch.clamp(rel, min=eps, max=1.0 - eps).to(dtype=dtype)

    def _jsd(self, expert_logits: torch.Tensor, amateur_logits: torch.Tensor) -> torch.Tensor:
        expert_probs = F.softmax(expert_logits.float(), dim=-1)
        amateur_probs = F.softmax(amateur_logits.float(), dim=-1)
        mixture = 0.5 * (expert_probs + amateur_probs)
        mixture_log = torch.log(mixture.clamp_min(1e-9))
        expert_probs = expert_probs.clamp_min(1e-9)
        amateur_probs = amateur_probs.clamp_min(1e-9)
        return 0.5 * (
            F.kl_div(mixture_log, expert_probs, reduction="batchmean", log_target=False)
            + F.kl_div(mixture_log, amateur_probs, reduction="batchmean", log_target=False)
        )

    def _dynamic_beta(self, expert_logits: Sequence[torch.Tensor], amateur_logits: torch.Tensor) -> float:
        if self.beta_mode == "fixed":
            return float(self.beta0)
        if self.beta_mode != "dynamic":
            raise ValueError(f"Unsupported PCED beta_mode: {self.beta_mode!r}")
        jsds = torch.stack([self._jsd(logit, amateur_logits) for logit in expert_logits])
        if self.beta_reduce == "mean":
            beta = torch.mean(jsds)
        elif self.beta_reduce == "max":
            beta = torch.max(jsds)
        else:
            raise ValueError(f"Unsupported PCED beta_reduce: {self.beta_reduce!r}")
        return float(max(float(beta.item()), float(self.beta_warmup)))

    def decode(self, example: HotpotExample, retrieval_scores: Optional[List[float]] = None) -> Dict[str, object]:
        """
        Batched PCED decode: all N document experts + 1 amateur are run in a single
        batched forward pass per step, giving an N+1x speedup over the sequential version.

        Sequences are left-padded to the same length so that logits_to_keep=1 always
        returns the correct last real-token logit for every sequence in the batch.
        """
        ids_list, masks_list = self._prepare_states(example)
        if len(ids_list) <= 1:
            text, timing = generate_with_single_model(
                self.lm_model,
                self.tokenizer,
                self._prompt(example, include_docs=False),
                max_new_tokens=self.max_new_tokens,
                max_length=self.max_length,
                use_generate=False,
                return_timing=True,
                stop_strings=self.stop_strings,
            )
            extracted = extract_final_answer(text)
            em, f1 = compute_em_f1(extracted, example.answer)
            return {
                "text": text,
                "extracted": extracted,
                "em": em,
                "f1": f1,
                "timing": {
                    "prefill_s": timing.prefill_s if timing else None,
                    "decode_s": timing.decode_s if timing else None,
                    "total_s": timing.total_s if timing else None,
                },
            }

        input_device = get_model_input_device(self.lm_model)
        eos_id = self.tokenizer.eos_token_id
        pad_token = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos_id
        generated: List[int] = []
        text_parts: List[str] = []
        num_seqs = len(ids_list)

        # Left-pad so every sequence's last real token is at position -1; this makes
        # logits_to_keep=1 return the correct logit for every sequence in the batch.
        max_seq_len = max(t.shape[1] for t in ids_list)
        padded_ids: List[torch.Tensor] = []
        padded_masks: List[torch.Tensor] = []
        for ids, mask in zip(ids_list, masks_list):
            pad_len = max_seq_len - ids.shape[1]
            if pad_len > 0:
                ids = torch.cat([
                    torch.full((1, pad_len), pad_token, device=input_device, dtype=ids.dtype),
                    ids.to(input_device),
                ], dim=1)
                mask = torch.cat([
                    torch.zeros((1, pad_len), device=input_device, dtype=mask.dtype),
                    mask.to(input_device),
                ], dim=1)
            else:
                ids = ids.to(input_device)
                mask = mask.to(input_device)
            padded_ids.append(ids)
            padded_masks.append(mask)

        batch_input_ids = torch.cat(padded_ids, dim=0)   # [N+1, max_seq_len]
        batch_attention = torch.cat(padded_masks, dim=0)  # [N+1, max_seq_len]

        beta_value: Optional[float] = None
        relevance_values: Optional[torch.Tensor] = None
        selected_experts: List[int] = []

        def _pced_select(step_outputs) -> int:
            nonlocal beta_value, relevance_values
            # step_outputs.logits: [num_seqs, 1, V]  (with logits_to_keep=1)
            last_logits = [step_outputs.logits[i, -1:, :] for i in range(num_seqs)]
            device = last_logits[0].device
            dtype = last_logits[0].dtype
            s0 = last_logits[-1]            # amateur (last entry)
            expert_logits = last_logits[:-1]
            if beta_value is None:
                beta_value = self._dynamic_beta(expert_logits, s0)
            if relevance_values is None:
                relevance_values = self._normalize_relevance(
                    retrieval_scores, len(expert_logits), device=device, dtype=dtype,
                )
            log_r = torch.log(relevance_values)
            hats = [
                (1 + beta_value) * logit - beta_value * s0 + self.gamma * log_r[idx]
                for idx, logit in enumerate(expert_logits)
            ]
            stacked = torch.stack(hats, dim=0)               # [K, 1, V]
            fused_logits, expert_indices = torch.max(stacked, dim=0)
            next_token = torch.argmax(fused_logits, dim=-1)  # [1]
            selected_experts.append(int(expert_indices[0, int(next_token[0].item())].item()))
            return int(next_token[0].item())

        def _sync() -> None:
            synchronize_model(self.lm_model)

        _sync()
        prefill_start = time.perf_counter()
        with torch.no_grad():
            outputs = self.lm_model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention,
                use_cache=True,
                logits_to_keep=1,
            )
        _sync()
        prefill_time = time.perf_counter() - prefill_start

        past_key_values = outputs.past_key_values
        token_id = _pced_select(outputs)
        generated.append(token_id)
        text_parts.append(self.tokenizer.decode([token_id], skip_special_tokens=False))

        decode_start = time.perf_counter()
        for _ in range(self.max_new_tokens - 1):
            if token_id == eos_id:
                break
            current_text = "".join(text_parts)
            if self.stop_strings and any(stop and stop in current_text for stop in self.stop_strings):
                break

            next_tok = torch.full((num_seqs, 1), token_id, device=input_device, dtype=batch_input_ids.dtype)
            batch_attention = torch.cat(
                [batch_attention, torch.ones((num_seqs, 1), device=input_device, dtype=batch_attention.dtype)],
                dim=1,
            )
            with torch.no_grad():
                outputs = self.lm_model(
                    input_ids=next_tok,
                    attention_mask=batch_attention,
                    past_key_values=past_key_values,
                    use_cache=True,
                    logits_to_keep=1,
                )
            past_key_values = outputs.past_key_values
            token_id = _pced_select(outputs)
            generated.append(token_id)
            text_parts.append(self.tokenizer.decode([token_id], skip_special_tokens=False))

        _sync()
        decode_time = time.perf_counter() - decode_start

        text = "".join(text_parts)
        if self.stop_strings:
            for stop in self.stop_strings:
                if stop and stop in text:
                    text = text[:text.find(stop)]
        extracted = extract_final_answer(text)
        em, f1 = compute_em_f1(extracted, example.answer)
        return {
            "text": text,
            "extracted": extracted,
            "em": em,
            "f1": f1,
            "timing": {
                "prefill_s": prefill_time,
                "decode_s": decode_time,
                "total_s": prefill_time + decode_time,
            },
            "pced": {
                "beta": beta_value,
                "beta_mode": self.beta_mode,
                "beta_reduce": self.beta_reduce,
                "gamma": self.gamma,
                "relevance_mode": self.relevance_mode,
                "relevance": relevance_values.detach().float().cpu().tolist() if relevance_values is not None else None,
                "selected_experts": selected_experts,
            },
        }


class ParallelFusionDecoder:
    """
    Batched parallel decoding for K document experts + 1 amateur (query-only) using a **single** LM.

    All experts share weights and reuse a shared KV cache; we only swap the cached states between
    contexts during decoding, so no model reloads are needed even on a single GPU.

    Fusion modes:
      - weighted_sum: fused = lambda * amateur + (1-lambda)/K * sum(experts)
      - max: elementwise max across all K+1 logits
      - entropy: compute per-model weights from top-k entropy and softmax(-entropy / temperature)
    """

    def __init__(
        self,
        lm_model,
        tokenizer,
        *,
        fusion_mode: str = "weighted_sum",
        lambda_weight: float = 0.5,
        entropy_top_k: int = 10,
        entropy_temperature: float = 1.0,
        use_generate: bool = True,
        max_new_tokens: int = 128,
        max_length: int = 4096,
        prompt_builder: Optional[Callable[..., str]] = None,
        stop_strings: Optional[List[str]] = None,
    ) -> None:
        self.lm_model = lm_model
        self.tokenizer = tokenizer
        self.fusion_mode = fusion_mode
        self.lambda_weight = lambda_weight
        self.entropy_top_k = entropy_top_k
        self.entropy_temperature = max(entropy_temperature, 1e-5)
        self.use_generate = use_generate
        self.max_new_tokens = max_new_tokens
        self.max_length = max_length
        self.prompt_builder = prompt_builder
        self.stop_strings = stop_strings or []

    def _prompt(self, example: HotpotExample, include_docs: bool) -> str:
        if self.prompt_builder:
            try:
                sig = inspect.signature(self.prompt_builder)
                if len(sig.parameters) >= 3:
                    return self.prompt_builder(example, include_docs, self.tokenizer)
            except (TypeError, ValueError):
                pass
            return self.prompt_builder(example, include_docs)
        return example.prompt(include_docs=include_docs)

    def _prepare_states(self, example: HotpotExample) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        ids_list: List[torch.Tensor] = []
        masks_list: List[torch.Tensor] = []

        for doc in example.documents:
            single = HotpotExample(example.example_id, example.question, [doc], example.answer)
            prompt = self._prompt(single, include_docs=True)
            inputs = prepare_inputs(self.tokenizer, prompt, max_length=self.max_length, return_tensors="pt")
            input_device = get_model_input_device(self.lm_model)
            ids_list.append(inputs["input_ids"].to(input_device))
            masks_list.append(inputs["attention_mask"].to(input_device))

        amateur_prompt = self._prompt(example, include_docs=False)
        am_inputs = prepare_inputs(self.tokenizer, amateur_prompt, max_length=self.max_length, return_tensors="pt")
        input_device = get_model_input_device(self.lm_model)
        ids_list.append(am_inputs["input_ids"].to(input_device))
        masks_list.append(am_inputs["attention_mask"].to(input_device))
        return ids_list, masks_list

    @staticmethod
    def _entropy_topk_norm(logits: torch.Tensor, k: int) -> torch.Tensor:
        k = max(1, min(k, logits.shape[-1]))
        logprobs = torch.log_softmax(logits, dim=-1)
        vals, _ = torch.topk(logprobs, k=k, dim=-1)
        probs = torch.exp(vals)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        return -(probs * probs.clamp(min=1e-12).log()).sum(dim=-1)

    def _fuse_logits(self, last_logits: List[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(last_logits, dim=0)  # [K+1, 1, V]
        if self.fusion_mode == "max":
            fused, _ = torch.max(stacked, dim=0)
            return fused

        if self.fusion_mode == "entropy":
            entropies = []
            for logit in last_logits:
                ent = self._entropy_topk_norm(logit, self.entropy_top_k)
                entropies.append(ent.squeeze())
            entropy_tensor = torch.stack(entropies, dim=0)  # [K+1]
            weights = torch.softmax(-entropy_tensor / self.entropy_temperature, dim=0)
            fused = (weights.view(-1, 1, 1) * stacked).sum(dim=0)
            return fused

        # Default: weighted_sum
        amateur = stacked[-1]
        experts = stacked[:-1]
        if experts.numel() == 0:
            return amateur
        expert_weight = (1.0 - self.lambda_weight) / experts.shape[0]
        fused = self.lambda_weight * amateur + expert_weight * experts.sum(dim=0)
        return fused

    def _gather_last_logits(self, step_logits: torch.Tensor, lengths: List[int]) -> List[torch.Tensor]:
        vocab = step_logits.shape[-1]
        last_logits: List[torch.Tensor] = []
        for i, seq_len in enumerate(lengths):
            # For the initial prefill, step_logits has length=max_len; for cached steps, length=1.
            pos = seq_len - 1 if step_logits.shape[1] > 1 else -1
            last_logits.append(step_logits[i : i + 1, pos, :vocab])
        return last_logits

    def decode(self, example: HotpotExample) -> Dict[str, object]:
        if self.fusion_mode == "weighted_sum" and self.lambda_weight >= 1.0 - 1e-8:
            prompt = self._prompt(example, include_docs=False)
            text = generate_with_single_model(
                self.lm_model,
                self.tokenizer,
                prompt,
                max_new_tokens=self.max_new_tokens,
                max_length=self.max_length,
                use_generate=False,
                stop_strings=self.stop_strings,
            )
            extracted = extract_final_answer(text)
            em, f1 = compute_em_f1(extracted, example.answer)
            return {
                "text": text,
                "extracted": extracted,
                "em": em,
                "f1": f1,
                "timing": {
                    "prefill_s": 0.0,
                    "decode_s": 0.0,
                    "total_s": 0.0,
                },
            }
        ids_list, masks_list = self._prepare_states(example)
        eos_id = self.tokenizer.eos_token_id
        pad_token = self.tokenizer.pad_token_id or eos_id
        generated: List[int] = []
        text_parts: List[str] = []

        def _sync() -> None:
            synchronize_model(self.lm_model)

        max_len = max(t.shape[1] for t in ids_list)
        batch_ids = []
        batch_masks = []
        lengths = []
        for ids, mask in zip(ids_list, masks_list):
            lengths.append(ids.shape[1])
            pad_len = max_len - ids.shape[1]
            if pad_len > 0:
                pad_ids = torch.full((1, pad_len), pad_token, device=ids.device, dtype=ids.dtype)
                pad_mask = torch.zeros((1, pad_len), device=mask.device, dtype=mask.dtype)
                ids = torch.cat([ids, pad_ids], dim=1)
                mask = torch.cat([mask, pad_mask], dim=1)
            batch_ids.append(ids)
            batch_masks.append(mask)
        batch_input_ids = torch.cat(batch_ids, dim=0)
        batch_attention = torch.cat(batch_masks, dim=0)

        _sync()
        prefill_start = time.perf_counter()
        with torch.no_grad():
            outputs = self.lm_model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention,
                use_cache=True,
                logits_to_keep=1,
            )
        _sync()
        prefill_time = time.perf_counter() - prefill_start
        past_key_values = outputs.past_key_values
        logits = outputs.logits

        def _step(step_logits: torch.Tensor, lens: List[int]) -> Tuple[int, List[int]]:
            last_logits = self._gather_last_logits(step_logits, lens)
            fused = self._fuse_logits(last_logits)
            next_token = torch.argmax(fused, dim=-1)
            token_id = int(next_token.item())
            return token_id, [seq_len + 1 for seq_len in lens]

        token_id, lengths = _step(logits, lengths)
        generated.append(token_id)
        text_parts.append(self.tokenizer.decode([token_id], skip_special_tokens=False))

        decode_start = time.perf_counter()
        for _ in range(self.max_new_tokens - 1):
            next_token_tensor = torch.tensor([[token_id]], device=batch_input_ids.device)
            ones = torch.ones((len(ids_list), 1), device=batch_attention.device, dtype=batch_attention.dtype)
            batch_attention = torch.cat([batch_attention, ones], dim=1)

            with torch.no_grad():
                outputs = self.lm_model(
                    input_ids=next_token_tensor.expand(len(ids_list), 1),
                    attention_mask=batch_attention,
                    past_key_values=past_key_values,
                    use_cache=True,
                    logits_to_keep=1,
                )
            past_key_values = outputs.past_key_values
            logits = outputs.logits  # [K+1, 1, V]

            step_logits = torch.cat([logits[i : i + 1, -1:, :] for i in range(len(ids_list))], dim=0)
            token_id, lengths = _step(step_logits, lengths)
            generated.append(token_id)
            text_parts.append(self.tokenizer.decode([token_id], skip_special_tokens=False))

            if self.stop_strings:
                current_text = "".join(text_parts)
                for stop in self.stop_strings:
                    if stop and stop in current_text:
                        cutoff = current_text.find(stop)
                        current_text = current_text[:cutoff]
                        text_parts = [current_text]
                        token_id = eos_id
                        break

            if token_id == eos_id:
                break

        _sync()
        decode_time = time.perf_counter() - decode_start
        text = "".join(text_parts) if text_parts else self.tokenizer.decode(generated, skip_special_tokens=True)
        extracted = extract_final_answer(text)
        em, f1 = compute_em_f1(extracted, example.answer)
        return {
            "text": text,
            "extracted": extracted,
            "em": em,
            "f1": f1,
            "timing": {
                "prefill_s": prefill_time,
                "decode_s": decode_time,
                "total_s": prefill_time + decode_time,
            },
        }


class BiParallelFusionDecoder:
    """
    Two identical LMs; each gets a subset of retrieved passages. Fusion mirrors FixedLambdaFusionDecoder
    (weighted_sum / max / entropy). Prompts are built per-subset; KV caches are reused without reloading weights.
    """

    def __init__(
        self,
        lm_a,
        lm_b,
        tokenizer,
        *,
        fusion_mode: str = "weighted_sum",
        lambda_weight: float = 0.5,
        entropy_top_k: int = 10,
        entropy_scale: float = 1.0,
        allocation: str = "alternate",  # alternate or headtail
        head_docs_for_a: Optional[int] = None,
        max_new_tokens: int = 128,
        max_length: int = 4096,
        prompt_builder: Optional[Callable[..., str]] = None,
        stop_strings: Optional[List[str]] = None,
    ) -> None:
        self.lm_a = lm_a
        self.lm_b = lm_b
        self.tokenizer = tokenizer
        self.fusion_mode = fusion_mode
        self.lambda_weight = lambda_weight
        self.entropy_top_k = entropy_top_k
        self.entropy_scale = entropy_scale
        self.allocation = allocation
        self.head_docs_for_a = head_docs_for_a
        self.max_new_tokens = max_new_tokens
        self.max_length = max_length
        self.prompt_builder = prompt_builder
        self.stop_strings = stop_strings or []

    def _prompt(self, example: HotpotExample, include_docs: bool) -> str:
        if self.prompt_builder:
            try:
                sig = inspect.signature(self.prompt_builder)
                if len(sig.parameters) >= 3:
                    return self.prompt_builder(example, include_docs, self.tokenizer)
            except (TypeError, ValueError):
                pass
            return self.prompt_builder(example, include_docs)
        return example.prompt(include_docs=include_docs)

    def _split_docs(self, docs: List[str]) -> Tuple[List[str], List[str]]:
        if self.allocation == "headtail":
            k = self.head_docs_for_a if self.head_docs_for_a is not None else max(1, len(docs) // 2)
            k = min(max(k, 0), len(docs))
            return docs[:k], docs[k:]
        # default: alternate
        a_docs: List[str] = []
        b_docs: List[str] = []
        for idx, doc in enumerate(docs):
            (a_docs if idx % 2 == 0 else b_docs).append(doc)
        return a_docs, b_docs

    @staticmethod
    def _entropy_topk_norm(logits: torch.Tensor, k: int) -> torch.Tensor:
        k = max(1, min(k, logits.shape[-1]))
        logprobs = torch.log_softmax(logits, dim=-1)
        vals, _ = torch.topk(logprobs, k=k, dim=-1)
        probs = torch.exp(vals)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        return -(probs * probs.clamp(min=1e-12).log()).sum(dim=-1)

    def _fuse_logits(self, logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
        vocab = min(logits_a.shape[-1], logits_b.shape[-1])
        logits_a = logits_a[..., :vocab]
        logits_b = logits_b[..., :vocab]
        if logits_a.device != logits_b.device:
            logits_a = logits_a.to(logits_b.device)
        if self.fusion_mode == "max":
            return torch.maximum(logits_a, logits_b)
        if self.fusion_mode == "entropy":
            ent_a = self._entropy_topk_norm(logits_a, self.entropy_top_k)
            ent_b = self._entropy_topk_norm(logits_b, self.entropy_top_k)
            gate = torch.sigmoid(ent_b - ent_a)  # >0.5 favors A when A is more confident (lower entropy)
            lambda_w = self.entropy_scale * (gate - 0.5) + 0.5
            lambda_w = torch.clamp(lambda_w, 0.0, 1.0)
            return lambda_w * logits_a + (1.0 - lambda_w) * logits_b
        # weighted_sum
        return self.lambda_weight * logits_a + (1.0 - self.lambda_weight) * logits_b

    def decode(self, example: HotpotExample) -> Dict[str, object]:
        docs_a, docs_b = self._split_docs(example.documents)
        ex_a = HotpotExample(example.example_id, example.question, docs_a or [""], example.answer)
        ex_b = HotpotExample(example.example_id, example.question, docs_b or [""], example.answer)

        prompt_a = self._prompt(ex_a, include_docs=True)
        prompt_b = self._prompt(ex_b, include_docs=True)

        inputs_a = prepare_inputs(self.tokenizer, prompt_a, max_length=self.max_length, return_tensors="pt")
        inputs_b = prepare_inputs(self.tokenizer, prompt_b, max_length=self.max_length, return_tensors="pt")

        lm_a_device = get_model_input_device(self.lm_a)
        lm_b_device = get_model_input_device(self.lm_b)
        ids_a = inputs_a["input_ids"].to(lm_a_device)
        mask_a = inputs_a["attention_mask"].to(lm_a_device)
        ids_b = inputs_b["input_ids"].to(lm_b_device)
        mask_b = inputs_b["attention_mask"].to(lm_b_device)

        eos_id = self.tokenizer.eos_token_id
        generated: List[int] = []
        text_parts: List[str] = []
        terminated = False

        def _sync() -> None:
            if torch.cuda.is_available():
                synchronize_model(self.lm_a)
                if self.lm_b is not self.lm_a:
                    synchronize_model(self.lm_b)

        can_parallel = (
            torch.cuda.is_available()
            and lm_a_device.type == "cuda"
            and lm_b_device.type == "cuda"
            and lm_a_device != lm_b_device
        )

        _sync()
        prefill_start = time.perf_counter()
        with torch.no_grad():
            out_a = self.lm_a(input_ids=ids_a, attention_mask=mask_a, use_cache=True, logits_to_keep=1)
            out_b = self.lm_b(input_ids=ids_b, attention_mask=mask_b, use_cache=True, logits_to_keep=1)
        _sync()
        prefill_time = time.perf_counter() - prefill_start
        past_a = out_a.past_key_values
        past_b = out_b.past_key_values

        stream_a = torch.cuda.Stream(device=lm_a_device) if can_parallel else None
        stream_b = torch.cuda.Stream(device=lm_b_device) if can_parallel else None

        decode_start = time.perf_counter()
        for _ in range(self.max_new_tokens):
            logits_a = out_a.logits[:, -1, :]
            logits_b = out_b.logits[:, -1, :]
            fused = self._fuse_logits(logits_a, logits_b)
            next_token = torch.argmax(fused, dim=-1)
            token_id = int(next_token[0].item())
            generated.append(token_id)
            text_parts.append(self.tokenizer.decode([token_id], skip_special_tokens=False))
            current_text = "".join(text_parts)
            if self.stop_strings:
                for stop in self.stop_strings:
                    if stop and stop in current_text:
                        cutoff = current_text.find(stop)
                        current_text = current_text[:cutoff]
                        text_parts = [current_text]
                        terminated = True
                        token_id = eos_id
                        break
            if token_id == eos_id:
                terminated = True
                break

            next_token_tensor = next_token.unsqueeze(0)
            ids_a = torch.cat([ids_a, next_token_tensor.to(ids_a.device)], dim=1)
            ids_b = torch.cat([ids_b, next_token_tensor.to(ids_b.device)], dim=1)
            ones_a = torch.ones_like(next_token_tensor, device=mask_a.device, dtype=mask_a.dtype)
            ones_b = torch.ones_like(next_token_tensor, device=mask_b.device, dtype=mask_b.dtype)
            mask_a = torch.cat([mask_a, ones_a], dim=1)
            mask_b = torch.cat([mask_b, ones_b], dim=1)

            if can_parallel:
                with torch.no_grad():
                    with torch.cuda.stream(stream_a):
                        out_a = self.lm_a(
                            input_ids=next_token_tensor.to(lm_a_device),
                            attention_mask=mask_a,
                            past_key_values=past_a,
                            use_cache=True,
                            logits_to_keep=1,
                        )
                    with torch.cuda.stream(stream_b):
                        out_b = self.lm_b(
                            input_ids=next_token_tensor.to(lm_b_device),
                            attention_mask=mask_b,
                            past_key_values=past_b,
                            use_cache=True,
                            logits_to_keep=1,
                        )
                synchronize_model(self.lm_a)
                synchronize_model(self.lm_b)
            else:
                with torch.no_grad():
                    out_a = self.lm_a(
                        input_ids=next_token_tensor.to(lm_a_device),
                        attention_mask=mask_a,
                        past_key_values=past_a,
                        use_cache=True,
                        logits_to_keep=1,
                    )
                    out_b = self.lm_b(
                        input_ids=next_token_tensor.to(lm_b_device),
                        attention_mask=mask_b,
                        past_key_values=past_b,
                        use_cache=True,
                        logits_to_keep=1,
                    )
            past_a = out_a.past_key_values
            past_b = out_b.past_key_values

        _sync()
        decode_time = time.perf_counter() - decode_start
        text = "".join(text_parts) if text_parts else self.tokenizer.decode(generated, skip_special_tokens=True)
        extracted = extract_final_answer(text)
        em, f1 = compute_em_f1(extracted, example.answer)
        return {
            "text": text,
            "extracted": extracted,
            "em": em,
            "f1": f1,
            "terminated": terminated,
            "timing": {
                "prefill_s": prefill_time,
                "decode_s": decode_time,
                "total_s": prefill_time + decode_time,
            },
        }
