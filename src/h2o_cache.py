from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers.cache_utils import DynamicCache

from .fusion import GenerationTiming
from .memory import reset_cuda_peak_memory, snapshot_cuda_memory
from .models import get_model_input_device, synchronize_model


LegacyPast = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]


@dataclass
class H2OGenerationResult:
    text: str
    timing: GenerationTiming
    stats: Dict[str, Any]


class H2OPruner:
    """Heavy-hitter + recent KV cache eviction.

    This follows the eviction rule in FMInference/H2O's real-drop implementation:
    accumulate attention mass per layer/head, keep the highest-scoring heavy-hitter
    positions plus the most recent positions, then physically prune KV tensors.

    Qwen2.5 uses grouped-query attention, so attention heads are reduced into KV heads
    before scoring. Each KV head may keep a different heavy-hitter set, matching the
    head-wise spirit of the official implementation.
    """

    def __init__(self, *, heavy_hitter_size: int, recent_size: int) -> None:
        if heavy_hitter_size < 0 or recent_size < 0:
            raise ValueError("heavy_hitter_size and recent_size must be >= 0.")
        if heavy_hitter_size + recent_size <= 0:
            raise ValueError("H2O cache size must be positive.")
        self.heavy_hitter_size = int(heavy_hitter_size)
        self.recent_size = int(recent_size)
        self.cache_size = self.heavy_hitter_size + self.recent_size
        self.hh_scores: List[Optional[torch.Tensor]] = []
        self.total_evictions = 0
        self.max_cache_len = 0

    @staticmethod
    def _legacy_cache(past_key_values: Any) -> LegacyPast:
        if hasattr(past_key_values, "to_legacy_cache"):
            return past_key_values.to_legacy_cache()
        return tuple(past_key_values)

    @staticmethod
    def _dynamic_cache(legacy: LegacyPast) -> DynamicCache:
        return DynamicCache.from_legacy_cache(legacy)

    @staticmethod
    def _reduce_attention_to_kv_heads(attn: torch.Tensor, kv_heads: int) -> torch.Tensor:
        # attn: [batch, attention_heads, query_len, kv_len]
        scores = attn.detach().float().sum(dim=(0, 2))
        attn_heads, kv_len = scores.shape
        if attn_heads == kv_heads:
            return scores
        if attn_heads % kv_heads != 0:
            # Conservative fallback for unexpected layouts.
            return scores.mean(dim=0, keepdim=True).expand(kv_heads, kv_len).contiguous()
        groups = attn_heads // kv_heads
        return scores.view(kv_heads, groups, kv_len).sum(dim=1)

    def _update_scores(self, layer_idx: int, attn: torch.Tensor, kv_heads: int, prev_len: int) -> torch.Tensor:
        scores = self._reduce_attention_to_kv_heads(attn, kv_heads)
        while len(self.hh_scores) <= layer_idx:
            self.hh_scores.append(None)
        previous = self.hh_scores[layer_idx]
        if previous is not None and prev_len > 0:
            overlap = min(previous.shape[-1], prev_len, scores.shape[-1])
            scores[:, :overlap] += previous[:, :overlap].to(scores.device)
        self.hh_scores[layer_idx] = scores
        return scores

    def _keep_indices(self, scores: torch.Tensor) -> torch.Tensor:
        kv_heads, seq_len = scores.shape
        recent_size = min(self.recent_size, seq_len)
        heavy_budget = min(self.heavy_hitter_size, max(seq_len - recent_size, 0))
        pieces = []
        if heavy_budget > 0:
            candidate_scores = scores[:, : seq_len - recent_size]
            heavy_idx = torch.topk(candidate_scores, heavy_budget, dim=-1).indices.sort(dim=-1).values
            pieces.append(heavy_idx)
        if recent_size > 0:
            recent_idx = torch.arange(seq_len - recent_size, seq_len, device=scores.device).expand(kv_heads, recent_size)
            pieces.append(recent_idx)
        if not pieces:
            return torch.empty((kv_heads, 0), device=scores.device, dtype=torch.long)
        return torch.cat(pieces, dim=-1)

    @staticmethod
    def _gather_kv(tensor: torch.Tensor, keep_idx: torch.Tensor) -> torch.Tensor:
        # tensor: [batch, kv_heads, seq_len, head_dim], keep_idx: [kv_heads, keep_len]
        batch, kv_heads, _, head_dim = tensor.shape
        gather_idx = keep_idx.to(tensor.device).view(1, kv_heads, -1, 1).expand(batch, kv_heads, -1, head_dim)
        return torch.gather(tensor, dim=2, index=gather_idx)

    def prune(self, past_key_values: Any, attentions: Optional[Sequence[torch.Tensor]], *, query_len: int) -> Any:
        if past_key_values is None or attentions is None:
            return past_key_values
        legacy = self._legacy_cache(past_key_values)
        pruned_layers: List[Tuple[torch.Tensor, torch.Tensor]] = []
        evicted_this_call = 0
        for layer_idx, (key, value) in enumerate(legacy):
            seq_len = int(key.shape[2])
            kv_heads = int(key.shape[1])
            self.max_cache_len = max(self.max_cache_len, seq_len)
            prev_len = max(seq_len - int(query_len), 0)
            scores = self._update_scores(layer_idx, attentions[layer_idx], kv_heads, prev_len)
            if seq_len <= self.cache_size:
                pruned_layers.append((key, value))
                continue
            keep_idx = self._keep_indices(scores)
            pruned_key = self._gather_kv(key, keep_idx)
            pruned_value = self._gather_kv(value, keep_idx)
            self.hh_scores[layer_idx] = torch.gather(scores, dim=1, index=keep_idx.to(scores.device))
            evicted_this_call += seq_len - int(keep_idx.shape[-1])
            pruned_layers.append((pruned_key, pruned_value))
        self.total_evictions += evicted_this_call
        return self._dynamic_cache(tuple(pruned_layers))

    def stats(self) -> Dict[str, Any]:
        return {
            "heavy_hitter_size": self.heavy_hitter_size,
            "recent_size": self.recent_size,
            "cache_size": self.cache_size,
            "total_evicted_layer_tokens": self.total_evictions,
            "max_cache_len_before_prune": self.max_cache_len,
        }


def _cache_len(past_key_values: Any) -> int:
    if past_key_values is None:
        return 0
    legacy = H2OPruner._legacy_cache(past_key_values)
    if not legacy:
        return 0
    return int(legacy[0][0].shape[2])


def _force_eager_attention(model) -> None:
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("eager")
    if hasattr(model, "config"):
        setattr(model.config, "attn_implementation", "eager")
        setattr(model.config, "_attn_implementation", "eager")


def generate_with_h2o_teacher(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    max_length: int,
    heavy_hitter_size: int,
    recent_size: int,
    prefill_chunk_size: int = 256,
    stop_strings: Optional[Sequence[str]] = None,
    measure_memory: bool = False,
) -> H2OGenerationResult:
    _force_eager_attention(model)
    input_device = get_model_input_device(model)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length, padding=False)
    input_ids = encoded["input_ids"].to(input_device)
    if input_ids.shape[0] != 1:
        raise ValueError("H2O teacher generation currently expects batch_size=1.")
    pruner = H2OPruner(heavy_hitter_size=heavy_hitter_size, recent_size=recent_size)
    generated_ids: List[int] = []
    stop_strings = list(stop_strings or [])
    eos_id = tokenizer.eos_token_id

    synchronize_model(model)
    start = time.perf_counter()
    prefill_start = time.perf_counter()
    past = None
    next_logits = None
    total_seen_tokens = 0
    phase_memory: Dict[str, Any] = {}
    if measure_memory:
        reset_cuda_peak_memory()
    with torch.no_grad():
        for start_idx in range(0, input_ids.shape[1], prefill_chunk_size):
            chunk = input_ids[:, start_idx : start_idx + prefill_chunk_size]
            cache_len = _cache_len(past)
            attention_mask = torch.ones((1, cache_len + chunk.shape[1]), device=input_device, dtype=torch.long)
            # Use total_seen_tokens for RoPE positions, not cache_len: after pruning, cache_len
            # shrinks to the budget but the actual sequence continues at total_seen_tokens.
            cache_position = torch.arange(total_seen_tokens, total_seen_tokens + chunk.shape[1], device=input_device, dtype=torch.long)
            position_ids = cache_position.unsqueeze(0)
            outputs = model(
                input_ids=chunk,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                past_key_values=past,
                use_cache=True,
                output_attentions=True,
                return_dict=True,
                logits_to_keep=1,
            )
            past = pruner.prune(outputs.past_key_values, outputs.attentions, query_len=int(chunk.shape[1]))
            next_logits = outputs.logits[:, -1, :]
            total_seen_tokens += int(chunk.shape[1])
    synchronize_model(model)
    prefill_s = time.perf_counter() - prefill_start
    if measure_memory:
        phase_memory["prefill"] = snapshot_cuda_memory()
        reset_cuda_peak_memory()

    decode_start = time.perf_counter()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            assert next_logits is not None
            next_token_id = int(torch.argmax(next_logits, dim=-1).item())
            if eos_id is not None and next_token_id == eos_id:
                break
            generated_ids.append(next_token_id)
            current_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            if stop_strings and any(stop in current_text for stop in stop_strings):
                break

            token = torch.tensor([[next_token_id]], device=input_device, dtype=input_ids.dtype)
            cache_len = _cache_len(past)
            attention_mask = torch.ones((1, cache_len + 1), device=input_device, dtype=torch.long)
            # Use total_seen_tokens for RoPE position, not cache_len (which is the pruned budget).
            cache_position = torch.tensor([total_seen_tokens], device=input_device, dtype=torch.long)
            position_ids = cache_position.unsqueeze(0)
            outputs = model(
                input_ids=token,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                past_key_values=past,
                use_cache=True,
                output_attentions=True,
                return_dict=True,
                logits_to_keep=1,
            )
            past = pruner.prune(outputs.past_key_values, outputs.attentions, query_len=1)
            next_logits = outputs.logits[:, -1, :]
            total_seen_tokens += 1
    synchronize_model(model)
    decode_s = time.perf_counter() - decode_start
    if measure_memory:
        phase_memory["decode"] = snapshot_cuda_memory()
    total_s = time.perf_counter() - start
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    stats = pruner.stats()
    stats.update(
        {
            "prompt_tokens": int(input_ids.shape[1]),
            "generated_tokens": len(generated_ids),
            "total_seen_tokens": int(total_seen_tokens),
            "prefill_chunk_size": int(prefill_chunk_size),
            "final_cache_len": _cache_len(past),
            "phase_memory": phase_memory,
        }
    )
    return H2OGenerationResult(
        text=text,
        timing=GenerationTiming(prefill_s=prefill_s, decode_s=decode_s, total_s=total_s),
        stats=stats,
    )
