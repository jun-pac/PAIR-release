from __future__ import annotations

from collections.abc import Callable, Sequence

import torch


def build_block_attention_mask(segment_ids: Sequence[int], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Build a causal 4D mask that blocks document-to-document cross attention.

    Segment id 0 is shared/query space and can attend across all prior tokens.
    Positive segment ids are passage blocks and can attend only within the same
    passage plus shared/query tokens.
    """
    seg = torch.tensor(segment_ids, device=device, dtype=torch.long)
    shared = seg == 0
    same_segment = seg.unsqueeze(0) == seg.unsqueeze(1)
    shared_connection = shared.unsqueeze(0) | shared.unsqueeze(1)
    allowed = same_segment | shared_connection
    allowed = torch.tril(allowed)
    blocked = torch.full((seg.numel(), seg.numel()), torch.finfo(dtype).min, device=device, dtype=dtype)
    blocked.masked_fill_(allowed, 0)
    return blocked.unsqueeze(0).unsqueeze(0)


def build_block_attention_decode_mask(total_kv_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # Generated answer tokens are treated as shared/query tokens.
    return torch.zeros((1, 1, 1, total_kv_len), device=device, dtype=dtype)


def build_pcw_position_ids(segment_ids: Sequence[int], *, device: torch.device | None = None) -> tuple[torch.Tensor, int]:
    """Build PCW-style position ids for Qwen-style RoPE.

    Positive segment ids are passages. Each passage receives local positions
    1..C_i. Segment 0 is shared/task space (instruction, query, output format,
    generated answer), and starts at C+1 where C is the maximum passage token
    count in the current prompt.
    """
    passage_lengths: dict[int, int] = {}
    for raw_seg in segment_ids:
        seg = int(raw_seg)
        if seg > 0:
            passage_lengths[seg] = passage_lengths.get(seg, 0) + 1
    max_passage_len = max(passage_lengths.values(), default=0)

    counters: dict[int, int] = {0: max_passage_len + 1}
    position_ids: list[int] = []
    for raw_seg in segment_ids:
        seg = int(raw_seg)
        pos = counters.get(seg, 1)
        position_ids.append(pos)
        counters[seg] = pos + 1
    next_shared_position = counters.get(0, max_passage_len + 1)
    return torch.tensor([position_ids], dtype=torch.long, device=device), next_shared_position


def build_pcw_decode_position_builder(prompt_len: int, next_shared_position: int) -> Callable[[int, torch.device], torch.Tensor]:
    """Return position ids for generated tokens as a continuation of shared/query space."""

    def _builder(total_kv_len: int, device: torch.device) -> torch.Tensor:
        generated_index = max(0, int(total_kv_len) - int(prompt_len) - 1)
        return torch.tensor([[int(next_shared_position) + generated_index]], dtype=torch.long, device=device)

    return _builder
