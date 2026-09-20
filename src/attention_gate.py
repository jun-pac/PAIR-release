from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn


_EPS = 1e-6
_CONTEXT_MARKER = "Context:\n"
_QUESTION_MARKERS = ("\n\nQuestion:\n", "\n\nQuestion:", "\nQuestion:\n", "\nQuestion:")
_TAIL_MARKERS = (
    "\n\nOutput format (STRICT):\nFinal Answer: <answer>\n",
    "\n\nAnswer:",
)


def _infer_context_end_char(prompt: str, context_start_char: int) -> Optional[int]:
    """Infer context end by anchoring to prompt tail markers, then fallback safely."""
    # Preferred path: locate tail marker near end and then nearest preceding question marker.
    for tail_marker in _TAIL_MARKERS:
        tail_idx = prompt.rfind(tail_marker)
        if tail_idx <= context_start_char:
            continue
        q_idx = -1
        for q_marker in _QUESTION_MARKERS:
            cand = prompt.rfind(q_marker, context_start_char, tail_idx)
            if cand > q_idx:
                q_idx = cand
        if q_idx >= 0:
            return q_idx

    # Fallback: pick the rightmost question marker after context start.
    rightmost = -1
    for q_marker in _QUESTION_MARKERS:
        cand = prompt.rfind(q_marker, context_start_char)
        if cand > rightmost:
            rightmost = cand
    if rightmost >= 0:
        return rightmost

    # Last fallback: previous behavior.
    candidates = [prompt.find(m, context_start_char) for m in _QUESTION_MARKERS]
    candidates = [idx for idx in candidates if idx >= 0]
    if candidates:
        return min(candidates)
    return None


def infer_context_token_span(prompt: str, tokenizer, prompt_seq_len: int) -> Optional[Tuple[int, int]]:
    """Infer [start, end) token span of long-context documents in the SLM prompt.

    The implementation is marker-based and tokenizer-aware, then aligned to the actual
    model input length to account for optional special tokens.
    """
    start_marker_idx = prompt.find(_CONTEXT_MARKER)
    if start_marker_idx < 0:
        return None

    context_start_char = start_marker_idx + len(_CONTEXT_MARKER)
    context_end_char = _infer_context_end_char(prompt, context_start_char)
    if context_end_char is None:
        context_end_char = len(prompt)
    if context_end_char <= context_start_char:
        return None

    prefix_len = len(tokenizer(prompt[:context_start_char], add_special_tokens=False, truncation=False).input_ids)
    prefix_ctx_len = len(tokenizer(prompt[:context_end_char], add_special_tokens=False, truncation=False).input_ids)
    full_plain_len = len(tokenizer(prompt, add_special_tokens=False, truncation=False).input_ids)

    # Most chat/tokenizer setups differ by a small special-token offset.
    offset = max(0, int(prompt_seq_len) - int(full_plain_len))
    ctx_start = min(int(prompt_seq_len), prefix_len + offset)
    ctx_end = min(int(prompt_seq_len), prefix_ctx_len + offset)
    if ctx_end <= ctx_start:
        return None
    return ctx_start, ctx_end


def _last_token_attention_heads(attentions: Sequence[torch.Tensor], kv_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-layer/per-head context-ready attention tensor for last query token.

    Output shape: [num_layers, num_heads, kv_len]
    """
    per_layer = []
    for attn in attentions:
        # [batch, heads, q_len, kv_len]
        a = attn[0, :, -1, :kv_len].to(dtype=torch.float32)
        per_layer.append(a)
    all_heads = torch.stack(per_layer, dim=0)
    return all_heads, all_heads.sum(dim=-1)


def extract_attention_head_masses(
    attentions: Sequence[torch.Tensor],
    *,
    context_span: Optional[Tuple[int, int]],
    kv_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-layer/head masses over context vs shared positions.

    Returns:
      - ctx_heads: [num_layers, num_heads]
      - shr_heads: [num_layers, num_heads]
    """
    if context_span is None:
        attn_all, totals = _last_token_attention_heads(attentions, kv_len)
        ctx = torch.zeros_like(totals)
        shr = totals
        return ctx, shr

    ctx_start = max(0, min(int(context_span[0]), int(kv_len)))
    ctx_end = max(ctx_start, min(int(context_span[1]), int(kv_len)))

    attn_all, totals = _last_token_attention_heads(attentions, kv_len)
    if ctx_end <= ctx_start:
        ctx = torch.zeros_like(totals)
    else:
        ctx = attn_all[:, :, ctx_start:ctx_end].sum(dim=-1)
    shr = torch.clamp(totals - ctx, min=0.0)
    return ctx, shr


def attention_ctx_ratio(ctx_heads: torch.Tensor, shr_heads: torch.Tensor) -> torch.Tensor:
    ctx = ctx_heads.sum()
    shr = shr_heads.sum()
    return ctx / (ctx + shr + _EPS)


def build_attention_stat_features(ctx_heads: torch.Tensor, shr_heads: torch.Tensor) -> torch.Tensor:
    """Compressed fixed-size features (shape [10]) from per-head attention masses."""
    ratio_lh = ctx_heads / (ctx_heads + shr_heads + _EPS)
    flat = ratio_lh.reshape(-1)
    ratio_sorted, _ = torch.sort(flat)
    n = int(flat.numel())
    tail = max(1, int(0.1 * n))

    layer_ratio = ratio_lh.mean(dim=-1)
    thirds = max(1, int(layer_ratio.numel() / 3))
    early = layer_ratio[:thirds].mean()
    late = layer_ratio[-thirds:].mean()

    feats = torch.stack(
        [
            attention_ctx_ratio(ctx_heads, shr_heads),
            flat.mean(),
            flat.std(unbiased=False),
            flat.max(),
            flat.min(),
            ratio_sorted[int(0.9 * (n - 1))],
            ratio_sorted[int(0.5 * (n - 1))],
            layer_ratio.std(unbiased=False),
            late - early,
            ratio_sorted[-tail:].mean(),
        ],
        dim=0,
    )
    return feats.to(dtype=torch.float32)


class AttentionLinearCalibrator(nn.Module):
    """Per-layer/per-head linear calibrator for lambda.

    Learns head weights and a scalar affine transform on weighted (ctx - shared).
    """

    def __init__(self, num_layers: int, num_heads: int) -> None:
        super().__init__()
        if num_layers <= 0 or num_heads <= 0:
            raise ValueError("num_layers and num_heads must be positive.")
        self.raw_head_weights = nn.Parameter(torch.zeros(num_layers, num_heads))
        self.scale = nn.Parameter(torch.tensor(4.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, ctx_heads: torch.Tensor, shr_heads: torch.Tensor) -> torch.Tensor:
        # ctx_heads/shr_heads: [B, L, H]
        weights = torch.sigmoid(self.raw_head_weights).unsqueeze(0)
        norm = weights.sum(dim=(-1, -2), keepdim=False).clamp(min=_EPS)
        ctx = (ctx_heads * weights).sum(dim=(-1, -2)) / norm
        shr = (shr_heads * weights).sum(dim=(-1, -2)) / norm
        z = self.scale * (ctx - shr) + self.bias
        return torch.sigmoid(z).unsqueeze(-1)


class AttentionRatioMLPGate(nn.Module):
    """Learn head importance and map weighted context-ratio with a 1D MLP."""

    def __init__(self, num_layers: int, num_heads: int, hidden_dim: int = 32, mlp_layers: int = 1) -> None:
        super().__init__()
        if num_layers <= 0 or num_heads <= 0:
            raise ValueError("num_layers and num_heads must be positive.")
        self.raw_head_weights = nn.Parameter(torch.zeros(num_layers, num_heads))
        if mlp_layers < 1:
            raise ValueError(f"Unsupported mlp_layers={mlp_layers}; expected an integer >= 1.")
        layers = [nn.Linear(1, hidden_dim), nn.GELU()]
        for _ in range(mlp_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        layers.extend([nn.Linear(hidden_dim, 1), nn.Sigmoid()])
        self.mlp = nn.Sequential(*layers)

    def forward(self, ctx_heads: torch.Tensor, shr_heads: torch.Tensor) -> torch.Tensor:
        # ctx_heads/shr_heads: [B, L, H]
        weights = torch.sigmoid(self.raw_head_weights).unsqueeze(0)
        ratio_lh = ctx_heads / (ctx_heads + shr_heads + _EPS)
        norm = weights.sum(dim=(-1, -2), keepdim=False).clamp(min=_EPS)
        weighted_ratio = (ratio_lh * weights).sum(dim=(-1, -2)) / norm
        return self.mlp(weighted_ratio.unsqueeze(-1))


class AttentionStatsGateMLP(nn.Module):
    """MLP gate over compressed attention statistics."""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, mlp_layers: int = 1) -> None:
        super().__init__()
        if mlp_layers < 1:
            raise ValueError(f"Unsupported mlp_layers={mlp_layers}; expected an integer >= 1.")
        layers = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        for _ in range(mlp_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        layers.extend([nn.Linear(hidden_dim, 1), nn.Sigmoid()])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
