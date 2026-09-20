"""SpecPrefill (official) — training-free speculative prefill: a small speculator's attention picks the
contextually important tokens; the big model prefills ONLY those (at their ORIGINAL position ids) + the query.

Faithful to "Speculative Prefill: Turbocharging TTFT with Lightweight and Training-Free Token Importance
Estimation" (Liu et al., ICML 2025; arXiv 2502.02789) and the official repo's `config_p1_full_lah8.yaml`:

  §3.2   importance = attention of the LAST prompt token + N look-ahead decoded tokens w.r.t. the context
         a_ij := Softmax(Q_{M+j} K^T)_i          (M = context length, N = look-ahead steps)
  §3.2.1 look-ahead (N=8) mitigates position bias (sink / proximity)
  §3.2.2 aggregation over the [N, L, S, H] score tensor = **MAX over H and L**, **MEAN over N**
  §3.2.3 denoise: 1D average pooling (kernel 13) to smooth, then chunk the context contiguously
         (chunk_size 32), average within chunk, select **Top-K chunks** (K = keep_pct * n_chunks)
  §3.2.4 restore position ids: kept tokens carry their ORIGINAL (non-contiguous) position ids, and the
         DECODING positions continue from the ORIGINAL full prompt length (not the compressed length)

Official defaults (config_p1_full_lah8.yaml): chunk=True, chunk_size=32, percentage=0.1, look_ahead_cnt=8,
pool_kernel_size=13  →  that config is the paper's "SpecPrefill Full LAH".

IMPLEMENTATION NOTE (documented deviation): the official code monkey-patches vLLM and explicitly stores the
speculator's decoded-token queries to compute Softmax(QK^T) without materialising the full S×S attention.
Here we get the identical quantity from HF by (a) prefilling all but the last prompt token WITHOUT attentions,
then (b) forwarding ONE token at a time with output_attentions=True — a single-row attention [1,H,1,S] per
layer, so nothing O(S^2) is materialised. Mathematically the same scores; different plumbing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

# official config_p1_full_lah8.yaml
DEFAULT_CHUNK_SIZE = 32
DEFAULT_POOL_KERNEL = 13
DEFAULT_LOOK_AHEAD = 8


@dataclass
class SpecPrefillPlan:
    """What the main model should prefill."""
    keep_ids: torch.Tensor        # [1, K] token ids actually sent to the main model
    keep_pos: torch.Tensor        # [1, K] their ORIGINAL position ids (non-contiguous)
    orig_len: int                 # original full prompt length (decoding positions continue from here)
    ctx_lo: int
    ctx_hi: int
    n_ctx_tokens: int             # context tokens BEFORE selection
    n_ctx_kept: int               # context tokens AFTER selection
    n_chunks: int = 0
    n_chunks_kept: int = 0
    scores: Optional[torch.Tensor] = field(default=None, repr=False)

    @property
    def keep_rate_ctx(self) -> float:
        return self.n_ctx_kept / max(1, self.n_ctx_tokens)

    @property
    def keep_rate_prompt(self) -> float:
        return int(self.keep_ids.shape[1]) / max(1, self.orig_len)


class _attn_impl:
    """Temporarily force a specific attention backend on a model."""

    def __init__(self, model, impl):
        self.cfg = getattr(model, "config", None); self.impl = impl; self.prev = None

    def __enter__(self):
        if self.cfg is not None:
            self.prev = getattr(self.cfg, "_attn_implementation", None)
            self.cfg._attn_implementation = self.impl
        return self

    def __exit__(self, *exc):
        if self.cfg is not None and self.prev is not None:
            self.cfg._attn_implementation = self.prev
        return False


def sdpa_for_gappy_positions(model):
    """★ REQUIRED whenever a forward carries NON-CONTIGUOUS position_ids.

    transformers' `_is_packed_sequence()` treats ANY position_ids that are not exactly
    [min, min+1, min+2, ...] (batch=1) as a PACKED multi-sequence batch, and flash-attention then
    segments the input at every position gap — so SpecPrefill's selected chunks would not attend to
    each other (silently wrong; it also crashes when the derived cu_seq_lens is degenerate).
    sdpa has no such inference: position_ids only drive RoPE and the mask stays plain causal, which is
    exactly SpecPrefill's semantics. Single-token decode steps are trivially contiguous, so they can
    stay on flash. (The official implementation sidesteps this by controlling vLLM's kernel directly.)
    """
    return _attn_impl(model, "sdpa")


class _eager_attn:
    """Temporarily switch a model to EAGER attention.

    Needed because sdpa/flash return `attentions=None` (transformers >=4.5x does NOT silently fall back).
    We only wrap the SINGLE-TOKEN forwards, whose attention is one row [B,H,1,S] — so nothing O(S^2) is built;
    the big prefill still runs on the fast backend.
    """

    def __init__(self, model):
        self.cfg = getattr(model, "config", None)
        self.prev = None

    def __enter__(self):
        if self.cfg is not None:
            self.prev = getattr(self.cfg, "_attn_implementation", None)
            self.cfg._attn_implementation = "eager"
        return self

    def __exit__(self, *exc):
        if self.cfg is not None and self.prev is not None:
            self.cfg._attn_implementation = self.prev
        return False


@torch.no_grad()
def speculate_scores(model, ids: torch.Tensor, ctx_lo: int, ctx_hi: int,
                     look_ahead: int = DEFAULT_LOOK_AHEAD, return_past: bool = False,
                     window: int = 0):
    """Per-context-token importance from the speculator (§3.2 + §3.2.1 + §3.2.2).

    Returns a float tensor [ctx_hi - ctx_lo] on CPU, or (scores, past_key_values) if `return_past`.

    Observation modes (which query positions' attention onto the context define importance):
      • window==0 (DEFAULT, official): the LAST prompt token + `look_ahead` greedily-DECODED tokens. The look-ahead
        is a decode loop — EXPENSIVE at small main-model scale (it dominates the speculator cost; see the TTFT bench).
      • window>0 (SnapKV-style, NO decode): the last `window` prompt tokens (the trailing query). Attention averaged
        over those w observation positions. NO look-ahead generation → the speculator is just prefill + one attn read,
        far cheaper at 14B+3B scale. (look_ahead=0 with window=0 is the cheapest official variant: last token only.)
    Aggregation = max over heads and layers, mean over observation positions.
    """
    dev = ids.device
    n = int(ids.shape[1])
    assert 0 <= ctx_lo < ctx_hi <= n, f"bad context span {ctx_lo}:{ctx_hi} for len {n}"

    if window and window > 0:
        # SnapKV-style query-window: forward the last `window` tokens WITH attention (no decode).
        w = min(window, n - 1) if n > 1 else 1
        if w < n:
            out = model(ids[:, :-w], use_cache=True); past = out.past_key_values
            win_ids = ids[:, -w:]
        else:
            past = None; win_ids = ids
        with _eager_attn(model):
            kw = dict(input_ids=win_ids, use_cache=True, output_attentions=True)
            if past is not None: kw["past_key_values"] = past
            out = model(**kw)
        if not out.attentions or out.attentions[0] is None:
            raise RuntimeError("speculator returned no attentions (query-window mode)")
        layer_max = None
        for att in out.attentions:                       # [B, H, w, S]
            a = att[0, :, :, ctx_lo:ctx_hi].float()      # [H, w, C]
            m = a.max(dim=0).values                       # max over heads -> [w, C]
            # Gemma-3 interleaves sliding-window layers whose attention rows cover only
            # the local window; only GLOBAL layers span the whole context span.
            if m.shape[-1] != (ctx_hi - ctx_lo):
                continue
            layer_max = m if layer_max is None else torch.maximum(layer_max, m)
        if layer_max is None:
            raise RuntimeError("no global-attention layer spans the context span — "
                               "SpecPrefill needs at least one full-attention layer")
        scores = layer_max.mean(0).detach().float().cpu()   # mean over the w window positions
        past = out.past_key_values
        if return_past:
            try: past.crop(n)
            except Exception: past = None
            return scores, past
        return scores

    # (a) prefill everything except the last token WITHOUT attentions (cheap, flash/sdpa)
    out = model(ids[:, :-1], use_cache=True)
    past = out.past_key_values
    cur = ids[:, -1:]

    per_obs = []          # each: [C] max-over-(L,H) attention on the context span
    with _eager_attn(model):                    # 1-row attention only -> cheap, never O(S^2)
        for step in range(look_ahead + 1):      # last prompt token + N look-ahead tokens
            out = model(cur, past_key_values=past, use_cache=True, output_attentions=True)
            past = out.past_key_values
            if not out.attentions or out.attentions[0] is None:
                raise RuntimeError("speculator returned no attentions — cannot run SpecPrefill "
                                   "(attn backend must yield weights for the 1-token steps)")
            # attentions: tuple(L) of [B, H, 1, S_t]; slice the context span
            layer_max = None
            for att in out.attentions:
                a = att[0, :, -1, ctx_lo:ctx_hi].float()     # [H, C]
                m = a.max(dim=0).values                       # max over heads
                # only GLOBAL-attention layers span the full context span
                if m.shape[-1] != (ctx_hi - ctx_lo):
                    continue
                layer_max = m if layer_max is None else torch.maximum(layer_max, m)   # max over layers
            per_obs.append(layer_max)
            if step == look_ahead:
                break
            cur = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True).to(dev)    # greedy look-ahead
    scores = torch.stack(per_obs, 0).mean(0).detach().float().cpu()                   # mean over N
    if return_past:
        try:
            past.crop(n)        # drop the look-ahead tokens -> exactly the original prompt's KV
        except Exception:
            past = None         # caller must re-prefill
        return scores, past
    return scores


def select_chunks(scores: torch.Tensor, keep_pct: float,
                  chunk_size: int = DEFAULT_CHUNK_SIZE,
                  pool_kernel: int = DEFAULT_POOL_KERNEL,
                  keep_tokens: int = 0):
    """§3.2.3 — smooth with 1D avg-pool, chunk contiguously, keep the Top-K chunks.

    Returns (keep_mask[bool, C], n_chunks, n_chunks_kept).
    `keep_tokens>0` overrides keep_pct with a FIXED token budget (n_keep = ceil(keep_tokens/chunk_size)
    chunks, independent of context length) — for the extreme fixed-budget setting (e.g. 320 tokens).
    """
    C = int(scores.numel())
    if C == 0:
        return torch.zeros(0, dtype=torch.bool), 0, 0
    x = scores[None, None].float()
    k = max(1, min(int(pool_kernel), C if C % 2 == 1 else C - 1))
    if k > 1:
        x = F.avg_pool1d(x, kernel_size=k, stride=1, padding=k // 2, count_include_pad=False)
    sm = x[0, 0][:C]

    n_chunks = math.ceil(C / chunk_size)
    pad = n_chunks * chunk_size - C
    padded = torch.cat([sm, torch.full((pad,), float("-inf"))]) if pad else sm
    chunk_score = padded.view(n_chunks, chunk_size)
    valid = torch.isfinite(chunk_score)
    chunk_mean = torch.where(valid, chunk_score, torch.zeros_like(chunk_score)).sum(1) / valid.sum(1).clamp(min=1)

    if keep_tokens and keep_tokens > 0:
        n_keep = max(1, math.ceil(keep_tokens / chunk_size))   # FIXED token budget, ctx-length-independent
    else:
        n_keep = max(1, math.ceil(keep_pct * n_chunks))
    n_keep = min(n_keep, n_chunks)
    top = torch.topk(chunk_mean, n_keep).indices
    mask = torch.zeros(n_chunks * chunk_size, dtype=torch.bool)
    for c in top.tolist():
        mask[c * chunk_size:(c + 1) * chunk_size] = True
    return mask[:C], n_chunks, n_keep


@torch.no_grad()
def build_plan(spec_model, ids: torch.Tensor, ctx_lo: int, ctx_hi: int, keep_pct: float,
               chunk_size: int = DEFAULT_CHUNK_SIZE, pool_kernel: int = DEFAULT_POOL_KERNEL,
               look_ahead: int = DEFAULT_LOOK_AHEAD, return_past: bool = False, window: int = 0,
               keep_tokens: int = 0):
    """Run the speculator and produce the main model's compressed prefill plan (§3.2.4 position restoration).

    Everything OUTSIDE [ctx_lo, ctx_hi) (the instruction header and the trailing question) is ALWAYS kept —
    the paper compresses the context and leaves the query/template intact.
    `window>0` uses the SnapKV-style query-window observation (no look-ahead decode); see speculate_scores.
    Returns the plan, or (plan, speculator_past) if `return_past` (see speculate_scores).
    """
    n = int(ids.shape[1])
    if return_past:
        scores, past = speculate_scores(spec_model, ids, ctx_lo, ctx_hi, look_ahead=look_ahead, window=window, return_past=True)
    else:
        scores, past = speculate_scores(spec_model, ids, ctx_lo, ctx_hi, look_ahead=look_ahead, window=window), None
    mask, n_chunks, n_keep = select_chunks(scores, keep_pct, chunk_size, pool_kernel, keep_tokens=keep_tokens)

    keep = torch.zeros(n, dtype=torch.bool)
    keep[:ctx_lo] = True                      # instruction / header
    keep[ctx_hi:] = True                      # question (+ anything after the context)
    keep[ctx_lo:ctx_hi] = mask
    idx = torch.nonzero(keep, as_tuple=False).flatten()          # ORIGINAL positions, ascending
    plan = SpecPrefillPlan(
        keep_ids=ids[:, idx.to(ids.device)],
        keep_pos=idx[None].to(ids.device),
        orig_len=n, ctx_lo=ctx_lo, ctx_hi=ctx_hi,
        n_ctx_tokens=ctx_hi - ctx_lo, n_ctx_kept=int(mask.sum()),
        n_chunks=n_chunks, n_chunks_kept=n_keep, scores=scores,
    )
    return (plan, past) if return_past else plan


def find_span(tok, prompt: str, ids: torch.Tensor, start_marker: str, end_marker: str):
    """Token span [lo, hi) of the compressible CONTEXT inside an already-tokenised prompt.

    Uses char offsets (fast tokenizer) so it matches EXACTLY the ids the pipeline feeds the model.
    Falls back to (0, len) semantics if a marker is missing (caller should treat that as an error).
    """
    c0 = prompt.find(start_marker)
    c1 = prompt.rfind(end_marker)
    if c0 < 0 or c1 < 0 or c1 <= c0:
        return None
    c0 += len(start_marker)
    enc = tok(prompt, return_offsets_mapping=True, add_special_tokens=False)
    offs = enc["offset_mapping"]
    n = int(ids.shape[1])
    if len(offs) != n:                       # truncation happened -> only trust the prefix
        offs = offs[:n]
    lo = next((i for i, (s, e) in enumerate(offs) if e > c0), None)
    hi = next((i for i, (s, e) in enumerate(offs) if s >= c1), None)
    if lo is None or hi is None or hi <= lo:
        return None
    return int(lo), int(hi)
