"""An HQQ KV cache that quantizes each token EXACTLY ONCE (2026-08-31).

WHY THIS EXISTS. HF's `QuantizedLayer.update()` does, on every residual flush:

    keys_to_return = cat(dequantize(_quantized_keys), keys, new_keys)
    if keys.shape[-2] + 1 >= residual_length:
        _quantized_keys = quantize(keys_to_return)        # <- THE WHOLE CACHE, again

so a token committed early is dequantized and re-quantized once per ~`residual_length` tokens
appended after it. Over a 30-turn conversation that is dozens of round trips, and the damage is
measured (`scripts/probe_hqq_requantization_drift.py`, job 3055723): the reconstruction error of a
FIXED committed block grows x3.4 over 30 turns at int8 on the 7B (5.5e-3 -> 1.9e-2), x3.1 at int4,
and x2.9 on the 32B at int8. That is an accuracy loss which GROWS WITH CONVERSATION DEPTH and which
belongs to this implementation, not to KV quantization — and it biases every accumulate-benchmark
quant arm DOWNWARD, i.e. in our own method's favour, which is the direction a baseline must never be
biased.

THE FIX. Keep quantized data in append-only BLOCKS. New tokens land in an fp residual; when the
residual reaches `residual_length` it is quantized into its own block and appended. Existing blocks
are never read back and re-quantized, so every token is quantized exactly once, for real.

WHAT THIS DOES NOT FIX: speed. Reads still dequantize the whole cache every step (that is HF's
attention contract — the kernel receives fp16), so this changes accuracy, not throughput. A fused
dequant-in-attention kernel is the separate thing that would change throughput.

ROLLBACK: blocks are immutable, so a snapshot is (list-copy of block refs, residual clone,
cumulative_length) and restore is exact by construction — the same contract
`mtrag_accum.QuantStateful` already relies on, and measured exact 30/30 turns for the HF layer too.
"""
from __future__ import annotations

import torch
from transformers.cache_utils import HQQQuantizedLayer, QuantizedCache


class AppendOnlyHQQLayer(HQQQuantizedLayer):
    """HQQ layer whose committed blocks are immutable — each token quantized exactly once."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._blocks_k: list = []      # list of quantized block handles, oldest first
        self._blocks_v: list = []

    # ---- the one method whose contract we change --------------------------------------------
    def update(self, key_states, value_states, cache_kwargs=None):
        self.cumulative_length += key_states.shape[-2]
        if not self.is_initialized:
            self.lazy_initialization(key_states)

        res_k = (torch.cat([self.keys, key_states], dim=-2)
                 if self.keys.dim() == 4 else key_states)
        res_v = (torch.cat([self.values, value_states], dim=-2)
                 if self.values.dim() == 4 else value_states)

        # flush FULL blocks out of the residual, quantizing ONLY the newly-flushed tokens
        while res_k.shape[-2] >= self.residual_length:
            n = (res_k.shape[-2] // self.residual_length) * self.residual_length
            self._blocks_k.append(self._quantize(res_k[:, :, :n].contiguous(), axis=self.axis_key))
            self._blocks_v.append(self._quantize(res_v[:, :, :n].contiguous(), axis=self.axis_value))
            res_k, res_v = res_k[:, :, n:], res_v[:, :, n:]

        self.keys, self.values = res_k, res_v
        parts_k = [self._dequantize(b) for b in self._blocks_k]
        parts_v = [self._dequantize(b) for b in self._blocks_v]
        if res_k.shape[-2]:
            parts_k.append(res_k)
            parts_v.append(res_v)
        return torch.cat(parts_k, dim=-2), torch.cat(parts_v, dim=-2)

    # ---- rollback surface, exact because blocks are immutable --------------------------------
    def snapshot(self):
        return (list(self._blocks_k), list(self._blocks_v),
                self.keys.clone(), self.values.clone(), self.cumulative_length)

    def restore(self, s):
        self._blocks_k, self._blocks_v = list(s[0]), list(s[1])
        self.keys, self.values, self.cumulative_length = s[2], s[3], s[4]

    def crop(self, max_length):
        raise NotImplementedError(
            "an append-only quantized cache cannot be cropped — roll back with snapshot()/restore()")


class AppendOnlyHQQCache(QuantizedCache):
    """QuantizedCache whose layers never re-quantize committed tokens."""

    def __init__(self, config, nbits=4, axis_key=0, axis_value=0,
                 q_group_size=64, residual_length=128):
        super().__init__("hqq", config, nbits, axis_key, axis_value, q_group_size, residual_length)
        kw = dict(nbits=nbits, axis_key=axis_key, axis_value=axis_value,
                  q_group_size=q_group_size, residual_length=residual_length)
        self.layers = [AppendOnlyHQQLayer(**kw) for _ in range(len(self.layers))]


# ── the switch every quant path goes through ────────────────────────────────────────────────────
# DEFAULT IS THE CORRECT ONE. `QUANT_CACHE=hf_requant` reproduces HF's re-quantize-on-flush
# behaviour, which exists only so the artefact's cost on a real benchmark can be MEASURED rather
# than assumed. Every run stamps which cache it used (`axis_quant_cache`), so no log is ambiguous
# and no old result is silently reinterpreted.
def make_quant_cache(config, nbits, q_group_size=64, residual_length=128):
    """returns (cache, mode_label)"""
    import os
    mode = os.environ.get("QUANT_CACHE", "append_only")
    if mode == "hf_requant":
        from transformers.cache_utils import HQQQuantizedCache
        return HQQQuantizedCache(config=config, nbits=nbits, q_group_size=q_group_size,
                                 residual_length=residual_length), "hf_requant"
    if mode != "append_only":
        raise SystemExit(f"❌ QUANT_CACHE={mode!r}: expected 'append_only' or 'hf_requant'")
    return AppendOnlyHQQCache(config=config, nbits=nbits, q_group_size=q_group_size,
                              residual_length=residual_length), "append_only"


def layer_snapshot(layer):
    """rollback state for either layout (append-only blocks, or HF's single re-quantized buffer)"""
    if hasattr(layer, "snapshot"):
        return layer.snapshot()
    if not getattr(layer, "is_initialized", False):
        return None
    return (layer._quantized_keys, layer._quantized_values,
            layer.keys.clone(), layer.values.clone(), layer.cumulative_length)


def layer_restore(layer, s):
    if s is None:
        return
    if hasattr(layer, "restore"):
        layer.restore(s)
        return
    (layer._quantized_keys, layer._quantized_values,
     layer.keys, layer.values, layer.cumulative_length) = s
