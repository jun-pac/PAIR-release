#!/usr/bin/env python
"""Is committed KV quantized EXACTLY ONCE, as our code comment claims? (2026-08-31)

THE CLAIM UNDER TEST. `mtrag_accum.QuantStateful` documents "committed KV quantized EXACTLY ONCE
(as a real KV-quant decoder does)" and rolls back with snapshot/restore precisely so that a
crop-then-requantize cannot compound error. But that discipline only governs OUR rollback. HF's own
`QuantizedLayer.update()` does this on every residual flush:

    keys_to_return = cat(dequantize(_quantized_keys), keys, key_states)
    if keys.shape[-2] + 1 >= residual_length:
        _quantized_keys = quantize(keys_to_return)      # <-- the WHOLE cache, again

so every ~128 appended tokens the ENTIRE committed cache is dequantized and re-quantized. Over a
30-turn conversation the oldest tokens go through that round trip dozens of times. If the error
compounds, then a degradation that grows with accumulation depth is an artefact of THIS
implementation and says nothing about KV quantization — the same class of finding as its throughput.

WHAT THIS MEASURES, with no model loaded (synthetic KV on the real head geometry):
  1. DRIFT: keep a fixed committed block X, append turns, and after each turn compare the cache's
     dequantized reconstruction of X against (a) its FIRST reconstruction and (b) the true fp16 X.
     Quantized-exactly-once predicts a FLAT line; repeated requantization predicts a rising one.
  2. ROLLBACK: around each turn, snapshot/restore exactly as QuantStateful does, and assert the
     restored cache is bit-identical to the pre-generation cache and that cumulative_length is
     restored. That tests the specific failure a reviewer would suspect first.
"""
import json
import os
import sys

import torch
sys.path.insert(0, ".")
from transformers import AutoConfig
from transformers.cache_utils import HQQQuantizedCache

MODEL = os.environ.get("PROBE_MODEL", "Qwen/Qwen2.5-7B-Instruct")   # the fusion READER
NBITS = int(os.environ.get("PROBE_NBITS", "8"))
CTX = int(os.environ.get("PROBE_CTX", "4096"))
TURNS = int(os.environ.get("PROBE_TURNS", "30"))
GEN = int(os.environ.get("PROBE_GEN", "200"))
HIST = int(os.environ.get("PROBE_HIST", "120"))     # committed history block per turn


def snapshot(layer):
    """the five fields mtrag_accum.QuantStateful.snapshot() saves, or the layer's own if it has one"""
    if hasattr(layer, "snapshot"):
        return layer.snapshot()
    return (layer._quantized_keys, layer._quantized_values,
            layer.keys.clone(), layer.values.clone(), layer.cumulative_length)


def restore(layer, s):
    if hasattr(layer, "restore"):
        layer.restore(s)
        return
    (layer._quantized_keys, layer._quantized_values,
     layer.keys, layer.values, layer.cumulative_length) = s


def dequant_all(layer):
    """full fp reconstruction of everything the layer holds, in order — both cache layouts"""
    has_res = layer.keys.dim() == 4 and layer.keys.shape[-2] > 0
    if hasattr(layer, "_blocks_k"):                       # append-only: immutable blocks + residual
        parts = [layer._dequantize(b) for b in layer._blocks_k]
        if has_res:
            parts.append(layer.keys)
        return torch.cat(parts, dim=-2)
    dq = layer._dequantize(layer._quantized_keys)         # HF: one re-quantized buffer + residual
    return torch.cat([dq, layer.keys], dim=-2) if has_res else dq


def main():
    dev = "cuda"
    cfg = AutoConfig.from_pretrained(MODEL, cache_dir=os.environ.get("HF_HOME"))
    KV = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    HD = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    torch.manual_seed(0)
    if os.environ.get("PROBE_FIXED") == "1":
        from src.quantized_cache_append_only import AppendOnlyHQQCache
        cache = AppendOnlyHQQCache(config=cfg, nbits=NBITS, q_group_size=64, residual_length=128)
        print("[cache] AppendOnlyHQQCache (each token quantized exactly once)")
    else:
        cache = HQQQuantizedCache(config=cfg, nbits=NBITS, q_group_size=64, residual_length=128)
        print("[cache] HF HQQQuantizedCache (re-quantizes the whole cache on every flush)")
    layer = cache.layers[0]

    # ---- commit the context, in the harness's chunk sizes ----
    X = torch.randn(1, KV, CTX, HD, dtype=torch.bfloat16, device=dev)
    pos = 0
    while pos < CTX:
        n = min(4096, CTX - pos)
        cache.update(X[:, :, pos:pos + n], X[:, :, pos:pos + n], 0)
        pos += n
    cache.update(torch.randn(1, KV, 8, HD, dtype=torch.bfloat16, device=dev),
                 torch.randn(1, KV, 8, HD, dtype=torch.bfloat16, device=dev), 0)   # flush trigger
    D0 = dequant_all(layer)[:, :, :CTX].float().clone()
    Xf = X.float()
    base_err = (D0 - Xf).abs().mean().item()
    print(f"model={MODEL} nbits={NBITS} ctx={CTX} turns={TURNS}")
    print(f"turn  |X-dequant| (vs true fp16)   drift-from-first   rollback-exact   cum_len")
    print(f"   0   {base_err:.6e}              0.000000e+00        —              {layer.cumulative_length}")

    rows = [dict(turn=0, err_vs_true=base_err, drift=0.0, rollback_ok=None,
                 cumulative_length=layer.cumulative_length)]
    for t in range(1, TURNS + 1):
        # --- a turn: question + generation, rolled back exactly as QuantStateful does ---
        pre_len = layer.cumulative_length
        pre_dq = dequant_all(layer).clone()
        s = snapshot(layer)
        cache.update(torch.randn(1, KV, 40, HD, dtype=torch.bfloat16, device=dev),
                     torch.randn(1, KV, 40, HD, dtype=torch.bfloat16, device=dev), 0)
        for _ in range(GEN):
            cache.update(torch.randn(1, KV, 1, HD, dtype=torch.bfloat16, device=dev),
                         torch.randn(1, KV, 1, HD, dtype=torch.bfloat16, device=dev), 0)
        restore(layer, s)
        post_dq = dequant_all(layer)
        ok = bool(layer.cumulative_length == pre_len and post_dq.shape == pre_dq.shape
                  and torch.equal(post_dq, pre_dq))
        # --- commit this turn's history, which is what actually grows the cache ---
        cache.update(torch.randn(1, KV, HIST, HD, dtype=torch.bfloat16, device=dev),
                     torch.randn(1, KV, HIST, HD, dtype=torch.bfloat16, device=dev), 0)
        Dk = dequant_all(layer)[:, :, :CTX].float()
        err = (Dk - Xf).abs().mean().item()
        drift = (Dk - D0).abs().mean().item()
        rows.append(dict(turn=t, err_vs_true=err, drift=drift, rollback_ok=ok,
                         cumulative_length=layer.cumulative_length))
        if t <= 3 or t % 5 == 0 or t == TURNS:
            print(f"{t:4d}   {err:.6e}              {drift:.6e}        {ok!s:5s}          "
                  f"{layer.cumulative_length}")
    out = f"results/timing/hqq_requant_drift_{MODEL.split('/')[-1]}_int{NBITS}.json"
    json.dump(rows, open(out, "w"), indent=2)
    bad = [r for r in rows if r["rollback_ok"] is False]
    print(f"\nROLLBACK: {'ALL EXACT' if not bad else f'{len(bad)} TURNS BROKEN'}")
    print(f"REQUANTIZATION DRIFT over {TURNS} turns: "
          f"{rows[0]['err_vs_true']:.6e} -> {rows[-1]['err_vs_true']:.6e} "
          f"(x{rows[-1]['err_vs_true'] / max(rows[0]['err_vs_true'], 1e-30):.3f})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
