"""Compiled decode on a StaticCache backend for the batched accum harness (2026-08-19).

WHY. The eager decode loop runs 97-149 ms/step against a ~22 ms bandwidth roofline; the gap is
CPU-side per-layer dispatch. torch.compile can only remove it under stable shapes AND stable tensor
addresses; a growing DynamicCache provides neither (measured: compile+DynamicCache made decode
17-22% SLOWER, job 2977045). A first design that COPIED the committed KV into a per-turn throwaway
StaticCache was rejected on arithmetic: it duplicates the committed KV (teacher LoCoMo B=3: +23 GiB
-> OOM).

DESIGN. `StaticBatchedStateful` swaps the storage: ONE preallocated StaticCache per model for the
whole conversation batch (no duplication), a fixed-width [B, L] attention-mask buffer, and
  * forward(): the parent's chunked eager path, writing the mask in place (prefill shapes vary ->
    eager; identical math to the parent);
  * step(): a 1-token forward through torch.compile with persistent input buffers - constant
    shapes, constant addresses, so capture happens once;
  * crop(n): POINTER REWIND - cache_len/mask are cut to n; stale KV beyond n is never attended
    (mask zero) and is overwritten by the next append at those cache_positions. This preserves the
    harness's exact-rollback contract without touching DynamicCache semantics.
bf16 kernel differences under compile may flip some greedy tokens vs eager (user: noise) -
STATIC_DECODE=1 is a TIMING arm until scores are re-validated on it.

Enable: STATIC_DECODE=1 [STATIC_MAXLEN=36864] [STATIC_COMPILE_MODE=reduce-overhead].
"""
from __future__ import annotations

import os
import torch

from src.batched_stateful import BatchedStateful


class StaticBatchedStateful(BatchedStateful):
    def __init__(self, model, tok, dev, batch, max_len=None):
        super().__init__(model, tok, dev, batch)
        from transformers import StaticCache
        self.L = int(max_len or os.environ.get("STATIC_MAXLEN", "31744"))
        self.cache = StaticCache(config=model.config, max_batch_size=batch,
                                 max_cache_len=self.L, device=dev, dtype=model.dtype)
        self.fullmask = torch.zeros(batch, self.L, dtype=torch.long, device=dev)
        self.mask = self.fullmask[:, :0]
        self._ids = torch.zeros(batch, 1, dtype=torch.long, device=dev)
        self._posb = torch.zeros(batch, 1, dtype=torch.long, device=dev)
        self._cpos = torch.zeros(1, dtype=torch.long, device=dev)
        mode = os.environ.get("STATIC_COMPILE_MODE", "reduce-overhead")
        self._compiled = torch.compile(model.forward, mode=mode, fullgraph=False, dynamic=False)

    @torch.no_grad()
    def forward(self, ids, chunk_mask, chunk=None):
        """Parent's chunked append, on the static cache + in-place mask (eager: prefill shapes vary)."""
        from src.batched_stateful import PREFILL_CHUNK
        C = chunk if chunk is not None else PREFILL_CHUNK
        last = None
        for s, e in self._slices(chunk_mask, C):
            part_ids, part_mask = ids[:, s:e], chunk_mask[:, s:e]
            n = e - s
            assert self.cache_len + n <= self.L, f"STATIC_MAXLEN {self.L} exceeded at {self.cache_len}+{n}"
            self.fullmask[:, self.cache_len:self.cache_len + n] = part_mask
            attn = self.fullmask[:, :self.cache_len + n]
            out = self.model(part_ids, past_key_values=self.cache, use_cache=True, logits_to_keep=1,
                             position_ids=self._position_ids(part_mask),
                             cache_position=torch.arange(self.cache_len, self.cache_len + n, device=self.dev),
                             attention_mask=attn)
            self.cache_len += n
            self.mask = attn
            self.pos = self.pos + part_mask.sum(-1)
            last = out.logits[:, -1, :]
            del out
        return last

    @torch.no_grad()
    def step(self, ids, alive):
        """Compiled 1-token step: fixed shapes ([B,1] ids, [B,L] mask), fixed addresses."""
        assert self.cache_len + 1 <= self.L, f"STATIC_MAXLEN {self.L} exceeded"
        self._ids.copy_(ids)
        self.fullmask[:, self.cache_len:self.cache_len + 1] = alive
        self._posb.copy_(self.pos[:, None] * alive)
        self._cpos.fill_(self.cache_len)
        out = self._compiled(input_ids=self._ids, attention_mask=self.fullmask,
                             past_key_values=self.cache, use_cache=True,
                             position_ids=self._posb, cache_position=self._cpos)
        self.cache_len += 1
        self.mask = self.fullmask[:, :self.cache_len]
        self.pos = self.pos + alive[:, 0]
        # .clone(): the compiled step's logits live in a CUDA-graph output buffer that the NEXT replay
        # (e.g. the other fusion branch's step) overwrites — reading it later raises "accessing tensor
        # output of CUDAGraphs that has been overwritten" (job 2977420). [B, V] bf16 ~ 1 MB per step.
        return out.logits[:, -1, :].clone()

    @torch.no_grad()
    def crop(self, n):
        self.fullmask[:, n:].zero_()
        self.cache_len = n
        self.mask = self.fullmask[:, :n]


def make_stateful(model, tok, dev, batch, max_len=None, quant_nbits=None, is_reader=False):
    """max_len sizes the preallocated cache. Job 2977230 OOM'd because BOTH fusion branches got the
    full-context length: the LM branch (no-accum: instruction+question+generation only) was handed a
    27 GiB static cache it can never fill. Callers pass the branch's true bound."""
    if quant_nbits is not None:
        # HQQ-quantized persistent KV (2026-08-31). Mutually exclusive with the fixed-path backends:
        # both preallocate a dense fp buffer, which is the opposite of what a quantized cache is for,
        # and silently ignoring the request would produce a row labelled "quant" that is not.
        if os.environ.get("FKV_DECODE", "0") == "1" or os.environ.get("STATIC_DECODE", "0") == "1":
            raise SystemExit("\u274c quant_nbits with FKV_DECODE/STATIC_DECODE: the fixed-path decoders "
                             "preallocate a dense fp KV buffer and cannot hold a quantized cache. "
                             "Run the quant arm on the default batched path (no FKV_DECODE).")
        from src.batched_stateful import BatchedQuantStateful
        return BatchedQuantStateful(model, tok, dev, batch, nbits=quant_nbits)
    if os.environ.get("FKV_DECODE", "0") == "1":
        # compact per-row cache + flash_attn_with_kvcache (2026-08-28, probes 3041327/3041349:
        # exact and reader x3.26 / teacher x2.10). teacher/floor/ours only; presses refuse.
        from src.flash_kvcache_decode import FlashKVStateful
        return FlashKVStateful(model, tok, dev, batch, max_len=max_len, is_reader=is_reader)
    if os.environ.get("STATIC_DECODE", "0") == "1":
        return StaticBatchedStateful(model, tok, dev, batch, max_len=max_len)
    return BatchedStateful(model, tok, dev, batch)
