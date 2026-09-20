"""Compact per-row KV + flash_attn_with_kvcache backend for the batched accum harness (2026-08-28).

WHY (HANDOFF_260827 A4.4). The eager path spends 31-38% of its GPU time recopying the DynamicCache
every step, and HF's static path forces sdpa whose decode attention costs 13x flash — and HF's own
FA2 cannot fix it because its cache handling unpads with host-side indices (CUDA-graph-unsound).
The sound kernel is flash_attn.flash_attn_with_kvcache: preallocated cache, DEVICE-side per-row
lengths, in-kernel append. Probes 3041327/3041349: EXACT (100% argmax vs the deployed path,
max|dlogit|=0.0000), reader x3.26 / teacher x2.10 under one CUDA graph.

WHY THE HARNESS CANNOT PASS ITS MASK: BatchedStateful keeps a rectangular cache by LEFT-padding
every appended chunk, so a row's dead slots sit scattered at chunk boundaries mid-cache; the kernel
has no arbitrary-mask input (cache_leftpad covers only one contiguous left pad). So this backend
stores each row COMPACTLY — row b holds only its real tokens, `lens[b]` of them — and the geometry
works out with NO masking machinery at all:

  * appends stay LEFT-padded [B, T] chunks. The chunk's real k/v are first written compactly at
    [lens[b], lens[b]+t_b) (a per-row copy; appends are rare), then ONE kernel call attends the
    whole chunk against the cache with causal=True. The kernel's causal mask is aligned BOTTOM-RIGHT:
    query column j of row b sees cache slots [0, lens_b + t_b - T + j + 1) — for the row's real
    columns (j >= T - t_b) that is exactly "the past plus my own prefix", and a pad column's output
    is garbage that no real position ever attends (pads are never written to the cache) and the
    harness never reads (logits_to_keep=1 reads the last column, which left-padding makes real).
  * decode steps pass k=/v= and the kernel appends at lens[b] and attends in one call. Rows the
    harness has frozen still get their pad token appended (their own later outputs are discarded and
    no other row can see them — the same argument _turn_blocked already relies on); restore() puts
    `lens` back so the turn's rollback stays exact.
  * crop(n) recovers each row's compact length from the rectangular mask: lens = mask[:, :n].sum(1).
    The rectangular mask/cache_len/pos bookkeeping is kept UNCHANGED so the harness cannot tell the
    storage changed.

Positions: the harness's per-row position_ids drive RoPE before this backend ever sees k, so
compact storage does not disturb positions (SpecPrefill's gappy positions would too — but press /
specprefill arms are NOT supported here; they keep the eager path).

Enable: FKV_DECODE=1 (mutually exclusive with STATIC_DECODE). FKV_GRAPH=1 additionally captures
the 1-token step in one CUDA graph per model, once per process (Phase 2): the graph reads lens /
positions / the fed token from fixed-address buffers, so it survives turns, appends and crops —
which is also why `lens` is only ever updated IN PLACE. Replay = the identical kernel sequence to
the FKV eager loop, so FKV_GRAPH generations must be byte-identical to FKV-eager (gated).
"""
from __future__ import annotations

import os

import torch

from src.batched_stateful import BatchedStateful


class FlashKVStateful(BatchedStateful):
    def __init__(self, model, tok, dev, batch, max_len=None, is_reader=False):
        super().__init__(model, tok, dev, batch)
        from flash_attn import flash_attn_with_kvcache  # noqa: F401  (fail fast if missing)
        # ★★ CAPACITY INVARIANT (2026-08-31). This used to be a DATASET-WIDE constant applied to
        # every slot of every layer, which reserved storage no sequence in the batch would ever use
        # and cut B_max — i.e. it made OUR OWN method look slower than it is. That is never an
        # acceptable default: a paper does not nerf its own method's implementation.
        #
        # `flash_attn_with_kvcache` needs *a* preallocated buffer, not a dataset-wide-maximum one,
        # and the CUDA graph needs fixed shapes only for the LIFE OF ONE STATEFUL — which the
        # harness builds per batch. So the batch's own requirement is the correct bound, and the
        # caller knows it (the contexts are tokenised before the decoder is constructed).
        #
        # Under-estimating must NOT be able to kill a run, so the buffer GROWS instead of asserting:
        # grow() reallocates, copies the live prefix and drops the captured graph for re-capture.
        # Measured waste is reported (`capacity_report`) so the invariant is auditable rather than
        # asserted — on the 2026-08-31 audit the old global constant over-reserved 2-9% on the runs
        # that were actually published.
        _fkv_is_reader = bool(is_reader)
        self.L = int(max_len or os.environ.get("STATIC_MAXLEN", "31744"))
        # FKV_INIT_LEN (2026-08-31) — TEST HOOK, no effect unless set. It caps the INITIAL
        # reservation so `_grow()` is forced to run, and exists because _grow() had never executed
        # in any real run: every published FKV log reports fkv_grows=0. The accum benches (LooGLE's
        # ~37k contexts, LoCoMo's 30 accumulating turns) are exactly where it would first fire, and
        # discovering a defect there means discarding a long run. With this set, a short job can
        # force the grow and check the answers are unchanged, which is a gate worth minutes rather
        # than a landmine worth hours. It only ever LOWERS the initial bound; the buffer still grows
        # to whatever the run needs, so the run's result must be identical either way.
        # It SETS the initial length rather than capping it (changed 2026-09-01). As a cap it could
        # only ever lower the bound, which is enough to force a grow but not enough to build the
        # control that separates "the copy corrupts" from "the buffer LENGTH changes the kernel's
        # reduction order": that control needs one leg that STARTS at the length the other leg grows
        # to. Test-only either way — unset, this line does nothing.
        # `role` is set by the harness right after construction, so the branch is identified by the
        # caller instead: FKV_INIT_LEN_S targets the reader (the branch that actually grows) and
        # FKV_INIT_LEN the whole decoder. Splitting them 2026-09-01 after a control OOM'd: setting
        # 32768 to build a same-final-length comparison also inflated the LM branch, which is a 32B
        # at 264.64 KiB per token — 16.9 GiB for two rows — while in the leg being compared against
        # only the READER had grown there. The control did not hold, and the leg died before the
        # script's own length-mismatch guard could say so.
        _init_cap = os.environ.get("FKV_INIT_LEN_S") if _fkv_is_reader else None
        _init_cap = _init_cap or os.environ.get("FKV_INIT_LEN")
        if _init_cap:
            self.L = int(_init_cap)
        self.L0 = self.L                      # what was reserved at construction
        self.peak_len = 0                     # the largest cache length any row actually reached
        self.grows = 0
        cfg = model.config
        hkv = cfg.num_key_value_heads
        hd = cfg.hidden_size // cfg.num_attention_heads
        self.kbuf = [torch.zeros(batch, self.L, hkv, hd, dtype=model.dtype, device=dev)
                     for _ in range(cfg.num_hidden_layers)]
        self.vbuf = [torch.zeros(batch, self.L, hkv, hd, dtype=model.dtype, device=dev)
                     for _ in range(cfg.num_hidden_layers)]
        self.lens = torch.zeros(batch, dtype=torch.int32, device=dev)   # per-row REAL kv length
        self._orig_impl = model.config._attn_implementation
        # Phase 2 (FKV_GRAPH=1): the 1-token step captured in ONE CUDA graph, ONCE per process.
        # Everything the step depends on that changes over time — lens, positions, the fed token —
        # is read from fixed-address device tensors, so the SAME graph stays valid across turns,
        # appends and crops; only the input buffers are refreshed before each replay. The replay
        # sequence is the identical kernel sequence to the FKV eager loop, so generations must be
        # byte-identical to FKV-eager (gated by run_fkv_validate; probes 3041327/3041398 already
        # showed graphed == eager-kernel argmax at every position).
        self._graph_on = os.environ.get("FKV_GRAPH", "0") == "1"
        self._graph = None
        self._register()

    # ---- the attention override ------------------------------------------------------------------
    def _register(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        name = f"fkv_{id(self)}"
        backend = self

        def fkv_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None,
                          **kwargs):
            # query [B,H,T,D]; key/value = THIS chunk only, post-rope (past_key_values is never
            # passed by this backend, so HF hands us just the new tokens).
            from flash_attn import flash_attn_with_kvcache
            li = module.layer_idx
            q = query.transpose(1, 2)
            T = q.shape[1]
            if T == 1:
                # decode step: in-kernel append at lens[b] + attend, one call
                o = flash_attn_with_kvcache(
                    q, backend.kbuf[li], backend.vbuf[li],
                    k=key.transpose(1, 2), v=value.transpose(1, 2),
                    cache_seqlens=backend.lens, causal=True, softmax_scale=scaling)
                if li == len(backend.kbuf) - 1:
                    backend.lens.add_(1)
                return o, None
            # append path: write the chunk's REAL tokens compactly, then attend the whole chunk
            # against the cache with bottom-right causal alignment (see module docstring).
            k, v = key.transpose(1, 2), value.transpose(1, 2)
            treal = backend._chunk_real                 # [B] python list, set by forward()
            lens_l = backend._lens_list                 # python list snapshot (pre-append)
            for b in range(backend.B):
                t = treal[b]
                if t == 0:
                    continue
                lo = lens_l[b]
                backend.kbuf[li][b, lo:lo + t] = k[b, T - t:]
                backend.vbuf[li][b, lo:lo + t] = v[b, T - t:]
            new_lens = backend._new_lens                # [B] int32 device (pre-computed)
            o = flash_attn_with_kvcache(
                q, backend.kbuf[li], backend.vbuf[li],
                cache_seqlens=new_lens, causal=True, softmax_scale=scaling)
            return o, None

        ALL_ATTENTION_FUNCTIONS[name] = fkv_attention
        self._impl_name = name

    @torch.no_grad()
    def _grow(self, need):
        """Reallocate to hold `need` tokens, preserving what is already cached.

        Growing costs one copy of the live prefix and one CUDA-graph re-capture, both rare by
        construction (the caller sizes the buffer from the batch it is about to run). It exists so a
        low estimate degrades into a small cost instead of killing a job mid-run, which is what the
        old assert did."""
        new_L = max(int(need * 1.25), self.L * 2)
        for buf in (self.kbuf, self.vbuf):
            for i, old in enumerate(buf):
                grown = torch.zeros(old.shape[0], new_L, old.shape[2], old.shape[3],
                                    dtype=old.dtype, device=old.device)
                grown[:, :self.L] = old
                buf[i] = grown
        self.L = new_L
        self._graph = None          # shapes changed: the captured step must be re-captured
        self.grows += 1
        print(f"[FKV] grew cache {self.L0} -> {new_L} (needed {need}); "
              f"grow #{self.grows}", flush=True)

    def capacity_report(self):
        """what was reserved vs what was used — the invariant, reported rather than assumed"""
        return dict(fkv_reserved_len=self.L, fkv_initial_len=self.L0,
                    fkv_peak_len=self.peak_len, fkv_grows=self.grows,
                    fkv_unused_frac=(None if not self.L else
                                     round(1 - self.peak_len / self.L, 4)))

    @torch.no_grad()
    def release(self):
        """Free the preallocated buffers AND drop this backend's global registration.

        ★ WHY THIS EXISTS (job 3055450, 2026-08-31). `ALL_ATTENTION_FUNCTIONS` is a MODULE-LEVEL dict
        and the closure above captures `backend = self`, so registering pins the object forever: the
        buffers are unreachable to the caller but never collected. A harness that builds one stateful
        per BATCH therefore leaks a FULL preallocated cache every batch — 11.8 GiB per batch for the
        32B at STATIC_MAXLEN=11264 — and the second batch OOMs no matter how small the batch is. It
        never showed on LoCoMo because those runs passed exactly B conversations and so only ever
        built ONE. The musique 96-question ladder OOM'd at EVERY rung, including teacher B=4 and
        floor-7B B=64, which is the signature: an OOM that does not improve as the batch shrinks is
        not a capacity limit, it is a leak.
        """
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        ALL_ATTENTION_FUNCTIONS.pop(getattr(self, "_impl_name", ""), None)
        self.kbuf = []
        self.vbuf = []
        self._graph = None
        self._graph_buffers = None

    # ---- BatchedStateful surface -----------------------------------------------------------------
    def _devguard(self):
        """torch.cuda.device(self.dev) when FKV_DEVGUARD=1, else a no-op.

        `flash_attn_with_kvcache` is a custom CUDA op, not an ATen one, so it is not covered by the
        dispatcher's automatic device guard the way every matmul in the model is. Whether flash-attn
        2.8.3 sets its own guard internally decides whether a second card works at all here, and the
        installed build is a compiled .so -- so this is a switch to MEASURE against, not a fix to
        assume. Default off keeps every existing single-card run byte-identical.
        """
        import contextlib
        # DEFAULT ON since 2026-09-01. `_capture()` is reached from inside step(), so this context is
        # what puts the CUDA-graph capture on the module's own card; probe 3062126 without it captured
        # an EMPTY graph on cuda:1 (replay a no-op, logits frozen, lens 8 short after 8 steps) and
        # probe 3062128 with it is EXACT at every prefill block and every step, under all three
        # capture bindings. On one card it sets the device that is already current, so single-card
        # runs are unchanged. FKV_DEVGUARD=0 exists only to reproduce the defect.
        return (contextlib.nullcontext() if os.environ.get("FKV_DEVGUARD", "1") == "0"
                else torch.cuda.device(self.dev))

    @torch.no_grad()
    def forward(self, ids, chunk_mask, chunk=None):
        with self._devguard():
            return self._forward(ids, chunk_mask, chunk)

    @torch.no_grad()
    def step(self, ids, alive):
        with self._devguard():
            return self._step(ids, alive)

    @torch.no_grad()
    def _forward(self, ids, chunk_mask, chunk=None):
        from src.batched_stateful import PREFILL_CHUNK
        C = chunk if chunk is not None else PREFILL_CHUNK
        last = None
        self.model.config._attn_implementation = self._impl_name
        try:
            for s, e in self._slices(chunk_mask, C):
                part_ids, part_mask = ids[:, s:e], chunk_mask[:, s:e]
                n = e - s
                treal = part_mask.sum(-1)
                need = int(self.lens.max()) + n
                if need > self.L:
                    self._grow(need)
                self.peak_len = max(self.peak_len, need)
                self._chunk_real = [int(t) for t in treal.tolist()]
                self._lens_list = [int(x) for x in self.lens.tolist()]
                self._new_lens = (self.lens + treal.to(torch.int32))
                out = self.model(part_ids, use_cache=False, logits_to_keep=1,
                                 position_ids=self._position_ids(part_mask))
                self.lens.copy_(self._new_lens)   # in place: the graph holds this address
                self.cache_len += n
                self.mask = torch.cat([self.mask, part_mask], dim=1)
                self.pos = self.pos + part_mask.sum(-1)
                last = out.logits[:, -1, :]
                del out
        finally:
            self.model.config._attn_implementation = self._orig_impl
        return last

    @torch.no_grad()
    def _step(self, ids, alive):
        """1-token decode. The kernel appends per row at lens[b] (lens advances for EVERY row,
        including frozen ones being fed pads — their outputs are discarded, no other row attends
        them, and restore() rolls lens back)."""
        need = int(self.lens.max()) + 1
        if need > self.L:
            self._grow(need)
        self.peak_len = max(self.peak_len, need)
        if self._graph_on:
            logits = self._graph_step(ids, alive)
        else:
            self.model.config._attn_implementation = self._impl_name
            try:
                out = self.model(ids, use_cache=False, logits_to_keep=1,
                                 position_ids=(self.pos[:, None] * alive))
            finally:
                self.model.config._attn_implementation = self._orig_impl
            logits = out.logits[:, -1, :]
        self.cache_len += 1
        self.mask = torch.cat([self.mask, alive], dim=1)
        self.pos = self.pos + alive[:, 0]
        return logits

    def _graph_step(self, ids, alive):
        if self._graph is None:
            self._capture(ids, alive)
        self._gids.copy_(ids)
        self._gpos.copy_(self.pos[:, None] * alive)
        with torch.cuda.device(self.dev):          # replay uses the CURRENT device, see _capture
            self._graph.replay()
        # clone: the graph's output buffer is overwritten by this branch's NEXT replay; a stale
        # reference held across steps was exactly the 2977420 crash class. [B, V] bf16 ~ 2.4 MB.
        return self._gout.clone()

    @torch.no_grad()
    def _capture(self, ids, alive):
        """Capture once. Warmup + capture both APPEND garbage into the cache at lens..lens+3 and
        advance lens; the pre-capture state is restored afterwards, and those slots are only ever
        rewritten by later real appends (nothing attends beyond lens), so the rollback is exact."""
        st = (self.lens.clone(), self.cache_len, self.mask, self.pos)
        self._gids = ids.clone()
        self._gpos = (self.pos[:, None] * alive).clone()
        self.model.config._attn_implementation = self._impl_name
        try:
            # ★ CAPTURING ON A SECOND CARD (2026-09-01). With the LM on cuda:1 this capture produced
            # an EMPTY graph -- torch says so itself ("The CUDA Graph is empty. This usually means
            # that the graph was attempted to be captured on wrong device or stream") -- so replay
            # was a no-op, `_gout` stayed frozen at its capture-time value and `lens` never advanced.
            # Probe 3062126 measured it exactly: forward() is EXACT on cuda:1 (max|dlogit| 0.0000 at
            # every prefill block, so flash_attn_with_kvcache and the compact store are innocent),
            # step0 is exact, step1 onward diverge by |dlogit| up to 35, and lens ends 8 short after
            # 8 steps -- one missing append per step. That is what made the two-card split answer
            # differently on every turn and look "2x faster": the 32B branch was not running.
            # FKV_CAPMODE selects how the capture is bound to this module's device; the default is
            # the one the probe showed correct. "devguard" is kept only to reproduce the defect.
            _mode = os.environ.get("FKV_CAPMODE", "stream")
            _prev_dev = torch.cuda.current_device()
            if _mode in ("setdev", "stream"):
                torch.cuda.set_device(self.dev)
            _devctx = torch.cuda.device(self.dev); _devctx.__enter__()
            s = torch.cuda.Stream(self.dev)
            s.wait_stream(torch.cuda.current_stream(self.dev))
            with torch.cuda.stream(s):
                for _ in range(3):
                    self.model(self._gids, use_cache=False, logits_to_keep=1,
                               position_ids=self._gpos)
            torch.cuda.current_stream(self.dev).wait_stream(s)
            torch.cuda.synchronize(self.dev)
            self.lens.copy_(st[0])                      # rewind warmup's 3 appends
            g = torch.cuda.CUDAGraph()
            _cap = torch.cuda.Stream(self.dev)
            with (torch.cuda.graph(g, stream=_cap) if _mode == "stream" else torch.cuda.graph(g)):
                out = self.model(self._gids, use_cache=False, logits_to_keep=1,
                                 position_ids=self._gpos)
                self._gout = out.logits[:, -1, :]
            self._graph = g
        finally:
            self.model.config._attn_implementation = self._orig_impl
            _devctx.__exit__(None, None, None)
            if _mode in ("setdev", "stream"):
                torch.cuda.set_device(_prev_dev)
        self.lens.copy_(st[0])                          # rewind the capture's append too
        self.cache_len, self.mask, self.pos = st[1], st[2], st[3]

    @torch.no_grad()
    def crop(self, n):
        self.cache_len = n
        self.mask = self.mask[:, :n]
        self.lens.copy_(self.mask.sum(1).to(torch.int32))   # in place: the graph holds this address

    def state(self):
        return (self.cache_len, self.pos.clone(), self.mask.clone(), self.lens.clone())

    def restore(self, st):
        n, pos, mask, lens = st
        self.cache_len = n
        self.pos, self.mask = pos.clone(), mask.clone()
        self.lens.copy_(lens)                               # in place: the graph holds this address

    # press / specprefill ingest paths are NOT implemented on this backend on purpose: they splice
    # foreign KV into the cache with their own layouts. Refuse loudly instead of being subtly wrong.
    def compress_append(self, *a, **k):
        raise NotImplementedError("FKV_DECODE supports teacher/floor/ours only; run press arms on "
                                  "the eager path")

    def specprefill_append(self, *a, **k):
        raise NotImplementedError("FKV_DECODE supports teacher/floor/ours only; run SpecPrefill on "
                                  "the eager path")
