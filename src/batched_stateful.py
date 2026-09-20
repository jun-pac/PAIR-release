"""Batched stateful KV decoding for the accumulate harness — B conversations at once.

WHY. `mtrag_accum.py` has always run at BATCH = 1: one conversation's KV cache, appended per turn, rolled
back after generation, reference answer committed. That wastes GPU on every experiment, and at batch 1 the
flash-attention graph does not compile, so any timing measured there is an artefact rather than the method's
speed. The standing instruction is that every experiment runs at batch >= 2; it was never enforced in code
for this harness, which is why it kept not happening.

THE DESIGN, and the one invariant that makes it simple:

    **every sequence in the batch is kept at the SAME cache length at all times.**

That is achieved by LEFT-padding every chunk that is appended (the context prefill, each question block,
each committed answer). Left-padding — not right — is what makes it work:
  * the last position of the chunk is the last REAL token for every sequence, so `logits_to_keep=1` returns
    each sequence's true next-token logits;
  * cache lengths stay equal, so `DynamicCache.crop(n)` is an exact per-sequence rollback rather than a
    ragged one;
  * pads are masked out of attention, so they cannot influence any real token.

Positions must still be per-sequence: sequence i has consumed a different number of REAL tokens than
sequence j, so `position_ids` is built per row from each sequence's own running `pos`, while
`cache_position` is shared (the cache is rectangular).

Finished sequences are frozen: they are fed a pad with mask 0 and their `pos` does not advance, so their KV
and positions stop changing while the rest of the batch continues.

Correctness is not assumed — `verify_batched_equivalence()` decodes the same conversations at B=1 and B=N
and asserts the generated token ids are identical.
"""
from __future__ import annotations

import os
import time as _time

import torch
from transformers import DynamicCache

# columns fed per forward when appending a long block; see BatchedStateful.forward
PREFILL_CHUNK = int(os.environ.get("PREFILL_CHUNK", "4096"))

# How many decode steps run between device->host syncs. 1 = the original loop: every step ends with a
# `.tolist()` on the chosen tokens, which STOPS THE CPU until the GPU has finished that step, so the
# launch chain for step t+1 can only begin after step t has executed. With K>1 the chosen tokens stay on
# the GPU and are fed straight back as the next input, and the CPU-side stop bookkeeping runs once per
# block on a [B, K] tolist(). See BatchedFusion.turn for why the generated tokens are unchanged.
DECODE_SYNC_EVERY = int(os.environ.get("DECODE_SYNC_EVERY", "1"))


class BatchedStateful:
    """One model, B sequences, a rectangular KV cache, exact rollback."""

    def __init__(self, model, tok, dev, batch):
        self.model, self.tok, self.dev, self.B = model, tok, dev, batch
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        self.cache = DynamicCache()
        self.cache_len = 0
        self.pos = torch.zeros(batch, dtype=torch.long, device=dev)      # REAL tokens consumed per sequence
        self.mask = torch.zeros((batch, 0), dtype=torch.long, device=dev)

    def ids_of(self, texts):
        """tokenize a list of B strings -> a LEFT-padded [B, T] block plus its validity mask"""
        import os
        if os.environ.get("DUMP_BLOCKS"):
            with open(os.environ["DUMP_BLOCKS"], "a") as _f:
                _f.write(f"<<<BAT|{getattr(self,'role','?')}>>>{list(texts)[0]!r}\n")
        seqs = [self.tok(t, return_tensors="pt", add_special_tokens=False).input_ids[0] for t in texts]
        T = max(int(s.shape[0]) for s in seqs)
        ids = torch.full((self.B, T), self.pad, dtype=torch.long)
        m = torch.zeros((self.B, T), dtype=torch.long)
        for i, s in enumerate(seqs):
            n = int(s.shape[0])
            ids[i, T - n:] = s                    # LEFT pad: the real tokens end at the last column
            m[i, T - n:] = 1
        return ids.to(self.dev), m.to(self.dev)

    def _position_ids(self, chunk_mask):
        """real tokens continue each sequence's own position counter; pads reuse position 0 (masked out)"""
        # positions within the chunk: 0,1,2... over the REAL tokens only
        within = (chunk_mask.cumsum(-1) - 1).clamp(min=0)
        return (self.pos[:, None] + within) * chunk_mask

    def _slices(self, chunk_mask, C):
        """Split [B,T] into column ranges of ~C, never emitting a range where SOME ROW IS ALL PAD.

        A row with no real token in the slice would attend to nothing on the first forward (empty cache)
        and produce NaN, so a range is extended until every row has at least one real token in it. With
        left padding the last column is real for every row, so the final range is always valid.
        """
        B, T = chunk_mask.shape
        out, s = [], 0
        while s < T:
            e = min(s + C, T)
            while e < T and not bool(chunk_mask[:, s:e].sum(1).gt(0).all()):
                e = min(e + C, T)
            out.append((s, e))
            s = e
        return out

    @torch.no_grad()
    def forward(self, ids, chunk_mask, chunk=None):
        """Append a block. Long blocks (the context prefill) are fed in COLUMN SLICES: activation memory
        scales with the slice, not with the 23k-token context, which is what OOM'd 32B+7B at batch 3 on a
        95 GiB card (75.4 GiB is weights). The KV built is identical either way — this is not an
        approximation, only a different order of the same forwards."""
        C = chunk if chunk is not None else PREFILL_CHUNK
        last = None
        for s, e in self._slices(chunk_mask, C):
            part_ids, part_mask = ids[:, s:e], chunk_mask[:, s:e]
            n = e - s
            attn = torch.cat([self.mask, part_mask], dim=1)
            out = self.model(part_ids, past_key_values=self.cache, use_cache=True, logits_to_keep=1,
                             position_ids=self._position_ids(part_mask),
                             cache_position=torch.arange(self.cache_len, self.cache_len + n, device=self.dev),
                             attention_mask=attn)
            self.cache = out.past_key_values
            self.cache_len += n
            self.mask = attn
            self.pos = self.pos + part_mask.sum(-1)
            last = out.logits[:, -1, :]
            del out
        return last


    @torch.no_grad()
    def compress_append(self, passage_texts, qcond_texts, make_press, press_method, ratio, query_dep):
        """Batched twin of Stateful.compress_append — the *_frozen / h2o ingest.

        ★ WHY THIS IS NOT A ONE-LINER. The press keeps `ratio x block_len` tokens, and block_len differs per
        row, so the rows end up with DIFFERENT kept counts. BatchedStateful requires every sequence to share
        one `cache_len` — that shared length is what makes crop() an exact per-sequence rollback and
        logits_to_keep=1 return each row's true last-token logits. So each row is compressed on its own (the
        press is query-dependent for snapkv/pyramidkv, so it cannot be run as one padded batch without the
        padding entering the observation window), then every row is LEFT-PADDED to K = max_i keep_i and the
        pad columns are marked 0 in the mask. Shared length restored; the cost is dead columns, not wrong
        results.

        `pos` advances by each row's FULL block length, not by what was kept, exactly as the sequential path
        does — later turns must sit at their true original positions.
        """
        import torch as _t
        from transformers import DynamicCache
        per_row, keeps, fulls = [], [], []
        for i, (ptxt, qtxt) in enumerate(zip(passage_texts, qcond_texts)):
            p_ids = self.tok(ptxt, return_tensors="pt", add_special_tokens=False).input_ids.to(self.dev)
            if query_dep:
                q_ids = self.tok(qtxt, return_tensors="pt", add_special_tokens=False).input_ids.to(self.dev)
                ids = _t.cat([p_ids, q_ids], dim=1); window = min(int(q_ids.shape[1]), 64)
            else:
                ids = p_ids; window = 0
            n = int(ids.shape[1]); pos_i = int(self.pos[i])
            # A press cannot keep fewer tokens than its own observation window: at high removal
            # ratios on a short row int(n*(1-ratio)) < window, get_seq_length()-window goes negative
            # and the left-pad allocation fails (CLUTRR r0.95, job 2969722). Floor the per-row budget
            # at window + 16 kept tokens (ratio 0 when the row is shorter than that); the EFFECTIVE
            # kept count is what `keeps` returns, so the log still records what actually happened.
            eff_ratio = min(ratio, max(0.0, 1.0 - (window + 16) / max(n, 1)))
            with make_press(press_method, eff_ratio, window)(self.model):
                out = self.model(ids, past_key_values=DynamicCache(), use_cache=True, logits_to_keep=1,
                                 position_ids=_t.arange(pos_i, pos_i + n, device=self.dev)[None],
                                 cache_position=_t.arange(n, device=self.dev))
            comp = out.past_key_values
            keep_i = max(int(comp.get_seq_length()) - window, 0)
            per_row.append(comp); keeps.append(keep_i); fulls.append(int(p_ids.shape[1]))
            del out
        K = max(keeps)
        mask = _t.zeros((self.B, K), dtype=_t.long, device=self.dev)
        for i, keep_i in enumerate(keeps):
            mask[i, K - keep_i:] = 1                      # LEFT pad, like ids_of
        for L in range(len(self.cache.layers)):
            ks, vs = [], []
            for i, keep_i in enumerate(keeps):
                k_i = per_row[i].layers[L].keys[:, :, :keep_i, :]
                v_i = per_row[i].layers[L].values[:, :, :keep_i, :]
                if keep_i < K:                            # left-pad this row's KV with zeros (masked out)
                    pad = (0, 0, K - keep_i, 0)           # pad the SEQ dim on the left
                    k_i = _t.nn.functional.pad(k_i, pad); v_i = _t.nn.functional.pad(v_i, pad)
                ks.append(k_i); vs.append(v_i)
            self._append_layer_kv(L, _t.cat(ks, dim=0), _t.cat(vs, dim=0))
        self.cache_len += K
        self.mask = _t.cat([self.mask, mask], dim=1)
        self.pos = self.pos + _t.tensor(fulls, device=self.dev, dtype=self.pos.dtype)
        return keeps, K

    @torch.no_grad()
    def specprefill_append(self, passage_texts, qcond_texts, spec_model, keep,
                           chunk, pool, look_ahead):
        """Batched twin of Stateful.specprefill_append (2026-08-24) — SpecPrefill was the LAST arm
        stuck on the sequential path, which meant its timing could not share a table with the batched
        teacher/ours/press arms (different decode path, no flash graph). Its selection is per row (the
        speculator scores THIS row's context against THIS row's question), but the LM prefill of the
        selected tokens is ONE batched forward, exactly like every other append here:
          * per row: speculator -> chunk scores -> Top-K chunk mask -> selected token indices sel_i;
          * the ragged sel_i are LEFT-PADDED to K = max_i |sel_i| (pad slots masked 0), restoring the
            shared cache length the batched cache is built on;
          * position_ids carry each selected token's ORIGINAL position (self.pos[i] + sel), so later
            turns still sit at their true positions — the §3.2.4 bookkeeping of the sequential path;
          * gappy positions REQUIRE sdpa (flash would read the gaps as packed-sequence boundaries).
        `pos` advances by the FULL block length per row; the cache grows by K.
        """
        import torch as _t
        from src import specprefill as SP
        sels, fulls = [], []
        for i, (ptxt, qtxt) in enumerate(zip(passage_texts, qcond_texts)):
            p_ids = self.tok(ptxt, return_tensors="pt", add_special_tokens=False).input_ids.to(self.dev)
            q_ids = self.tok(qtxt, return_tensors="pt", add_special_tokens=False).input_ids.to(self.dev)
            P = int(p_ids.shape[1])
            scores = SP.speculate_scores(spec_model, _t.cat([p_ids, q_ids], dim=1), 0, P,
                                         look_ahead=look_ahead)
            keep_mask, _, _ = SP.select_chunks(scores, keep, chunk, pool)
            sel = _t.nonzero(keep_mask, as_tuple=False).flatten()
            if sel.numel() == 0:
                sel = _t.arange(min(chunk, P))
            sels.append((p_ids, sel.to(self.dev)))
            fulls.append(P)
        K = max(int(s.numel()) for _, s in sels)
        ids = _t.full((self.B, K), self.pad, dtype=_t.long, device=self.dev)
        msk = _t.zeros((self.B, K), dtype=_t.long, device=self.dev)
        pos = _t.zeros((self.B, K), dtype=_t.long, device=self.dev)
        for i, (p_ids, sel) in enumerate(sels):
            k = int(sel.numel())
            ids[i, K - k:] = p_ids[0, sel]                       # LEFT pad, like ids_of
            msk[i, K - k:] = 1
            pos[i, K - k:] = int(self.pos[i]) + sel              # ORIGINAL positions of the kept tokens
        attn = _t.cat([self.mask, msk], dim=1)
        with SP.sdpa_for_gappy_positions(self.model):
            out = self.model(ids, past_key_values=self.cache, use_cache=True, logits_to_keep=1,
                             position_ids=pos,
                             cache_position=_t.arange(self.cache_len, self.cache_len + K,
                                                      device=self.dev),
                             attention_mask=attn)
        self.cache = out.past_key_values
        self.cache_len += K
        self.mask = attn
        self.pos = self.pos + _t.tensor(fulls, device=self.dev, dtype=self.pos.dtype)
        return [int(s.numel()) for _, s in sels], K

    @torch.no_grad()
    def _append_layer_kv(self, L, k, v):
        """Write one already-built [B, H, K, D] KV block into layer L's store.

        The press/SpecPrefill ingests do not run a forward that the cache could intercept — they hand
        the cache a block of KV that was computed elsewhere — so they write it here. It is a hook
        rather than two inline `cat`s because a QUANTIZED store cannot be concatenated into: see
        BatchedQuantStateful._append_layer_kv, which is what makes press+quant composition possible."""
        lay = self.cache.layers[L]
        lay.keys = torch.cat([lay.keys, k], dim=2)
        lay.values = torch.cat([lay.values, v], dim=2)

    @torch.no_grad()
    def crop(self, n):
        """exact rollback — valid because every sequence shares the same cache length by construction"""
        self.cache.crop(n)
        self.cache_len = n
        self.mask = self.mask[:, :n]

    def state(self):
        return (self.cache_len, self.pos.clone(), self.mask.clone())

    def restore(self, st):
        n, pos, mask = st
        self.crop(n)
        self.pos, self.mask = pos.clone(), mask.clone()


class BatchedQuantStateful(BatchedStateful):
    """BatchedStateful whose persistent KV cache is HQQ-quantized — the batched twin of
    `mtrag_accum.QuantStateful`.

    WHY IT EXISTS (2026-08-31). The quant baseline was the last accuracy arm still forced onto the
    sequential path, and that had two costs, not one: (a) a quant arm could never appear in a
    throughput table at all, because `throughput_eval.py` refuses batch < 2 and a batch-1 wall is an
    artefact (the flash-attention graph does not compile) — so "how fast is the KV-quant baseline"
    was an unanswerable question rather than a measured one; (b) every quant accuracy run paid ~9 s
    per example where a batched arm pays a fraction of that. CLAUDE.md 0원칙 is explicit that a path
    which cannot batch gets FIXED rather than run at batch 1.

    THE ONE DIFFERENCE from the parent. A quantized cache cannot be `crop`ped: committed tokens are
    packed into per-layer quantized buffers with an fp residual of the most recent `residual_length`
    positions, and there is no exact way to cut that in the middle. So rollback is snapshot/restore
    of the buffer references plus a clone of the fp residual, exactly as the sequential QuantStateful
    does — which is also what keeps the committed KV quantized EXACTLY ONCE. Crop-then-requantize
    would compound the quantization error every turn and would wreck low-bit arms unfairly; that bug
    class is on record (int4 0.314 -> 0.601 when it was fixed).

    Everything else — left padding, the shared cache length, per-row positions, chunked prefill — is
    inherited unchanged, so a quant arm and a teacher arm differ only in the cache's dtype story.
    """

    def __init__(self, model, tok, dev, batch, nbits, q_group_size=64, residual_length=128):
        super().__init__(model, tok, dev, batch)
        from src.quantized_cache_append_only import make_quant_cache
        self.nbits = nbits
        self.cache, self.cache_mode = make_quant_cache(model.config, nbits, q_group_size,
                                                       residual_length)

    @torch.no_grad()
    def _append_layer_kv(self, L, k, v):
        """press / SpecPrefill ingest into a QUANTIZED store — the composition arm.

        The parent concatenates the block onto `layer.keys`, which for a QuantizedLayer is only the
        fp RESIDUAL, so the compressed context would silently stay full precision and the row would
        be labelled "snapKV+int8" while storing fp KV. Routing through `update()` instead puts the
        block into the same store every other quant arm uses. Note HF's cache is LAZY — a large block
        lands in the residual and is quantized by the NEXT update (the question block), exactly as the
        plain quant arms behave; resident bytes are therefore read POST-FLUSH (`kv_gib_flushed`)."""
        self.cache.layers[L].update(k, v)

    def crop(self, n):
        raise NotImplementedError(
            "a quantized KV cache cannot be cropped — rollback goes through state()/restore(), "
            "which snapshots the quantized buffers instead of cutting them")

    @torch.no_grad()
    def state(self):
        """Per-layer rollback state plus the batched bookkeeping. Handles both cache layouts: HF's
        single re-quantized buffer (whose update() REASSIGNS rather than mutates) and the
        append-only blocks (immutable by construction). Measured exact 30/30 turns for both."""
        from src.quantized_cache_append_only import layer_snapshot
        return ([layer_snapshot(l) for l in self.cache.layers],
                self.cache_len, self.pos.clone(), self.mask.clone())

    @torch.no_grad()
    def restore(self, st):
        from src.quantized_cache_append_only import layer_restore
        snap, n, pos, mask = st
        for layer, s in zip(self.cache.layers, snap):
            layer_restore(layer, s)
        self.cache_len = n
        self.pos, self.mask = pos.clone(), mask.clone()


class BatchedFusion:
    """ours, batched: the reader holds INSTRUCTION+context, the LM holds INSTRUCTION only (query-blind)."""

    def __init__(self, slm, slm_tok, lm, lm_tok, lam, dev, batch, instruction, lam_answer=-1.0,
                 max_len=None, reader_quant=None, lm_max_len=None, lm_dev=None):
        from src.static_decode import make_stateful
        # reader_quant=8|4|2 (2026-08-31): the COMPOSITION arm — ours with a quantized reader cache.
        # It ran sequentially at batch=1 only because nobody had passed the flag down; the rollback
        # this class already uses (state()/restore(), not crop()) is exactly the interface
        # BatchedQuantStateful implements, so no other line changes. batch=1 is an unconditional
        # throughput loss (CLAUDE.md 0원칙: fix the path, do not run it at 1), and it also made the
        # composition arm unpublishable in any timing table (throughput_eval refuses batch < 2).
        # The FKV bound is meaningless for a quantized cache (no dense preallocation), so it is
        # dropped rather than passed and ignored.
        self.reader_quant = reader_quant
        # TWO-DEVICE SPLIT (2026-09-01). lm_dev places the LM branch on a SECOND GPU so the two
        # forwards of a decode step can run at once instead of one after the other. It is the only
        # thing a second card can do for this method that it cannot do for a baseline: a baseline
        # given a second card gets a second replica — twice the throughput at the SAME per-answer
        # latency — whereas ours can make one answer arrive sooner.
        #
        # It matters most exactly where FKV helps least. Measured on the fixed decode path: on
        # hotpot the teacher spends 63% of its wall in prefill, where ours is 3.4x faster, and ours
        # wins x2.39; on LooGLE one document's prefill is shared by ~5 questions so prefill is only
        # 30% of the teacher's wall, the answer phase is 85% of OURS, and the win falls to x1.56.
        # The answer phase is what running the branches concurrently would cut.
        self.dev_s = torch.device(dev)
        self.dev_l = torch.device(lm_dev) if lm_dev is not None else self.dev_s
        self.split = self.dev_l != self.dev_s
        self.S = make_stateful(slm, slm_tok, dev, batch,
                               max_len=(None if reader_quant else max_len),
                               quant_nbits=reader_quant, is_reader=True)      # reader: this batch's bound
        # lm_max_len: the LM branch's own batch-local bound (2026-08-31). It stayed on the fixed
        # STATIC_MAXLEN_LM env constant for half a day after the reader branch was moved off a
        # dataset-wide constant for exactly this reason, and measured 1024 reserved against a peak
        # of 321 -- 68.7% unused, 2.4 GiB held back from the batch at B=14 on a 32B. The env var is
        # kept only as the fallback when a caller does not compute a bound.
        self.L = make_stateful(lm, lm_tok, self.dev_l, batch,
                               max_len=int(lm_max_len or os.environ.get("STATIC_MAXLEN_LM", "4096")))
        self.S.role, self.L.role = "S", "L"
        self.lam, self.dev, self.B, self.instr = lam, dev, batch, instruction
        # λ SCHEDULE (2026-08-27). lam_answer >= 0 switches this row's mixing weight once its own
        # generation has emitted "Final Answer:", mirroring run_ours' sequential LAM_ANSWER. Rows
        # switch INDEPENDENTLY, so the weight becomes a [B,1] vector rather than a scalar. When the
        # axis is off the scalar path below is untouched, so existing runs stay byte-identical.
        self.lam_answer = float(lam_answer)
        self._warmed = False   # see the serial first step in turn()
        # A split ALWAYS runs the pair concurrently: with the branches on different cards there is
        # no reason to serialise them, and serialising would make the split pointless.
        # ★ A two-card split does NOT imply stream overlap, and tying the two together was a design
        # mistake that cost three correctness gates (2026-08-31/09-01: 0/23 raw identical, every answer
        # diverging at the FIRST generated token). The gain from a second card is CAPACITY — moving the
        # 32B off the reader's card frees ~60 GiB there, so the reader's batch grows several-fold — and
        # capacity needs no overlap whatever. Sequential-across-two-cards differs from one card only by
        # tensor copies, which are exact. Overlap stays reachable with FUSION_PAR=1 but is UNVERIFIED
        # across devices: do not put it in a measured run until its own gate passes.
        self._par = os.environ.get("FUSION_PAR", "0") == "1" and torch.cuda.is_available()
        if self._par:
            from concurrent.futures import ThreadPoolExecutor
            self._pool = ThreadPoolExecutor(max_workers=1)
            self._stream_s = torch.cuda.Stream(device=self.dev_s)
            self._stream_l = torch.cuda.Stream(device=self.dev_l)

    def prefill(self, contexts):
        """★ The instruction and the context are forwarded as SEPARATE blocks, and the LM branch gets the
        SAME instruction block (trailing "\\n\\n" included) as the reader.

        Both details are equivalence requirements, not style. The sequential path does
        `forward(_ids(INSTRUCTION() + "\\n\\n"))` then `forward(_ids(fmt(passages)))`; concatenating the two
        strings before tokenising merges tokens across the boundary, so the model sees different ids. And
        the first draft prefilled the LM with `instr` while the reader got `instr + "\\n\\n"`, which made the
        two branches disagree about their own instruction. Together these put the batched path at 66.7%
        identical generations against the sequential one (2026-08-12 gate) — the reasoning cited different
        sessions, not merely different wording."""
        s_i, s_m = self.S.ids_of([self.instr + "\n\n"] * self.B)
        c_i, c_m = self.S.ids_of(list(contexts))
        l_i, l_m = self.L.ids_of([self.instr + "\n\n"] * self.B)
        if self._par:
            # The LM's instruction block is ~250 tokens against the reader's ~37k of context, so it
            # looks free — but a 32B forward must stream 59.6 GiB of WEIGHTS whatever its length, and
            # that read is not free (user, 2026-09-01). Overlap it with the reader's context prefill,
            # which is the longest single forward in the run. The per-turn question forward and every
            # decode step already go through _forward_pair; this closes the last serial point.
            ev = torch.cuda.Event(); ev.record(torch.cuda.current_stream(self.dev_l))

            def _lm():
                self._stream_l.wait_event(ev)
                with torch.cuda.device(self.dev_l), torch.cuda.stream(self._stream_l):
                    return self.L.forward(l_i, l_m)

            fut = self._pool.submit(_lm)
            with torch.cuda.stream(self._stream_s):
                self.S.forward(s_i, s_m)
                self.S.forward(c_i, c_m)
            fut.result()
            torch.cuda.current_stream(self.dev_s).wait_stream(self._stream_s)
            torch.cuda.current_stream(self.dev_l).wait_stream(self._stream_l)
        else:
            self.S.forward(s_i, s_m)
            self.S.forward(c_i, c_m)
            self.L.forward(l_i, l_m)

    def _lm_seq(self, ids, mask, use_step=False):
        """the LM branch, run sequentially, correct whether or not the two branches share a card.

        On one card this is exactly what the code did before. On two, the ids go to the LM's card and
        the [B, V] logits come back to the reader's, because the caller mixes them with the reader's
        logits and torch will not do that across devices. Nothing here is asynchronous: a .to() on the
        current stream is ordered against the work around it, which is the whole reason this path is
        the one being measured while the overlapped path is not."""
        if self.split:
            ids = ids.to(self.dev_l)
            mask = mask.to(self.dev_l) if mask is not None else None
        if self.split:
            # A CUDA graph replays on the CURRENT DEVICE, and setting a stream does not set the
            # device. Without this guard the LM's graph, captured on its own card, is replayed from
            # the reader's card context.
            with torch.cuda.device(self.dev_l):
                out = (self.L.step(ids, mask) if use_step and hasattr(self.L, "step")
                       else self.L.forward(ids, mask))
            return out.to(self.dev_s)
        return (self.L.step(ids, mask) if use_step and hasattr(self.L, "step")
                else self.L.forward(ids, mask))

    @torch.no_grad()
    def turn(self, qblocks, max_new, stop_ids, block_fn=None, done_fn=None):
        """decode one turn for all B sequences; returns a list of B token-id lists

        Decode-loop cost note (2026-08-19): measured per-step wall was 149 ms (B=3) against a ~22 ms
        bandwidth roofline — the loop is dominated by CPU-side per-layer dispatch plus per-row
        `.item()` syncs, NOT by the two models' arithmetic (85.5 GB/step read ≈ the teacher's 86.2).
        Fixes here: ONE `.tolist()` sync per step (was B syncs), Python-list liveness bookkeeping
        (was a device `any()` per step), and FUSION_PAR=1 runs the two branch forwards in two threads
        on two CUDA streams so their launch/dispatch chains overlap. Token selection is UNCHANGED —
        greedy argmax over the same fused logits, same stop logic — so generations must stay
        byte-identical with FUSION_PAR on or off (gated by the bench's equivalence check)."""
        # ★ Time to the first output token, measured from BEFORE this turn's question block is
        # forwarded through either branch — those two forwards are the work, and the overlap is
        # exactly what shortens them. Placed after them in the first draft, it read 0.4 ms on a
        # 32B+7B pair, which is impossible; that number was withdrawn rather than reported.
        _ttft_t0 = _time.perf_counter()
        self.ttft_s = None
        self.first_tok_t = None      # absolute perf_counter of the first token on the host (TTFT probe, 2026-09-14)
        si, sm = self.S.ids_of(qblocks)
        li, lm_ = self.L.ids_of(qblocks)
        if self._par:
            sl, ll = self._forward_pair(si, sm, li, lm_)
        else:
            sl = self.S.forward(si, sm)
            ll = self._lm_seq(li, lm_)
        V = min(sl.shape[-1], ll.shape[-1])
        if DECODE_SYNC_EVERY > 1 and not self._par:
            return self._turn_blocked(sl, ll, V, max_new, stop_ids, block_fn, done_fn)
        gen = [[] for _ in range(self.B)]
        alive = [True] * self.B
        # ★ TTFT (2026-09-01, user): "하나의 입력 -> 첫 output". Measured from the top of turn() —
        # which is where this turn's question block is forwarded through BOTH branches — to the moment
        # the first token is in hand. It is the half of latency that a two-card split can actually
        # shorten, because the two branch forwards are what get overlapped; the shared context prefill
        # is reported separately as prefill_s_batch and is NOT folded in here.
        sched = self.lam_answer >= 0
        # per-row weight; only allocated when the schedule is on, so the scalar path is unchanged
        lam_t = (torch.full((self.B, 1), self.lam, device=self.dev, dtype=sl.dtype) if sched else None)
        in_ans = [False] * self.B
        for step in range(max_new):
            if sched:
                fused = lam_t * sl[..., :V] + (1 - lam_t) * ll[..., :V]
            else:
                fused = self.lam * sl[..., :V] + (1 - self.lam) * ll[..., :V]
            if block_fn is not None:
                fused = block_fn(fused, step)
            nxt = torch.argmax(fused, dim=-1)                       # [B]
            nxt_l = nxt.tolist()                                    # ONE device->host sync per step
            if self.ttft_s is None:      # .tolist() already synced, so this is a real wall time
                _now = _time.perf_counter()
                self.ttft_s = round(_now - _ttft_t0, 4)
                self.first_tok_t = _now
            for i in range(self.B):
                if not alive[i]:
                    continue
                t = nxt_l[i]
                if t in stop_ids or (done_fn is not None and done_fn(gen[i] + [t])):
                    alive[i] = False
                    if t not in stop_ids:
                        gen[i].append(t)
                    continue
                gen[i].append(t)
                # this row's answer has begun -> its weight moves for the REST of this row's turn.
                # Checked per row and only until it fires, matching run_ours' sequential condition.
                if sched and not in_ans[i] and "Final Answer" in self.S.tok.decode(gen[i]):
                    in_ans[i] = True
                    lam_t[i, 0] = self.lam_answer
            if not any(alive):
                break
            step_ids = torch.tensor([[nxt_l[i] if alive[i] else self.S.pad] for i in range(self.B)],
                                    device=self.dev, dtype=torch.long)
            step_mask = torch.tensor([[1 if alive[i] else 0] for i in range(self.B)],
                                     device=self.dev, dtype=torch.long)
            if self._par and not self._warmed:
                # ★ THE FIRST STEP OF A RUN IS TAKEN SERIALLY, ON PURPOSE (2026-09-01). Each FKV
                # branch captures its CUDA graph lazily on its first step(), and `torch.cuda.graph`
                # defaults to capture_error_mode="global", which treats CUDA work launched by ANY
                # OTHER THREAD during the capture as an error. Under FUSION_PAR the other branch is
                # running in a worker thread at exactly that moment, which is what killed job
                # 3062188's B=16 leg. Taking step 0 serially gets both graphs captured before any
                # overlap begins; every later step overlaps. The computation is identical either
                # way, so generations are unchanged -- only this one step is not overlapped.
                sl = self.S.step(step_ids, step_mask) if hasattr(self.S, "step") else \
                     self.S.forward(step_ids, step_mask)
                ll = self._lm_seq(step_ids.clone(), step_mask, use_step=True)
                self._warmed = True
            elif self._par:
                # use_step: on the FKV backend this replays the captured graph, which is the whole
                # point of testing a cross-device overlap — falling back to forward() here would
                # measure the eager path and answer a question nobody asked.
                sl, ll = self._forward_pair(step_ids, step_mask, step_ids.clone(), step_mask,
                                            use_step=True)
            elif hasattr(self.S, "step"):
                sl = self.S.step(step_ids, step_mask)
                ll = self._lm_seq(step_ids, step_mask, use_step=True)
            else:
                sl = self.S.forward(step_ids, step_mask)
                ll = self._lm_seq(step_ids.clone(), step_mask)
        return gen

    @torch.no_grad()
    def _turn_blocked(self, sl, ll, V, max_new, stop_ids, block_fn, done_fn):
        """DECODE_SYNC_EVERY=K: K steps per device->host sync, tokens fed back GPU-resident.

        WHY the generated tokens are IDENTICAL to the K=1 loop. Attention is per row: row i's step-s
        logits depend only on row i's own KV and its own fed token. In the K=1 loop a row that has
        stopped is frozen (fed a pad at mask 0); here it keeps being fed its own argmax at mask 1, so
        its OWN cache and `pos` run past its stop — but nothing of row i ever enters row j's
        computation, so every still-live row sees exactly the inputs it saw before. The over-run KV is
        discarded wholesale by the harness's `restore()` at the end of the turn, which is the same
        rollback that already drops the whole generation. The CPU-side stop decisions (`stop_ids`,
        `done_fn`) are replayed over the block IN ORDER, so `gen` is truncated at the same token.

        COST of K>1: after the last live row stops, the block still finishes — up to K-1 wasted steps
        per turn. That is the price for letting the CPU run K launches ahead of the GPU.
        """
        gen = [[] for _ in range(self.B)]
        alive = [True] * self.B
        ones = torch.ones((self.B, 1), dtype=torch.long, device=self.dev)
        step = 0
        while step < max_new:
            n = min(DECODE_SYNC_EVERY, max_new - step)
            buf = []
            for j in range(n):
                fused = self.lam * sl[..., :V] + (1 - self.lam) * ll[..., :V]
                if block_fn is not None:
                    fused = block_fn(fused, step + j)
                nxt = torch.argmax(fused, dim=-1)                   # [B], stays on the device
                buf.append(nxt)
                ids = nxt[:, None]
                if hasattr(self.S, "step"):
                    sl = self.S.step(ids, ones); ll = self._lm_seq(ids, ones, use_step=True)
                else:
                    sl = self.S.forward(ids, ones); ll = self._lm_seq(ids.clone(), ones)
            toks = torch.stack(buf, dim=1).tolist()                 # ONE sync per K steps -> [B][n]
            for j in range(n):
                for i in range(self.B):
                    if not alive[i]:
                        continue
                    t = toks[i][j]
                    if t in stop_ids or (done_fn is not None and done_fn(gen[i] + [t])):
                        alive[i] = False
                        if t not in stop_ids:
                            gen[i].append(t)
                        continue
                    gen[i].append(t)
                if not any(alive):
                    break
            step += n
            if not any(alive):
                break
        return gen

    def _forward_pair(self, si, sm, li, lm_, use_step=False):
        """Run the reader and LM forwards concurrently — reader on a worker thread, LM on the
        calling thread, each on its own side stream.

        SAME DEVICE: both streams belong to that device, and one event orders them against the
        current stream. CROSS DEVICE: each branch's inputs are moved to its own card first, the
        events are per-device, and the LM's logits come back to the reader's device before the two
        are mixed. The [B, V] logit copy is the only cross-card traffic per step; the KV caches never
        move."""
        cur_s = torch.cuda.current_stream(self.dev_s)
        ev_s = torch.cuda.Event(); ev_s.record(cur_s)
        if self.split:
            # BLOCKING, deliberately. With non_blocking=True these cross-device copies are
            # issued asynchronously and nothing on the consuming side waits for them, so the LM can
            # read inputs that have not landed and the reader can read logits that have not landed.
            # That is what job 3062035 measured: the split leg generated 2.07x the tokens in HALF
            # the answer time -- 2.54 ms for a step running a 32B and a 7B, which cannot be -- and
            # matched the single-card answers on only 7 of 55 turns. The same _forward_pair with two
            # streams and graph replay is exactly correct on ONE card (23/23 identical, raw
            # included, job 3062047), so the fault was never the streams or the graph; it was these
            # copies. A [B, V] logit tensor is ~2.4 MB against a 32B step's ~60 GiB of weight
            # traffic, so making them synchronous costs nothing worth measuring.
            li = li.to(self.dev_l)
            lm_ = lm_.to(self.dev_l) if lm_ is not None else None
            cur_l = torch.cuda.current_stream(self.dev_l)
            ev_l = torch.cuda.Event(); ev_l.record(cur_l)
        else:
            cur_l, ev_l = cur_s, ev_s

        def _s():
            self._stream_s.wait_event(ev_s)
            with torch.cuda.stream(self._stream_s):
                return (self.S.step(si, sm) if use_step and hasattr(self.S, "step")
                        else self.S.forward(si, sm))

        fut = self._pool.submit(_s)
        # ★ torch.cuda.stream() sets the STREAM, not the current DEVICE — and a CUDA graph replays
        # on the CURRENT DEVICE. Running the LM's graph from a cuda:0 context while it was captured
        # on cuda:1 is why the split kept producing different text (0/46 raw identical) even after
        # the copies were made synchronous, and why the one-card isolation passed 23/23: with one
        # card the device context is right by accident. torch.cuda.device() fixes the context.
        self._stream_l.wait_event(ev_l)
        with torch.cuda.device(self.dev_l), torch.cuda.stream(self._stream_l):
            ll = (self.L.step(li, lm_) if use_step and hasattr(self.L, "step")
                  else self.L.forward(li, lm_))
        sl = fut.result()
        cur_s.wait_stream(self._stream_s)
        if self.split:
            cur_l.wait_stream(self._stream_l)
            torch.cuda.current_stream(self.dev_l).synchronize()
            ll = ll.to(self.dev_s)
        else:
            cur_s.wait_stream(self._stream_l)
        return sl, ll


class BatchedSingle:
    """teacher / floor, batched: ONE model that sees instruction + context + the accumulated dialogue.

    Same contract as BatchedFusion so the harness can treat them alike — the ceiling and floor arms are the
    cheapest runs in every comparison and there is no reason for them to be the only ones left at batch 1.
    """

    def __init__(self, model, tok, dev, batch, instruction, quant_nbits=None, max_len=None):
        from src.static_decode import make_stateful
        # max_len = THIS batch's requirement, not a dataset-wide constant (2026-08-31). Ignored by
        # the growable eager/quant caches; used by FKV to size its preallocation.
        self.S = make_stateful(model, tok, dev, batch, max_len=max_len, quant_nbits=quant_nbits)
        self.dev, self.B, self.instr = dev, batch, instruction

    def prefill(self, contexts):
        # separate blocks, to match the sequential path's tokenisation splits — see BatchedFusion.prefill
        ids, m = self.S.ids_of([self.instr + "\n\n"] * self.B); self.S.forward(ids, m)
        ids, m = self.S.ids_of(list(contexts)); self.S.forward(ids, m)

    @torch.no_grad()
    def turn(self, qblocks, max_new, stop_ids, block_fn=None, done_fn=None):
        # ttft_s / first_tok_t as in BatchedFusion.turn (2026-09-14): question forward + step 0 until the first
        # token reaches the host; first_tok_t is the absolute instant, for a one-wall TTFT from the prefill start
        _ttft_t0 = _time.perf_counter()
        self.ttft_s = None
        self.first_tok_t = None
        ids, m = self.S.ids_of(qblocks)
        lg = self.S.forward(ids, m)
        if DECODE_SYNC_EVERY > 1:
            # the SAME sync-free decode the fusion arm gets — a speed fix applied to only one arm would
            # invalidate every throughput comparison it appears in
            return self._turn_blocked(lg, max_new, stop_ids, block_fn, done_fn)
        gen = [[] for _ in range(self.B)]
        alive = [True] * self.B
        for step in range(max_new):
            if block_fn is not None:
                lg = block_fn(lg, step)
            nxt = torch.argmax(lg, dim=-1)
            nxt_l = nxt.tolist()                                    # ONE device->host sync per step
            if self.ttft_s is None:      # .tolist() already synced, so this is a real wall time
                _now = _time.perf_counter()
                self.ttft_s = round(_now - _ttft_t0, 4)
                self.first_tok_t = _now
            for i in range(self.B):
                if not alive[i]:
                    continue
                t = nxt_l[i]
                if t in stop_ids or (done_fn is not None and done_fn(gen[i] + [t])):
                    alive[i] = False
                    if t not in stop_ids:
                        gen[i].append(t)
                    continue
                gen[i].append(t)
            if not any(alive):
                break
            step_ids = torch.tensor([[nxt_l[i] if alive[i] else self.S.pad] for i in range(self.B)],
                                    device=self.dev, dtype=torch.long)
            step_mask = torch.tensor([[1 if a else 0] for a in alive], device=self.dev, dtype=torch.long)
            lg = (self.S.step(step_ids, step_mask) if hasattr(self.S, "step")
                  else self.S.forward(step_ids, step_mask))
        return gen


    @torch.no_grad()
    def _turn_blocked(self, lg, max_new, stop_ids, block_fn, done_fn):
        """single-model twin of BatchedFusion._turn_blocked — see there for the equivalence argument"""
        gen = [[] for _ in range(self.B)]
        alive = [True] * self.B
        ones = torch.ones((self.B, 1), dtype=torch.long, device=self.dev)
        step = 0
        while step < max_new:
            n = min(DECODE_SYNC_EVERY, max_new - step)
            buf = []
            for j in range(n):
                if block_fn is not None:
                    lg = block_fn(lg, step + j)
                nxt = torch.argmax(lg, dim=-1)
                buf.append(nxt)
                ids = nxt[:, None]
                lg = (self.S.step(ids, ones) if hasattr(self.S, "step") else self.S.forward(ids, ones))
            toks = torch.stack(buf, dim=1).tolist()
            for j in range(n):
                for i in range(self.B):
                    if not alive[i]:
                        continue
                    t = toks[i][j]
                    if t in stop_ids or (done_fn is not None and done_fn(gen[i] + [t])):
                        alive[i] = False
                        if t not in stop_ids:
                            gen[i].append(t)
                        continue
                    gen[i].append(t)
                if not any(alive):
                    break
            step += n
            if not any(alive):
                break
        return gen


def verify_batched_equivalence(build_single, build_batched, contexts, qblocks, max_new, stop_ids):
    """decode the same conversations at B=1 and B=len(contexts); the token ids must match exactly.

    A batched harness that quietly changes the outputs is worse than no batching at all — every existing
    number would stop being comparable. This is the gate the batched path has to pass before it is used.
    """
    single = []
    for c, q in zip(contexts, qblocks):
        d = build_single([c])
        d.prefill([c])
        single.append(d.turn([q], max_new, stop_ids)[0])
    d = build_batched(contexts)
    d.prefill(contexts)
    batched = d.turn(qblocks, max_new, stop_ids)
    bad = [i for i, (a, b) in enumerate(zip(single, batched)) if a != b]
    return dict(ok=not bad, mismatched=bad, single=single, batched=batched)
