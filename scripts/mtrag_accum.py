#!/usr/bin/env python
"""mtRAG-accumulate — ONE stateful, INTERLEAVED harness for all methods (fair, GPU-efficient).

Every method replays a conversation in the faithful accumulate order [INSTR][P1][Q1 A1_ref][P2][Q2 A2_ref]...,
reusing the KV cache across turns (append-only) EXCEPT snapkv_fresh which by definition recompresses each turn.
Answers are generated, scored, then rolled back; the reference answer is teacher-forced into history.

methods:
  teacher   : append passages FULL, reuse KV (O(n))
  ours      : SLM accumulates passages+dialogue; LM sees dialogue/query only (no passages); logits fused; reuse KV
  snapkv_frozen : compress each new passage block ONCE on its arrival Q, merge+reuse (O(n))
  snapkv_fresh  : each turn, rebuild [P1..Pt] compressing every block on the CURRENT Qt (O(n^2)) -- the recompute cost
  h2o       : stream heavy-hitter+recent eviction as passages arrive; query-independent -> ONE version (O(n), eager)
Budget for snapkv/h2o = keep 18.75% of passage tokens (matched to ours' 3B-KV).
"""
import os, sys, json, re, hashlib, argparse, time, collections, torch

# The ONLY --ref values that may be silently replaced when --bench pins a different one: the harness default
# and the launcher default, which are the same string and are passed for every benchmark whether or not
# anyone chose them. Every other value was typed on purpose — see the FOOTGUN GUARD below.
_TOLERATED_DEFAULT_REFS = {"reference.jsonl"}
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import DynamicCache
from transformers.cache_utils import HQQQuantizedCache
from src.models import load_causal_lm
from src.eval import compute_best_em_f1, extract_final_answer
import src.qa_prompts as qa_prompts
from src.qa_prompts import QA_FULL_CONTEXT_INSTRUCTION_REASON as INSTRUCTION_TERSE
from src.qa_prompts import QA_MTRAG_INSTRUCTION as INSTRUCTION_FULL
from scripts.mab_eval import done_final_answer, q_turn as q_turn_terse, ref_answer
import scripts.reason_fix as RF
import scripts.bench_config as BC
from src import specprefill as SP
from src.gate_kl import TokenAgnosticGateMLP, build_gate_features   # v3 learned per-token fusion gate
import kvpress

# ---- per-turn timing/memory (2026-08-06, additive fields only — never changes behavior) ----
# _turn_begin() at the top of every turn loop; _ingest_done() after the passage-commit block;
# _ans_begin() at the start of the answer generation. _rec() attaches ans_s / ingest_s (real
# perf_counter wall-clock, CUDA-synced) + peak alloc/reserved GiB summed over all GPUs since turn start.
_TURN = {}
def _kv_gib(*caches):
    """ACTUAL KV bytes held in the caches — not peak_reserved, which is KV + activations + allocator slack
    and is dominated by activations. Measuring the tensors is the only way to answer 'how much KV does the
    reader hold vs the teacher', which is the whole memory claim: per-token KV is 256 KiB for the 32B
    (64 layers x 8 kv-heads x 128) against 56 KiB for the 7B (28 x 4 x 128), a 4.57x ratio that a
    peak-reserved reading cannot see."""
    def _deep(o, depth=0):
        if o is None or depth > 4: return 0
        if torch.is_tensor(o): return o.numel() * o.element_size()
        if isinstance(o, dict): return sum(_deep(v, depth + 1) for v in o.values())
        if isinstance(o, (list, tuple)): return sum(_deep(v, depth + 1) for v in o)
        if hasattr(o, "__dict__"):
            return sum(_deep(v, depth + 1) for k, v in vars(o).items() if not k.startswith("__"))
        return 0
    tot = 0
    for c in caches:
        if c is None: continue
        try:
            for layer in c.layers:
                for t in (getattr(layer, "keys", None), getattr(layer, "values", None)):
                    if t is not None: tot += t.numel() * t.element_size()
                # HQQ-quantized layers: packed weights + scale/zero meta live under _quantized_*
                # (HF's single-buffer layout) or _blocks_k/_blocks_v (AppendOnlyHQQLayer's immutable
                # block list). ★ The block list was NOT walked until 2026-08-31, so every append-only
                # quant arm reported only its fp RESIDUAL — hotq int8 read 0.138 GiB for a 12-row
                # cache that holds ~7.7 GiB, about 5% of the truth. Direction of the error: it made
                # the quantization baseline look SMALLER in memory than it is, i.e. it flattered the
                # baseline and understated our composition — the conservative direction, but wrong.
                for _attr in ("_quantized_keys", "_quantized_values", "_blocks_k", "_blocks_v"):
                    tot += _deep(getattr(layer, _attr, None))
        except Exception:
            return None
    return round(tot / 2**30, 3)


def _mem_gib():
    if not torch.cuda.is_available(): return (None, None)
    a = sum(torch.cuda.max_memory_allocated(d) for d in range(torch.cuda.device_count()))
    r = sum(torch.cuda.max_memory_reserved(d) for d in range(torch.cuda.device_count()))
    return (round(a / 2**30, 3), round(r / 2**30, 3))
def _turn_begin():
    global _TURN
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        for d in range(torch.cuda.device_count()): torch.cuda.reset_peak_memory_stats(d)
    _TURN = {"t0": time.perf_counter(), "ingest_s": None, "ans_t0": None}
def _ingest_done():
    if torch.cuda.is_available(): torch.cuda.synchronize()
    _TURN["ingest_s"] = time.perf_counter() - _TURN.get("t0", time.perf_counter())
def _ans_begin():
    if torch.cuda.is_available(): torch.cuda.synchronize()
    _TURN["ans_t0"] = time.perf_counter()

# ---- SESSION-MEMO accumulation (2026-08-09, env MEMO=1, method ours only; additive, off by default) ----
# Motivation (Q32B7B_RESULTS §6): the unrecovered LoCoMo mass is deep-session binding — a READING skill
# (gap-set F1: 32B-reads 0.878 / 14B-reads 0.519 / 7B-reads 0.132). At ingest the reader generates a
# COMPLETE query-independent session index ("[Session k | date]: events") which is teacher-forced into
# BOTH branches' KV (~1k tok): depth->shallow for the reader, and the blind LM gains veto material on
# binding tokens. Differs from M3 (net-negative) by being complete + query-independent, not stale-selected.
MEMO_MODE = os.environ.get("MEMO", "0") == "1"
MEMO_PROMPT = ("\nList every session of the conversation above, one line each, exactly in the format "
               "'[Session k | date]: main events in one short clause'. Cover ALL sessions in order. "
               "Output nothing else.\n")
MEMO_MAX_NEW = int(os.environ.get("MEMO_MAX_NEW", "600"))
def _memo_generate(S, tok, dev, stop_ids):
    """Greedy-generate the session memo from the reader's current state; rolled back after."""
    sb, sp = S.cache_len, S.pos
    logits = S.forward(S._ids(MEMO_PROMPT)).logits[:, -1, :]
    gen = []
    for _ in range(MEMO_MAX_NEW):
        nxt = int(torch.argmax(logits, -1).item())
        if nxt in stop_ids: break
        gen.append(nxt)
        if len(gen) > 8 and gen[-4:] == gen[-8:-4]: break   # crude repetition stop
        logits = S.forward(torch.tensor([[nxt]], device=dev)).logits[:, -1, :]
    S.crop(sb); S.pos = sp
    return tok.decode(gen, skip_special_tokens=True).strip()

# ---- per-STEP fusion token logging (2026-08-09, env TOKLOG=<path>; ours only; additive) ----
# Operationalizes the access/operation/trajectory trichotomy (Q32B7B_RESULTS ★★★): per decode step record
# the token, both branches' entropies, KL(SLM||LM), whether the branches' argmaxes differ, and which branch
# the fused argmax agreed with. Written as JSONL: one record per turn with a per-step list.
TOKLOG = os.environ.get("TOKLOG", "")
_TOKF = None
def _tok_stats(sl, ll, lam, V, nxt):
    import torch.nn.functional as _F
    sp = _F.log_softmax(sl[..., :V].float(), -1); lp = _F.log_softmax(ll[..., :V].float(), -1)
    se = float(-(sp.exp() * sp).sum()); le = float(-(lp.exp() * lp).sum())
    kl = float((sp.exp() * (sp - lp)).sum())
    sa, la = int(sl[..., :V].argmax()), int(ll[..., :V].argmax())
    return dict(t=nxt, se=round(se, 3), le=round(le, 3), kl=round(kl, 3),
                agree=int(sa == la), win=("s" if nxt == sa else "l" if nxt == la else "x"))

# v3 learned fusion gate (loaded in main from --gate; None -> plain fixed-λ fusion).
FUSION_GATE = None
GATE_TOP_K = 10
# ★ RESTRICTED FUSION (2026-08-12). On the token log (120 LoCoMo turns, 7,588 steps, λ0.85) the branches
# disagree on 14.9% of steps, and on 9.7% of THOSE the fused argmax is a token that is NEITHER branch's
# top-1 — a compromise the mixture invents. It is not rare or concentrated: 47% of turns contain at least
# one, and mean F1 falls monotonically with how many a turn has (0 steps 0.5546 / 1 0.5487 / 2 0.4411 /
# 3+ 0.3448, N=64/24/18/14), which turn length does not explain (60.3 vs 66.6 steps).
# FUSE_RESTRICT=1 keeps the mixture for SCORING but restricts the choice to {SLM top-1, LM top-1}, so the
# fusion can prefer either branch's proposal but can no longer invent a third token.
FUSE_RESTRICT = os.environ.get("FUSE_RESTRICT", "0") == "1"

# ---- BATCH SIZE ----------------------------------------------------------------------------------------
# This harness has never batched: it holds one conversation's KV cache, appends each turn, rolls the
# generated tokens back and commits the reference answer. Batching means running B conversations
# concurrently with a padded, per-sequence-rollback cache — a rewrite of StatefulLM/run_ours, not a flag.
# Until that lands, BATCH_SIZE is 1 and every timing this file records is marked invalid, because at batch 1
# the flash-attention graph does not compile and the measured speed is not the method's speed.
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))


def pick_from_fused(fused, sl, ll, V):
    """argmax over the fused logits, or — under FUSE_RESTRICT — over only the two branches' top-1 tokens.
    The candidates are taken from the SAME (possibly reason-blocked) tensors that scored them, so a token
    the reasoning guard has masked can never be resurrected as a candidate."""
    if not FUSE_RESTRICT:
        return int(torch.argmax(fused, -1).item())
    cand = torch.tensor(sorted({int(sl[..., :V].argmax()), int(ll[..., :V].argmax())}), device=fused.device)
    return int(cand[int(torch.argmax(fused[..., :V].reshape(-1)[cand]))])


def fuse_logits(sl, ll, lam, V):
    """Per-token fusion: fixed λ, or the learned gate λ_t = g(SLM_topk, LM_topk) if FUSION_GATE is set."""
    sl, ll = sl[..., :V], ll[..., :V]
    if FUSION_GATE is None:
        return lam * sl + (1 - lam) * ll
    lam_t = FUSION_GATE(build_gate_features(sl, ll, gate_top_k=GATE_TOP_K))   # [1,1] in (0,1)
    return lam_t * sl + (1 - lam_t) * ll

# SpecPrefill (official ICML'25) globals — set in main() for methods specprefill / specprefill_ours.
SPEC_MODEL = None                       # the 3B speculator (same model we use as the SLM)
SPEC_KEEP = 0.1                         # official default keep percentage
SPEC_CHUNK = SP.DEFAULT_CHUNK_SIZE      # 32
SPEC_POOL = SP.DEFAULT_POOL_KERNEL      # 13
SPEC_LAH = SP.DEFAULT_LOOK_AHEAD        # 8
# ★ QUERY-AGNOSTIC (generic-query) SpecPrefill (user idea, 2026-07-24): select tokens with a GENERIC query so
# the selection does NOT depend on which question arrives → the selected LM-context is REUSABLE across turns
# (removes the query-dependence that makes plain SpecPrefill snapKV-like and collapse under KV-reuse). Fusion
# then uses the REAL query. NOTE: the SLM's persistent fusion KV never holds the generic query — scoring is a
# transient SPEC_MODEL forward — so the "drop the summarize-query KV, append the real query" step is automatic.
SPEC_USE_GENERIC = False
SPEC_GENERIC_QUERY = "Summarize the key facts, entities, names, dates, and numbers stated in the text above."
# method families (query-dependent + query-agnostic variants)
SP_SINGLE = {"specprefill", "specprefill_generic"}               # SpecPrefill, no fusion (LM over selected ctx)
SP_FUSION = {"specprefill_ours", "specprefill_generic_ours"}     # SpecPrefill + SLM/LM fusion
SP_GENERIC = {"specprefill_generic", "specprefill_generic_ours"} # query-AGNOSTIC selection
SP_ALL = SP_SINGLE | SP_FUSION
REASON_FIX = os.environ.get("REASON_FIX", "1") == "1"   # 2026-07-17: structurally guaranteed reasoning (see reason_fix.py)
LAM_SCHEDULE = None                                     # 2026-07-28: per-turn λ list (turn-adaptive fusion); set in main()
LAM_ANSWER = float(os.environ.get("LAM_ANSWER", "-1"))  # 2026-07-29: within-answer λ drop — after 'Final Answer:' switch λ
                                                        # to LAM_ANSWER so the reasoned LM commits (SLM-dominant only while
                                                        # reasoning). <0 disables (default). Mirrors specprefill_eval LAM_ANSWER.
# REASON_HIST: what answer goes in the accumulated history. 'gen' = the model's own generated answer (coherent w/ its
# reasoning but PROPAGATES errors); 'ref' = the model's reasoning + the REFERENCE gold answer (no error propagation).
REASON_HIST = os.environ.get("REASON_HIST", "gen")
# compression press for the *_frozen/*_fresh/h2o methods (set in main from the method name).
PRESS_METHOD = "snapkv"
RATIO_GLOBAL = 0.8125                          # set in main() from --ratio; read by run_single_batched
QUERY_DEP = {"snapkv", "pyramidkv"}            # use the query as observation window; others are query-independent
def make_press(name, ratio, window):
    if name == "snapkv":            return kvpress.SnapKVPress(compression_ratio=ratio, window_size=window)
    if name == "pyramidkv":         return kvpress.PyramidKVPress(compression_ratio=ratio, window_size=window)
    if name == "expected_attention":return kvpress.ExpectedAttentionPress(compression_ratio=ratio)   # H2O-like, query-independent
    raise ValueError(f"unknown press {name}")
def press_for_method(m):
    if m.startswith("pyramidkv"): return "pyramidkv"
    if m == "h2o":                return "expected_attention"
    return "snapkv"
def _hist_reason(qt, raw, aref):
    if REASON_HIST == "ref":
        rp = RF.reason_prefix(raw)
        return qt + (rp + "\n" if rp else "") + f"Final Answer: {aref}\n"
    return qt + str(raw).rstrip() + "\n"

# --fullans switches the prompt/answer-format ONLY (mtRAG references are full conversational sentences ~90w,
# not terse spans). Everything else (KV-reuse mechanics) is identical, so the prompt is the sole variable.
# --instruction NAME overrides the terse-path instruction with any qa_prompts constant (e.g. a QASPER concise
# prompt) while keeping the terse "Final Answer:" decode/extract — used for the QASPER multi-question reuse run.
FULLANS = False
INSTR_OVERRIDE = None
# CHAT_WRAP=1 (2026-08-30): ChatML turn-wrapping for chat-ONLY families. OLMo-3 under this
# harness's raw prompt emits EOS at step 0 (smoke 3050092: every arm generated "" — the old
# cross-family recipe required USE_CHAT=1 for exactly this reason). Opt-in and OFF by default, so
# every Qwen/gemma prompt stays byte-identical. The wrap opens ONE user turn at the instruction,
# switches to the assistant turn right before each decode, and re-opens a user turn after each
# committed history block; passages/questions stream inside the open user turn unchanged.
# 2026-09-05: the markers live in scripts/chat_wrap.py (CHAT_WRAP=1 ChatML for OLMo-3, CHAT_WRAP=gemma
# for Gemma-3's <bos>+<start_of_turn> format), shared with the teacher generator and both SFT trainers.
import scripts.chat_wrap as _CW
CHAT_WRAP = _CW.ON
_CW_OPEN, _CW_ASSIST, _CW_REOPEN = _CW.OPEN, _CW.ASSIST, _CW.REOPEN
def INSTRUCTION():
    base = INSTRUCTION_FULL if FULLANS else (INSTR_OVERRIDE or INSTRUCTION_TERSE)
    return (_CW_OPEN + base) if CHAT_WRAP else base
def q_turn(q):
    base = f"\n\nQuestion: {q}\nAnswer:" if FULLANS else q_turn_terse(q)
    return (base + _CW_ASSIST) if CHAT_WRAP else base
def hist_block(q, a):
    if FULLANS:
        base = f"\n\nQuestion: {q}\nAnswer: {a}\n"
        return (base + "") if not CHAT_WRAP else (q_turn(q) + f" {a}\n" + _CW_REOPEN)
    if CHAT_WRAP:
        return q_turn(q) + ref_answer(a) + _CW_REOPEN
    return q_turn_terse(q) + ref_answer(a)
def extract(text):
    if FULLANS: return text.split("\nQuestion:")[0].strip()
    return RF.extract_answer(text) if REASON_FIX else extract_final_answer(text)   # robust to reasoning-ramble
def stop_gen(tok, gen):     # early-stop: next-turn hallucination (full) or 'Final Answer:' complete (terse)
    if FULLANS:  return len(gen) >= 2 and "\nQuestion:" in tok.decode(gen)
    return done_final_answer(tok, gen)


def norm(t): return re.sub(r"\s+", " ", (t or "").strip()).lower()
def pkey(c): return (c.get("document_id"), hashlib.sha1(norm(c.get("text", "")).encode()).hexdigest())
def fmt(cs):
    out = "\n\n".join("[Evidence Passage]\nText:\n" + (c.get("text") or "") for c in cs)
    # CTX_CAP (chars ~ 4x tokens): truncate the CONTEXT BLOCK ONLY, for controlled TTFT-vs-ctx sweeps.
    # Questions/instruction untouched. Recorded in provenance as ctx_cap_tokens.
    if CTX_CAP:
        out = out[: CTX_CAP * 4]
    return out

CTX_CAP = int(os.environ.get("CTX_CAP", "0") or 0)
def last_user(inp):
    for m in reversed(inp or []):
        if m.get("speaker") == "user": return m.get("text", "")
    return ""



# ★ DUMP_BLOCKS=<path> writes every text block committed to a model, in order, from BOTH the sequential and
# the batched path. Diffing the two files answers "do the two paths feed the model the same thing?" directly,
# instead of inferring it from divergent outputs. A previous divergence in this harness (2026-08-12) turned
# out to be exactly this — instruction+context tokenised together merged tokens across the boundary.
_DUMPF = None
def _dump_block(tag, text, row=None):
    import os
    path = os.environ.get("DUMP_BLOCKS")
    if not path:
        return
    global _DUMPF
    if _DUMPF is None:
        _DUMPF = open(path, "a")
    if row is not None and row != 0:
        return                      # batched: record row 0 only, to compare against a single sequential run
    _DUMPF.write(f"<<<{tag}>>>{text!r}\n"); _DUMPF.flush()

class Stateful:
    """Single-model persistent KV cache with append / compress-append / H2O-prune / exact rollback."""
    def __init__(self, model, tok, dev, ratio):
        self.model, self.tok, self.dev, self.ratio = model, tok, dev, ratio
        self.cache = DynamicCache(); self.cache_len = 0; self.pos = 0
    def _ids(self, text):
        _dump_block(f"SEQ|{getattr(self, 'role', '?')}", text)
        return self.tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.dev)
    @torch.no_grad()
    def forward(self, ids, want_attn=False):
        n = ids.shape[1]
        out = self.model(ids, past_key_values=self.cache, use_cache=True, logits_to_keep=1,
                         position_ids=torch.arange(self.pos, self.pos + n, device=self.dev)[None],
                         cache_position=torch.arange(self.cache_len, self.cache_len + n, device=self.dev),
                         attention_mask=torch.ones((1, self.cache_len + n), device=self.dev, dtype=torch.long),
                         output_attentions=want_attn)
        self.cache = out.past_key_values; self.cache_len += n; self.pos += n
        return out
    @torch.no_grad()
    def compress_append(self, passages_text, qcond_text):
        """Isolated from-scratch prefill of passages at the current offset WITH the configured press, then MERGE the
        compressed passage KV into the persistent cache. query-DEPENDENT presses (snapkv/pyramidkv) use the qcond as
        the observation window (then drop it); query-INDEPENDENT (expected_attention=H2O-like) compress passages alone."""
        p_ids = self._ids(passages_text)
        if PRESS_METHOD in QUERY_DEP:
            q_ids = self._ids(qcond_text); ids = torch.cat([p_ids, q_ids], dim=1); n = ids.shape[1]; window = min(q_ids.shape[1], 64)
        else:
            ids = p_ids; n = p_ids.shape[1]; window = 0
        temp = DynamicCache()
        with make_press(PRESS_METHOD, self.ratio, window)(self.model):
            out = self.model(ids, past_key_values=temp, use_cache=True, logits_to_keep=1,
                             position_ids=torch.arange(self.pos, self.pos + n, device=self.dev)[None],
                             cache_position=torch.arange(n, device=self.dev))
        comp = out.past_key_values; keep_p = comp.get_seq_length() - window
        for L in range(len(comp.layers)):
            self.cache.layers[L].keys = torch.cat([self.cache.layers[L].keys, comp.layers[L].keys[:, :, :keep_p, :]], dim=2)
            self.cache.layers[L].values = torch.cat([self.cache.layers[L].values, comp.layers[L].values[:, :, :keep_p, :]], dim=2)
        self.cache_len += keep_p; self.pos += p_ids.shape[1]
    @torch.no_grad()
    def specprefill_append(self, passages_text, qcond_text):
        """SpecPrefill (official) in the ACCUMULATE setting: the 3B speculator scores this passage block w.r.t.
        the observation query, we keep the Top-K chunks, and the LM prefills ONLY those tokens — at their
        ORIGINAL positions. Bookkeeping mirrors compress_append: the KV grows by the KEPT count while `pos`
        advances by the FULL block, so every later turn sits at its true original position (§3.2.4).

        If SPEC_USE_GENERIC, the observation query is the fixed GENERIC query (query-AGNOSTIC selection) instead
        of the turn's real question — so the SAME kept tokens serve every later question (reuse-safe)."""
        if SPEC_USE_GENERIC:
            qcond_text = SPEC_GENERIC_QUERY
        p_ids = self._ids(passages_text)
        q_ids = self._ids(qcond_text)
        P = p_ids.shape[1]
        ids = torch.cat([p_ids, q_ids], dim=1).to(SPEC_MODEL.device if hasattr(SPEC_MODEL, "device") else self.dev)
        scores = SP.speculate_scores(SPEC_MODEL, ids, 0, P, look_ahead=SPEC_LAH)
        mask, n_chunks, n_keep = SP.select_chunks(scores, SPEC_KEEP, SPEC_CHUNK, SPEC_POOL)
        sel = torch.nonzero(mask, as_tuple=False).flatten()
        if sel.numel() == 0:
            sel = torch.arange(min(SPEC_CHUNK, P))
        sel_dev = sel.to(self.dev)
        sel_ids = p_ids[:, sel_dev]
        K = int(sel.numel())
        pos = (self.pos + sel_dev).unsqueeze(0)                       # ORIGINAL positions inside this block
        # non-contiguous position_ids => MUST NOT use flash (it would treat them as packed sequences and
        # segment the block at every gap). See SP.sdpa_for_gappy_positions.
        with SP.sdpa_for_gappy_positions(self.model):
            out = self.model(sel_ids, past_key_values=self.cache, use_cache=True, logits_to_keep=1,
                             position_ids=pos,
                             cache_position=torch.arange(self.cache_len, self.cache_len + K, device=self.dev),
                             attention_mask=torch.ones((1, self.cache_len + K), device=self.dev, dtype=torch.long))
        self.cache = out.past_key_values
        self.cache_len += K            # KV holds ONLY the selected tokens
        self.pos += P                  # positions advance by the FULL block (original positions preserved)
        return K, P, n_keep, n_chunks

    @torch.no_grad()
    def h2o_prune_last(self, block_len, keep):
        """After appending a passage block of length block_len, keep only `keep` of the last-block key positions by
        accumulated attention (heavy hitters) + most-recent; old cache (before the block) untouched. Query-independent."""
        # re-forward is not needed: we already have the block in cache. Score by the block's own attention would need
        # attentions from the append; here we approximate H2O with attention over the block from the most recent query.
        pass  # (unused: h2o uses forward(want_attn) path below)
    @torch.no_grad()
    def crop(self, n):
        self.cache.crop(n); self.cache_len = n
    @torch.no_grad()
    def generate(self, last_logits, max_new, stop_ids):
        gen = []; logits = last_logits
        for step in range(max_new):
            lg = RF.block_logits(logits, step) if REASON_FIX else logits
            nxt = int(torch.argmax(lg, -1).item())
            if nxt in stop_ids: break
            gen.append(nxt)
            if (RF.reason_stopped(self.tok, gen) if REASON_FIX else stop_gen(self.tok, gen)): break
            logits = self.forward(torch.tensor([[nxt]], device=self.dev)).logits[:, -1, :]
        return gen


# ── KV-quantization baseline ────────────────────────────────────────────────────────────────────
# Same "teacher" flow (append passages FULL, reuse across turns) but the persistent KV cache is stored
# QUANTIZED (HQQ int8 / int4). Faithful: HF's QuantizedLayer.update() dequantizes on every read, so the
# attention forward operates on the lossy-reconstructed KV exactly as a real quant-KV decoder does. The
# ONLY thing HF's QuantizedLayer.crop() does not handle is resetting cumulative_length / the quantized
# buffer, so we override crop() to rebuild the quantized state from the sliced dequantized KV.
QUANT_NBITS = None  # set in main() for --method quant_int8 / quant_int4
# READER_QUANT=8|4|3|2 (2026-08-30, ours+quant composition study): quantize the FUSION READER's
# KV cache with the same HQQ discipline (per-turn rollback via snapshot/restore, never
# crop-requantize — MEASURED exact, 30/30 turns). NOTE the correction in QuantStateful's docstring:
# HF's update() still re-quantizes the whole cache on every residual flush, so committed KV is NOT
# quantized once end-to-end and accumulate-depth degradation through this path is an implementation
# artefact. Sequential run_ours path only — excluded from
# _batchable like the other single-conversation axes. The LM branch stays fp16.
READER_QUANT_NBITS = int(os.environ["READER_QUANT"]) if os.environ.get("READER_QUANT") else None
# COMPOSITION arm (2026-08-31): a press / SpecPrefill / teacher arm whose persistent KV cache is ALSO
# HQQ-quantized — "snapKV 30% kept + int8", the strongest baseline available, and the one that tells us
# whether ours+reader-int8 still wins once the baseline is allowed to compose the two savings too.
# The compressed block is written through QuantizedLayer.update() (BatchedQuantStateful._append_layer_kv),
# never concatenated onto the fp residual, so the row's label matches what is stored.
PRESS_QUANT_NBITS = int(os.environ["PRESS_QUANT"]) if os.environ.get("PRESS_QUANT") else None

class QuantStateful(Stateful):
    """Persistent KV cache stored HQQ-quantized; attention reads dequantize on the fly.

    ★★ CORRECTION 2026-08-31 — this docstring used to claim committed tokens are "quantized EXACTLY
    ONCE". That is TRUE of our rollback and FALSE end to end, and the difference matters for every
    accumulate benchmark. Our per-turn rollback is snapshot/restore of the quantized buffers + fp
    residual + cumulative_length (never crop, which would leave the quantized store stale) and it is
    MEASURED EXACT: 30/30 turns bit-identical at int8 and int4
    (`scripts/probe_hqq_requantization_drift.py`, job 3055723). But HF's own
    `QuantizedLayer.update()` runs, on every residual flush,

        _quantized_keys = quantize(cat(dequantize(_quantized_keys), keys, new_keys))

    i.e. it re-quantizes THE WHOLE CACHE roughly every `residual_length` appended tokens. So a token
    committed at turn 1 is re-quantized dozens of times across a 30-turn conversation, and the same
    probe measures the damage: the reconstruction error of a FIXED committed block grows x3.4 over 30
    turns at int8 (5.5e-3 -> 1.9e-2) and x3.1 at int4 (7.3e-2 -> 2.2e-1).

    CONSEQUENCE: any accuracy degradation that GROWS WITH ACCUMULATION DEPTH through this cache is an
    artefact of this implementation, not a property of KV quantization, and may not be reported as
    one. Single-turn arms are unaffected (one flush). A correct implementation would append new
    quantized blocks and leave already-quantized ones immutable."""
    def __init__(self, model, tok, dev, ratio, nbits):
        super().__init__(model, tok, dev, ratio)
        self.nbits = nbits
        from src.quantized_cache_append_only import make_quant_cache
        self.cache, self.cache_mode = make_quant_cache(model.config, nbits)
    @torch.no_grad()
    def snapshot(self):
        """Exact rollback state, per layer. Measured exact 30/30 turns for BOTH cache layouts
        (probe_hqq_requantization_drift, jobs 3055723/3055728): HF's update() REASSIGNS its buffers
        rather than mutating them, and the append-only layout's committed blocks are immutable by
        construction, so saved references stay valid across the generation that follows."""
        from src.quantized_cache_append_only import layer_snapshot
        return ([layer_snapshot(l) for l in self.cache.layers], self.cache_len, self.pos)
    @torch.no_grad()
    def restore(self, state):
        from src.quantized_cache_append_only import layer_restore
        snap, clen, cpos = state
        for layer, s in zip(self.cache.layers, snap):
            layer_restore(layer, s)
        self.cache_len, self.pos = clen, cpos

def _mk_stateful(model, tok, dev, ratio):
    if QUANT_NBITS is not None:
        return QuantStateful(model, tok, dev, ratio, QUANT_NBITS)
    return Stateful(model, tok, dev, ratio)


def _abort_if_empty(done, out_path, n_conv):
    """A run that produced ZERO answered turns must FAIL, not exit 0 with an empty file.
    (2026-08-12: a ref with lowercase 'answerable' skipped every turn; the job showed COMPLETED
    and the empty .jsonl looked like a finished arm.)"""
    if done:
        return
    raise SystemExit(
        f"\n❌ 0 answered turns over {n_conv} conversations -> {out_path} is EMPTY.\n"
        f"   The turn filter is `Answerability[0].upper() == 'ANSWERABLE'`; check the ref's\n"
        f"   Answerability / input / turn fields. Exiting 3 so this is never mistaken for a result.")


def prep(tasks):
    """One conversation -> list of turns with newly-arrived passages, question, reference answer."""
    seen = set(); turns = []
    for t in tasks:
        cur = t.get("contexts") or []
        newk = [c for c in cur if pkey(c) not in seen]
        for c in newk: seen.add(pkey(c))
        turns.append({"newk": newk, "reuse_exposed": len(cur) > len(newk),
                      # .upper(): a ref written with lowercase 'answerable' silently skipped EVERY turn and
                      # the job still exited 0 with an empty output file (2026-08-12, locomo_ep10_60).
                      "ans": ((t.get("Answerability") or [""])[0] or "").upper(),
                      "q": last_user(t.get("input")),
                      "aref": (t.get("targets") or [{}])[0].get("text", ""),
                      "conv": t["conversation_id"], "turn": int(t["turn"])})
    # MAX_TURNS: truncate each conversation to its first K turns — TIMING/MEMORY probes only
    # (a probe needs the context ingest + a turn or two, not all 30; scores from truncated runs
    # are partial by construction and must never enter an accuracy table).
    _mt = int(os.environ.get("MAX_TURNS", "0") or 0)
    if _mt:
        turns = turns[:_mt]
    return turns


MERGE_LORA = os.environ.get("MERGE_LORA", "0") == "1"   # merge adapters into weights at load (timing runs)

PROV = None   # set in main(): {prompt_name, prompt_sha, decoding, max_new, model, reason_fix, ...}

def _rec(tr, pred, ct, raw=None):
    em, f1 = compute_best_em_f1(pred, [tr["aref"]])
    full = FULLANS or (INSTR_OVERRIDE is not None)  # store untruncated pred+gold for full-answer / QASPER scoring
    gcap = None if full else 150
    pcap = None if full else 80
    d = {"conv": tr["conv"], "turn": tr["turn"], "q": tr["q"], "gold": tr["aref"][:gcap],
         "reuse_exposed": tr["reuse_exposed"], "acc_f1": float(f1), "acc_em": int(em),
         "acc_pred": pred[:pcap], "acc_ctx_tok": ct}
    if raw is not None: d["raw"] = raw   # full generation incl. reasoning (for diagnosis)
    if _TURN.get("ans_t0") is not None:
        if torch.cuda.is_available(): torch.cuda.synchronize()
        d["ans_s"] = round(time.perf_counter() - _TURN["ans_t0"], 3)
    if _TURN.get("ingest_s") is not None: d["ingest_s"] = round(_TURN["ingest_s"], 3)
    if _TURN.get("ingest_reps"): d["ingest_reps"] = _TURN["ingest_reps"]
    a_gib, r_gib = _mem_gib()
    if a_gib is not None: d["peak_alloc_gib"], d["peak_reserved_gib"] = a_gib, r_gib
    # ★★ BATCH-1 TIMINGS ARE NOT THE METHOD'S TIMINGS. At batch 1 the flash-attention graph does not
    # compile, so ans_s / ingest_s / peak_*_gib measured here are an artefact of a broken kernel path.
    # This harness decodes one conversation at a time (BATCH_SIZE == 1), so the flag is always set today;
    # it disappears the moment batched decoding lands. Accuracy is unaffected (greedy decode per turn).
    # 2026-09-12 (user): the flag is about the eager/sequential batch-1 path. FORCE_BATCHED=1 on the FKV
    # path runs the batched class at B=1 — same code, same captured graph as B>=2 — so a B=1 timing there
    # is a measurement (the arm's largest fitting batch, e.g. teacher-32B at d320 on one card), not an artefact.
    if BATCH_SIZE < 2 and not (FORCED_BATCHED and os.environ.get("FKV_DECODE", "0") == "1"):
        d["timing_invalid_batch1"] = True
    if PROV is not None: d["_provenance"] = PROV
    return d


def run_single(model, tok, dev, ratio, tasks, max_new, stop_ids, method):
    """teacher / snapkv_frozen : incremental stateful.  snapkv_fresh : rebuild each turn."""
    turns = prep(tasks); out = []
    # quant_int8 / quant_int4: teacher flow (append passages FULL, reuse) but the persistent KV cache is HQQ-quantized.
    # Answer on snapshot/restore so committed KV is quantized ONCE (never re-quantized -> fair for low-bit int4).
    is_quant = method.startswith("quant")
    if is_quant:
        st = QuantStateful(model, tok, dev, ratio, QUANT_NBITS); st.forward(st._ids(INSTRUCTION() + "\n\n"))
        for tr in turns:
            _turn_begin()
            if tr["newk"]:
                st.forward(st._ids(fmt(tr["newk"])))          # commit passages (quantized once)
                _ingest_done()
            _raw = None
            if tr["ans"] == "ANSWERABLE" and tr["q"]:
                _ans_begin()
                snap = st.snapshot()                          # freeze committed quantized state
                last = st.forward(st._ids(q_turn(tr["q"]))).logits[:, -1, :]
                # kv_gib_flushed (2026-08-30): HF QuantizedLayer is LAZY — a big committed chunk
                # sits in the fp residual until the NEXT update flush-quantizes everything. The
                # post-rollback kv_gib therefore reads the pre-flush fp state on single-turn
                # shapes (98% of fp16, an artifact of measurement timing, NOT of what decode saw:
                # this question-forward triggered the flush, so generation reads quantized ctx).
                # Measure here, post-flush, for the honest retained-KV of an engaged quantizer.
                _kvf = _kv_gib(st.cache)
                gen = st.generate(last, max_new, stop_ids)
                st.restore(snap)                              # drop query+answer; committed KV untouched
                _raw = tok.decode(gen, skip_special_tokens=True)
                if REASON_FIX: _raw = RF.trim_leak(_raw)
                _r = _rec(tr, extract(_raw), st.cache_len, raw=_raw)
                _r['kv_gib'] = _kv_gib(st.cache)
                _r['kv_gib_flushed'] = _kvf
                out.append(_r)
            if REASON_FIX and _raw is not None:
                st.forward(st._ids(_hist_reason(q_turn(tr["q"]), _raw, tr["aref"])))   # commit history (quantized once)
            else:
                st.forward(st._ids(hist_block(tr["q"], tr["aref"])))
        return out
    # frozen-style (compress passage KV on arrival, reuse across turns): *_frozen (snapkv/pyramidkv) + h2o (query-indep).
    if method == "teacher" or method.endswith("_frozen") or method == "h2o" or method in SP_SINGLE:
        st = _mk_stateful(model, tok, dev, ratio); st.forward(st._ids(INSTRUCTION() + "\n\n"))
        for tr in turns:
            _turn_begin()
            if tr["newk"]:
                def _ingest_once():
                    if method == "teacher": st.forward(st._ids(fmt(tr["newk"])))
                    elif method in SP_SINGLE: st.specprefill_append(fmt(tr["newk"]), q_turn(tr["q"]))
                    else: st.compress_append(fmt(tr["newk"]), q_turn(tr["q"]))
                # INGEST_REPEATS: TTFT probes only — repeat the ingest, timing each, rolling the cache
                # back (exact crop + pos restore) so the run's semantics are untouched; the first repeat
                # absorbs kernel warmup. Times land in the row as `ingest_reps` next to the final `ingest_s`.
                for _ri in range(int(os.environ.get("INGEST_REPEATS", "0") or 0)):
                    _cb, _pb = st.cache_len, st.pos
                    _rt0 = time.perf_counter()
                    _ingest_once()
                    if torch.cuda.is_available(): torch.cuda.synchronize()
                    _TURN.setdefault("ingest_reps", []).append(round(time.perf_counter() - _rt0, 3))
                    st.crop(_cb); st.pos = _pb
                _ingest_once()
                if MEMO_MODE and method == "teacher":     # memo CONTROL for teacher/floor (single model = its own reader)
                    _memo = _memo_generate(st, tok, dev, stop_ids)
                    if _memo:
                        st.forward(st._ids(f"\n[SESSION MEMO]\n{_memo}\n"))
                        print(f"[memo] {len(_memo)} chars committed (single)", flush=True)
                _ingest_done()
            _raw = None
            if tr["ans"] == "ANSWERABLE" and tr["q"]:
                _ans_begin()
                base, bpos = st.cache_len, st.pos
                last = st.forward(st._ids(q_turn(tr["q"]))).logits[:, -1, :]
                gen = st.generate(last, max_new, stop_ids)
                st.crop(base); st.pos = bpos
                _raw = tok.decode(gen, skip_special_tokens=True)
                if REASON_FIX: _raw = RF.trim_leak(_raw)
                _r = _rec(tr, extract(_raw), st.cache_len, raw=_raw)
                _r['kv_gib'] = _kv_gib(st.cache)
                out.append(_r)
            # history: REASON_FIX -> teacher-force OUR generated reasoning+answer (so the model keeps reasoning across
            # turns); else the reference-answer accumulate. Non-answerable turns fall back to the reference.
            if REASON_FIX and _raw is not None:
                st.forward(st._ids(_hist_reason(q_turn(tr["q"]), _raw, tr["aref"])))
            else:
                st.forward(st._ids(hist_block(tr["q"], tr["aref"])))
    elif method.endswith("_fresh"):                          # rebuild + recompress every block on the CURRENT Q_t
        gen_hist = {}                                        # turn_idx -> OUR generated reasoning+answer (for REASON_FIX history)
        for i, tr in enumerate(turns):
            _turn_begin()
            if tr["ans"] == "ANSWERABLE" and tr["q"]:
                _ans_begin()   # ans_s here = the full O(n) rebuild + decode — that IS the _fresh method's cost
                st = Stateful(model, tok, dev, ratio); st.forward(st._ids(INSTRUCTION() + "\n\n"))
                for j in range(i + 1):                       # rebuild, compressing every block on the CURRENT Q_t
                    tj = turns[j]
                    if tj["newk"]: st.compress_append(fmt(tj["newk"]), q_turn(tr["q"]))
                    if j < i:
                        if REASON_FIX and j in gen_hist:
                            st.forward(st._ids(q_turn(tj["q"]) + gen_hist[j] + "\n"))
                        else:
                            st.forward(st._ids(q_turn(tj["q"]) + ref_answer(tj["aref"])))
                last = st.forward(st._ids(q_turn(tr["q"]))).logits[:, -1, :]
                gen = st.generate(last, max_new, stop_ids)
                _raw = tok.decode(gen, skip_special_tokens=True)
                if REASON_FIX: _raw = RF.trim_leak(_raw)
                gen_hist[i] = (RF.reason_prefix(_raw) + f"\nFinal Answer: {tr['aref']}") if REASON_HIST=="ref" else _raw.rstrip()
                _r = _rec(tr, extract(_raw), st.cache_len, raw=_raw)
                _r['kv_gib'] = _kv_gib(st.cache)
                out.append(_r); del st
    return out


def run_standard(model, tok, dev, ratio, tasks, max_new, stop_ids, method,
                 slm=None, slm_tok=None, lam=0.7):
    """★ STANDARD (NO-ACCUMULATION) setting — the CONTROL for the KV-reuse axis.

    Every question is answered INDEPENDENTLY on a FRESH cache: [INSTRUCTION][all passages][question].
    No dialogue history, no cross-turn KV reuse. This isolates a method's raw quality from the
    reuse penalty — which matters for SpecPrefill because its selection is QUERY-DEPENDENT: with
    accumulation it is anchored to whichever question arrived with the passages (a Q1 lottery, like
    snapkv_frozen), whereas here every question gets its OWN selection.

    Supports: teacher / specprefill / ours / specprefill_ours (fusion variants pass slm).
    """
    turns = prep(tasks); out = []
    all_ctx = []
    for tr in turns:
        all_ctx.extend(tr["newk"])                      # every passage of this conversation/paper
    ctx_text = fmt(all_ctx)
    fuse = method in ({"ours"} | SP_FUSION)
    use_sp = method in SP_ALL
    for tr in turns:
        if tr["ans"] != "ANSWERABLE" or not tr["q"]:
            continue
        _turn_begin(); _ans_begin()   # ans_s = fresh full prefill + decode (the standard setting's per-question cost)
        L = Stateful(model, tok, dev, ratio)            # FRESH cache per question — no reuse
        L.forward(L._ids(INSTRUCTION() + "\n\n"))
        if use_sp:
            L.specprefill_append(ctx_text, q_turn(tr["q"]))     # selection uses THIS question
        elif not fuse:
            L.forward(L._ids(ctx_text))                          # teacher: full context
        # (fuse and not use_sp) == plain 'ours': the LM sees NO passages
        ll = L.forward(L._ids(q_turn(tr["q"]))).logits[:, -1, :]
        if fuse:
            S = Stateful(slm, slm_tok, dev, ratio)
            S.forward(S._ids(INSTRUCTION() + "\n\n")); S.forward(S._ids(ctx_text))
            sl = S.forward(S._ids(q_turn(tr["q"]))).logits[:, -1, :]
            V = min(sl.shape[-1], ll.shape[-1]); gen = []; _ans = False
            for step in range(max_new):
                _lam = LAM_ANSWER if (LAM_ANSWER >= 0 and _ans) else lam   # within-answer λ drop at 'Final Answer:'
                fused = fuse_logits(sl, ll, _lam, V)
                if REASON_FIX:
                    fused = RF.block_logits(fused, step)
                    sl, ll = RF.block_logits(sl, step), RF.block_logits(ll, step)
                nxt = pick_from_fused(fused, sl, ll, V)
                if nxt in stop_ids: break
                gen.append(nxt)
                if LAM_ANSWER >= 0 and not _ans and "Final Answer" in slm_tok.decode(gen, skip_special_tokens=True): _ans = True
                if (RF.reason_stopped(slm_tok, gen) if REASON_FIX else stop_gen(slm_tok, gen)): break
                tt = torch.tensor([[nxt]], device=dev)
                sl = S.forward(tt).logits[:, -1, :]; ll = L.forward(tt).logits[:, -1, :]
            del S
        else:
            gen = L.generate(ll, max_new, stop_ids)
        _raw = tok.decode(gen, skip_special_tokens=True)
        if REASON_FIX: _raw = RF.trim_leak(_raw)
        out.append(_rec(tr, extract(_raw), L.cache_len, raw=_raw))
        del L
    return out



EFFECTIVE_BATCH = None      # set in main() once _batchable is known; None until then
FORCED_BATCHED = False      # FORCE_BATCHED=1 diagnostic: the batched class at B=1


def run_ours_batched(slm, slm_tok, lm, lm_tok, dev, lam, tasklists, max_new, stop_ids, lm_no_accum=False):
    """`ours` over B conversations at once — the batched twin of run_ours.

    Only the plain fusion path is batched (no specprefill / memo / lam-schedule / toklog): those axes are
    single-conversation experiments and fall back to run_ours. Every sequence is held at the SAME cache
    length by LEFT-padding each appended chunk, which is what makes logits_to_keep=1 return each sequence's
    true last-token logits and DynamicCache.crop an exact per-sequence rollback (src/batched_stateful.py).

    Conversations in a batch may have different turn counts; a finished conversation is fed a masked pad.

    lm_no_accum=True is ONE line here (the LM's history commit is skipped), exactly as in run_ours: the SLM
    still accumulates passages+dialogue, the LM's committed state stays at the instruction, and each turn's
    query is rolled back by restore(). It was excluded from _batchable purely because nobody had wired it,
    and that exclusion silently demoted the run to batch=1 — which never compiles the flash-attention graph
    and is therefore an unconditional throughput loss (CLAUDE.md 0원칙).
    """
    from src.batched_stateful import BatchedFusion
    B = len(tasklists)
    preps = [prep(t) for t in tasklists]
    # contexts BEFORE the decoder, so FKV sizes the READER branch from THIS batch (2026-08-31)
    ctxs = []
    for turns in preps:
        first = next((t for t in turns if t["newk"]), None)
        ctxs.append(fmt(first["newk"]) if first else "")
    # FUSION_LM_DEV=cuda:1 puts the LM branch on a SECOND card so the two forwards of a decode step
    # overlap instead of running one after the other (2026-09-01). Unset -> both on one card, the
    # default, and nothing changes. Under the 0원칙 a second card needs a reason, and this is the
    # only one that applies to ours and not to a baseline: a baseline given a second card gets a
    # second replica, i.e. twice the throughput at the SAME per-answer latency, while ours can make
    # one answer arrive sooner.
    # ★ EACH BRANCH'S DEVICE IS READ FROM ITS OWN WEIGHTS, not from an env var and not from the
    # caller's `dev` (2026-09-01). `dev` at the call site is `next(model.parameters()).device`, i.e.
    # the LM's — so putting the LM on a second card silently moved the READER's stateful there too,
    # while the 7B weights stayed on cuda:0, and the first split run died in the embedding lookup
    # with "index is on cuda:1, other tensors on cuda:0". Reading each device from the module that
    # owns it cannot disagree with reality; FUSION_LM_DEV now only decides where the LM LOADS.
    _dev_s = next(slm.parameters()).device
    _dev_l = next(lm.parameters()).device
    D = BatchedFusion(slm, slm_tok, lm, lm_tok, lam, _dev_s, B, INSTRUCTION(), lam_answer=LAM_ANSWER,
                      max_len=_fkv_batch_bound(slm_tok, INSTRUCTION(), ctxs, preps, max_new),
                      lm_max_len=_fkv_lm_bound(lm_tok, INSTRUCTION(), preps, max_new, lm_no_accum),
                      reader_quant=READER_QUANT_NBITS,
                      lm_dev=(_dev_l if _dev_l != _dev_s else None))
    if _dev_l != _dev_s:
        print(f"[split] reader on {_dev_s}, LM on {_dev_l} — branches run concurrently", flush=True)
    # ★ TIMING (2026-08-18). The single-conversation paths call _turn_begin/_ans_begin and their records
    #   carry ans_s / ingest_s; this batched path never did, so every run since batch 3 became the default
    #   wrote scores with NO time columns at all — `results/clutrr_accum/*` has ans_s, `clb3_*`/`c30b3_*`
    #   do not. That blocks the E2E/TTFT columns of the results table.
    #   In a batch the generation is ONE call covering B conversations, so what is measurable is the
    #   BATCH's wall clock. Both are recorded: `batch_ans_s` as measured, and `ans_s` = batch_ans_s /
    #   (active rows) as the per-example share — the derived one is labelled so it is never mistaken for a
    #   single-sequence latency. `prefill_s` is the shared context ingest, likewise per batch.
    # ★★ TOTAL WALL (2026-08-28, user). Timing was assembled from PIECES — `batch_ans_s` for the
    # answer and `prefill_s_batch` for the context ingest — and the per-turn history commit below fell
    # between them and was counted by NEITHER. Summing pieces cannot be audited: whatever is not
    # wrapped in a timer silently disappears, and here it disappeared in the baselines' favour (they
    # append to a long-context cache every turn, we do not). `conv_wall_s` times the WHOLE batch of
    # conversations from before the context prefill to after the last commit, so nothing can be
    # omitted by construction. It is the only wall a throughput number may be divided by.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        for d in range(torch.cuda.device_count()): torch.cuda.reset_peak_memory_stats(d)
    _t_conv = time.perf_counter()
    _t_pref = time.perf_counter()
    D.prefill(ctxs)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    prefill_s = round(time.perf_counter() - _t_pref, 3)
    out = [[] for _ in range(B)]
    for ti in range(max(len(t) for t in preps)):
        rows = [(t[ti] if ti < len(t) else None) for t in preps]
        qb = [(q_turn(r["q"]) if r and r["ans"] == "ANSWERABLE" and r["q"] else "\n") for r in rows]
        st_s, st_l = D.S.state(), D.L.state()
        if torch.cuda.is_available(): torch.cuda.synchronize()
        _t_ans = time.perf_counter()
        gens = D.turn(qb, max_new, stop_ids,
                      block_fn=(RF.block_logits if REASON_FIX else None),
                      done_fn=(lambda g: RF.reason_stopped(slm_tok, g)) if REASON_FIX else
                              (lambda g: stop_gen(slm_tok, g)))
        if torch.cuda.is_available(): torch.cuda.synchronize()
        batch_ans_s = round(time.perf_counter() - _t_ans, 3)
        n_active = max(1, sum(1 for r in rows if r is not None))
        # MEASURED KV bytes of the whole batch, read AFTER D.turn()'s question forward and BEFORE the
        # rollback. For a quantized reader that forward is what flush-quantizes the committed context,
        # so reading later (or earlier) reports the fp residual instead — the same 98%-of-fp16 artefact
        # the single-model runner documents. Split by branch because the composition claim is about the
        # READER's cache: the LM branch is query-only and its bytes are not what is being compressed.
        _kv_s, _kv_l = _kv_gib(D.S.cache), _kv_gib(D.L.cache)
        D.S.restore(st_s); D.L.restore(st_l)             # drop the generated KV, keep the committed state
        hist = []
        for i, r in enumerate(rows):
            if r is None:
                hist.append("\n"); continue
            raw = RF.trim_leak(slm_tok.decode(gens[i], skip_special_tokens=True)) if REASON_FIX \
                else slm_tok.decode(gens[i], skip_special_tokens=True)
            if r["ans"] == "ANSWERABLE" and r["q"]:
                _d = _rec(r, extract(raw), D.S.cache_len, raw=raw)
                _d["batch_ans_s"] = batch_ans_s          # MEASURED: wall clock of the whole batch's turn
                _d["batch_rows"] = n_active
                _d["ans_s_per_example"] = round(batch_ans_s / n_active, 4)   # DERIVED, labelled as such
                _d["prefill_s_batch"] = prefill_s        # MEASURED: shared context ingest for this batch
                # MEASURED: question-block forward through both branches + step 0, i.e. time to the
                # FIRST generated token of this turn. Separate from prefill_s_batch on purpose.
                _d["ttft_s"] = getattr(D, "ttft_s", None)
                # MEASURED, ONE WALL (2026-09-14, user: a real TTFT, not the prefill): from the synchronized instant
                # before the context prefill to the first generated token of the FIRST turn arriving on the host —
                # both branches' prefill + question forward + decode step 0, nothing summed. Turn 0 only; a
                # single-sequence latency only when batch_rows == 1 (the TTFT probe runs FORCE_BATCHED=1 B=1).
                _ft = getattr(D, "first_tok_t", None)
                _d["ttft_from_prefill_s"] = (round(_ft - _t_pref, 4) if (ti == 0 and _ft) else None)
                _d["kv_gib_batch_reader"] = _kv_s     # MEASURED tensor bytes, reader branch, whole batch
                _d["kv_gib_batch_lm"] = _kv_l         # MEASURED tensor bytes, LM branch, whole batch
                _d["kv_gib_batch"] = (None if (_kv_s is None or _kv_l is None)
                                      else round(_kv_s + _kv_l, 4))
                # bytes PER CONTEXT TOKEN is the only cross-run-comparable form: batches are
                # length-bucketed, so a per-seq number compares different batches' lengths (the error
                # that made a 53%-sized cache read as 89% on 2026-08-31).
                _slots = D.B * int(getattr(D.S, "cache_len", 0) or 0)
                _d["kv_kib_per_cache_token_reader"] = (None if (_kv_s is None or not _slots)
                                                       else round(_kv_s * 1024 * 1024 / _slots, 3))
                _d["kv_cache_slots_reader"] = _slots or None
                for _st, _pre in ((getattr(D, "S", None), "fkvS_"), (getattr(D, "L", None), "fkvL_")):
                    _cr = getattr(_st, "capacity_report", None)
                    if _cr:      # FKV only: reserved vs actually used, so the invariant is auditable
                        _d.update({_pre + k[4:]: v for k, v in _cr().items()})
                out[i].append(_d)
            hist.append(_hist_reason(q_turn(r["q"]), raw, r["aref"]) if REASON_FIX
                        else hist_block(r["q"], r["aref"]))
        ids, m = D.S.ids_of(hist); D.S.forward(ids, m)
        if not lm_no_accum:                       # ← the whole of the lm_no_accum axis, mirroring run_ours
            ids, m = D.L.ids_of(hist); D.L.forward(ids, m)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    conv_wall_s = round(time.perf_counter() - _t_conv, 3)
    for _rows in out:
        for _d in _rows:
            _d["conv_wall_s"] = conv_wall_s       # MEASURED: the whole batch, nothing outside a timer
            _d["conv_batch_rows"] = sum(len(x) for x in out)
    _release_batched(D)
    return out


def _release_batched(D):
    """Drop a batched decoder's preallocated buffers + global registrations before the next batch.

    The FKV backend registers itself in transformers' MODULE-LEVEL ALL_ATTENTION_FUNCTIONS via a
    closure that captures the object, so a per-batch stateful is pinned forever and every batch leaks
    a whole preallocated cache (job 3055450: the musique ladder OOM'd at EVERY rung, teacher B=4 and
    floor-7B B=64 included — an OOM that does not improve as the batch shrinks is a leak, not a
    capacity limit). Cheap and harmless for the backends that do not need it.
    """
    import gc as _gc
    for st in (getattr(D, "S", None), getattr(D, "L", None)):
        rel = getattr(st, "release", None)
        if rel is not None:
            rel()
    _gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _fkv_lm_bound(tok, instruction, tasklists, max_new, lm_no_accum):
    """The LM branch's own capacity need — the twin of `_fkv_batch_bound` for the query-only branch.

    ★ 2026-08-31. The reader branch was moved off a dataset-wide constant that morning because
    reserving storage no sequence uses cuts B_max and makes OUR OWN method look slower. The LM
    branch was left on `STATIC_MAXLEN_LM`, a fixed env constant, and the very next measurement
    showed it reserving 1024 slots for a peak of 321 -- 68.7% unused. At B=14 on a 32B (256 KiB per
    token) that is 3.5 GiB reserved against 1.1 GiB used: 2.4 GiB held back from the batch, on the
    same method, for the same reason, found and then not fixed. Fixing it here.

    Under `lm_no_accum` (the canonical setting) the LM never accumulates dialogue: it holds the
    instruction, the current question, and the generation being decoded, and is rolled back after
    each turn. So the bound is instruction + longest question + max_new, plus slack. With
    accumulation on, the committed history grows per turn exactly as the reader's does.
    """
    def n(s):
        return len(tok(s, add_special_tokens=False).input_ids) if s else 0
    instr = n(instruction + "\n\n")
    q_max = 0
    for tl in tasklists:
        for r in tl:
            q_max = max(q_max, n(str(r.get("q") or "")))
    if lm_no_accum:
        return int((instr + q_max + max_new + 64) * 1.15)
    turns = max((len(t) for t in tasklists), default=1)
    per_turn = q_max + max_new + 32
    return int((instr + turns * per_turn + max_new + 64) * 1.15)


def _fkv_batch_bound(tok, instruction, ctxs, tasklists, max_new):
    """The cache capacity THIS batch actually needs, for the FKV backend to size itself from.

    Replaces a dataset-wide `STATIC_MAXLEN` applied to every slot, which reserved storage no
    sequence in the batch would use and cut B_max — i.e. it made our own method look slower than it
    is (2026-08-31 audit: 2-9% over-reservation on the runs that were published).

    Bound = instruction + the batch's longest context + one committed history block per turn +
    the generation being decoded. History blocks contain generated reasoning whose length is not
    known in advance, so they are bounded by `max_new` plus the reference answer, and the whole
    figure carries 15% slack. Under-estimating is not fatal: FlashKVStateful grows.
    """
    def n(s):
        return len(tok(s, add_special_tokens=False).input_ids) if s else 0
    instr = n(instruction + "\n\n")
    ctx_max = max((n(c) for c in ctxs), default=0)
    turns = max((len(t) for t in tasklists), default=1)
    per_turn = 0
    for tl in tasklists:
        for r in tl[:4]:                       # sample a few turns; blocks are near-uniform in size
            per_turn = max(per_turn, n(str(r.get("q") or "")) + n(str(r.get("aref") or "")))
    growth = turns * (per_turn + max_new + 32)
    return int((instr + ctx_max + growth + max_new + 64) * 1.15)


def run_single_batched(model, tok, dev, tasklists, max_new, stop_ids, compress=False, spec=False,
                       quant_nbits=None):
    """`teacher` / floor, and — with compress=True — the FROZEN compression arms, over B conversations.

    compress=True routes the context through BatchedStateful.compress_append instead of a plain forward:
    the press keeps ratio x block_len per row, the rows are left-padded to the largest kept count and the
    pad columns masked, which restores the shared cache length the batched cache is built on.

    ★ WHY IT IS WORTH THE CODE. Left at batch 1 these arms never trigger the flash-attention graph compile,
    which is an unconditional throughput loss (CLAUDE.md 0원칙), and — measured on 2026-08-13 — a batch-1
    arm is also on a DIFFERENT DECODE PATH from a batched one (`PREFILL_CHUNK`, §7s), so a sequential
    compression baseline may not share a table with batched fusion arms. Batching them fixes both at once.
    quant_nbits routes the persistent cache through BatchedQuantStateful (HQQ, committed KV quantized
    exactly once, snapshot/restore rollback because a quantized cache cannot be cropped); specprefill
    still takes its own append path because its selection is query-dependent per turn."""
    from src.batched_stateful import BatchedSingle
    B = len(tasklists)
    preps = [prep(t) for t in tasklists]
    # contexts BEFORE the decoder, so FKV can size its preallocation from THIS batch (2026-08-31)
    ctxs, qc = [], []
    for turns in preps:
        first = next((t for t in turns if t["newk"]), None)
        ctxs.append(fmt(first["newk"]) if first else "")
        qc.append(q_turn(first["q"]) if (first and first.get("q")) else "\n")
    D = BatchedSingle(model, tok, dev, B, INSTRUCTION(), quant_nbits=quant_nbits,
                      max_len=_fkv_batch_bound(tok, INSTRUCTION(), ctxs, preps, max_new))
    # ★ SAME TIMER AS run_ours_batched (2026-09-14). The fusion path synchronizes before its prefill timer
    # starts and again before it stops; this path did neither, so its prefill_s stopped while the tail of
    # the last PREFILL_CHUNK-token slice of the context forward could still be running on the GPU (FKV
    # syncs on the host only at the START of each slice). Every teacher/floor prefill_s written before
    # this line read low by at most that slice; conv_wall_s was always synchronized and is unaffected.
    if torch.cuda.is_available(): torch.cuda.synchronize()
    _t0 = time.perf_counter()
    if compress:
        ids, m = D.S.ids_of([D.instr + "\n\n"] * B); D.S.forward(ids, m)
        keeps, K = D.S.compress_append(ctxs, qc, make_press, PRESS_METHOD, RATIO_GLOBAL,
                                       PRESS_METHOD in QUERY_DEP)
        print(f"[compress] kept per row {keeps} -> padded to {K} (ratio {RATIO_GLOBAL})", flush=True)
    elif spec:
        ids, m = D.S.ids_of([D.instr + "\n\n"] * B); D.S.forward(ids, m)
        keeps, K = D.S.specprefill_append(ctxs, qc, SPEC_MODEL, SPEC_KEEP,
                                          SPEC_CHUNK, SPEC_POOL, SPEC_LAH)
        print(f"[specprefill] selected per row {keeps} -> padded to {K} (keep {SPEC_KEEP})", flush=True)
    else:
        D.prefill(ctxs)
    # same timing convention as run_ours_batched: prefill_s_batch MEASURED (shared context ingest),
    # batch_ans_s MEASURED per turn, ans_s_per_example DERIVED (batch / active rows), labelled so.
    # conv_wall_s (below) is the only wall a throughput number may be divided by — see the note in
    # run_ours_batched: the per-turn history commit sits outside batch_ans_s and was counted nowhere.
    if torch.cuda.is_available(): torch.cuda.synchronize()     # close the prefill timer on the GPU, as the fusion path does
    prefill_s = round(time.perf_counter() - _t0, 3)
    out = [[] for _ in range(B)]
    for ti in range(max(len(t) for t in preps)):
        rows = [(t[ti] if ti < len(t) else None) for t in preps]
        qb = [(q_turn(r["q"]) if r and r["ans"] == "ANSWERABLE" and r["q"] else "\n") for r in rows]
        st = D.S.state()
        _t_ans = time.perf_counter()
        gens = D.turn(qb, max_new, stop_ids,
                      block_fn=(RF.block_logits if REASON_FIX else None),
                      done_fn=(lambda g: RF.reason_stopped(tok, g)) if REASON_FIX else
                              (lambda g: stop_gen(tok, g)))
        batch_ans_s = round(time.perf_counter() - _t_ans, 3)
        # MEASURED KV bytes of the whole batch, read AFTER D.turn()'s question forward — for a
        # quantized cache THAT forward is what flush-quantizes the committed context, so reading
        # earlier reports the pre-flush fp residual (the 98%-of-fp16 measurement artefact, 08-30).
        _kv_batch = _kv_gib(D.S.cache)
        n_active = sum(1 for r in rows if r is not None and r["ans"] == "ANSWERABLE" and r["q"]) or 1
        D.S.restore(st)
        hist = []
        for i, r in enumerate(rows):
            if r is None:
                hist.append("\n"); continue
            raw = tok.decode(gens[i], skip_special_tokens=True)
            if REASON_FIX: raw = RF.trim_leak(raw)
            if r["ans"] == "ANSWERABLE" and r["q"]:
                _d = _rec(r, extract(raw), D.S.cache_len, raw=raw)
                _d["batch_ans_s"] = batch_ans_s
                _d["batch_rows"] = n_active
                _d["ans_s_per_example"] = round(batch_ans_s / n_active, 4)
                _d["prefill_s_batch"] = prefill_s
                _d["ttft_s"] = getattr(D, "ttft_s", None)           # question forward + step 0 (as run_ours_batched)
                # MEASURED, ONE WALL: synchronized instant before the prefill -> first token on the host, turn 0
                # (see run_ours_batched); single-sequence only at batch_rows == 1
                _ft = getattr(D, "first_tok_t", None)
                _d["ttft_from_prefill_s"] = (round(_ft - _t0, 4) if (ti == 0 and _ft) else None)
                for _st in (getattr(D, "S", None), getattr(D, "L", None)):
                    _cr = getattr(_st, "capacity_report", None)
                    if _cr:                       # FKV only: reserved vs actually used, per the invariant
                        _pre = "fkvS_" if _st is getattr(D, "S", None) else "fkvL_"
                        _d.update({_pre + k[4:]: v for k, v in _cr().items()})
                _d["kv_gib_batch"] = _kv_batch                        # MEASURED tensor bytes, whole batch
                _d["kv_gib_per_seq"] = (None if _kv_batch is None     # DERIVED: batch bytes / B rows
                                        else round(_kv_batch / B, 4))
                # ★ kv_gib_per_seq IS NOT COMPARABLE ACROSS RUNS and reading it as if it were cost a
                # published claim on 2026-08-31: batches are LENGTH-BUCKETED, and `longest_first`
                # switches on only when an arm's ladder has more than one rung — so a one-rung arm
                # writes its SHORTEST batch first and a two-rung arm its LONGEST. Comparing row 0 of
                # each put the teacher's shortest batch against int8's longest and made a 53%-sized
                # cache look like 89%. Bytes PER TOKEN divide that difference out, so this is the
                # field a comparison may use.
                # ★ THIS FIELD WAS ALWAYS None (found 2026-08-31 while validating hotq's first 60
                # rows). `rows` are the TASK rows from prep(); `acc_ctx_tok` is added later by _rec(),
                # so the sum was 0 every time and the guard returned None. Every log ever written by
                # this path carries kv_kib_per_token=None — the byte axis the composition and press
                # comparisons are supposed to be read on has never actually been recorded.
                # The denominator is now the cache's own token capacity: B rows x cache_len positions
                # is exactly how many token slots the measured tensors hold, so the quotient is the
                # per-token KV size at that precision and is comparable across arms and batches.
                # (Dividing by CONTEXT tokens instead would have mixed in a second axis — how much of
                # each sequence is instruction and question — which differs between benches.)
                _slots = B * int(getattr(D.S, "cache_len", 0) or 0)
                _d["kv_kib_per_cache_token"] = (None if (_kv_batch is None or not _slots)
                                                else round(_kv_batch * 1024 * 1024 / _slots, 3))
                _d["kv_cache_slots"] = _slots or None
                out[i].append(_d)
            hist.append(_hist_reason(q_turn(r["q"]), raw, r["aref"]) if REASON_FIX
                        else hist_block(r["q"], r["aref"]))
        ids, m = D.S.ids_of(hist); D.S.forward(ids, m)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    conv_wall_s = round(time.perf_counter() - _t0, 3)
    for _rows in out:
        for _d in _rows:
            _d["conv_wall_s"] = conv_wall_s       # MEASURED: the whole batch, nothing outside a timer
            _d["conv_batch_rows"] = sum(len(x) for x in out)
    _release_batched(D)
    return out


def run_ours(slm, slm_tok, lm, lm_tok, dev, lam, tasks, max_new, stop_ids, specprefill=False, lm_no_accum=False):
    """Stateful fusion: SLM accumulates passages+dialogue; LM accumulates dialogue/query only (no passages).

    specprefill=True → the LM ALSO prefills the speculator-SELECTED passage tokens (at their ORIGINAL
    positions) instead of seeing no context at all. That is the 'SpecPrefill + fusion' configuration: on an
    EXTRACTIVE task the query-only LM has no spans to contribute (why QASPER closeness is low), so handing it
    the important ~10% of the context is the targeted intervention.

    lm_no_accum=True (2026-07-27) → the LM does NOT accumulate the Q/A(/R) dialogue history at all; each turn it
    sees ONLY the instruction + the current query (rolled back after). Rationale: ours' LM barely prefills context,
    so accumulating prior Q/A only hurts — the LM read the history as few-shot examples and IGNORED the reasoning
    instruction (which forced the ad-hoc C-Q-R-A reasoning-injection). With no accumulated history a plain
    'reason about Q' prompt actually reasons. The SLM (reader) still accumulates as before; ONLY the LM changes.
    """
    turns = prep(tasks); out = []
    if READER_QUANT_NBITS:
        assert not int(os.environ.get("INGEST_REPEATS", "0") or 0), \
            "READER_QUANT is incompatible with INGEST_REPEATS (crop would requantize committed KV)"
        S = QuantStateful(slm, slm_tok, dev, 0.0, READER_QUANT_NBITS)
    else:
        S = Stateful(slm, slm_tok, dev, 0.0)
    L = Stateful(lm, lm_tok, dev, 0.0)
    S.role, L.role = "S", "L"
    S.forward(S._ids(INSTRUCTION() + "\n\n")); L.forward(L._ids(INSTRUCTION() + "\n\n"))
    for tr in turns:
        _turn_begin()
        if tr["newk"]:
            def _ingest_once():
                S.forward(S._ids(fmt(tr["newk"])))                      # SLM reads the FULL passages
                if specprefill:
                    L.specprefill_append(fmt(tr["newk"]), q_turn(tr["q"]))  # LM reads only the SELECTED tokens
            for _ri in range(int(os.environ.get("INGEST_REPEATS", "0") or 0)):   # TTFT probes only
                _sb, _sp, _lb, _lp = S.cache_len, S.pos, L.cache_len, L.pos
                _rt0 = time.perf_counter()
                _ingest_once()
                if torch.cuda.is_available(): torch.cuda.synchronize()
                _TURN.setdefault("ingest_reps", []).append(round(time.perf_counter() - _rt0, 3))
                S.crop(_sb); S.pos = _sp; L.crop(_lb); L.pos = _lp
            _ingest_once()
            if MEMO_MODE:                                               # session-index memo -> BOTH branches
                _memo = _memo_generate(S, slm_tok, dev, stop_ids)
                if _memo:
                    _mblk = f"\n[SESSION MEMO]\n{_memo}\n"
                    S.forward(S._ids(_mblk)); L.forward(L._ids(_mblk))
                    print(f"[memo] {len(_memo)} chars committed to both branches", flush=True)
            _ingest_done()
        _raw = None
        if tr["ans"] == "ANSWERABLE" and tr["q"]:
            _ans_begin()
            # turn-adaptive λ (2026-07-28): the SpecPrefill reuse-collapse is TURN-DRIVEN (turn 1 fresh selection → SpecPrefill
            # fine; turn 2+ reused selection stale → collapse). LAM_SCHEDULE=lam1,lam2,... sets λ by turn index (last value repeats).
            lam_t = LAM_SCHEDULE[min(int(tr["turn"]) - 1, len(LAM_SCHEDULE) - 1)] if LAM_SCHEDULE else lam
            sb, sp = S.cache_len, S.pos; lb, lp = L.cache_len, L.pos
            s_state = S.snapshot() if READER_QUANT_NBITS else None
            sl = S.forward(S._ids(q_turn(tr["q"]))).logits[:, -1, :]
            ll = L.forward(L._ids(q_turn(tr["q"]))).logits[:, -1, :]
            V = min(sl.shape[-1], ll.shape[-1]); gen = []; _ans = False; _steps = []
            for step in range(max_new):
                _lam = LAM_ANSWER if (LAM_ANSWER >= 0 and _ans) else lam_t   # within-answer λ drop at 'Final Answer:'
                fused = fuse_logits(sl, ll, _lam, V)
                if REASON_FIX:   # SLM injects facts, LM reasons over them, first
                    fused = RF.block_logits(fused, step)
                    sl, ll = RF.block_logits(sl, step), RF.block_logits(ll, step)
                nxt = pick_from_fused(fused, sl, ll, V)
                if nxt in stop_ids: break
                if TOKLOG: _steps.append(_tok_stats(sl, ll, _lam, V, nxt))
                gen.append(nxt)
                if LAM_ANSWER >= 0 and not _ans and "Final Answer" in slm_tok.decode(gen, skip_special_tokens=True): _ans = True
                if (RF.reason_stopped(slm_tok, gen) if REASON_FIX else stop_gen(slm_tok, gen)): break
                tt = torch.tensor([[nxt]], device=dev)
                sl = S.forward(tt).logits[:, -1, :]; ll = L.forward(tt).logits[:, -1, :]
            if READER_QUANT_NBITS:
                S.restore(s_state)                    # exact rollback; committed quantized KV untouched
            else:
                S.crop(sb); S.pos = sp
            L.crop(lb); L.pos = lp
            _raw = RF.trim_leak(slm_tok.decode(gen, skip_special_tokens=True)) if REASON_FIX else slm_tok.decode(gen, skip_special_tokens=True)
            if TOKLOG:
                global _TOKF
                if _TOKF is None: _TOKF = open(TOKLOG, "w")
                _TOKF.write(json.dumps({"conv": tr["conv"], "turn": tr["turn"],
                                        "toks": [slm_tok.decode([s["t"]]) for s in _steps],
                                        "steps": _steps}) + "\n"); _TOKF.flush()
            _r = _rec(tr, extract(_raw), S.cache_len, raw=_raw)
            _r['kv_gib_reader'], _r['kv_gib_lm'] = _kv_gib(S.cache), _kv_gib(L.cache)
            _r['kv_gib'] = None if _r['kv_gib_reader'] is None else round(_r['kv_gib_reader'] + _r['kv_gib_lm'], 3)
            out.append(_r)
        # history into the SLM (reader) always; into the LM only if lm_no_accum is OFF (2026-07-27 axis).
        if REASON_FIX and _raw is not None:
            hgen = _hist_reason(q_turn(tr["q"]), _raw, tr["aref"])
            S.forward(S._ids(hgen))
            if not lm_no_accum: L.forward(L._ids(hgen))
        else:
            S.forward(S._ids(hist_block(tr["q"], tr["aref"])))
            if not lm_no_accum: L.forward(L._ids(hist_block(tr["q"], tr["aref"])))
    return out


TOKEN_INJECT_K     = int(os.environ.get("TOKEN_INJECT_K", "384"))     # per-step injected token budget
TOKEN_INJECT_CHUNK = int(os.environ.get("TOKEN_INJECT_CHUNK", "16"))
TOKEN_INJECT_POOL  = int(os.environ.get("TOKEN_INJECT_POOL", "13"))
TOKEN_INJECT_WIN   = int(os.environ.get("TOKEN_INJECT_WIN", "8"))     # avg SLM attn over last W decode steps


def content_mask(ctx_ids, tok, device):
    """1.0 for content tokens, 0.0 for structural/sink tokens — raw attention concentrates on sinks; masking lets
    real content surface in the top-k (validated in the single-turn probe)."""
    m = torch.ones(len(ctx_ids), device=device)
    for i, tid in enumerate(ctx_ids.tolist()):
        s = tok.decode([tid]).strip()
        if s == "" or s.isdigit() or all(not c.isalnum() for c in s) or s in ("Document", "Context", "###", "Question", "Text", "Passage", "Evidence"):
            m[i] = 0.0
    return m


@torch.no_grad()
def run_token_inject_accum(slm, slm_tok, lm, lm_tok, dev, tasks, max_new, stop_ids):
    """Fusion-free / λ-free SLM→LM transfer in the ACCUM setting (2026-07-30, user idea). The SLM reads the FULL
    accumulated context; each DECODE step it selects the top-k currently-important context CHUNKS (by recent attention)
    and the LM prefills ONLY those, at their TRUE positions, in front of its cached instruction+query (after_ctx
    layout — the winning probe layout). The LM ALONE emits (no λ). Injected KV is dropped each step. Dynamic per-step
    re-selection is exactly what static SpecPrefill can't do under reuse. Mirrors the single-turn probe token_inject."""
    from collections import deque
    turns = prep(tasks); out = []
    S = Stateful(slm, slm_tok, dev, 0.0)
    S.forward(S._ids(INSTRUCTION() + "\n\n"))
    instr_ids = lm_tok(INSTRUCTION() + "\n\n", return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
    ctx_chunks = []                                  # accumulated PASSAGE blocks: (start_pos, ids[1D]) — injectable evidence
    def ctx_tensors():
        if not ctx_chunks: return torch.zeros(0, dtype=torch.long, device=dev), torch.zeros(0, dtype=torch.long, device=dev)
        cids = torch.cat([ids for _, ids in ctx_chunks])
        cpos = torch.cat([torch.arange(st, st + ids.shape[0], device=dev) for st, ids in ctx_chunks])
        return cids, cpos
    def ctx_attn(o, CPOS):                            # max over heads AND layers, sliced to passage positions (cache idx==pos)
        lmax = None
        for a in o.attentions:
            mm = a[0, :, -1, :].float().max(0).values
            lmax = mm if lmax is None else torch.maximum(lmax, mm)
        return lmax[CPOS] if CPOS.numel() else lmax[:0]
    for tr in turns:
        _turn_begin()
        if tr["newk"]:
            pids = S._ids(fmt(tr["newk"]))
            ctx_chunks.append((S.pos, pids[0].clone()))
            S.forward(pids)
            _ingest_done()
        _raw = None
        if tr["ans"] == "ANSWERABLE" and tr["q"]:
            _ans_begin()
            C_END = S.pos
            CIDS, CPOS = ctx_tensors()
            cmask = content_mask(CIDS, slm_tok, dev) if CIDS.numel() else CIDS.float()
            # L (fresh): instruction+query PE AFTER the context [C_END, C_END+m)  (after_ctx = winning probe layout)
            qa_ids = torch.cat([instr_ids, lm_tok(q_turn(tr["q"]), return_tensors="pt", add_special_tokens=False).input_ids.to(dev)], 1)
            m = qa_ids.shape[1]
            with SP.sdpa_for_gappy_positions(lm):
                o = lm(qa_ids, position_ids=torch.arange(C_END, C_END + m, device=dev)[None], use_cache=True,
                       cache_position=torch.arange(m, device=dev), logits_to_keep=1)
            logits_l = o.logits[:, -1, :]
            pk = [ly.keys.clone() for ly in o.past_key_values.layers]; pv = [ly.values.clone() for ly in o.past_key_values.layers]; Ln = len(pk)
            # S: forward the query WITH attention (seed the recent-attention window); rolled back after the turn
            sb, sp = S.cache_len, S.pos
            with SP._eager_attn(slm):
                so = S.forward(S._ids(q_turn(tr["q"])), want_attn=True)
            dq = deque([ctx_attn(so, CPOS)], maxlen=TOKEN_INJECT_WIN)
            gen = []; gen_base = C_END + m
            for step in range(max_new):
                t = int(torch.argmax(logits_l[0]))
                if t in stop_ids: break
                gen.append(t)
                if (RF.reason_stopped(slm_tok, gen) if REASON_FIX else stop_gen(slm_tok, gen)): break
                if CIDS.numel():
                    imp = (torch.stack(list(dq)).mean(0) * cmask).float().cpu()
                    cmk, _, _ = SP.select_chunks(imp, 0.0, chunk_size=TOKEN_INJECT_CHUNK, pool_kernel=TOKEN_INJECT_POOL, keep_tokens=TOKEN_INJECT_K)
                    sel = torch.nonzero(cmk, as_tuple=False).squeeze(-1).to(dev)
                else:
                    sel = torch.zeros(0, dtype=torch.long, device=dev)
                ki = int(sel.numel())
                comb = DynamicCache()
                if ki:
                    with SP.sdpa_for_gappy_positions(lm):
                        oi = lm(CIDS[sel][None], position_ids=CPOS[sel][None], use_cache=True,
                                cache_position=torch.arange(ki, device=dev), logits_to_keep=1)
                    ik = [ly.keys for ly in oi.past_key_values.layers]; iv = [ly.values for ly in oi.past_key_values.layers]
                    for l in range(Ln): comb.update(torch.cat([ik[l], pk[l]], 2), torch.cat([iv[l], pv[l]], 2), l)
                else:
                    for l in range(Ln): comb.update(pk[l], pv[l], l)
                P = pk[0].shape[2]
                with SP.sdpa_for_gappy_positions(lm):
                    o = lm(torch.tensor([[t]], device=dev), past_key_values=comb,
                           position_ids=torch.tensor([[gen_base + step]], device=dev),
                           cache_position=torch.tensor([ki + P], device=dev), logits_to_keep=1)
                logits_l = o.logits[:, -1, :]
                nk = [ly.keys for ly in o.past_key_values.layers]; nv = [ly.values for ly in o.past_key_values.layers]
                pk = [nk[l][:, :, ki:, :].clone() for l in range(Ln)]; pv = [nv[l][:, :, ki:, :].clone() for l in range(Ln)]
                with SP._eager_attn(slm):
                    so = S.forward(torch.tensor([[t]], device=dev), want_attn=True)
                dq.append(ctx_attn(so, CPOS))
            S.crop(sb); S.pos = sp
            _raw = RF.trim_leak(slm_tok.decode(gen, skip_special_tokens=True)) if REASON_FIX else slm_tok.decode(gen, skip_special_tokens=True)
            _r = _rec(tr, extract(_raw), S.cache_len, raw=_raw)
            _r['kv_gib_reader'], _r['kv_gib_lm'] = _kv_gib(S.cache), _kv_gib(L.cache)
            _r['kv_gib'] = None if _r['kv_gib_reader'] is None else round(_r['kv_gib_reader'] + _r['kv_gib_lm'], 3)
            out.append(_r)
        # history into the SLM reader (passages already committed above); LM is rebuilt fresh each turn
        if REASON_FIX and _raw is not None:
            S.forward(S._ids(_hist_reason(q_turn(tr["q"]), _raw, tr["aref"])))
        elif tr["ans"] == "ANSWERABLE" and tr["q"]:
            S.forward(S._ids(hist_block(tr["q"], tr["aref"])))
    return out


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# SPEC VALIDATION BEFORE ANY MODEL LOAD (2026-08-31)
#
# Job 3058332 loaded a 32B and a 7B, then died on `--sweep` containing method 'ours'. The guard was
# correct; it just lived ~220 lines and one three-minute load AFTER the models were on the GPU. A
# malformed spec must cost seconds on a login node, not a GPU allocation and a place in the queue —
# on a partition running 25 free GPUs against 640 pending jobs, losing the slot costs far more than
# the minutes.
#
# It also rejects keys the runner would IGNORE, which is the more dangerous failure. `--sweep` does
# not read `reader_quant` (only `--kv-probe` does), so a sweep asking for a quantized reader would
# have run in fp16 and written rows LABELLED int8/int4 — a silent mislabel, not a crash. An ignored
# key is therefore a hard error, not a warning.
_SWEEP_KEYS = {"tag", "method", "ratio", "keep", "press_quant", "papers", "papers_pool",
               "batch", "longest_first"}
_KVPROBE_KEYS = {"tag", "method", "ratio", "keep", "floor", "reader_quant"}
_SWEEP_METHODS = ("teacher", "snapkv_frozen", "h2o",
                  "quant_int8", "quant_int4", "quant_int3", "quant_int2")


def validate_specs(args):
    """Check --sweep / --kv-probe before anything is loaded. Raises SystemExit on any problem."""
    for flag, raw, keys, req in (("--sweep", getattr(args, "sweep", None), _SWEEP_KEYS,
                                  {"tag", "batch"}),
                                 ("--kv-probe", getattr(args, "kv_probe", None), _KVPROBE_KEYS,
                                  {"tag", "method"})):
        if not raw:
            continue
        try:
            items = json.loads(raw)
        except json.JSONDecodeError as e:
            raise SystemExit(f"❌ {flag} is not valid JSON: {e}")
        if not isinstance(items, list) or not items:
            raise SystemExit(f"❌ {flag} must be a non-empty JSON list")
        for i, it in enumerate(items):
            if not isinstance(it, dict):
                raise SystemExit(f"❌ {flag}[{i}] is not an object")
            unknown = set(it) - keys
            if unknown:
                raise SystemExit(
                    f"❌ {flag}[{i}] tag={it.get('tag')!r}: key(s) {sorted(unknown)} are NOT read by "
                    f"this runner and would be SILENTLY IGNORED. Known keys: {sorted(keys)}. "
                    f"(reader_quant is a --kv-probe key; on --sweep it would leave the arm in fp16 "
                    f"while the rows carried a quantized label.)")
            missing = req - set(it)
            if missing:
                raise SystemExit(f"❌ {flag}[{i}]: missing required key(s) {sorted(missing)}")
            if flag == "--sweep" and "method" in it and it["method"] not in _SWEEP_METHODS:
                raise SystemExit(
                    f"❌ {flag}[{i}] method {it['method']!r}: only the single-model family "
                    f"{list(_SWEEP_METHODS)} can share one model load. 'ours' needs two models — run "
                    f"it as separate invocations (one per setting), which is what the composition "
                    f"jobs do.")
            if "batch" in it:
                lad = it["batch"] if isinstance(it["batch"], list) else [it["batch"]]
                if not lad or any(int(b) < 2 for b in lad):
                    raise SystemExit(f"❌ {flag}[{i}]: batch ladder {lad} contains a batch < 2 "
                                     f"(CLAUDE.md 0원칙: batch=1 never compiles the flash graph)")
    # {rank} in --out is substituted with this process's torchrun RANK. Needed only for the
    # tensor-parallel baseline (scripts/run_teacher_tp_true.slurm): torchrun sets RANK inside each
    # CHILD, so the launcher's shell cannot expand it, and every rank runs the identical program
    # on the identical inputs — they must not all append to one file. Ranks other than 0 write a
    # throwaway copy; rank 0's file is the result. A run without RANK is unaffected.
    if "{rank}" in str(getattr(args, "out", "")):
        args.out = args.out.replace("{rank}", os.environ.get("RANK", "0"))
    if getattr(args, "validate_only", False):
        print("✅ spec validation passed (--validate-only: nothing loaded, nothing run)")
        raise SystemExit(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="/work/hdd/myproject/anon/mtrag/reference.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--method", required=True, choices=["teacher", "ours",
                    "snapkv_frozen", "snapkv_fresh", "pyramidkv_frozen", "pyramidkv_fresh", "h2o",
                    "quant_int8", "quant_int4", "quant_int3", "quant_int2",
                    "specprefill", "specprefill_ours", "specprefill_generic", "specprefill_generic_ours",
                    "token_inject"])
    ap.add_argument("--keep", type=float, default=0.1,
                    help="SpecPrefill keep percentage (official default 0.1) for specprefill / specprefill_ours")
    ap.add_argument("--lm-no-accum", action="store_true",
                    help="ACCUM ours: the LM does NOT accumulate Q/A(/R) history — only sees instruction+current query "
                         "each turn (SLM reader unchanged). 2026-07-27: removes the few-shot-mimicry that suppressed reasoning.")
    ap.add_argument("--no-accum", action="store_true",
                    help="STANDARD setting: answer every question independently on a FRESH cache (no history, "
                         "no cross-turn KV reuse). The control for the accumulation axis — and the setting where "
                         "SpecPrefill's QUERY-DEPENDENT selection is per-question instead of anchored to Q1.")
    ap.add_argument("--spec-chunk", type=int, default=SP.DEFAULT_CHUNK_SIZE)
    ap.add_argument("--spec-pool", type=int, default=SP.DEFAULT_POOL_KERNEL)
    ap.add_argument("--spec-lah", type=int, default=SP.DEFAULT_LOOK_AHEAD)
    ap.add_argument("--slm-model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--lm-lora", default=None, help="PEFT/LoRA adapter for the LM (fusion branch) — eval a fusion-FT'd LM (scripts/fusion_distill_train.py)")
    ap.add_argument("--slm-lora", default=None, help="PEFT/LoRA adapter for the SLM (reader) — joint SLM+LM fusion-FT (--train-slm)")
    ap.add_argument("--gate", default=None, help="v3: a fusion_gate.pt (learned per-token gate) to REPLACE fixed λ")
    ap.add_argument("--lam", type=float, default=0.7)
    ap.add_argument("--lam-schedule", default=None,
                    help="turn-adaptive λ: comma list e.g. '0.3,0.85' (turn1=0.3, turn2+=0.85; last value repeats). Overrides --lam.")
    ap.add_argument("--ratio", type=float, default=0.8125)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true",
                    help="keep the COMPLETE conversations already in --out and decode only the rest; a "
                         "partially-written conversation is dropped and redone (its KV accumulates)")
    ap.add_argument("--max-conv", type=int, default=110)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--fullans", action="store_true", help="mtRAG full conversational-answer prompt/format (else terse reason_then_answer)")
    ap.add_argument("--bench", default=None, help="benchmark key in bench_config (locomo/qasper/...)")
    ap.add_argument("--instruction", default=None, help="qa_prompts constant name to use as the terse-path instruction (e.g. QA_FULL_CONTEXT_INSTRUCTION_REASON_QASPER_V2)")
    ap.add_argument("--ref-override", default=None,
                    help="use THIS ref instead of bench_config's, keeping every other canonical setting. "
                         "Recorded in the provenance so build_table refuses to mix it with the canonical slice.")
    ap.add_argument("--papers", default=None, help="comma-separated conversation_ids to restrict to (diagnosis)")
    ap.add_argument("--kv-probe", default=None, help=(
        "JSON list of arms for the RESIDENT-KV measurement pass (2026-08-30): each item "
        "{tag, method, ratio?, keep?, nbits?, reader_quant?} runs SEQUENTIALLY over the first "
        "--max-conv conversations with max_new=8 and records kv_gib = the MEASURED bytes of every "
        "key/value (and quantized+meta) tensor the committed cache holds at answer time. "
        "Output per arm: --out with {tag}. One model load for every arm."))
    ap.add_argument("--sweep", default=None, help=(
        "IN-PROCESS sweep (2026-08-28, user: '한번 로딩한다음에 여러 실험 최대한 많이'): JSON list of "
        "items run back-to-back on the ONE loaded model, so a retention curve costs one weight load "
        "instead of one per point. Item: {\"tag\": str, \"batch\": int|[ladder...], \"ratio\"?: float, "
        "\"keep\"?: float, \"method\"?: teacher|snapkv_frozen|h2o, \"papers\"?: \"id,id\"}. batch as a "
        "list = descend-until-fit (OOM caught in-process, next batch tried). Requires '{tag}' in --out; "
        "batched single-model methods only; no --resume. Provenance (ratio/keep/batch/method) is "
        "re-stamped per item."))
    ap.add_argument("--validate-only", action="store_true",
                    help="parse and validate the spec, load nothing, exit 0. Run this on the login "
                         "node before every sbatch: a malformed --sweep must cost seconds, not a GPU "
                         "allocation and a queue slot.")
    args = ap.parse_args()
    validate_specs(args)      # BEFORE any model load — see validate_specs' docstring
    global FULLANS, INSTR_OVERRIDE, PROV, PRESS_METHOD, QUANT_NBITS, LAM_SCHEDULE, REASON_HIST, RATIO_GLOBAL, BATCH_SIZE
    global PRESS_QUANT_NBITS
    LAM_SCHEDULE = [float(x) for x in re.split(r"[,+]", args.lam_schedule)] if args.lam_schedule else None  # '+' survives sbatch --export
    PRESS_METHOD = press_for_method(args.method)             # snapkv / pyramidkv / expected_attention(=h2o); unused for quant/teacher
    RATIO_GLOBAL = args.ratio                                # the batched compress path reads this; keep it in step with --ratio
    if args.method.startswith("quant"):
        # HQQ nbits for the persistent KV cache. int3 = 18.75% mem = MEMORY-MATCHED to ours + snapkv/h2o (keep-18.75%);
        # int4 (25%) and int8 (50%) use MORE memory (higher-precision references, NOT the fair equal-memory point).
        QUANT_NBITS = {"quant_int8": 8, "quant_int4": 4, "quant_int3": 3, "quant_int2": 2}[args.method]
    FULLANS = args.fullans
    # ★ FOOTGUN GUARD (2026-07-28): if --bench is given but --instruction is omitted, auto-apply the benchmark's
    # CANONICAL instruction from bench_config instead of silently falling back to the generic terse prompt. This
    # is what caused the QASPER prompt-mix — `--bench qasper` without `--instruction` ran with the generic
    # QA_FULL_CONTEXT_INSTRUCTION_REASON (f92f6e17) instead of the canonical QA_REASON_V3 (c1a237e0), and because
    # the fingerprint was computed from the WRONG prompt, build_table could not catch it against another f92f6e17 run.
    if not args.instruction and args.bench and not args.fullans:
        args.instruction = BC.get(args.bench)["instruction"]
        print(f"★ --bench {args.bench}: auto-applying canonical instruction {args.instruction} from bench_config "
              f"(pass --instruction to override).", flush=True)
    # ★ FOOTGUN GUARD (2026-07-28): --bench must also fix the DATASET (ref file), not just the prompt. Otherwise the
    # mtRAG default --ref leaks into a qasper/locomo run (observed: --bench qasper ran on mtRAG convs, zero overlap).
    # bench_config is the source of truth for which conversations a benchmark uses.
    if args.bench:
        _bref = BC.get(args.bench).get("ref")
        if args.ref_override:
            # A DIFFERENT SLICE of the same benchmark (e.g. LoCoMo at 100 questions/conversation instead of
            # the canonical 30) is a DIFFERENT comparison set, not a variant of the canonical one. It is
            # recorded in the fingerprint below so build_table.py refuses to mix the two Ns.
            print(f"★ --bench {args.bench}: REF-OVERRIDE {args.ref_override} (canonical was {_bref}).", flush=True)
            args.ref = args.ref_override
        elif _bref and _bref != args.ref:
            # ★ FOOTGUN GUARD (2026-08-13): a DELIBERATE --ref must never be silently discarded.
            # The line below used to print and carry on, which is right when the launcher's own default
            # leaks in (run_mtrag_accum.slurm defaults REF to mtRAG's reference.jsonl for every benchmark)
            # but WRONG when someone asked for a different slice on purpose. Two reader-alone runs
            # (2939260/2939261) were launched with REF=locomo_ep10_60.jsonl, silently ran on the canonical
            # LoCoMo-30 ref instead, and produced 300-turn logs that are not comparable to the LoCoMo-10
            # arms they were built to explain. The ★ line was printed and was not enough.
            # So: the launcher default is tolerated and replaced; anything else ABORTS and names the fix.
            if os.path.basename(args.ref) not in _TOLERATED_DEFAULT_REFS:
                sys.stderr.write(
                    f"[ABORT] --bench {args.bench} pins ref {_bref}, but --ref {args.ref} was passed "
                    f"deliberately.\n        A different ref is a DIFFERENT comparison set, not a variant: "
                    f"LoCoMo-30 and LoCoMo-10\n        have different question mixes and different "
                    f"accumulation depths.\n        If you mean it, pass --ref-override / REF_OVERRIDE= — it "
                    f"is recorded in the fingerprint so\n        build_table.py refuses to mix the two Ns.\n")
                raise SystemExit(3)
            print(f"★ --bench {args.bench}: using bench_config ref {_bref} (replacing the launcher default "
                  f"{os.path.basename(args.ref)}).", flush=True)
            args.ref = _bref
    # ★ FOOTGUN GUARD (2026-08-12): --bench must also fix MAX_NEW and REASON_HIST, not just the prompt and the
    # ref. It did not, so a launcher default (MAXNEW=128, and REASON_HIST's env default 'gen') silently
    # overrode the canonical 200/'ref' and produced a SECOND, non-comparable family of LoCoMo logs — the
    # F3/F4 format cells and a v9 floor all landed in it, and the F3-vs-F1 comparison built from them was
    # invalid. bench_config is the source of truth; a harness that only RECORDS its settings instead of
    # APPLYING them lets the drift through.
    if args.bench:
        _bc = BC.get(args.bench)
        if args.max_new != _bc["max_new"]:
            if os.environ.get("MAX_NEW_OVERRIDE"):
                print(f"★ --bench {args.bench}: max_new {args.max_new} kept (MAX_NEW_OVERRIDE set; canonical "
                      f"is {_bc['max_new']}) — this run is NOT comparable to canonical ones.", flush=True)
            else:
                print(f"★ --bench {args.bench}: max_new {args.max_new} -> canonical {_bc['max_new']} from "
                      f"bench_config (set MAX_NEW_OVERRIDE=1 to keep a non-canonical value).", flush=True)
                args.max_new = _bc["max_new"]
        _brh = _bc.get("reason_hist")
        if _brh and REASON_HIST != _brh:
            if os.environ.get("REASON_HIST_OVERRIDE"):
                print(f"★ --bench {args.bench}: reason_hist '{REASON_HIST}' kept (override set; canonical is "
                      f"'{_brh}') — NOT comparable to canonical runs.", flush=True)
            else:
                print(f"★ --bench {args.bench}: reason_hist '{REASON_HIST}' -> canonical '{_brh}'.", flush=True)
                REASON_HIST = _brh
    if args.instruction:
        INSTR_OVERRIDE = getattr(qa_prompts, args.instruction)
    _pname = args.instruction or ("QA_MTRAG_INSTRUCTION" if args.fullans else "QA_FULL_CONTEXT_INSTRUCTION_REASON")
    if not args.bench:
        print('★★★ WARNING: no --bench -> provenance INCOMPLETE -> build_table.py will REFUSE this run. Pass --bench <key>.', flush=True)
    if args.bench:
        PROV = BC.run_fingerprint(args.bench, args.model, args.method, _pname, INSTRUCTION(), args.max_new,
                                  args.ratio, REASON_FIX, REASON_HIST, slm_model=getattr(args,"slm_model",None), lam=getattr(args,"lam",None))
        if args.ref_override:
            PROV = dict(PROV); PROV["ref_override"] = os.path.basename(args.ref_override)
    else:
        PROV = RF.provenance(_pname, INSTRUCTION(), "manual-greedy", args.max_new, args.model, REASON_FIX,
                             method=args.method, slm_model=getattr(args,"slm_model",None), lam=getattr(args,"lam",None), ratio=args.ratio)
    # ★★ EXPERIMENT AXES — recorded on EVERY row so variants can NEVER be silently mixed in a table.
    # (accumulation / reasoning / specprefill / fusion are independent axes; see SPECPREFILL_PLAN.md registry.)
    if PROV is not None:
        PROV = dict(PROV)
        PROV["axis_accumulation"] = (not args.no_accum)          # True = multi-turn KV reuse; False = standard
        PROV["axis_lm_accum"] = (not args.lm_no_accum)           # 2026-07-27: False = LM sees only current query (no Q/A history)
        PROV["axis_lam_schedule"] = args.lam_schedule            # 2026-07-28: per-turn λ list (None = fixed --lam)
        PROV["axis_lam_answer"] = (LAM_ANSWER if LAM_ANSWER >= 0 else None)  # 2026-07-29: within-answer λ drop at 'Final Answer:'
        PROV["axis_reasoning"] = bool(REASON_FIX)                # reason_then_answer enforced
        # ★ decide the DECODE PATH here, before provenance is written, so provenance can record what will
        # actually run rather than what was requested. FORCE_BATCHED=1 runs the BATCHED CLASS at B=1: a
        # DIAGNOSTIC that separates an implementation difference from the batch-shape effect. Never a run
        # mode — B=1 does not compile the flash-attention graph.
        global EFFECTIVE_BATCH, FORCED_BATCHED
        FORCED_BATCHED = os.environ.get("FORCE_BATCHED", "0") == "1"
        # specprefill joined the batchable set 2026-08-24 (BatchedStateful.specprefill_append):
        # it was the last arm forced onto the sequential path, which barred it from every batched
        # timing table. Only the QUERY-CONDITIONED single-model variant is batched (SP_SINGLE);
        # the fusion variants keep the sequential path.
        # quant_int* joined the batchable set 2026-08-31 (BatchedQuantStateful): it was the last
        # ACCURACY arm still forced onto the sequential path, which (a) made a quant THROUGHPUT row
        # impossible — throughput_eval refuses batch<2 and a batch-1 wall is an artefact — and
        # (b) cost ~9 s/example where a batched arm pays a fraction. 0원칙: fix the path, never run
        # at batch 1 and "fix it later".
        _BATCHABLE_SINGLE = ("teacher", "snapkv_frozen", "pyramidkv_frozen", "h2o", "specprefill",
                             "quant_int8", "quant_int4", "quant_int3", "quant_int2")
        # READER_QUANT (ours + quantized reader cache) joined this list on 2026-08-31: BatchedFusion
        # now builds its reader branch as BatchedQuantStateful and the turn rollback it already used
        # (state()/restore()) is that class's interface, so the axis batches like any other. It had
        # been excluded purely because the flag was never passed down — and that exclusion silently
        # cost ~4x on every composition run (0원칙) and kept the arm out of every timing table.
        _bt = ((BATCH_SIZE > 1 or FORCED_BATCHED) and args.method in (("ours",) + _BATCHABLE_SINGLE)
               and not args.no_accum and not LAM_SCHEDULE and not TOKLOG and not MEMO_MODE)
        if READER_QUANT_NBITS and _bt and int(os.environ.get("INGEST_REPEATS", "0") or 0):
            raise SystemExit("❌ READER_QUANT with INGEST_REPEATS: the TTFT probe rolls back with "
                             "crop(), which a quantized cache cannot do.")
        EFFECTIVE_BATCH = (1 if FORCED_BATCHED else BATCH_SIZE) if _bt else 1
        if PRESS_QUANT_NBITS is not None and not _bt:
            # composition lives only on the batched decoder; the sequential path would ignore the
            # request and emit rows labelled "+int8" that stored fp KV (the exact mislabel class the
            # nbits re-stamping guard exists for)
            raise SystemExit("❌ PRESS_QUANT set but this configuration is NOT batchable "
                             f"(method={args.method}, BATCH_SIZE={BATCH_SIZE}). The composition arm "
                             "exists only on the batched path — raise BATCH_SIZE or drop PRESS_QUANT.")
        PROV["axis_fusion"] = args.method in ({"ours"} | SP_FUSION)
        PROV["axis_reader_quant_nbits"] = READER_QUANT_NBITS
        PROV["axis_quant_nbits"] = QUANT_NBITS        # HQQ bits of the arm's OWN persistent cache
        # WHICH quantized-cache implementation ran. 'append_only' = each token quantized exactly
        # once (default, correct). 'hf_requant' = HF's re-quantize-the-whole-cache-on-every-flush,
        # kept only to measure that artefact's cost. A quant row without this axis predates the
        # distinction and was produced on hf_requant.
        PROV["axis_quant_cache"] = (os.environ.get("QUANT_CACHE", "append_only")
                                    if (QUANT_NBITS or READER_QUANT_NBITS or PRESS_QUANT_NBITS)
                                    else None)
        # BYTE-WALKER VERSION (2026-08-31). _kv_gib walked only `keys`/`values` and HF's
        # `_quantized_*`, never AppendOnlyHQQLayer's `_blocks_k`/`_blocks_v`, so every append-only
        # quant arm reported its fp RESIDUAL as if it were the cache (hotq int8: 0.138 GiB for a
        # 12-row cache holding ~7.7 GiB). Accuracy is untouched — the walker only measures — but any
        # kv_gib_* column is only trustworthy at v2. A row WITHOUT this axis was written by v1 and
        # its byte columns must not be compared against a v2 row.
        PROV["axis_kv_bytes_walker"] = "v2_counts_append_only_blocks"
        PROV["axis_press_quant_nbits"] = PRESS_QUANT_NBITS   # composition: press/spec arm + quantized KV
        PROV["axis_specprefill"] = args.method in SP_ALL
        PROV["axis_specprefill_generic"] = args.method in SP_GENERIC
        PROV["axis_specprefill_keep"] = (args.keep if PROV["axis_specprefill"] else None)
        PROV["axis_reason_hist"] = (REASON_HIST if not args.no_accum else None)
        PROV["axis_memo"] = MEMO_MODE                             # 2026-08-09: session-index memo committed to both branches
        PROV["axis_fuse_restrict"] = FUSE_RESTRICT                # 2026-08-12: choice restricted to the two branches' top-1
        # ★ batch size is a COMPARISON AXIS, not a throughput knob (2026-08-12). Three identical copies of
        # one conversation in one batch agree with each other 10/10 but match the true batch-1 run 0/10,
        # with no padding anywhere — the batch SHAPE changes the generated text. Recorded here so a table
        # can never merge two different batch sizes, exactly as it cannot merge two decoding modes.
        from src.batched_stateful import PREFILL_CHUNK
        # ★ the EFFECTIVE batch, not the requested one (2026-08-13). snapkv_fresh and h2o were submitted
        # with BATCH_SIZE=2, silently demoted to the sequential path, and still recorded "2" — so a table
        # built from provenance alone could not see that they ran on a DIFFERENT DECODE PATH than the
        # batched fusion arms they were compared against. The sequential and batched paths are NOT
        # interchangeable (bisect: at the same B=1 they generate 263 vs 148 chars and score 0.2738 vs
        # 0.2937), so this field has to say what actually ran.
        PROV["axis_batch_size"] = EFFECTIVE_BATCH if EFFECTIVE_BATCH is not None else BATCH_SIZE
        PROV["merge_lora"] = MERGE_LORA
        # ★ YaRN (2026-09-11): the extended context axis (hotpotqa_st200..320_full) runs both models
        # with ROPE_YARN_FACTOR=4 (src/models.py: rope_scaling yarn over original 32768). A run past
        # the native window without it would be silently wrong, so the factor is stamped here — and
        # a native-window run stamps None, which is how the two halves of that axis are told apart.
        PROV["axis_rope_yarn_factor"] = (float(os.environ["ROPE_YARN_FACTOR"])
                                         if os.environ.get("ROPE_YARN_FACTOR") else None)
        PROV["ctx_cap_tokens"] = CTX_CAP or None
        PROV["axis_batch_requested"] = BATCH_SIZE
        PROV["axis_decode_path"] = "batched" if (EFFECTIVE_BATCH or 1) > 1 or FORCED_BATCHED else "sequential"
        # ★ the NODE, because a timing is only comparable if the arms got comparable hardware. In the
        # 2026-08-12 episode wave the two arms that landed on the same node ran at 4.1 and 6.2 turns/min
        # while the same class of arm alone on a node ran at 9.1 — co-location roughly halves throughput.
        # Scores are unaffected (same computation, just slower); speed columns are not.
        PROV["slurm_node"] = os.environ.get("SLURMD_NODENAME") or os.environ.get("SLURM_NODELIST")
        PROV["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
        PROV["axis_prefill_chunk"] = (PREFILL_CHUNK if BATCH_SIZE > 1 else None)
        # ★ 2026-08-12: record WHICH ADAPTERS were decoded. Without this a log cannot say what configuration
        # produced it, the results board has to be hand-maintained, and a VOID (gold-supervised) adapter can
        # be pulled back into a table — which happened.
        PROV["slm_lora"] = os.path.basename(str(getattr(args, "slm_lora", "") or "")) or None
        PROV["lm_lora"] = os.path.basename(str(getattr(args, "lm_lora", "") or "")) or None
        PROV["fusion_par"] = os.environ.get("FUSION_PAR", "0") == "1"
        PROV["decode_compile"] = os.environ.get("DECODE_COMPILE", "0") == "1"
        PROV["static_decode"] = os.environ.get("STATIC_DECODE", "0") == "1"
    cache = os.environ.get("HF_HOME", "/work/hdd/myproject/anon/hf")
    _lm_dev = os.environ.get("FUSION_LM_DEV") or "cuda:0"   # see FUSION_LM_DEV in run_ours_batched
    model, tok = load_causal_lm(args.model, device_map=_lm_dev, cache_dir=cache); model.eval()
    if getattr(args, "lm_lora", None):
        from src.lora_load import load_lora_stack
        model = load_lora_stack(model, args.lm_lora, label="LM-LoRA (fusion LM branch)",
                                merge_final=MERGE_LORA); model.eval()
    # DECODE_COMPILE=1: torch.compile the step forward. Measured 2026-08-19: the un-compiled decode
    # loop runs at 97-149 ms/step against a ~22 ms bandwidth roofline — CPU-side per-layer dispatch
    # dominates, so kernels alone (flash) cannot fix it. Timing runs only; greedy paths may flip a
    # few tokens vs eager, so accuracy tables must not mix compiled and un-compiled arms.
    DECODE_COMPILE = os.environ.get("DECODE_COMPILE", "0") == "1"
    _CMODE = os.environ.get("DECODE_COMPILE_MODE", "default")  # reduce-overhead uses cudagraphs, which
    if DECODE_COMPILE:                                         # thrashes on a growing DynamicCache
        try:
            model = torch.compile(model, mode=_CMODE, dynamic=True)
            print(f"[Info] DECODE_COMPILE: LM torch.compile({_CMODE}) ON", flush=True)
        except Exception as e:
            print(f"[Warn] DECODE_COMPILE failed on LM ({e}); continuing eager", flush=True)
    dev = next(model.parameters()).device
    stop_ids = set(x for x in [tok.eos_token_id] if x is not None)
    if CHAT_WRAP:                       # chat families end turns with their turn-end token, not EOS
        _ie = _CW.end_token_id(tok)
        if _ie is not None:
            stop_ids.add(_ie)
    slm = slm_tok = None
    needs_slm = args.method in ({"ours", "token_inject"} | SP_ALL)   # 3B = fusion reader / speculator / injector
    if needs_slm:
        # the speculator/injector must be able to return attention weights on its 1-token steps → sdpa base (eager per-read)
        if args.method in SP_ALL or args.method == "token_inject":
            os.environ["ATTN_IMPL"] = "sdpa"
        slm, slm_tok = load_causal_lm(args.slm_model, device_map="cuda:0", cache_dir=cache); slm.eval()
        if args.method == "token_inject" and slm_tok.get_vocab() != tok.get_vocab():
            raise SystemExit("❌ token_inject: SLM and LM must share a vocabulary (ids injected across models)")
        os.environ.pop("ATTN_IMPL", None)
        if getattr(args, "slm_lora", None):
            from src.lora_load import load_lora_stack
            slm = load_lora_stack(slm, args.slm_lora, label="SLM-LoRA (fusion reader branch)",
                                  merge_final=MERGE_LORA); slm.eval()
        if DECODE_COMPILE and args.method == "ours":
            try:
                slm = torch.compile(slm, mode=_CMODE, dynamic=True)
                print(f"[Info] DECODE_COMPILE: SLM torch.compile({_CMODE}) ON", flush=True)
            except Exception as e:
                print(f"[Warn] DECODE_COMPILE failed on SLM ({e}); continuing eager", flush=True)
    if args.method in SP_ALL:
        global SPEC_MODEL, SPEC_KEEP, SPEC_CHUNK, SPEC_POOL, SPEC_LAH, SPEC_USE_GENERIC
        SPEC_USE_GENERIC = args.method in SP_GENERIC
        SPEC_MODEL, SPEC_KEEP = slm, args.keep
        SPEC_CHUNK, SPEC_POOL, SPEC_LAH = args.spec_chunk, args.spec_pool, args.spec_lah
        if slm_tok.get_vocab() != tok.get_vocab():
            raise SystemExit("❌ speculator and main model do not share a vocabulary — ids not transferable")
        print(f"[Info] SpecPrefill: keep={SPEC_KEEP} chunk={SPEC_CHUNK} pool={SPEC_POOL} lah={SPEC_LAH} "
              f"speculator={args.slm_model}", flush=True)
    if getattr(args, "gate", None):
        global FUSION_GATE, GATE_TOP_K
        _g = torch.load(args.gate, map_location=dev)
        GATE_TOP_K = int(_g.get("gate_top_k", 10))
        FUSION_GATE = TokenAgnosticGateMLP(input_dim=2*GATE_TOP_K, hidden_dim=int(_g.get("gate_hidden",64)), mlp_layers=1).to(dev)
        FUSION_GATE.load_state_dict(_g["state_dict"]); FUSION_GATE.eval()
        print(f"[Info] learned fusion GATE loaded: {args.gate} (top_k={GATE_TOP_K}) — replaces fixed λ", flush=True)
    rows = [json.loads(l) for l in open(args.ref) if l.strip()]
    by = defaultdict(list)
    for r in rows: by[r["conversation_id"]].append(r)
    if args.papers:
        keep = set(re.split(r"[,+]", args.papers))   # '+' delimiter is safe through sbatch --export (comma = its separator)
        by = {k: v for k, v in by.items() if k in keep}
    # ★ MEMORY BASELINE (2026-08-12): peak_reserved on a turn is weights + KV, and weights dominate
    # (32B+7B is ~78 GiB of weights against the teacher's ~64), so the raw peak says nothing about the KV
    # story. Record the reserved figure AFTER the models are loaded and BEFORE any decoding, so KV can be
    # reported as a MEASURED difference (peak - baseline) rather than an estimate from parameter counts.
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    _base_a, _base_r = _mem_gib()
    if PROV is not None and _base_r is not None:
        PROV["mem_baseline_alloc_gib"], PROV["mem_baseline_reserved_gib"] = _base_a, _base_r
    print(f"[Mem] after-load baseline: alloc {_base_a} GiB / reserved {_base_r} GiB (weights only, no KV)",
          flush=True)
    if BATCH_SIZE < 2 and FORCED_BATCHED and os.environ.get("FKV_DECODE", "0") == "1":
        print("★ BATCH_SIZE=1 on the FKV batched class (FORCE_BATCHED=1): same code path and graph as B>=2, so\n"
              "   the timing is a measurement at this arm's largest fitting batch (user, 2026-09-12).", flush=True)
    elif BATCH_SIZE < 2:
        print("🚫 BATCH_SIZE=1. At batch 1 the flash-attention graph does not compile, so EVERY timing this\n"
              "   run records (ans_s / ingest_s / peak_*_gib) is an artefact and is flagged\n"
              "   `timing_invalid_batch1` on every row. Accuracy is unaffected (greedy decode per turn), so\n"
              "   the run proceeds — but do not publish speed or memory from it. Batched decoding across\n"
              "   conversations is the fix; see BATCH_SIZE in this file.", flush=True)
    # ★ RESUME (2026-08-12). `open(out, "w")` truncated, so any job that died mid-run lost everything it
    # had decoded. Eleven jobs died at once on a full home quota, throwing away up to 540 of 600 turns each.
    # A conversation is the resume unit because the KV accumulates across its turns: a conversation whose
    # rows are all present is skipped, a partially-written one is redone from its first turn.
    done_convs = set()
    if args.resume and os.path.exists(args.out):
        have = collections.Counter()
        for line in open(args.out):
            try:
                r = json.loads(line)
            except Exception:
                continue
            have[r.get("conv")] += 1
        want = {cid: sum(1 for t in tasks
                         if ((t.get("Answerability") or [""])[0] or "").upper() == "ANSWERABLE"
                         and last_user(t.get("input")))
                for cid, tasks in by.items()}
        done_convs = {c for c, n in have.items() if n >= want.get(c, 10 ** 9)}
        print(f"★ --resume: {len(done_convs)} of {len(by)} conversations already complete in {args.out} "
              f"({sum(have[c] for c in done_convs)} turns kept); the rest are redone from their first turn.",
              flush=True)
    if done_convs:
        # rewrite the file with ONLY the complete conversations, then append — a partially-written
        # conversation's rows must not survive, or the run would mix two different accumulations.
        keep = [l for l in open(args.out)
                if (json.loads(l).get("conv") if l.strip() else None) in done_convs]
        with open(args.out, "w") as fh:
            fh.writelines(keep)
        # ★ SLICE FIRST, THEN DROP THE DONE ONES (2026-09-05, job 3085983). Filtering `by` before the
        # downstream `[:args.max_conv]` slices let a resumed run REPLACE its finished conversations
        # with the next ones from the pool: a LooGLE-48 arm whose OOM'd rung had completed the four
        # longest conversations resumed onto a 48 that excluded them and included four shorter ones,
        # i.e. a different workload. The intended set is the first max_conv of the pool; a resume
        # runs the not-yet-done members of THAT set and nothing else.
        by = {c: t for c, t in list(by.items())[:args.max_conv] if c not in done_convs}
    fout = open(args.out, "a" if done_convs else "w")
    done = sum(1 for _ in open(args.out)) if done_convs else 0
    if args.kv_probe:
        import traceback as _tb
        import gc as _gc          # main() has a LOCAL `import gc` further down, which makes the
                                  # bare name unbound HERE (the wave-killer of probes 1 and 2)

        def _probe_one(it, tag, m, out_path):
            with open(out_path, "w") as pf:
                pn = 0
                for cid, tasks in list(by.items())[:args.max_conv]:
                    tasks.sort(key=lambda r: int(r["turn"]))
                    if m in ({"ours"} | SP_FUSION):
                        recs = run_ours(slm, slm_tok, model, tok, dev, args.lam, tasks, 8, stop_ids,
                                        specprefill=(m in SP_FUSION), lm_no_accum=args.lm_no_accum)
                    elif it.get("floor"):     # the floor arm runs the SLM weights as a solo teacher
                        recs = run_single(slm, slm_tok, dev, args.ratio, tasks, 8, stop_ids, "teacher")
                    else:
                        # kvpress refuses a PEFT-wrapped model (and the accuracy presses ran on the
                        # bare 32B anyway): unwrap for the single-model arms. KV shapes unchanged.
                        _bm = model.get_base_model() if hasattr(model, "get_base_model") else model
                        recs = run_single(_bm, tok, dev, args.ratio, tasks, 8, stop_ids, m)
                    for r in recs:
                        r["_kvprobe"] = dict(tag=tag, method=m, ratio=args.ratio,
                                             nbits=QUANT_NBITS, reader_quant=it.get("reader_quant"),
                                             keep=it.get("keep"))
                        pf.write(json.dumps(r) + "\n"); pn += 1
            print(f"=== KVPROBE {tag} method={m} -> {out_path} ({pn} rows) ===", flush=True)

        for it in json.loads(args.kv_probe):
            tag = it["tag"]; m = it["method"]
            args.method = m
            if "ratio" in it: args.ratio = float(it["ratio"]); RATIO_GLOBAL = args.ratio
            PRESS_METHOD = press_for_method(m)
            QUANT_NBITS = {"quant_int8": 8, "quant_int4": 4, "quant_int3": 3, "quant_int2": 2}.get(m)
            globals()["READER_QUANT_NBITS"] = it.get("reader_quant")
            if m in SP_ALL:
                globals()["SPEC_MODEL"] = slm; globals()["SPEC_KEEP"] = float(it.get("keep", args.keep))
            try:
                _probe_one(it, tag, m, args.out.replace("{tag}", tag))
            except Exception:
                print(f"=== KVPROBE FAIL {tag} ===", flush=True); _tb.print_exc()
            _gc.collect(); torch.cuda.empty_cache()
        print("[Done] kv-probe complete", flush=True)
        return

    # ★ BATCHED PATH: BATCH_SIZE>1 runs B conversations at once. Restricted to plain `ours` — the axes that
    # are single-conversation experiments (specprefill / memo / lam-schedule / toklog) keep the
    # sequential path so their semantics are untouched.
    _force_b, _batchable = FORCED_BATCHED, (EFFECTIVE_BATCH or 1) > 1 or FORCED_BATCHED
    if BATCH_SIZE > 1 and not _batchable:
        print(f"★ BATCH_SIZE={BATCH_SIZE} requested but this configuration is not batchable "
              f"(method={args.method}); running sequentially at batch 1. "
              f"★ THIS IS A DIFFERENT DECODE PATH from a batched arm — they may not share a table.",
              flush=True)
    if _batchable:
        if _force_b and BATCH_SIZE <= 1:
            print("★ FORCE_BATCHED=1: running the BATCHED CLASS at B=1" + (" on the FKV path: a measurement at this arm's largest fitting batch (2026-09-12)." if os.environ.get("FKV_DECODE", "0") == "1" else " (diagnostic only)."), flush=True)

        def _batched_pass(by_d, fh, skip_convs=()):
            """One full pass over by_d at the CURRENT globals (BATCH_SIZE/RATIO_GLOBAL/SPEC_KEEP/
            PRESS_METHOD/PROV), writing rows to fh. Extracted so --sweep can run it repeatedly on
            the one loaded model.

            skip_convs = conversations already written by an earlier rung of the batch ladder.
            ★ WHY (2026-08-31, user): the ladder used to DELETE the partial file and restart the arm
            from zero at the next rung. Job 3057915 finished 588 of 600 examples and the OOM handler
            erased all of them. That is only defensible if batch were a treatment whose value had to
            be constant across an arm — and it is not (see CANONICAL_RUNS §2e-note: arms are compared
            across different batches everywhere, because batch cannot cause a systematic shift). So
            the first 480 examples at B=12 and the rest at B=8 is a perfectly good arm, and throwing
            away finished work to keep one number constant is pure waste."""
            done_n = 0
            items = [(cid, t) for cid, t in list(by_d.items())[:args.max_conv]
                     if cid not in skip_convs]
            # ★ LENGTH-BUCKET the batch. Every appended chunk is LEFT-PADDED to the batch's longest, so a
            # conversation batched with a much longer one carries thousands of pad slots: wasted compute, and
            # empirically the row with the MOST padding is the one whose generations diverge most from the
            # sequential path (2026-08-12 gate: 5/10 turns on the shortest context vs 1/10 on the longest).
            # Sorting by context length puts similar lengths together, which minimises both. The key is
            # (length, conversation id) so the grouping is deterministic and the run reproduces.
            items.sort(key=lambda kv: (sum(len(c.get("text", "")) for t in kv[1]
                                           for c in (t.get("contexts") or [])), kv[0]))
            # LONGEST-FIRST (2026-08-29): with descend-on-OOM at full-N, ascending order puts the
            # worst bucket LAST — an infeasible batch burns a ~full pass before failing (mufsnap
            # lost 3 arms x ~1h each). Longest-first fails in the first minutes instead.
            # LONGEST_FIRST=1 extends this to PLAIN (non-sweep) runs (2026-08-31). It was a sweep-only
            # global, so a plain full-N run at an unproven batch processed shortest-first and only met
            # its worst bucket at the very end: job 3057915 reached batch 48 of 50 -- 98% of the work --
            # before OOMing, and the ladder then deleted the partial. Fail in the first two minutes
            # instead. Ordering does not change any per-example result; it only changes which batch
            # is met first (and batches are re-formed from the same deterministic length sort).
            if globals().get("_SWEEP_LONGEST_FIRST") or os.environ.get("LONGEST_FIRST") == "1":
                items.reverse()
            for b in range(0, len(items), BATCH_SIZE):
                chunk = items[b: b + BATCH_SIZE]
                for cid, tasks in chunk:
                    tasks.sort(key=lambda r: int(r["turn"]))
                if args.method in _BATCHABLE_SINGLE:
                    _isq = args.method.startswith("quant_")
                    recs = run_single_batched(model, tok, dev, [t for _, t in chunk], args.max_new, stop_ids,
                                              compress=(args.method not in ("teacher", "specprefill")
                                                        and not _isq),
                                              spec=(args.method == "specprefill"),
                                              quant_nbits=(QUANT_NBITS if _isq else PRESS_QUANT_NBITS))
                else:
                    recs = run_ours_batched(slm, slm_tok, model, tok, dev, args.lam,
                                            [t for _, t in chunk], args.max_new, stop_ids,
                                            lm_no_accum=args.lm_no_accum)
                for (cid, _), rs in zip(chunk, recs):
                    for r in rs:
                        fh.write(json.dumps(r) + "\n"); done_n += 1
                fh.flush()
                print(f"[batch {b//BATCH_SIZE}] {len(chunk)} conversations, answerable so far={done_n}",
                      flush=True)
            return done_n, len(items)

        if args.sweep:
            fout.close()
            if os.path.exists(args.out) and os.path.getsize(args.out) == 0:
                # NEVER os.remove a result path. This one is provably empty, but the rule is
                # categorical after 2026-08-31: nothing in this repo deletes a file under
                # results/. Move it aside instead. (Job 3057915 lost 588 finished examples to an
                # os.remove on an OOM path. Provenance cannot protect a file from deletion --
                # it is written INSIDE the file.)
                os.replace(args.out, os.path.join("results/quarantine",
                           os.path.basename(args.out) + ".empty"))
            if "{tag}" not in args.out:
                raise SystemExit("❌ --sweep requires '{tag}' in --out")
            if args.resume:
                raise SystemExit("❌ --sweep does not support --resume (each item overwrites its file)")
            for it in json.loads(args.sweep):
                tag = it["tag"]
                out_path = args.out.replace("{tag}", tag)
                if "method" in it:
                    _SWEEPABLE = ("teacher", "snapkv_frozen", "h2o",
                                  "quant_int8", "quant_int4", "quant_int3", "quant_int2")
                    if it["method"] not in _SWEEPABLE:
                        raise SystemExit(f"❌ sweep method {it['method']!r}: only the single-model "
                                         "family (teacher/snapkv_frozen/h2o/quant_int*) can share one load")
                    args.method = it["method"]
                    PRESS_METHOD = press_for_method(args.method)
                    # ★ quant became sweepable 2026-08-31 ONLY because nbits is re-stamped here. The
                    # earlier refusal existed because a sweep that changed `method` without changing
                    # QUANT_NBITS would write rows LABELLED int4 that were produced at int8.
                    QUANT_NBITS = {"quant_int8": 8, "quant_int4": 4,
                                   "quant_int3": 3, "quant_int2": 2}.get(args.method)
                if "ratio" in it:
                    args.ratio = float(it["ratio"]); RATIO_GLOBAL = args.ratio
                if "keep" in it:
                    SPEC_KEEP = float(it["keep"])
                if "press_quant" in it:
                    PRESS_QUANT_NBITS = (int(it["press_quant"]) if it["press_quant"] else None)
                by_i = by
                if it.get("papers"):
                    _keep = set(str(it["papers"]).split(","))
                    by_i = {k: v for k, v in by.items() if k in _keep}
                ladder = it["batch"] if isinstance(it["batch"], list) else [it["batch"]]
                # DEFAULT longest_first whenever a descend ladder exists (2026-08-30): ascending
                # bucket order let every infeasible batch burn a near-full pass before OOMing —
                # the mufsnap/mufexp waves lost ~10 GPU-hours to exactly this. Explicit
                # longest_first: false restores ascending order.
                globals()["_SWEEP_LONGEST_FIRST"] = bool(
                    it.get("longest_first", len(ladder) > 1))
                ok = False
                for _B in ladder:
                    BATCH_SIZE = int(_B); EFFECTIVE_BATCH = BATCH_SIZE
                    # papers_pool: 'the B longest at batch B' convention under a LADDER — the
                    # conversation subset follows the batch being attempted (first B of the pool).
                    # A fixed `papers` list with a ladder was the 3043852 crash class.
                    if it.get("papers_pool"):
                        _pool = str(it["papers_pool"]).split(",")
                        by_i = {k: by[k] for k in _pool[:BATCH_SIZE] if k in by}
                    PROV = dict(PROV)
                    PROV["axis_batch_size"] = BATCH_SIZE
                    PROV["ratio"] = args.ratio
                    PROV["method"] = args.method
                    PROV["axis_quant_nbits"] = QUANT_NBITS
                    PROV["axis_press_quant_nbits"] = PRESS_QUANT_NBITS
                    # ★ axis_quant_cache is decided in main() from the ENV, before the sweep exists,
                    # so a press+quant arm whose nbits comes from a SWEEP ITEM (not from PRESS_QUANT
                    # in the environment) was stamped cache=None. Found 2026-08-31 on job 3058334's
                    # first arm: the row said press_quant=8 and cache=None. That is not cosmetic —
                    # the Ledger reads a missing axis_quant_cache as "HF re-quantizing cache" and
                    # would mark a perfectly good append-only row VOID. Re-stamp it here alongside
                    # the nbits it belongs to.
                    if QUANT_NBITS or READER_QUANT_NBITS or PRESS_QUANT_NBITS:
                        PROV["axis_quant_cache"] = os.environ.get("QUANT_CACHE", "append_only")
                    if args.method == "specprefill":
                        PROV["spec_keep"] = SPEC_KEEP
                    print(f"=== SWEEP {tag} method={args.method} ratio={args.ratio} "
                          f"keep={SPEC_KEEP} B={BATCH_SIZE} -> {out_path} ===", flush=True)
                    # RESUME, never restart: keep whatever earlier rungs finished.
                    _have = set()
                    if os.path.exists(out_path):
                        with open(out_path) as _f:
                            for _l in _f:
                                if _l.strip():
                                    try: _have.add(json.loads(_l)["conv"])
                                    except Exception: pass
                    if _have:
                        print(f"=== SWEEP RESUME {tag} at B={BATCH_SIZE}: "
                              f"{len(_have)} conversations already done, keeping them ===", flush=True)
                    try:
                        with open(out_path, "a" if _have else "w") as fh:
                            dn, ni = _batched_pass(by_i, fh, skip_convs=_have)
                        _tot = sum(1 for _ in open(out_path)) if os.path.exists(out_path) else dn
                        _abort_if_empty(_tot, out_path, ni + len(_have))
                        print(f"=== SWEEP OK {tag} B={BATCH_SIZE} "
                              f"({dn} new turns, {_tot} total) ===", flush=True)
                        ok = True
                        break
                    except torch.cuda.OutOfMemoryError:
                        _kept = sum(1 for _ in open(out_path)) if os.path.exists(out_path) else 0
                        print(f"=== SWEEP OOM {tag} B={BATCH_SIZE} — KEEPING {_kept} finished rows, "
                              f"dropping to the next rung ===", flush=True)
                        import gc
                        gc.collect(); torch.cuda.empty_cache()
                if not ok:
                    print(f"=== SWEEP FAIL {tag}: no batch in {ladder} fit ===", flush=True)
            print("[Done] sweep complete", flush=True)
            return

        dn, ni = _batched_pass(by, fout)
        done += dn
        fout.close(); print(f"[Done] {args.method} {done} turns -> {args.out}", flush=True)
        _abort_if_empty(done, args.out, ni)
        return
    for cid, tasks in list(by.items())[:args.max_conv]:
        tasks.sort(key=lambda r: int(r["turn"]))
        if args.method == "token_inject":
            recs = run_token_inject_accum(slm, slm_tok, model, tok, dev, tasks, args.max_new, stop_ids)
        elif args.no_accum:
            recs = run_standard(model, tok, dev, args.ratio, tasks, args.max_new, stop_ids, args.method,
                                slm=slm, slm_tok=slm_tok, lam=args.lam)
        elif args.method in ({"ours"} | SP_FUSION):
            recs = run_ours(slm, slm_tok, model, tok, dev, args.lam, tasks, args.max_new, stop_ids,
                            specprefill=(args.method in SP_FUSION), lm_no_accum=args.lm_no_accum)
        else:
            recs = run_single(model, tok, dev, args.ratio, tasks, args.max_new, stop_ids, args.method)
        for r in recs: fout.write(json.dumps(r) + "\n"); done += 1
        fout.flush(); print(f"[conv {cid}] answerable so far={done}", flush=True)
    fout.close(); print(f"[Done] {args.method} {done} turns -> {args.out}", flush=True)
    _abort_if_empty(done, args.out, len(list(by.items())[:args.max_conv]))


if __name__ == "__main__":
    main()
