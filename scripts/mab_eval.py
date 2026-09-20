#!/usr/bin/env python
"""
Stateful multi-turn evaluator for MAB LME(S*) episodes (MULTI_TURN_EVAL.md).

Reference-history protocol (per turn): append Q_i -> snapshot -> generate A_i' -> score
-> ROLLBACK (discard generated KV) -> teacher-force A_i_ref -> continue. Every method sees the
same logical transcript; each manages physical KV per its own semantics.

Single-model methods implemented here (share one causal LM):
  teacher            full-KV accumulation (no compression)
  snapkv_frozen      compress the shared context ONCE at Q1, freeze it, accumulate Q/A tail in full KV
  snapkv_fresh       re-prefill the full logical transcript every turn, compress on the current question
  pyramidkv_frozen   same as snapkv_frozen with PyramidKVPress (layer-wise budget)
  pyramidkv_fresh    same as snapkv_fresh with PyramidKVPress

Rollback is EXACT: generation only appends KV; we slice the legacy K/V back to the pre-generation
length. RoPE position (`pos`) and physical cache length (`cache_len`) are tracked separately so a
compressed context (pos>cache_len) decodes correctly (mirrors snapkv_reuse_qasper.decode_from_context).

Usage (SLURM, 1 GPU):
  python scripts/mab_eval.py --episodes results/mab_episodes/lme_28k.jsonl --model Qwen/Qwen2.5-14B-Instruct \
     --methods teacher,snapkv_frozen,snapkv_fresh --ratio 0.5 --max-episodes 6 --max-new 64 \
     --out results/mab/lme28k_14b.jsonl
"""
import os, sys, json, time, argparse
os.environ.setdefault("HF_HOME", "/work/hdd/myproject/anon/hf")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from transformers import DynamicCache
from src.models import load_causal_lm, set_use_chat_template, prepare_inputs
from src.eval import compute_best_em_f1, extract_final_answer
import src.qa_prompts as _qp
import scripts.reason_fix as RF

# 2026-07-17: REASON_FIX makes reasoning structurally guaranteed (see scripts/reason_fix.py). The prior setup
# silently produced NO reasoning (97% straight-to-'Final Answer:') -> the fusion collapsed to the SLM floor.
REASON_FIX = os.environ.get("REASON_FIX", "1") == "1"     # default ON now; set REASON_FIX=0 to reproduce the (buggy) old runs
INSTRUCTION = _qp.QA_REASON_V2 if REASON_FIX else _qp.QA_FULL_CONTEXT_INSTRUCTION_REASON

# reason_then_answer turn format: the model reasons over the (SLM-injected) facts then emits "Final Answer:".
# This reasoning step is exactly how the query-only LM contributes in the fusion (it reasons over the tokens
# the SLM surfaces). A bare "Answer:" prompt removes it -> ours degenerates to the SLM.
def q_turn(q):
    return RF.q_turn_reason(q) if REASON_FIX else f"\n\nQuestion: {q}\n"

def ref_answer(gold):
    return f"Final Answer: {gold}\n"

# 2026-07-18: UNIFY multi-turn history with mtrag_accum/LoCoMo/QASPER (reason_hist=ref). The old eventqa history was
# ANSWER-ONLY (ref_answer = 'Final Answer: gold') which suppresses reasoning in later turns. reason_hist=ref prepends
# the model's OWN generated reasoning + the REFERENCE gold answer (no error-propagation: the answer is always gold).
REASON_HIST = os.environ.get("REASON_HIST", "ref")
def reason_hist_block(aref, gen_text):
    """aref = 'Final Answer: gold\\n'. reason_hist=ref -> model's reasoning (from gen_text) + gold; else answer-only."""
    if REASON_FIX and REASON_HIST == "ref":
        reason = RF.reason_prefix(gen_text or "").rstrip()
        if reason:
            return reason + "\n" + aref
    return aref

def done_final_answer(tok, gen):
    """Stop once the 'Final Answer:' line is complete (a newline appeared after it). Robust to \\n\\n
    being a single token: decodes the running text every step (gen is short, <= max_new)."""
    if len(gen) < 2:
        return False
    txt = tok.decode(gen)
    return "Final Answer:" in txt and "\n" in txt.split("Final Answer:", 1)[1]

PRESS = {
    "snapkv": "SnapKVPress", "pyramidkv": "PyramidKVPress",
    "tova": "TOVAPress", "streamingllm": "StreamingLLMPress",
    "h2o": "ExpectedAttentionPress", "expected_attention": "ExpectedAttentionPress",  # query-independent (H2O-like)
}


def build_press(name, ratio):
    import kvpress
    cls = getattr(kvpress, PRESS[name])
    return cls(compression_ratio=ratio)


def to_legacy(cache):
    return [[k.detach(), v.detach()] for (k, v) in cache.to_legacy_cache()]


def legacy_len(kv):
    return kv[0][0].shape[2]


def truncate(kv, n):
    return [[k[:, :, :n, :].contiguous(), v[:, :, :n, :].contiguous()] for (k, v) in kv]


class StatefulLM:
    """One causal LM with manual KV accumulation + exact rollback + optional one-shot compression."""

    def __init__(self, model, tok, device):
        self.model = model
        self.tok = tok
        self.device = device
        self.cache = None   # persistent DynamicCache (updated in place by the model)
        self.pos = 0        # next RoPE position (logical)
        self.cache_len = 0  # physical KV length

    def _ids(self, text, chat=False):
        if chat:
            enc = prepare_inputs(self.tok, text)  # applies chat template if enabled
            return enc["input_ids"].to(self.device)
        return self.tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)

    def _crop(self, n):
        """Truncate the persistent cache to physical length n (exact rollback; O(slice), no copy)."""
        c = self.cache
        if hasattr(c, "crop"):
            c.crop(n)
        else:  # manual slice of the layer lists
            for i in range(len(c.key_cache)):
                c.key_cache[i] = c.key_cache[i][:, :, :n, :]
                c.value_cache[i] = c.value_cache[i][:, :, :n, :]
        self.cache_len = n

    @torch.no_grad()
    def prefill(self, ids, press=None):
        cache = DynamicCache()
        pos_ids = torch.arange(ids.shape[1], device=self.device).unsqueeze(0)
        ctx = press(self.model) if press is not None else _null()
        with ctx:
            out = self.model(ids, past_key_values=cache, use_cache=True, logits_to_keep=1,
                             position_ids=pos_ids, cache_position=torch.arange(ids.shape[1], device=self.device))
        self.cache = out.past_key_values
        self.cache_len = self.cache.get_seq_length()
        self.pos = ids.shape[1]                 # logical position after the (possibly compressed) prompt
        return out.logits[:, -1, :]

    @torch.no_grad()
    def forward(self, ids):
        n = ids.shape[1]
        pos_ids = torch.arange(self.pos, self.pos + n, device=self.device).unsqueeze(0)
        cache_pos = torch.arange(self.cache_len, self.cache_len + n, device=self.device)
        attn = torch.ones((1, self.cache_len + n), device=self.device, dtype=torch.long)
        out = self.model(ids, past_key_values=self.cache, use_cache=True, position_ids=pos_ids,
                         cache_position=cache_pos, attention_mask=attn, logits_to_keep=1)
        self.cache = out.past_key_values
        self.cache_len += n
        self.pos += n
        return out.logits[:, -1, :]

    @torch.no_grad()
    def generate_greedy(self, last_logits, max_new, stop_ids):
        # reason_then_answer: generate reasoning then stop once the "Final Answer:" line is complete.
        gen = []
        logits = last_logits
        for step in range(max_new):
            lg = RF.block_logits(logits, step) if REASON_FIX else logits   # force reasoning before 'Final Answer:'
            nxt = int(torch.argmax(lg, dim=-1).item())
            if nxt in stop_ids:
                break
            gen.append(nxt)
            if (RF.reason_stopped(self.tok, gen) if REASON_FIX else done_final_answer(self.tok, gen)):
                break
            logits = self.forward(torch.tensor([[nxt]], device=self.device))
        return gen


def _null():
    import contextlib
    return contextlib.nullcontext()


class FusedStateful:
    """ours: SLM ingests the long context (persistent), LM is query-only + dialogue history.
    Per-step logits fused lam*slm + (1-lam)*lm (same Qwen vocab). Both branches rolled back +
    reference-forced each turn. LM NEVER sees the long context (spec §4.2)."""

    def __init__(self, slm, slm_tok, lm, lm_tok, lam, dev):
        self.slm = StatefulLM(slm, slm_tok, dev)
        self.lm = StatefulLM(lm, lm_tok, dev)
        self.lam = lam
        self.dev = dev
        self.first = True

    def prefill_context(self, ctx):
        # SLM sees INSTRUCTION + memory (full context); LM sees ONLY the INSTRUCTION (query-only, no context).
        self.slm.prefill(self.slm._ids(INSTRUCTION + "\n\n" + ctx))
        self.lm.prefill(self.lm._ids(INSTRUCTION))

    @torch.no_grad()
    def turn(self, qblock, aref, max_new, stop_ids):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        sids = self.slm._ids(qblock)
        lids = self.lm._ids(qblock)
        sl = self.slm.forward(sids)
        ll = self.lm.forward(lids)
        torch.cuda.synchronize(); prefill_s = time.perf_counter() - t0
        base_s = (self.slm.cache_len, self.slm.pos)
        base_l = (self.lm.cache_len, self.lm.pos)
        V = min(sl.shape[-1], ll.shape[-1])   # shared vocab (Qwen2.5-3B=151936, 14B=152064)
        td = time.perf_counter()
        gen = []
        for step in range(max_new):
            fused = self.lam * sl[..., :V] + (1.0 - self.lam) * ll[..., :V]
            if REASON_FIX:
                fused = RF.block_logits(fused, step)     # force reasoning tokens (SLM injects facts, LM reasons) first
            nxt = int(torch.argmax(fused, dim=-1).item())
            if nxt in stop_ids:
                break
            gen.append(nxt)
            if (RF.reason_stopped(self.slm.tok, gen) if REASON_FIX else done_final_answer(self.slm.tok, gen)):
                break
            tok_t = torch.tensor([[nxt]], device=self.dev)
            sl = self.slm.forward(tok_t)
            ll = self.lm.forward(tok_t)
        torch.cuda.synchronize(); decode_s = time.perf_counter() - td
        # rollback both branches (discard generated KV)
        self.slm._crop(base_s[0]); self.slm.pos = base_s[1]
        self.lm._crop(base_l[0]); self.lm.pos = base_l[1]
        # teacher-force history into both (reason_hist=ref -> generated reasoning + reference gold)
        hist = reason_hist_block(aref, self.lm.tok.decode(gen, skip_special_tokens=True))
        aids_s = self.slm._ids(hist); self.slm.forward(aids_s)
        aids_l = self.lm._ids(hist); self.lm.forward(aids_l)
        return gen, prefill_s, decode_s


def run_episode(sm, ep, method, ratio, max_new, stop_ids):
    """Returns list of per-turn dicts. sm is a fresh StatefulLM (state reset per episode)."""
    model, tok, dev = sm.model, sm.tok, sm.device
    instr_ctx = INSTRUCTION + "\n\n" + ep["context"]   # reason_then_answer instruction + memory
    turns = ep["turns"]
    results = []
    press_name = ("snapkv" if "snapkv" in method else "pyramidkv" if "pyramidkv" in method
                  else "h2o" if method == "h2o" else None)
    frozen = method.endswith("frozen") or method == "h2o"   # h2o = query-independent, compress-once/reuse (frozen-style)
    fresh = method.endswith("fresh")
    ref_transcript = ""  # accumulated Q/A_ref text for the fresh rebuild

    # non-fresh: prefill context once (teacher) or defer to Q1 (frozen)
    if not fresh:
        if method == "teacher":
            sm.prefill(sm._ids(instr_ctx))
        # frozen: prefill deferred until turn 1 (compress [instr+ctx+Q1])

    for i, t in enumerate(turns):
        q = t["question"]
        golds = t["answer_ref"]
        qblock = q_turn(q)
        aref = ref_answer(golds[0] if golds else "")

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        if fresh:
            # rebuild full logical transcript, full prefill (+ compress on current Q), then generate
            full = instr_ctx + ref_transcript + qblock
            p = build_press(press_name, ratio) if press_name else None
            last = sm.prefill(sm._ids(full), press=p)
            torch.cuda.synchronize(); ttft = time.perf_counter() - t0
            gen = sm.generate_greedy(last, max_new, stop_ids)
            torch.cuda.synchronize(); dec_s = time.perf_counter() - t0 - ttft
        else:
            if method != "teacher" and i == 0:
                # frozen: compress [instr + ctx + Q1] once
                p = build_press(press_name, ratio) if press_name else None
                last = sm.prefill(sm._ids(instr_ctx + qblock), press=p)
            else:
                last = sm.forward(sm._ids(qblock))
            torch.cuda.synchronize(); ttft = time.perf_counter() - t0
            base_len, base_pos = sm.cache_len, sm.pos     # snapshot AFTER Q
            gen = sm.generate_greedy(last, max_new, stop_ids)
            torch.cuda.synchronize(); dec_s = time.perf_counter() - t0 - ttft
            # ROLLBACK: discard generated KV
            sm._crop(base_len)
            sm.pos = base_pos
            # teacher-force history (reason_hist=ref -> generated reasoning + reference gold)
            sm.forward(sm._ids(reason_hist_block(aref, tok.decode(gen, skip_special_tokens=True))))

        text = tok.decode(gen, skip_special_tokens=True)
        pred = extract_final_answer(text) if text else text
        em, f1 = compute_best_em_f1(pred or text, golds)
        peak = torch.cuda.max_memory_allocated() / 1e9
        if fresh:
            ref_transcript += qblock + reason_hist_block(aref, text)
        results.append({
            "episode_id": ep["episode_id"], "turn": i, "question_type": t["question_type"],
            "method": method, "em": int(em), "f1": float(f1),
            "pred": (pred or text)[:200], "raw": text[:2000], "gold": golds,  # keep FULL gen incl 'Final Answer:' (was [:400] -> truncated the marker for scoring)
            "ttft_s": round(ttft, 3), "decode_s": round(dec_s, 3),
            "cache_len": sm.cache_len, "peak_gib": round(peak, 2),
        })
    return results


@torch.no_grad()
def verify_rollback(model, tok, dev, ep, max_new, stop_ids):
    """Correctness (spec §14): after append-Q1 -> generate -> ROLLBACK -> teacher-force A1_ref -> append Q2,
    the incremental next-token logits must equal a FRESH full re-prefill of [C+Q1+A1_ref+Q2]."""
    ctx = ep["context"]; t0 = ep["turns"][0]; t1 = ep["turns"][1]
    ids = lambda s: tok(s, return_tensors="pt", add_special_tokens=False)["input_ids"].to(dev)
    q1 = f"\n\nQuestion: {t0['question']}\nAnswer:"; a1 = " " + (t0["answer_ref"][0] if t0["answer_ref"] else "")
    q2 = f"\n\nQuestion: {t1['question']}\nAnswer:"
    # incremental path
    sm = StatefulLM(model, tok, dev)
    sm.prefill(ids(ctx))
    last = sm.forward(ids(q1))
    base_len, base_pos = sm.cache_len, sm.pos
    sm.generate_greedy(last, max_new, stop_ids)           # generate + discard
    sm._crop(base_len); sm.pos = base_pos                 # rollback
    sm.forward(ids(a1))                                    # teacher-force reference
    logits_incr = sm.forward(ids(q2))
    # fresh reference path
    sm2 = StatefulLM(model, tok, dev)
    sm2.prefill(ids(ctx + q1 + a1 + q2))
    logits_ref = sm2._last if hasattr(sm2, "_last") else None
    # recompute fresh last-logits directly
    full = ids(ctx + q1 + a1 + q2)
    from transformers import DynamicCache as DC
    out = model(full, past_key_values=DC(), use_cache=True,
                position_ids=torch.arange(full.shape[1], device=dev).unsqueeze(0))
    logits_ref = out.logits[:, -1, :]
    # DECISIVE: my manual incremental forward vs HF-native cache forward (uncompressed) -> must be ~0
    # (proves manual position_ids/cache_position/attention_mask == HF's inferred handling)
    smM = StatefulLM(model, tok, dev)
    smM.prefill(ids(ctx))
    logits_manual = smM.forward(ids(q1))
    cacheH = DC()
    model(ids(ctx), past_key_values=cacheH, use_cache=True,
          position_ids=torch.arange(ids(ctx).shape[1], device=dev).unsqueeze(0))
    outH = model(ids(q1), past_key_values=cacheH, use_cache=True)   # HF infers positions
    logits_hf = outH.logits[:, -1, :]
    manual_vs_hf = float((logits_manual - logits_hf).abs().max())
    print(f"[VERIFY] MANUAL-incremental vs HF-native-incremental max|Δlogit|={manual_vs_hf:.5f} "
          f"-> {'OK (manual==HF)' if manual_vs_hf < 0.05 else 'BUG in position handling'}")

    # noise floor: two independent fresh re-prefills of the SAME text (flash/bf16 non-determinism)
    out2 = model(full, past_key_values=DC(), use_cache=True,
                 position_ids=torch.arange(full.shape[1], device=dev).unsqueeze(0))
    logits_ref2 = out2.logits[:, -1, :]
    floor = float((logits_ref - logits_ref2).abs().max())
    same_argmax = int(torch.argmax(logits_incr) == torch.argmax(logits_ref))
    maxdiff = float((logits_incr - logits_ref).abs().max())
    ok = same_argmax and maxdiff <= max(0.5, 3 * floor)
    print(f"[VERIFY] incr-vs-fresh max|Δlogit|={maxdiff:.4f}  fresh-vs-fresh NOISE FLOOR={floor:.4f}  "
          f"argmax_match={same_argmax} (incr cache_len={sm.cache_len}) -> {'OK (within noise)' if ok else 'CHECK'}")
    return same_argmax, maxdiff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="results/mab_episodes/lme_28k.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--slm-model", default="Qwen/Qwen2.5-3B-Instruct", help="SLM for the 'ours' fusion method")
    ap.add_argument("--lam", type=float, default=0.7, help="fusion lambda (SLM weight) for 'ours'")
    ap.add_argument("--methods", default="teacher,snapkv_frozen,snapkv_fresh")
    ap.add_argument("--ratio", type=float, default=0.5)
    ap.add_argument("--max-episodes", type=int, default=6)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--use-chat", type=int, default=0)
    ap.add_argument("--out", default="results/mab/smoke.jsonl")
    ap.add_argument("--verify", type=int, default=0)
    args = ap.parse_args()

    eps = [json.loads(l) for l in open(args.episodes)][:args.max_episodes]
    print(f"[mab_eval] {len(eps)} episodes, methods={args.methods}, model={args.model}, ratio={args.ratio}")
    if args.use_chat:
        set_use_chat_template(True)
    model, tok = load_causal_lm(args.model, device_map="cuda:0")
    model.eval()
    dev = next(model.parameters()).device
    stop_ids = set(x for x in [tok.eos_token_id] if x is not None)
    # add newline as a soft stop for short-answer turns
    nl = tok("\n", add_special_tokens=False)["input_ids"]
    if len(nl) == 1:
        stop_ids.add(nl[0])

    if args.verify:
        verify_rollback(model, tok, dev, eps[0], args.max_new, stop_ids)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fout = open(args.out, "w")
    import re
    methods = [m.strip() for m in re.split(r"[,+]", args.methods) if m.strip()]
    slm_model = slm_tok = None
    if "ours" in methods:
        slm_model, slm_tok = load_causal_lm(args.slm_model, device_map="cuda:0")
        slm_model.eval()
        print(f"[mab_eval] loaded SLM {args.slm_model} for fusion (lam={args.lam})")
    for method in methods:
        for ep in eps:
            t0 = time.perf_counter()
            if method == "ours":
                fs = FusedStateful(slm_model, slm_tok, model, tok, args.lam, dev)
                fs.prefill_context(ep["context"])
                rows = []
                for i, t in enumerate(ep["turns"]):
                    q = t["question"]; golds = t["answer_ref"]
                    qblock = q_turn(q); aref = ref_answer(golds[0] if golds else "")
                    torch.cuda.reset_peak_memory_stats()
                    gen, pref_s, dec_s = fs.turn(qblock, aref, args.max_new, stop_ids)
                    text = tok.decode(gen, skip_special_tokens=True)
                    pred = extract_final_answer(text) if text else text
                    em, f1 = compute_best_em_f1(pred or text, golds)
                    rows.append({"episode_id": ep["episode_id"], "turn": i, "question_type": t["question_type"],
                                 "method": method, "em": int(em), "f1": float(f1), "pred": (pred or text)[:200], "raw": text[:400],
                                 "gold": golds, "ttft_s": round(pref_s, 3), "decode_s": round(dec_s, 3),
                                 "cache_len": fs.slm.cache_len, "peak_gib": round(torch.cuda.max_memory_allocated()/1e9, 2)})
            else:
                sm = StatefulLM(model, tok, dev)
                rows = run_episode(sm, ep, method, args.ratio, args.max_new, stop_ids)
                del sm
            for r in rows:
                fout.write(json.dumps(r) + "\n")
            fout.flush()
            em = sum(r["em"] for r in rows) / len(rows)
            f1 = sum(r["f1"] for r in rows) / len(rows)
            print(f"  [{method}] {ep['episode_id']} ctx={ep['context_tokens']} nturns={len(rows)} "
                  f"EM={em:.2f} F1={f1:.2f} {time.perf_counter()-t0:.1f}s cache_end={rows[-1]['cache_len']}")
            torch.cuda.empty_cache()
    fout.close()
    print(f"[mab_eval] wrote {args.out}")


if __name__ == "__main__":
    main()
