#!/usr/bin/env python
"""
CLUTRR multi-question KV-reuse — teacher / snapkv-perquery / snapkv-reuse (single 14B model).
Same experiment as scripts/snapkv_reuse_qasper.py but on the CLEAN-METRIC CLUTRR benchmark
(scripts/clutrr_multiq_build.py). One padded story = shared context; several DERIVED kinship
questions; answer = one relation word -> relation-vocabulary EM (no length pathology).

Reuses the QASPER harness's KV primitives verbatim (prefill_uncompressed / snapkv_compress_doc /
decode_from_context) and the IDENTICAL prompt builder (build_full_context_qa_prompt, variant
reason_then_answer_clutrr) so teacher/ours(run_evidence)/snapKV all share one prompt. ours (14B+3B
fusion) is run separately via run_evidence (--dataset clutrr_multiq) — lossless reuse by construction.
"""
from __future__ import annotations
import argparse, json, os, random, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from snapkv_reuse_qasper import prefill_uncompressed, snapkv_compress_doc, decode_from_context, _extract  # noqa: E402
from src.models import load_causal_lm  # noqa: E402
from src.qa_prompts import build_full_context_qa_prompt  # noqa: E402
from src.clutrr_data import build_clutrr_question  # noqa: E402

# relation vocabulary, LONGEST-first so "grandmother"/"mother-in-law" match before "mother"
RELVOCAB = ["mother-in-law", "father-in-law", "daughter-in-law", "son-in-law", "sister-in-law", "brother-in-law",
            "grandmother", "grandfather", "granddaughter", "grandson", "mother", "father", "daughter", "son",
            "sister", "brother", "aunt", "uncle", "niece", "nephew", "wife", "husband"]


def rel_em(pred, golds):
    p = pred.lower()
    for rel in RELVOCAB:
        if rel in p:
            return float(rel == str(golds[0]).lower())
    return 0.0


def load_stories(n_stories, variant):
    data = json.load(open("results/clutrr_multiq/clutrr_multiq.json"))[:n_stories]
    stories = []
    for item in data:
        ctx = item["context"]
        qs = [(f"{item['story_id']}_{qi}", build_clutrr_question(q["b"], q["a"]), [q["rel"]])
              for qi, q in enumerate(item["questions"])]
        q0 = qs[0][1]
        full0 = build_full_context_qa_prompt(q0, [ctx], prompt_variant=variant)
        marker0 = f"\n\nQuestion: {q0}\n"
        assert full0.endswith(marker0), "prompt tail mismatch"
        stories.append({"story_id": item["story_id"], "prefix": full0[:-len(marker0)], "questions": qs})
    return stories


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--cache-dir", default=os.environ.get("HF_HOME", "/work/hdd/myproject/anon/hf"))
    ap.add_argument("--variant", default="reason_then_answer")  # standard prompt (custom clutrr variant made fusion answer-first)
    ap.add_argument("--stories", type=int, default=int(os.environ.get("CLUTRR_MULTIQ_STORIES", "100")))
    ap.add_argument("--keep-budget", type=int, default=64)
    # ★ FAIR-MEMORY mode: keep_budget = round(keep_ratio * doc_len) PER STORY, so snapKV's 14B-KV footprint matches
    # ours' 3B-KV footprint. keep_ratio = (3B KV/token)/(14B KV/token) = 18432/98304 = 0.1875 for Qwen2.5 3B vs 14B.
    ap.add_argument("--keep-ratio", type=float, default=0.0, help="if >0, per-story budget = round(keep_ratio*doc_len)")
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--max-doc-tokens", type=int, default=8000)
    ap.add_argument("--methods", default="teacher,snapkv_perquery,snapkv_reuse")
    ap.add_argument("--output", required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    methods = args.methods.replace("+", ",").split(",")  # '+' allowed so sbatch --export commas don't split the list

    print(f"[Info] loading {args.model}", flush=True)
    model, tok = load_causal_lm(args.model, device_map="cuda:0", cache_dir=args.cache_dir)
    model.eval()

    done = set()
    if args.resume and os.path.exists(args.output):
        for l in open(args.output):
            try:
                r = json.loads(l); done.add((r["example_id"], r["method"]))
            except Exception:
                pass
        print(f"[Info] resume: {len(done)} done", flush=True)

    stories = load_stories(args.stories, args.variant)
    print(f"[Info] {len(stories)} stories, variant={args.variant}, B={args.keep_budget}", flush=True)

    fout = open(args.output, "a")
    agg = {m: [] for m in methods}
    for si, story in enumerate(stories):
        doc_ids = tok(story["prefix"], return_tensors="pt", add_special_tokens=True,
                      truncation=True, max_length=args.max_doc_tokens).input_ids
        doc_len = int(doc_ids.shape[1])
        qs = story["questions"]
        suffix_ids = [tok(f"\n\nQuestion: {q}\n", return_tensors="pt", add_special_tokens=False).input_ids
                      for (_qid, q, _g) in qs]
        # ★ RANDOM anchor per story (seeded, reproducible) — the question snapKV-reuse compresses on. FIXES the
        # confound where the anchor was ALWAYS q_index 0 = the CLUTRR target (the full k-hop = hardest). With a random
        # anchor, both the anchor and the reused (non-anchor) questions are difficulty-representative, so the reuse
        # penalty is measured on a fair, unbiased set. Scoring is done on the NON-ANCHOR questions (see clutrr_multiq_score.py).
        anchor = random.Random(int(story["story_id"])).randrange(len(qs))
        kb = max(1, round(args.keep_ratio * doc_len)) if args.keep_ratio > 0 else args.keep_budget  # fair-memory budget per story
        ctxs, build_s = {}, {}
        if "teacher" in methods:
            ctxs["teacher"], build_s["teacher"] = prefill_uncompressed(model, doc_ids)
        if "snapkv_reuse" in methods:
            ctxs["snapkv_reuse"], build_s["snapkv_reuse"] = snapkv_compress_doc(model, doc_ids, suffix_ids[anchor], kb)

        for qi, (qid, q, golds) in enumerate(qs):
            for m in methods:
                if (qid, m) in done:
                    continue
                if m == "teacher":
                    text, glen, dec_s = decode_from_context(model, tok, ctxs["teacher"], doc_len, suffix_ids[qi], doc_len, args.max_new)
                    ctx_len = doc_len
                elif m == "snapkv_reuse":
                    if qi == anchor:
                        continue  # the anchor is what we compressed ON — it is NOT a reused question
                    ctx_len = int(ctxs["snapkv_reuse"][0][0].shape[2])
                    text, glen, dec_s = decode_from_context(model, tok, ctxs["snapkv_reuse"], doc_len, suffix_ids[qi], ctx_len, args.max_new)
                elif m == "snapkv_perquery":
                    pq, _ = snapkv_compress_doc(model, doc_ids, suffix_ids[qi], kb)
                    ctx_len = int(pq[0][0].shape[2])
                    text, glen, dec_s = decode_from_context(model, tok, pq, doc_len, suffix_ids[qi], ctx_len, args.max_new)
                else:
                    continue
                ext = _extract(text)
                em = rel_em(ext, golds)
                agg[m].append(em)
                rec = {"story_id": story["story_id"], "example_id": qid, "q_index": qi, "n_questions": len(qs),
                       "method": m, "anchor": anchor, "is_anchor": (qi == anchor), "question": q, "golds": golds,
                       "raw": text, "extracted": ext, "em": em,
                       "keep_budget": kb, "keep_ratio": args.keep_ratio, "doc_len": doc_len, "ctx_len": ctx_len, "gen_len": glen}
                fout.write(json.dumps(rec) + "\n"); fout.flush()
        if (si + 1) % 10 == 0:
            msg = " | ".join(f"{m}: em={sum(agg[m])/max(1,len(agg[m])):.3f} (n={len(agg[m])})" for m in methods)
            print(f"[{si+1}/{len(stories)}] {msg}", flush=True)
    fout.close()
    print("[Done]", flush=True)


if __name__ == "__main__":
    main()
