#!/usr/bin/env python
"""CANONICAL EVALUATION — the SINGLE authoritative way to score & compare runs.

WHY THIS EXISTS (2026-06-27, after a research-shaking class of bug):
  We repeatedly compared runs that were NOT comparable — different extraction code per runner,
  answers truncated by too-small max_new, different example subsets, stale/mixed jobs — and trusted
  each runner's INLINE score. That produced impossible results (compression > full teacher) that were
  either waved off as "near-lossless" or only caught when the direction looked wrong. Every past
  comparison is therefore suspect.

THE RULE THIS TOOL ENFORCES (see EVALUATION_PROTOCOL.md):
  1. NEVER trust inline scores for a comparison. ALWAYS re-score the RAW generated text here, with ONE
     canonical extraction+metric per dataset (imported from src.eval — identical to the official metric).
  2. REFUSE to emit any number unless the runs pass the INVARIANT GATE:
        - identical example_id set (score only the intersection = true shared-N)
        - identical prompt (prompt_sha256 per example; fusion uses its context-prompt hash)
        - identical decoding mode, max_new_tokens, max_length, dataset
        (model is ALLOWED to differ — that is the point of teacher-vs-SLM.)
  3. REFUSE / loudly flag TRUNCATED outputs (generation hit max_new without a clean stop) — a cut-off
     answer cannot be scored as the model's committed answer (this was THE bl64k bug).
  A comparison that fails the gate prints UNSAFE and exits nonzero. No contaminated number escapes.

USAGE:
  canonical_eval.py --dataset babilong --sandwich LM=fileA.jsonl OURS=fileB.jsonl SLM=fileC.jsonl \
                    [BASELINE=fileD.jsonl ...] [--allow-unverified-provenance]
  (Label any number of role=file pairs. Closeness = (OURS-SLM)/(LM-SLM) is printed if LM,OURS,SLM given.)

Exit 0 only if the gate passes AND no config is materially truncated.
"""
import argparse
import json
import sys

sys.path.insert(0, "/u/anon/SLM_LM")
from src.eval import (  # noqa: E402
    extract_final_answer,
    extract_final_answer_full,
    compute_em_f1,
    compute_best_em_f1,
    compute_musique_em_f1,
    compute_2wiki_em_f1,
    canonicalize_qasper_prediction,
)

# ── CANONICAL scoring registry: dataset -> (extraction, metric). FROZEN. ───────────────────────────
# extraction(raw_text) -> pred string ; metric(pred, golds) -> (em, f1) in [0,1]/[0,100]-agnostic floats.
EOS_MARKERS = ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "</s>")


def _answer_truncated(raw: str) -> bool:
    """Is the COMMITTED ANSWER cut off (the bl64k bug), as opposed to the model merely rambling past it?

    SAFE (not truncated) if EITHER: an EOS marker is present (clean stop), OR a 'Final Answer:' marker
    is followed by a newline (the answer line completed and the model moved on — even if it then rambled
    into fake Q&A). UNSAFE if the text ends WHILE STILL INSIDE the answer line (no newline / EOS after the
    'Final Answer:' content) — i.e. generation hit max_new mid-answer (this under-scored teacher-14B at
    max_new=32). When no marker exists at all, treat as safe only if it ends with sentence punctuation."""
    if not raw:
        return True
    if any(m in raw for m in EOS_MARKERS):
        return False
    low = raw.lower()
    idx = low.rfind("final answer:")
    if idx != -1:
        after = raw[idx + len("final answer:"):]
        return "\n" not in after  # answer line still open at the cutoff => truncated
    # no explicit marker: complete only if it ends cleanly (newline or sentence punctuation)
    return not raw.rstrip().endswith((".", "!", "?", "\n", '"'))


def _short_answer_metric(pred, golds):
    return compute_best_em_f1(pred, golds)


def _summary_metric(pred, golds):
    # rougeLsum for summarization; computed via the same path the runners use is not in src.eval,
    # so summarization comparison goes through score_rouge_sharedN.py, NOT here. Guard against misuse.
    raise SystemExit("Summarization (rougeLsum) is NOT scored here — use scripts/score_rouge_sharedN.py "
                     "(which already reads the SAME stored rougeLsum). canonical_eval covers short-answer EM/F1.")


REGISTRY = {
    "babilong": (extract_final_answer, _short_answer_metric),
    "clutrr": (extract_final_answer, _short_answer_metric),
    "musique": (extract_final_answer, compute_musique_em_f1),
    "hotpotqa": (extract_final_answer, compute_best_em_f1),
    "2wiki": (extract_final_answer, compute_2wiki_em_f1),
    "2wikimultihopqa": (extract_final_answer, compute_2wiki_em_f1),
    "qasper": (extract_final_answer, lambda p, g: compute_best_em_f1(canonicalize_qasper_prediction(p, g), g)),
    "longbench_passageretrieval": (extract_final_answer, compute_best_em_f1),
    "longbench_passagecount": (extract_final_answer, compute_best_em_f1),
}


def _block(rec):
    for k in ("teacher", "slm_lm", "ours", "lm", "slm"):
        b = rec.get(k)
        if isinstance(b, dict) and ("text" in b or "extracted" in b):
            return b
    return rec  # minference: top-level text/metrics


def _audit(rec, b):
    pa = b.get("prompt_audit") or rec.get("prompt_audit")
    pa = eval(pa) if isinstance(pa, str) else (pa or {})
    return pa


def _provenance(rec):
    b = _block(rec)
    pa = _audit(rec, b)
    sha = pa.get("prompt_sha256")
    maxlen = pa.get("max_length")
    if sha is None:  # fusion: context-prompt hash lives in slm_debug
        sd = b.get("slm_debug") if isinstance(b, dict) else None
        if isinstance(sd, dict):
            sha = sd.get("shared_prompt_sha256") or sd.get("slm_answer_prompt_sha256")
    raw = b.get("text", "") if isinstance(b, dict) else ""
    # The fusion path decodes with --stop-after-answer "Final Answer:", so a COMPLETE answer legitimately
    # ends mid-line with no newline/EOS — the text heuristic below calls that "truncated" and would refuse
    # every fusion comparison (2026-08-10: 100% false positives on musique/hotpot ours arms). When the
    # harness recorded its OWN termination ground truth, trust it; otherwise fall back to the heuristic.
    # This changes only the GATE verdict, never the extraction or the metric.
    term = b.get("terminated") if isinstance(b, dict) else None
    truncated = (not term) if isinstance(term, bool) else _answer_truncated(raw)
    return {
        "raw": raw,
        "gold": rec.get("answer", ""),
        "golds": rec.get("answers") or ([rec.get("answer")] if rec.get("answer") else []),
        "sha": sha, "maxlen": maxlen,
        "max_new": pa.get("max_new_tokens") or (b.get("max_new_tokens") if isinstance(b, dict) else None),
        "decoding": pa.get("decoding") or (b.get("decoding") if isinstance(b, dict) else None),
        "truncated": truncated,
    }


def load(path):
    rows = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            eid = r.get("example_id") or r.get("id")
            if eid:
                rows[eid] = _provenance(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sandwich", nargs="+", required=True, help="role=file.jsonl pairs (e.g. LM=a OURS=b SLM=c)")
    ap.add_argument("--allow-unverified-provenance", action="store_true",
                    help="proceed (with WARN) when a field like max_length isn't recorded in a log — use sparingly")
    ap.add_argument("--max-trunc-frac", type=float, default=0.05, help="fail if any config exceeds this truncation fraction")
    args = ap.parse_args()

    if args.dataset not in REGISTRY:
        sys.exit(f"dataset {args.dataset!r} not in canonical registry {list(REGISTRY)}")
    extract, metric = REGISTRY[args.dataset]

    pairs = []
    for tok in args.sandwich:
        if "=" not in tok:
            sys.exit(f"bad role=file token: {tok!r}")
        role, path = tok.split("=", 1)
        pairs.append((role, path))
    data = {role: load(path) for role, path in pairs}
    roles = [r for r, _ in pairs]
    hard_fail = False
    print(f"=== CANONICAL EVAL  dataset={args.dataset}  ({len(pairs)} configs) ===")
    for role, path in pairs:
        print(f"  {role:10} = {path.split('/')[-1]}  (n={len(data[role])})")

    # GATE 1: example identity
    shared = set.intersection(*[set(d) for d in data.values()])
    n = len(shared)
    union = set.union(*[set(d) for d in data.values()])
    print(f"\n[GATE 1] example identity: shared-N={n} / union={len(union)}")
    if any(set(data[r]) != shared for r in roles):
        print("   ⚠ configs cover different example sets; scoring the intersection only.")
    if n == 0:
        print("   ❌ no shared examples — cannot compare.")
        sys.exit(1)

    # GATE 2: prompt identity
    bad = sum(1 for e in shared if len({data[r][e]["sha"] for r in roles}) > 1 or None in {data[r][e]["sha"] for r in roles})
    if bad == 0:
        print(f"[GATE 2] prompt identity: ✅ identical prompt_sha256 on all {n} shared examples")
    else:
        print(f"[GATE 2] prompt identity: ❌ {bad}/{n} examples differ (or missing) → NOT COMPARABLE")
        hard_fail = True

    # GATE 3: decoding/length identity (where recorded)
    for field in ("maxlen", "max_new", "decoding"):
        vals = {}
        for r in roles:
            s = {data[r][e][field] for e in shared}
            vals[r] = next(iter(s)) if len(s) == 1 else ("MIXED" if s else None)
        present = {v for v in vals.values() if v not in (None, "MIXED")}
        if len(present) > 1:
            print(f"[GATE 3] {field}: ❌ differs across configs: { {r: vals[r] for r in roles} } → NOT COMPARABLE")
            hard_fail = True
        elif not present:
            msg = "not recorded in logs — UNVERIFIED" + ("" if args.allow_unverified_provenance else " (use --allow-unverified-provenance to override)")
            print(f"[GATE 3] {field}: ⚠ {msg}")
            if not args.allow_unverified_provenance:
                hard_fail = True
        else:
            print(f"[GATE 3] {field}: ✅ {present.pop()}")

    # GATE 4: truncation
    print("[GATE 4] truncation (cut-off answers are UNSAFE to score):")
    for r in roles:
        tr = sum(1 for e in shared if data[r][e]["truncated"]) / n
        flag = "✅" if tr <= args.max_trunc_frac else "❌"
        if tr > args.max_trunc_frac:
            hard_fail = True
        print(f"   {flag} {r:10}: {tr*100:.0f}% truncated")

    # RE-SCORE (canonical) on shared-N
    print(f"\n=== CANONICAL RE-SCORE (shared-N={n}, single extraction+metric) ===")
    res = {}
    for r in roles:
        ems = []
        for e in shared:
            pred = extract(data[r][e]["raw"])
            golds = data[r][e]["golds"] or [data[r][e]["gold"]]
            em, _f1 = metric(pred, golds)
            ems.append(float(em))
        res[r] = sum(ems) / n
        print(f"   {r:10}: EM={res[r]:.3f}")
    if {"LM", "OURS", "SLM"} <= set(res):
        denom = res["LM"] - res["SLM"]
        cl = 100 * (res["OURS"] - res["SLM"]) / denom if denom else float("nan")
        print(f"\n   closeness (OURS-SLM)/(LM-SLM) = {cl:.0f}%   [LM {res['LM']:.3f} / OURS {res['OURS']:.3f} / SLM {res['SLM']:.3f}]")
        if res["OURS"] > res["LM"] + 1e-9:
            print("   ⚠ OURS > LM-teacher — impossible direction, investigate (do NOT report).")

    print()
    if hard_fail:
        print("VERDICT: ❌ UNSAFE — gate failed. NO number above may be reported until fixed.")
        sys.exit(1)
    print("VERDICT: ✅ SAFE — gate passed; canonical re-scored numbers above are comparable.")
    sys.exit(0)


if __name__ == "__main__":
    main()
