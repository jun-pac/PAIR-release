#!/usr/bin/env python
"""build_table.py — the ONLY sanctioned way to turn result logs into a comparison table.

It ENFORCES fair comparison in code (not by trusting Claude):
  1. Every log must carry a `_provenance` fingerprint (from scripts/bench_config.py). No provenance -> REFUSE.
  2. All logs in the table must share the SAME bench_config.COMPARE_KEYS (prompt_sha, extractor, decoding,
     reason_fix/hist, max_new, ratio, doc_number, truncation, order, seed, metric). Any mismatch -> REFUSE, print diff.
  3. It computes the EXACT shared-N = intersection of example-ids across all logs (never per-log-N).
  4. It RE-SCORES every log on the shared-N with the ONE canonical extractor (reason_fix.extract_answer) and the
     bench's metric, so extraction/metric are identical for all rows.
  5. Only then does it emit the table (with N and the shared fingerprint). Refusal exits non-zero.

Usage: python scripts/build_table.py --bench locomo teacher=a.jsonl ours=b.jsonl floor=c.jsonl [--md out.md]
"""
import os, sys, json, argparse, statistics as st
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import scripts.bench_config as BC
import scripts.reason_fix as RF
from src.eval import compute_best_em_f1
try: from src.eval import canonicalize_qasper_prediction as _canon
except Exception: _canon = None
try: from src.eval import clutrr_relation_em as _clutrr_rel_em
except Exception: _clutrr_rel_em = None

def load(f):
    txt = open(f).read().strip()
    return json.loads(txt) if txt and txt[0] == "[" else [json.loads(l) for l in txt.splitlines() if l.strip()]

def rec_id(r):
    for keys in (("conv", "turn"), ("conversation_id", "turn"), ("episode_id", "turn")):
        if all(k in r for k in keys): return tuple(str(r[k]) for k in keys)
    return None

def score_one(r, metric):
    raw = r.get("raw", r.get("acc_pred", ""))
    pred = RF.extract_answer(raw)
    gold = str(r.get("gold", (r.get("golds") or [""])[0]))
    if metric == "clutrr_relation_em" and _clutrr_rel_em is not None:
        return _clutrr_rel_em(pred, [gold])          # relation-vocabulary exact match (clean single-word metric)
    if metric == "qasper_canonical_f1" and _canon is not None:
        pred = _canon(pred, [gold])
    return float(compute_best_em_f1(pred, [gold])[1])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("cols", nargs="+", help="role=path pairs (the methods being compared)")
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    named = {}
    for kv in a.cols:
        role, path = kv.split("=", 1); named[role] = path
    cfg = BC.get(a.bench); metric = cfg["metric"]

    # 1+2: provenance present + COMPARE_KEYS identical across all logs
    provs, rows = {}, {}
    fatal = []
    for role, f in named.items():
        if not os.path.exists(f): fatal.append(f"{role}: file missing {f}"); continue
        rs = load(f); rows[role] = rs
        p = next((x.get("_provenance") for x in rs if x.get("_provenance")), None)
        if p is None: fatal.append(f"{role}: NO _provenance -> not comparable (re-run with the current harness)")
        provs[role] = p
    if fatal:
        print("❌ REFUSED (cannot build a fair table):"); [print("   -", x) for x in fatal]; sys.exit(2)
    # every run must carry the FULL fingerprint (incomplete = pre-system run = not protocol-compliant)
    for role, p in provs.items():
        missing = [k for k in BC.COMPARE_KEYS if k not in p]
        if missing:
            fatal.append(f"{role}: INCOMPLETE provenance, missing {missing} — re-run with --bench {a.bench} (current harness)")
    if fatal:
        print(f"❌ REFUSED — runs not protocol-compliant (bench={a.bench}):"); [print("   -", x) for x in fatal]
        sys.exit(2)
    ref_role = next(iter(provs)); ref = provs[ref_role]
    for role, p in provs.items():
        for k in BC.COMPARE_KEYS:
            if str(p.get(k)) != str(ref.get(k)):
                fatal.append(f"MISMATCH '{k}': {ref_role}={ref.get(k)!r} vs {role}={p.get(k)!r}")
    if fatal:
        print(f"❌ REFUSED — these runs are NOT the same experiment (bench={a.bench}):")
        [print("   -", x) for x in fatal]
        print("   Fix: re-run the odd one(s) with the bench_config settings, then rebuild.")
        sys.exit(2)

    # 3: exact shared-N (intersection of example-ids)
    idsets = {role: set(filter(None, (rec_id(r) for r in rs))) for role, rs in rows.items()}
    shared = set.intersection(*idsets.values()) if idsets else set()
    if not shared:
        print("❌ REFUSED: no shared example-ids across the logs."); sys.exit(2)
    for role in rows:
        extra = len(idsets[role]) - len(shared)
        if extra: print(f"   note: {role} has {extra} example(s) outside the shared-N (dropped)")

    # 4: re-score on shared-N with the ONE canonical extractor + metric
    scores = {}
    for role, rs in rows.items():
        byid = {rec_id(r): r for r in rs if rec_id(r) in shared}
        scores[role] = st.mean([score_one(byid[i], metric) for i in shared])

    # 5: emit
    N = len(shared)
    print(f"✅ FAIR TABLE — bench={a.bench}, N(shared)={N}, metric={metric}")
    print(f"   settings: prompt={ref['prompt_name']}({ref['prompt_sha']}) extractor={ref['extractor']} "
          f"decoding={ref['decoding']} reason_fix={ref['reason_fix']}/{ref['reason_hist']} max_new={ref['max_new']} "
          f"ratio={ref['ratio']} doc={ref['doc_number']} trunc={ref['truncation']} order={ref['order']} seed={ref['seed']}")
    lines = [f"| method | {metric} |", "|---|--:|"]
    for role in named:
        print(f"   {role:16s} {scores[role]:.3f}"); lines.append(f"| {role} | {scores[role]:.3f} |")
    # common-sense sandwich if roles present
    def g(*names):
        for n in names:
            if n in scores: return scores[n]
        return None
    lm, ours, flo = g("teacher", "lm-teacher"), g("ours"), g("floor3b", "floor", "slm-teacher")
    if None not in (lm, ours, flo):
        ok = lm + 1e-9 >= ours >= flo - 1e-9
        clо = (ours - flo) / (lm - flo) if lm != flo else float("nan")
        print(f"   SANDWICH lm-teacher>=ours>=floor: {'✅' if ok else '❌ VIOLATED (investigate raw logs)'} | closeness={100*clо:.0f}%")
    if a.md:
        open(a.md, "w").write("\n".join(lines) + f"\n\n_N={N}, {metric}, prompt {ref['prompt_sha']}, extractor {ref['extractor']}_\n")
        print(f"   wrote {a.md}")

if __name__ == "__main__":
    main()
