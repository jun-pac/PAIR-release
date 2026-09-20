#!/usr/bin/env python
"""validate_experiment.py — THE experiment-integrity gate. Run on any result log (or a comparison SET of logs)
BEFORE any number is trusted / put in RESULTS_MASTER. It encodes the invariants that were only ever "in my head"
and that failed silently (the reason_then_answer bug: 97% straight-to-answer, `<one-line reasoning>` echoed).

Checks (per log):
  I1 REASONING     — for reason_then_answer runs, a MAJORITY of outputs must contain real reasoning before the
                     'Final Answer:' line (>= MIN_REASON_WORDS words, NOT the literal '<...reasoning...>' placeholder).
  I2 DEGENERATE    — empty preds, placeholder-echo, chat-template/role leaks ('<|im_end|>','Human:','Assistant:'),
                     repetition loops must all be rare.
  I3 PROVENANCE    — the run must record prompt name+sha, decoding mode, max_new, model, N (added by the harness).
                     Missing provenance => the run cannot be fairness-checked => NOT trustworthy for comparison.
Checks (across a comparison SET, when >1 log given):
  I4 FAIRNESS      — every log in the set shares the SAME prompt-sha, decoding mode, max_new, and example-id set.
  I5 SANDWICH      — if role tags are given (--roles lm-teacher=... ours=... slm-teacher=...), enforce
                     lm-teacher >= ours >= slm-teacher; big>=small; full>=compressed. Violation = suspect.

Exit code 0 = ALL PASS. Non-zero = at least one FAIL (the number is NOT trustworthy until fixed).
Usage:
  python scripts/validate_experiment.py LOG [LOG2 ...] [--expect-reasoning] [--metric-field acc_f1]
  python scripts/validate_experiment.py --set teacher=a.jsonl ours=b.jsonl floor=c.jsonl
"""
import sys, os, json, re, argparse, hashlib
from collections import defaultdict

MIN_REASON_WORDS = 5
REASON_FRAC_MIN = 0.50           # >=50% of outputs must show real reasoning (for reason_then_answer runs)
PLACEHOLDER_RE = re.compile(r"<\s*(one-line|brief|short)?\s*reasoning\s*>|<\s*answer\s*>|<reasoning>", re.I)
LEAK_RE = re.compile(r"<\|im_end\|>|<\|im_start\|>|\bHuman:|\bAssistant:|\buser\n|\bassistant\n")
RAWKEYS = ["raw", "generated", "output", "full_text", "gen", "completion", "response", "text"]
PREDKEYS = ["acc_pred", "pred", "prediction", "answer"]

def load(f):
    txt = open(f).read().strip()
    if not txt: return []
    return json.loads(txt) if txt[0] == "[" else [json.loads(l) for l in txt.splitlines() if l.strip()]

def rawfield(rows):
    return next((k for k in RAWKEYS if k in rows[0]), None)
def predfield(rows):
    return next((k for k in PREDKEYS if k in rows[0]), None)

def has_reasoning(raw):
    r = str(raw or "").strip()
    if PLACEHOLDER_RE.search(r): return "placeholder"          # emitted the literal template placeholder
    for marker in ("Final Answer:", "\nFinal answer:", "\nAnswer:"):
        if marker in r:
            before = r.split(marker, 1)[0].strip()
            before = PLACEHOLDER_RE.sub("", before).strip()
            return "reasoned" if len(before.split()) >= MIN_REASON_WORDS else "straight"
    # no marker at all: if it's a long free-form answer it may still be fine; treat >=MIN words as "answer, no marker"
    return "nomarker"

def repetition(raw, k=6):
    toks = str(raw or "").split()
    if len(toks) < 3*k: return False
    tail = toks[-k:]
    return toks[-2*k:-k] == tail and toks[-3*k:-2*k] == tail   # same k-gram thrice at the end = loop

def check_log(f, expect_reasoning):
    rows = load(f)
    if not rows: return {"file": f, "FAIL": ["empty log"]}
    n = len(rows); rk = rawfield(rows); pk = predfield(rows)
    fails, warns, info = [], [], {}
    # I2 degenerate (on pred + raw)
    empty = sum(1 for r in rows if pk and not str(r.get(pk) or "").strip())
    placeholder = sum(1 for r in rows if rk and PLACEHOLDER_RE.search(str(r.get(rk) or "")))
    leak = sum(1 for r in rows if rk and LEAK_RE.search(str(r.get(rk) or "")))
    loops = sum(1 for r in rows if rk and repetition(r.get(rk)))
    info.update(N=n, empty=empty, placeholder=placeholder, leak=leak, loops=loops)
    if pk and empty / n > 0.05: fails.append(f"empty preds {empty}/{n} ({100*empty/n:.0f}%)")
    if rk and placeholder / n > 0.02: fails.append(f"PLACEHOLDER-echo {placeholder}/{n} ({100*placeholder/n:.0f}%) — prompt template leaking")
    if rk and leak / n > 0.02: fails.append(f"chat/role LEAK {leak}/{n}")
    if rk and loops / n > 0.05: warns.append(f"repetition loops {loops}/{n}")
    # I1 reasoning
    if rk:
        cats = defaultdict(int)
        for r in rows: cats[has_reasoning(r.get(rk))] += 1
        reasoned = cats["reasoned"]; info["reasoned%"] = round(100*reasoned/n)
        info["reason_breakdown"] = dict(cats)
        if expect_reasoning and reasoned / n < REASON_FRAC_MIN:
            fails.append(f"REASONING SUPPRESSED: only {reasoned}/{n} ({100*reasoned/n:.0f}%) show reasoning "
                         f"(need >={int(100*REASON_FRAC_MIN)}%); breakdown={dict(cats)}")
    else:
        warns.append("no raw-text field saved -> cannot verify reasoning or degeneracy (SAVE raw going forward)")
    # I3 provenance
    prov = rows[0].get("_provenance")
    if not prov:
        warns.append("NO _provenance block (prompt sha / decoding / max_new / model) -> not fairness-checkable")
    else:
        info["provenance"] = prov
    # I3b every row's provenance, not just row 0 — a resumed log is written by more than one run
    notes, mixfails = check_mixed_provenance(rows)
    warns += notes; fails += mixfails
    return {"file": f, "FAIL": fails, "WARN": warns, "info": info}

def check_mixed_provenance(rows):
    """★ A RESUMED log carries rows from more than one run (2026-08-12). Every row has its own
    `_provenance`, but validate/build_table read only the FIRST row, so a file whose tail was produced
    under different code — or, worse, different settings — reads as homogeneous.

    Two things are reported separately because they matter differently:
      * a different `code_commit` alone is a NOTE: the resume may have run under later commits that did
        not touch the decode path.
      * a different SETTING (prompt sha, decoding, max_new, reason_hist, lam, model, batch size) is a
        FAILURE: those rows are not comparable with each other, let alone with another arm.
    Also note that a resumed run re-batches only the conversations it still owes, so its batch groupings
    differ from an uninterrupted run — a round-off-level effect on the generations, not a settings change,
    but it does mean a resumed log is not byte-reproducible from the same command.
    """
    SETTING = ("prompt_sha", "decoding", "max_new", "reason_hist", "lam", "model", "slm_model",
               "axis_batch_size", "slm_lora", "lm_lora", "ref_override")
    provs = [(r.get("_provenance") or {}) for r in rows]
    provs = [p for p in provs if p]
    if not provs:
        return [], []
    notes, fails = [], []
    commits = {p.get("code_commit") for p in provs}
    if len(commits) > 1:
        notes.append(f"rows span {len(commits)} code commits {sorted(map(str, commits))} — a resumed log; "
                     f"check that nothing between them touched the decode path")
    for k in SETTING:
        vals = {json.dumps(p.get(k), sort_keys=True) for p in provs}
        if len(vals) > 1:
            fails.append(f"rows disagree on `{k}`: {sorted(vals)} — the file mixes two configurations and "
                         f"is NOT a single experiment")
    return notes, fails

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="*")
    ap.add_argument("--set", nargs="*", default=[], help="role=log pairs for a comparison set (fairness+sandwich)")
    ap.add_argument("--expect-reasoning", action="store_true", help="assert this run uses reason_then_answer")
    ap.add_argument("--metric-field", default="acc_f1")
    a = ap.parse_args()
    named = {}
    for kv in a.set:
        role, path = kv.split("=", 1); named[role] = path
    logs = list(a.logs) + list(named.values())
    if not logs: ap.error("give at least one log (positional) or --set role=log ...")

    print("=" * 78); print("EXPERIMENT INTEGRITY VALIDATION"); print("=" * 78)
    any_fail = False; reports = {}
    for f in logs:
        rep = check_log(f, a.expect_reasoning)
        reports[f] = rep
        status = "❌ FAIL" if rep["FAIL"] else ("⚠️  WARN" if rep.get("WARN") else "✅ PASS")
        print(f"\n{status}  {f}")
        print(f"    info: {rep.get('info')}")
        for x in rep["FAIL"]: print(f"    ❌ {x}"); any_fail = True
        for x in rep.get("WARN", []): print(f"    ⚠️  {x}")

    # I4 fairness across the set
    if len(logs) > 1:
        print("\n" + "-" * 78); print("I4 FAIRNESS across the comparison set:")
        provs = {f: (load(f)[0].get("_provenance") if load(f) else None) for f in logs}
        if any(p is None for p in provs.values()):
            print("    ⚠️  provenance missing on >=1 log -> CANNOT verify same prompt/decoding/max_new. Add provenance + re-run.")
        else:
            for key in ["prompt_sha", "decoding", "max_new", "model"]:
                vals = {f: provs[f].get(key) for f in logs}
                if len(set(map(str, vals.values()))) > 1:
                    print(f"    ❌ MISMATCH {key}: {vals}"); any_fail = True
                else:
                    print(f"    ✅ same {key} = {next(iter(vals.values()))}")
        # same example set
        idsets = {}
        for f in logs:
            rows = load(f); k = "conv" if rows and "conv" in rows[0] else ("episode_id" if rows and "episode_id" in rows[0] else None)
            idsets[f] = set((r.get(k), r.get("turn")) for r in rows) if k else None
        if all(idsets.values()):
            common = set.intersection(*idsets.values())
            for f in logs:
                if len(idsets[f]) != len(common):
                    print(f"    ⚠️  {f} has {len(idsets[f])} ids vs shared {len(common)} -> score on the INTERSECTION, not per-log N")

    # I5 sandwich (needs the set with roles + metric)
    if named and all(os.path.exists(p) for p in named.values()):
        import statistics as st
        def score(f):
            rows = load(f); return st.mean([r[a.metric_field] for r in rows if a.metric_field in r]) if rows else None
        sc = {role: score(p) for role, p in named.items()}
        print("\n" + "-" * 78); print(f"I5 SANDWICH/ordering ({a.metric_field}): {{{', '.join(f'{r}={v:.3f}' for r,v in sc.items() if v is not None)}}}")
        lm, ours, slm = sc.get("lm-teacher") or sc.get("teacher"), sc.get("ours"), sc.get("slm-teacher") or sc.get("floor")
        if None not in (lm, ours, slm):
            if not (lm + 1e-9 >= ours >= slm - 1e-9):
                print(f"    ❌ SANDWICH VIOLATED: need lm-teacher({lm:.3f}) >= ours({ours:.3f}) >= floor({slm:.3f})"); any_fail = True
            else:
                clos = (ours - slm) / (lm - slm) if lm != slm else float('nan')
                print(f"    ✅ sandwich holds. closeness-to-LM = {100*clos:.0f}%")

    print("\n" + "=" * 78)
    print("VERDICT:", "❌ NOT TRUSTWORTHY — fix the FAILs before recording any number." if any_fail
          else "✅ passed integrity checks (still read the logs).")
    print("=" * 78)
    sys.exit(1 if any_fail else 0)

if __name__ == "__main__":
    main()
