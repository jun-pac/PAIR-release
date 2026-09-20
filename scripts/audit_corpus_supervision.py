#!/usr/bin/env python
"""Is this training corpus GOLD-SUPERVISED? Run before any adapter trained on it is put in a table.

WHY. `reader_binding_v2_r16_s600` surfaced as the best LoCoMo arm in the project (78.0% closeness at λ0.7)
and it is VOID: v1-v4 were built by `gen_binding_corpus_v3.py`, whose synthetic rows set

    target = f"Step 1 (Reasoning): ...\\nStep 2 (Answer):\\nFinal Answer: {gold}"

— an f-string written by the GENERATOR with the gold value interpolated. No model ever produced it. The
doctrine is behaviour cloning: supervision is the teacher's own generation and gold is never used. A
gold-supervised adapter is not a method result, however good its number.

THE TEST. Two signals, and the second is what decides:
  1. how often the gold string appears verbatim in the target — near 100% is suspicious but NOT proof, since
     a teacher that answers correctly also contains the gold (v5's replay slice is 87% and is clean),
  2. whether a model produced the target at all — `teacher_text` present in the row, OR the row's target was
     copied from a teacher-gen corpus. Rows failing BOTH are gold.

The `replay` slice is the trap: `gen_binding_corpus_v3.py` writes `target=r["teacher_text"]` from the ftmix
teacher-14B corpus and does NOT carry the `teacher_text` field forward, so a "no teacher_text ⇒ gold" rule
misclassifies it. Hence per-slice reporting, and hence this file rather than a one-off heuristic.

Usage:
  audit_corpus_supervision.py results/fusionft/binding_corpus_*.jsonl
"""
import argparse
import collections
import json
import os
import sys

# corpora whose targets are generator-written f-strings containing the gold (verified by reading
# gen_binding_corpus_v3.py::questions). Any adapter trained on these is VOID as a method claim.
KNOWN_GOLD = {"binding_corpus_v1.jsonl", "binding_corpus_v1b_seed8.jsonl", "binding_corpus_v2.jsonl",
              "binding_corpus_v3.jsonl", "binding_corpus_v4.jsonl", "binding_corpus_v4_synthetic.jsonl"}
# EXACT adapter-directory stems, not prefixes: "reader_binding_v1" is a prefix of "reader_binding_v10"
# and wrongly voided v10/v11/v12 in the results board until 2026-08-12.
VOID_ADAPTERS = ("reader_binding_v1_r16_s600", "reader_binding_v1_3b_r16_s600", "reader_binding_v2_r16_s600",
                 "reader_binding_v3_r16_s600", "reader_binding_v4_r16_s900", "teacher32b_binding_v4_r16_s900",
                 "stage2_lm_binding_v1_r16_s600")


def audit(path, limit=2000):
    rows = []
    for i, line in enumerate(open(path)):
        if i >= limit:
            break
        r = json.loads(line)
        if str(r.get("target", "")).strip():
            rows.append(r)
    if not rows:
        return None
    by = collections.defaultdict(lambda: [0, 0, 0])
    for r in rows:
        slice_ = (r.get("meta") or {}).get("fmt") or (r.get("meta") or {}).get("kind") or "synthetic"
        slice_ = "replay" if slice_ == "replay" else "synthetic"
        gold = str(r.get("gold", "")).strip().lower()
        b = by[slice_]
        b[0] += 1
        b[1] += bool(gold) and gold in str(r["target"]).lower()
        b[2] += "teacher_text" in r
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpora", nargs="+")
    a = ap.parse_args()
    bad = 0
    print(f"{'corpus':44s} {'slice':10s} {'rows':>6s} {'gold-in-target':>15s} {'teacher_text':>13s}  verdict")
    for p in a.corpora:
        by = audit(p)
        base = os.path.basename(p)
        if by is None:
            print(f"{base:44s} {'-':10s} {'0':>6s} {'(no targets yet)':>15s}")
            continue
        for slice_, (n, hit, tt) in sorted(by.items()):
            known = base in KNOWN_GOLD
            gold_like = known or (tt == 0 and hit / n > 0.98 and slice_ != "replay")
            verdict = "🔴 GOLD-SUPERVISED — VOID as a method arm" if gold_like else "✅ teacher-distilled"
            bad += gold_like
            print(f"{base:44s} {slice_:10s} {n:6d} {hit/n:14.0%} {tt/n:12.0%}  {verdict}")
    if bad:
        print(f"\n{bad} gold-supervised slice(s). Adapters trained on them must not appear in any results "
              f"table. Known-void adapter prefixes: {', '.join(VOID_ADAPTERS)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
