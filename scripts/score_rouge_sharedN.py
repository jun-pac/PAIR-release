#!/usr/bin/env python
"""Shared-N rougeLsum scorer for LongBench summarization logs (gov_report / multi_news).

Each result row stores the per-example score at teacher.metrics.rougeLsum (single-model
'teacher' logs) or ours.metrics.rougeLsum (fusion logs). We average over the INTERSECTION
of example_ids across all given files (fair shared-N), and also print mean output length
and the fraction whose first sentence is a meta-opener ("This report/document examines...").

Usage: score_rouge_sharedN.py a.jsonl b.jsonl [c.jsonl ...]
"""
import json
import re
import sys

META_OPENER = re.compile(r"^\s*(the |this )?(report|document|gao|study|paper)\b.{0,40}\b(examine|discuss|provide|present|describe|analyze|review|address|focus|cover|detail|explore|overview)", re.I)


def load(path):
    """Return {example_id: (rougeLsum, extracted_text)} for whichever role the file has."""
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            eid = r.get("example_id") or r.get("id")
            role = None
            for cand in ("ours", "slm_lm", "teacher", "slm", "lm"):
                if isinstance(r.get(cand), dict) and "metrics" in r[cand]:
                    role = cand
                    break
            if role is None:
                continue
            blk = r[role]
            m = blk["metrics"]
            if isinstance(m, str):
                m = eval(m)  # logs store the dict repr
            rl = float(m.get("rougeLsum", 0.0))
            txt = blk.get("extracted") or blk.get("text") or ""
            out[eid] = (rl, txt)
    return out


def main():
    paths = sys.argv[1:]
    if len(paths) < 1:
        print("usage: score_rouge_sharedN.py a.jsonl [b.jsonl ...]")
        sys.exit(1)
    data = {p: load(p) for p in paths}
    ids = set.intersection(*[set(d.keys()) for d in data.values()]) if data else set()
    n = len(ids)
    print(f"shared-N = {n}\n")
    print(f"{'file':<52} {'rougeLsum':>10} {'avg_words':>10} {'meta_open%':>10}")
    rows = []
    for p in paths:
        d = data[p]
        rls = [d[i][0] for i in ids]
        wls = [len(d[i][1].split()) for i in ids]
        mo = [1 if META_OPENER.search(d[i][1]) else 0 for i in ids]
        score = sum(rls) / n if n else 0.0
        rows.append((p, score))
        name = p.split("/")[-1]
        print(f"{name:<52} {score:>10.2f} {sum(wls)/n:>10.1f} {100*sum(mo)/n:>9.0f}%")
    if len(rows) == 2:
        diff = rows[0][1] - rows[1][1]
        print(f"\n{rows[0][0].split('/')[-1]} - {rows[1][0].split('/')[-1]} = {diff:+.2f}  "
              f"({'first>second' if diff > 0 else 'first<second' if diff < 0 else 'tie'})")


if __name__ == "__main__":
    main()
