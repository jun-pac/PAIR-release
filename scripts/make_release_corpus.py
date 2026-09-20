#!/usr/bin/env python
"""Cut the two shipped views of the fusion training corpus out of the full one.

The full corpus is 133 MB because 96% of it is the `context` field. Two views are released instead:

  binding_corpus_v5.supervision.jsonl   all 7,280 rows, every field except `context`
                                        (id, question, gold, the teacher's target, instruction name,
                                        source) plus `context_chars`, so every training item and the
                                        supervision it carries is visible.
  binding_corpus_v5_sample.jsonl        200 rows WITH their context, stratified over the three
                                        families in the corpus's own proportions, seed 0.

Rebuilding the contexts is described in data/README.md.
"""
import json, argparse, random, collections

FAMILY = lambda r: (r["meta"].get("fmt") or "dialogue")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="results/fusionft/binding_corpus_v5.jsonl")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.corpus)]
    by = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by[FAMILY(r)].append(i)

    sup = f"{a.out_dir}/binding_corpus_v5.supervision.jsonl"
    with open(sup, "w") as o:
        for r in rows:
            d = {k: v for k, v in r.items() if k != "context"}
            d["context_chars"] = len(r["context"])
            o.write(json.dumps(d, ensure_ascii=False) + "\n")

    # proportional over families, then a seeded shuffle inside each one
    take, left = [], a.sample
    fams = sorted(by, key=lambda f: -len(by[f]))
    for j, f in enumerate(fams):
        k = left if j == len(fams) - 1 else round(a.sample * len(by[f]) / len(rows))
        idx = by[f][:]
        random.Random(a.seed + j).shuffle(idx)
        take += idx[:k]
        left -= k
    take.sort()

    smp = f"{a.out_dir}/binding_corpus_v5_sample.jsonl"
    with open(smp, "w") as o:
        for i in take:
            o.write(json.dumps(rows[i], ensure_ascii=False) + "\n")

    print(f"{sup}: {len(rows)} rows")
    print(f"{smp}: {len(take)} rows, families " +
          ", ".join(f"{f} {sum(1 for i in take if FAMILY(rows[i]) == f)}" for f in fams))


if __name__ == "__main__":
    main()
