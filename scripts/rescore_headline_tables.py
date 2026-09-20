#!/usr/bin/env python
"""Re-score every headline table on ONE metric basis (2026-08-24).

Why: three normalization changes landed this week (hyphen->space, digit-comma->join [REGRESSION],
digit-comma->space [fix]). Published tables were computed under whichever basis was live at the
time, so they no longer share a basis. This recomputes each arm's headline number from its stored
predictions with the CURRENT src.eval metric and prints old vs new, so every table can be restated
on one basis. CPU only, no GPU, no re-generation.
"""
import json
import statistics
import sys

sys.path.insert(0, ".")
from src.eval import compute_best_em_f1
from scripts.bench_config import BENCHMARKS

ACCUM = {
    "LoCoMo-30 (λ0.85)": [("teacher", "c30_teacher32b"), ("floor-7B", "c30_floor7b"),
                          ("plain", "c30b3_7b_plain_LMNA"), ("S-solo", "c30b3_7b_Ssolo_LMNA"),
                          ("L-fusion", "c30b3_7b_SsoloLfus_LMNA")],
    "QASPER-accum (λ0.85)": [("teacher", "qab3_teacher32b"), ("floor-7B", "qab3_floor7b"),
                             ("plain", "qab3_plain_LMNA"), ("S-solo", "qab3_Ssolo_LMNA"),
                             ("L-fusion", "qab3_SsoloLfus_LMNA")],
    "LooGLE-accum (λ0.7)": [("teacher", "lg_teacher"), ("floor-7B", "lg_floor7"),
                            ("plain", "lg_plain"), ("S-solo λ0.95", "lg_Ssolo_l095"),
                            ("L-fusion λ0.7", "lg_SsoloLfus_l07"),
                            ("snapKV 40%", "lg_snapkv_r60"),
                            ("snapKV 21.9%", "lg_snapkv_r78125"), ("SpecPrefill 21.9%", "lg_spec_k21875")],
}


def load(tag):
    d = {}
    for l in open(f"results/fusionft/{tag}.jsonl"):
        if l.strip():
            r = json.loads(l)
            d[(r["conv"], int(r["turn"]))] = r
    return d


def main():
    for bench, arms in ACCUM.items():
        try:
            logs = {name: load(tag) for name, tag in arms}
        except FileNotFoundError as e:
            print(f"\n== {bench}: SKIP ({e.filename} missing)")
            continue
        shared = set.intersection(*[set(v) for v in logs.values()])
        print(f"\n== {bench} · shared-N={len(shared)}")
        print(f"{'arm':20s} {'logged (mixed basis)':>21s} {'current metric':>15s} {'delta':>8s}")
        for name, _ in arms:
            d = logs[name]
            old = statistics.mean(d[k]["acc_f1"] for k in shared)
            new = statistics.mean(compute_best_em_f1(d[k]["acc_pred"], [d[k]["gold"]])[1] for k in shared)
            print(f"{name:20s} {old:21.4f} {new:15.4f} {new-old:+8.4f}")


if __name__ == "__main__":
    main()
