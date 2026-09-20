#!/usr/bin/env python
"""Time to first token along the context axis at batch 1 -> results/timing/context_ttft_b1.json (2026-09-14, user).

Reads results/fusionft/cttf_d{D}_{teacher,floor7,ours}.jsonl (scripts/run_context_ttft_b1.slurm: one depth per
job, three arms on one card, the first 24 questions of the 96-question throughput set, FKV, FORCE_BATCHED=1
BATCH_SIZE=1, YaRN x4 from 200 documents). The quantity is `ttft_from_prefill_s`: ONE wall per question from the
synchronized instant before the context prefill to the first generated token on the host (every branch's prefill
+ question forward + decode step 0). Per arm and depth: n, median, mean, min, max, p90; beside it the two
partial walls the harness also stamps (prefill_s_batch, ttft_s = question forward + step 0) for reading, never
summed. x = the axis's ctx_ktok (results/timing/context_axis.json), the same x as the other two panels.

Refuses (prints REFUSED and skips the arm) unless every row ran at batch_rows == 1 on the batched path; prints
the node of every arm and flags a depth whose three arms are not on one node.
"""
import glob
import json
import os
import statistics as st
import sys

ARMS = ("teacher", "ours", "floor7")      # file suffixes; stored under teacher / ours / floor
KEYS = ("teacher", "ours", "floor")
OUT = "results/timing/context_ttft_b1.json"


def load(path):
    t = [json.loads(l) for l in open(path) if l.strip()]
    return [r for r in t if "acc_f1" in r]


def stats(v):
    v = sorted(v)
    return dict(n=len(v), median=round(st.median(v), 4), mean=round(st.fmean(v), 4), min=round(v[0], 4),
                max=round(v[-1], 4), p90=round(v[min(len(v) - 1, int(round(0.9 * (len(v) - 1))))], 4))


def main():
    axis = json.load(open("results/timing/context_axis.json"))
    store = json.load(open(OUT)) if os.path.exists(OUT) else {}
    depths = sorted({int(os.path.basename(f).split("_")[1][1:]) for f in glob.glob("results/fusionft/cttf_d*_*.jsonl")})
    for D in depths:
        k = f"d{D}"
        e = store.get(k, {})
        e["docs"] = D
        e["ctx_ktok"] = axis[k]["ctx_ktok"]
        e["yarn"] = bool(axis[k].get("yarn"))
        for arm in ARMS:
            f = f"results/fusionft/cttf_{k}_{arm}.jsonl"
            if not os.path.exists(f):
                continue
            rows = load(f)
            if not rows:
                continue
            pv = rows[0]["_provenance"]
            br = sorted({r.get("batch_rows") for r in rows})
            if br != [1] or pv.get("axis_batch_size") != 1 or pv.get("axis_decode_path") != "batched":
                print(f"REFUSED {k} {arm}: batch_rows={br} axis_batch_size={pv.get('axis_batch_size')} "
                      f"path={pv.get('axis_decode_path')} (needs 1 / 1 / batched)")
                continue
            v = [r["ttft_from_prefill_s"] for r in rows if r.get("ttft_from_prefill_s") is not None]
            if len(v) != len(rows):
                print(f"REFUSED {k} {arm}: {len(rows) - len(v)} rows without ttft_from_prefill_s")
                continue
            key = "floor" if arm == "floor7" else arm      # store keys as the other context stores: teacher / ours / floor
            e[key] = dict(batch=1, path="fkv", node=pv.get("slurm_node"), job=pv.get("slurm_job_id"),
                          yarn=pv.get("axis_rope_yarn_factor"),
                          ttft=stats(v),
                          prefill_s_median=round(st.median(r["prefill_s_batch"] for r in rows), 4),
                          q_and_step0_s_median=round(st.median(r["ttft_s"] for r in rows), 4),
                          ctx_tok_median=int(st.median(r["acc_ctx_tok"] for r in rows)),
                          gen_tok_mean=None)
        nodes = {e[a]["node"] for a in KEYS if a in e}
        e["one_node"] = len(nodes) == 1
        store[k] = e
    os.makedirs("results/timing", exist_ok=True)
    json.dump(store, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")
    print("  depth  ctx    | TTFT median s  teacher / ours / floor  (n)   | min-max teacher | min-max ours | node(s)")
    for k in sorted(store, key=lambda s: int(s[1:])):
        e = store[k]
        med = " / ".join(f"{e[a]['ttft']['median']:.3f}" if a in e else "—" for a in KEYS)
        n = ",".join(str(e[a]["ttft"]["n"]) for a in KEYS if a in e)
        mm = lambda a: (f"{e[a]['ttft']['min']:.2f}-{e[a]['ttft']['max']:.2f}" if a in e else "—")
        nodes = ",".join(sorted({e[a]["node"] or "?" for a in KEYS if a in e}))
        print(f"  {k:5s} {e['ctx_ktok']:5.1f}k | {med:28s} ({n}) | {mm('teacher'):12s} | {mm('ours'):12s} | {nodes}"
              + ("" if e["one_node"] else "  ★ MIXED NODES"))
    if "--ratio" in sys.argv:
        for k in sorted(store, key=lambda s: int(s[1:])):
            e = store[k]
            if "teacher" in e and "ours" in e:
                print(f"  {k}: teacher ÷ ours = ×{e['teacher']['ttft']['median'] / e['ours']['ttft']['median']:.2f} (medians, same node)")


if __name__ == "__main__":
    main()
