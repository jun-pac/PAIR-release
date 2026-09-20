#!/usr/bin/env python
"""Did the job spend its GPU hours MEASURING, or loading models? Run after every job.

WHY (2026-08-27, user). A LoCoMo curve job ran 14 arms and took 2h24m of GPU. The measured decode
wall inside it was 54 minutes. The other ~90 minutes were the 32B being loaded from disk FOURTEEN
TIMES, because the harness is one-arm-per-process. Nothing detected it: the existing GPU-efficiency
watchdog tracks sec/example against a config's own history, which cannot see time spent OUTSIDE the
examples. So the waste was invisible until a human asked why it was slow.

This computes the one number that would have caught it:

    USEFUL FRACTION = (summed measured wall inside the result logs) / (the job's SLURM elapsed)

and fails below a threshold. It also reports how many separate model loads the job paid for, which is
the usual cause.

Usage:
  python scripts/job_efficiency_check.py 3034080 results/fusionft/lcv_*.jsonl
  python scripts/job_efficiency_check.py --min-useful 0.6 <jobid> <logs...>
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys


def elapsed_seconds(jobid):
    out = subprocess.run(["sacct", "-j", str(jobid), "--format=Elapsed", "-n", "-P"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    if not out:
        return None
    t = out[0].strip()                      # [DD-]HH:MM:SS
    days = 0
    if "-" in t:
        d, t = t.split("-", 1); days = int(d)
    h, m, s = (int(x) for x in t.split(":"))
    return days * 86400 + h * 3600 + m * 60 + s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobid")
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--min-useful", type=float, default=0.60)
    a = ap.parse_args()

    paths = sorted({p for pat in a.logs for p in glob.glob(pat)})
    if not paths:
        sys.exit(f"no logs matched {a.logs}")

    total_measured = 0.0
    per_log, loads = [], 0
    for p in paths:
        rows = [json.loads(l) for l in open(p) if l.strip()]
        if not rows:
            continue
        pv = next((r.get("_provenance") for r in rows if r.get("_provenance")), {}) or {}
        if str(pv.get("slurm_job_id")) != str(a.jobid):
            continue                                   # a log from a different job
        loads += 1                                     # one process == one model load
        walls = {}
        prefill = 0.0
        for r in rows:
            if r.get("batch_ans_s") is not None:
                walls[r["turn"]] = r["batch_ans_s"]     # one wall per turn, never per row
            elif r.get("ans_s") is not None:
                walls[(r.get("conv"), r["turn"])] = r["ans_s"]
        seen_pref = set()
        for r in rows:
            k = r.get("prefill_s_batch") or r.get("prefill_s")
            if k is not None and k not in seen_pref:
                seen_pref.add(k); prefill += k
        w = sum(walls.values()) + prefill
        total_measured += w
        per_log.append((p.split("/")[-1], len(rows), w))

    el = elapsed_seconds(a.jobid)
    if el is None:
        sys.exit(f"sacct returned nothing for job {a.jobid}")
    frac = total_measured / el if el else 0.0

    print(f"job {a.jobid}: SLURM elapsed {el/60:.1f} min")
    print(f"  measured wall inside the logs   {total_measured/60:.1f} min over {len(per_log)} arms")
    print(f"  separate processes (model loads) {loads}")
    print(f"  USEFUL FRACTION                  {frac:.1%}")
    print(f"  overhead (load / import / idle)  {(el-total_measured)/60:.1f} min"
          f"  ≈ {(el-total_measured)/max(loads,1)/60:.1f} min per process")
    for name, n, w in sorted(per_log, key=lambda x: -x[2]):
        print(f"     {name:34s} rows={n:4d} measured={w/60:6.1f} min")

    if frac < a.min_useful:
        print(f"\n🔴 USEFUL FRACTION {frac:.1%} < {a.min_useful:.0%}. The job spent most of its GPU time\n"
              f"   NOT measuring. With {loads} processes at "
              f"~{(el-total_measured)/max(loads,1)/60:.1f} min of setup each, the fix is to load the\n"
              f"   model ONCE and sweep the arms in-process (mtrag_accum --sweep-ratio / --sweep-keep),\n"
              f"   or to group arms that share a model into one process.")
        sys.exit(1)
    print(f"\n✅ useful fraction {frac:.1%}")


if __name__ == "__main__":
    main()
