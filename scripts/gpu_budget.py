#!/usr/bin/env python
"""Campaign GPU-budget ledger — the mechanical answer to the 2026-08-30 estimation failure
(musique full-set campaign: quoted ~20 GPU-h, burned ~49).

Root causes that a promise cannot fix but this gate does:
  1. the quote extrapolated from 96-question TRUNCATED runs (shorter tail -> optimistic rate);
  2. retry cost was quoted at ZERO — ladder OOMs burned ~1 h per failed ascending attempt and
     TIMEOUTs made arms be paid for twice;
  3. nothing compared actual burn against the quote WHILE the campaign ran — the overrun was
     discovered by sacct after the fact.

Mechanics (state in results/timing/gpu_budget.json):
  quote <campaign> --hours H --rate-source "<log/job the per-example rate was MEASURED from>"
        REFUSES a quote without a rate source. Stores H (the caller must already include the
        1.5x failure margin — say so in --rate-source).
  add   <campaign> <jobid>...      attach every sbatch of the campaign the moment it is submitted
  check <campaign>                 sums sacct elapsed x ALLOCATED GPUs over attached jobs;
        EXIT 1 the moment spent > quote — meaning: STOP SUBMITTING, report the overrun to the
        user, re-quote with their approval. Exit 2 = campaign unknown (quote first).
  list                             show all campaigns, spent vs quote.
"""
import json
import os
import subprocess
import sys
import time

PATH = "results/timing/gpu_budget.json"


def load():
    return json.load(open(PATH)) if os.path.exists(PATH) else {}


def save(d):
    json.dump(d, open(PATH, "w"), indent=2)


def spent_hours(jobs):
    if not jobs:
        return 0.0
    out = subprocess.run(
        # ★ AllocTRES, not just ElapsedRaw: a 2-GPU job costs TWICE its elapsed time and this
        # ledger used to bill it once. Found 2026-09-03 auditing the two-card campaign, which the
        # ledger reported at 1.8 GPU-h against a true 2.9 -- a 1.6x undercount on exactly the
        # campaign that has a hard user-set cap. A budget tool that under-reports is worse than no
        # budget tool, because it reports safety.
        ["sacct", "-j", ",".join(jobs), "-X", "-n", "-P",
         "--format=JobID,ElapsedRaw,AllocTRES%80"],
        capture_output=True, text=True).stdout
    import re
    gpuh = 0.0
    for line in out.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 3 or not parts[1].isdigit():
            continue
        m = re.search(r"gres/gpu=(\d+)", parts[2])
        gpuh += int(parts[1]) * (int(m.group(1)) if m else 1) / 3600.0
    return gpuh


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cmd, d = sys.argv[1], load()
    if cmd == "quote":
        camp = sys.argv[2]
        args = dict(zip(sys.argv[3::2], sys.argv[4::2]))
        hours, src = args.get("--hours"), args.get("--rate-source", "")
        if not hours or len(src) < 20:
            print("REFUSED: a quote needs --hours and a real --rate-source naming the "
                  "MEASURED per-example rate (log/job) it came from, incl. the 1.5x margin.")
            sys.exit(1)
        d[camp] = dict(quote_h=float(hours), rate_source=src, jobs=d.get(camp, {}).get("jobs", []),
                       ts=time.strftime("%Y-%m-%d %H:%M"))
        save(d)
        print(f"quoted {camp}: {hours} GPU-h ({src})")
    elif cmd == "add":
        camp = sys.argv[2]
        if camp not in d:
            print(f"unknown campaign {camp} — quote it first")
            sys.exit(2)
        d[camp]["jobs"] = sorted(set(d[camp]["jobs"]) | set(sys.argv[3:]))
        save(d)
        print(f"{camp}: {len(d[camp]['jobs'])} jobs attached")
    elif cmd == "check":
        camp = sys.argv[2]
        if camp not in d:
            print(f"unknown campaign {camp} — quote it first")
            sys.exit(2)
        s, q = spent_hours(d[camp]["jobs"]), d[camp]["quote_h"]
        print(f"{camp}: spent {s:.1f} / quoted {q:.1f} GPU-h ({100 * s / q:.0f}%)")
        if s > q:
            print("OVER BUDGET — do not submit more jobs for this campaign; report the overrun "
                  "and re-quote with user approval.")
            sys.exit(1)
    elif cmd == "list":
        for camp, v in d.items():
            s = spent_hours(v["jobs"])
            flag = " ⚠OVER" if s > v["quote_h"] else ""
            print(f"{camp:24s} {s:6.1f} / {v['quote_h']:.1f} GPU-h "
                  f"({len(v['jobs'])} jobs){flag}")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
