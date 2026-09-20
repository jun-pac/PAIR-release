#!/usr/bin/env python
"""results/timing/context_throughput.json — THROUGHPUT ALONG THE CONTEXT AXIS (2026-09-04).

WHAT IT SCORES. Job 3081142 (`scripts/run_context_throughput.slurm`): teacher-32B / ours 32B+7B /
floor-7B at 40, 80, 120 and 160 hotpotQA documents, the SAME 96 questions at every depth, twelve
arms in ONE job on ONE node, each arm on a descending batch ladder. The accuracy runs of the same
axis (`results/timing/context_axis.json`) could not yield a rate: one arm per job on whatever node
it landed on, and ladders that descended mid-run so a turn carried two answer walls.

WHAT A NUMBER HERE IS. `answers/s` = questions ÷ the one end-to-end wall (`conv_wall_s`: context
prefill + decode + per-turn commit, nothing outside it), from `scripts/throughput_eval.measure()`,
the only sanctioned timing path. Two rates are written for every arm and they are the same number
whenever the batch divides 96:
  ans_s       every answer over every batch's wall (the makespan rate);
  ans_s_full  complete batches only — the steady-state rate, which is what the figure plots,
              because a partial last batch is a property of `96 mod B` and not of the arm.
The batch beside every rate is the one that RAN (`axis_batch_size`); `oom_at` is the rung above it
that failed, read from the job's own stdout, so the record says which of two facts B is:
`bracketed` (OOM'd one rung up) or `lower bound` (survived the top rung, never pushed).

WHAT THIS SCRIPT DOES NOT DO. It prints the three self-checks a throughput table owes — one node,
one workload, batches that differ across arms — as DATA next to the decision rule. It asserts no
conclusion (CANONICAL_RUNS §2y).
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_throughput_report import collect  # noqa: E402

LOGDIR = "/work/hdd/myproject/anon/slurm_logs"
DEPTHS = (40, 80, 120, 160)
ARM = {"teacher": "teacher", "ours": "ours", "floor7": "floor"}


def rungs_tried(job, prefix="ctp"):
    """(depth, arm) -> sorted list of batches that OOM'd, from the job's own stdout."""
    hits = glob.glob(f"{LOGDIR}/*-{job}.out")
    out = {}
    for log in hits:
        for ln in open(log, errors="replace"):
            m = re.match(rf"!! {prefix} d(\d+) (\S+) OOM at B=(\d+)", ln)
            if m:
                out.setdefault((int(m.group(1)), m.group(2)), []).append(int(m.group(3)))
    return {k: sorted(v) for k, v in out.items()}


def main():
    # --prefix ctpf (2026-09-06): the same twelve arms on the FKV production path
    # (scripts/run_context_throughput_fkv.slurm) -> results/timing/context_throughput_fkv.json.
    prefix = "ctp"
    if "--prefix" in sys.argv:
        prefix = sys.argv[sys.argv.index("--prefix") + 1]
    out_path = {"ctp": "results/timing/context_throughput.json",
                "ctpf": "results/timing/context_throughput_fkv.json",
                # ctpy (2026-09-11): the EIGHT-depth table, d40-d160 native + d200-d320 with YaRN x4,
                # one job on one node (scripts/run_context_throughput_yarn8.slurm). Its own store;
                # the canonical four-depth ctpf cells are untouched.
                "ctpy": "results/timing/context_throughput_yarn8.json"}.get(
        prefix, f"results/timing/context_throughput_{prefix}.json")
    axis = json.load(open("results/timing/context_axis.json"))
    # the label for a depth the accuracy scorer has not seen yet: the document block's median
    # tokens + ~1.0k of instruction and question, marked provisional (score_context_axis.YARN_DEPTHS)
    prov_ktok = {200: 29.0, 240: 34.8, 280: 40.7, 320: 46.5}
    depths = DEPTHS if prefix != "ctpy" else sorted({int(re.match(r".*_d(\d+)_", f).group(1))
                                                     for f in glob.glob(f"results/fusionft/{prefix}_d*_*.jsonl")})
    store, jobs = {}, set()
    # ctpy, per-depth jobs (2026-09-12): the one-job eight-depth run (3129892, 5 h 30 wall) sat in the queue
    # for a day with an estimated start two days out, while <=2.5 h jobs backfilled in minutes, so the YaRN
    # depths were resubmitted one depth per job (3134474-3134477) and the four native depths are NOT re-run:
    # they are the canonical ctpf cells (job 3102806, gh089), copied in here with their node and job. The
    # table's basis is therefore ONE NODE PER DEPTH (every ratio is same-node) and not one node per table;
    # what licenses that: the batch sweep on gh014 reproduced ctpf's gh089 top rungs within 1% (RESULTS_MASTER
    # 2026-09-11b), so the node-to-node spread on these GH200 cards is below the figure's resolution.
    if prefix == "ctpy" and os.path.exists("results/timing/context_throughput_fkv.json"):
        F = json.load(open("results/timing/context_throughput_fkv.json"))
        for k, e in F.items():
            if k.startswith("d") and int(k[1:]) not in depths:
                store[k] = dict(e, native_from="ctpf, job 3102806 (context_throughput_fkv.json), not re-run")
                depths.append(int(k[1:]))
        depths = sorted(depths)
    for d in depths:
        arms = collect(f"{prefix}_d{d}")
        if not arms:
            print(f"  d{d}: no scoreable arms")
            continue
        e = {"ctx_ktok": (axis[f"d{d}"]["ctx_ktok"] if f"d{d}" in axis else prov_ktok.get(d)),
             "docs": d, "ctx_ktok_provisional": f"d{d}" not in axis}
        for tag, m in arms.items():
            jobs.add(m["job"])
            try:      # the YaRN stamp, from the file's own provenance (None on the native depths)
                _pv = json.loads(open(f"results/fusionft/{m['file']}").readline())["_provenance"]
                e["yarn"] = _pv.get("axis_rope_yarn_factor")
            except Exception:
                pass
            e[ARM.get(tag, tag)] = dict(
                batch=m["batch"], ans_s=round(m["ans_s"], 4),
                ans_s_full=round(m["ans_s_full"] or m["ans_s"], 4),
                tail_rows=m["tail_rows"], n_batches=m["n_batches"],
                n_full_batches=m["n_full_batches"], answers=m["rows"],
                e2e_s=round(m["e2e"], 1), prefill_s=round(m["prefill"], 1),
                prefill_s_per_q=round(m["prefill"] / m["rows"], 3),
                f1_mean=round(m["inline_f1"], 4),
                correct_share={str(t): m["correct_share"][t] for t in (1.0, 0.8, 0.5)},
                node=m["node"], job=m["job"], file=m["file"])
        store[f"d{d}"] = e
    tried = {}
    for j in jobs:
        tried.update(rungs_tried(j, prefix))
    for k, e in store.items():
        d = e["docs"]
        for tag, arm in ARM.items():
            if arm not in e:
                continue
            oom = tried.get((d, tag), [])
            e[arm]["oom_at"] = oom[0] if oom else None
            e[arm]["bound"] = "bracketed" if oom else "lower bound - never OOM'd"
    store["_source"] = ("scripts/score_context_throughput.py — throughput_eval.measure() over the "
                        f"{prefix}_d*_{{teacher,ours,floor7}} logs of job(s) "
                        + ", ".join(sorted(jobs)) + "; hotpotQA d40 s42 reference, the first 96 "
                        "questions at every depth; complete-batch rate beside the all-batch rate."
                        + (" YaRN depths one job per depth (2026-09-12); native depths copied from the canonical "
                           "ctpf store (job 3102806, gh089); one node per depth." if prefix == "ctpy" else ""))
    if not any(k.startswith("d") for k in store):
        print(f"  nothing scoreable for prefix {prefix}; {out_path} not written"); return
    os.makedirs("results/timing", exist_ok=True)
    json.dump(store, open(out_path, "w"), indent=1)
    print(f"wrote {out_path}" + ("" if prefix == "ctp" else f"  (decode path: FKV, prefix {prefix})"))

    # ---- the table, and the three checks as data -------------------------------------------
    print("  context      | teacher-32B          | ours 32B+7B          | floor-7B             | ours/teacher")
    print("               | B    ans/s  (all)    | B    ans/s  (all)    | B    ans/s  (all)    | complete-batch")
    for k in (f"d{d}" for d in depths):
        if k not in store:
            continue
        e = store[k]
        cells = []
        for arm in ("teacher", "ours", "floor"):
            a = e.get(arm)
            if not a:
                cells.append("no rung fit (OOM at %s)" % (tried.get((e["docs"], next(t for t, x in ARM.items() if x == arm)), ["?"])[0]))
                continue
            mark = "" if a["oom_at"] else ">="
            cells.append("%-4s %-6.3f (%-6.3f)" % (f"{mark}{a['batch']}", a["ans_s_full"], a["ans_s"]))
        print("  %4.1fk %3d docs | %s | %s | %s | %s" % (
            e["ctx_ktok"], e["docs"], cells[0], cells[1], cells[2],
            ("x%.2f" % (e["ours"]["ans_s_full"] / e["teacher"]["ans_s_full"])) if "teacher" in e and "ours" in e else "—"))
    print("  prefill s per question (sum of batch prefill walls / 96, at the arm's batch):")
    for k in (f"d{d}" for d in depths):
        if k in store:
            e = store[k]
            print("    %4.1fk  teacher %s  ours %s  floor %s" % (
                e["ctx_ktok"], *("%.2f" % e[a]["prefill_s_per_q"] if a in e else "—" for a in ("teacher", "ours", "floor"))))
    nodes = {e[a]["node"] for e in store.values() if isinstance(e, dict) and "docs" in e
             for a in ("teacher", "ours", "floor") if a in e}
    per_depth = {k: sorted({e[a]["node"] for a in ("teacher", "ours", "floor") if a in e})
                 for k, e in store.items() if isinstance(e, dict) and "docs" in e}
    answers = {e[a]["answers"] for e in store.values() if isinstance(e, dict) and "docs" in e
               for a in ("teacher", "ours", "floor") if a in e}
    tails = [(k, a, e[a]["tail_rows"]) for k, e in store.items() if isinstance(e, dict) and "docs" in e
             for a in ("teacher", "ours", "floor") if a in e and e[a]["tail_rows"]]
    mixed_depths = [k for k, v in per_depth.items() if len(v) > 1]
    print(f"  nodes: {sorted(nodes)} -> " + ("one node" if len(nodes) == 1 else
          (f"one node PER DEPTH ({', '.join(f'{k}:{v[0]}' for k, v in sorted(per_depth.items(), key=lambda kv: int(kv[0][1:])))}) — every ratio is same-node"
           if not mixed_depths else f"MIXED NODES WITHIN A DEPTH {mixed_depths}, NOT A TABLE")))
    print(f"  answers per arm: {sorted(answers)} -> "
          f"{'one workload' if len(answers) == 1 else 'DIFFERENT WORKLOADS'}")
    print(f"  arms with a remainder batch: {tails or 'none'}")


if __name__ == "__main__":
    main()
