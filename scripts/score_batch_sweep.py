#!/usr/bin/env python
"""results/timing/batch_sweep_fkv.json — THROUGHPUT AS A FUNCTION OF BATCH (2026-09-11).

WHAT IT SCORES. scripts/run_batch_sweep_fkv.slurm: teacher-32B / ours 32B+7B / floor-7B at every
batch from 2 up to the rung that OOMs, hotpotQA d40 s42 at 40 and 160 documents (6.1k / 23.3k median
tokens), the SAME 96 questions everywhere, production FKV path, one job on one node. One file per
(depth, arm, batch): bsw_d<D>_<arm>_b<B>.jsonl.

WHAT A NUMBER HERE IS. Every rate is throughput_eval.measure() over one file: `ans_s` = 96 ÷ the sum
of every batch's whole wall (context prefill + decode + commit, one timer), `ans_s_full` = the same on
complete batches only (identical whenever B divides 96; the store carries both and the figure plots
the complete-batch rate, as every published throughput number does). `batch` is the batch that RAN
(axis_batch_size, checked against the file name). `oom` per arm is the rung that failed, read from the
job's own stdout ("=== SWEEP OOM <arm>_b<B>" for the swept single-model arms, "!! bsw d<D> ours OOM at
B=<B>" for ours), so the curve's end is a measured fact and not an assumption.

The script prints the three checks a throughput table owes — one node, one workload, the rung equals
the recorded batch — as data beside the decision rule; it asserts nothing else.
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from throughput_eval import load, measure  # noqa: E402

LOGDIR = "/work/hdd/myproject/anon/slurm_logs"
ARM = {"teacher": "teacher", "ours": "ours", "floor7": "floor"}
_TOK = None


def tok():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct",
                                             cache_dir=os.environ.get("HF_HOME", "/work/hdd/myproject/anon/hf"))
    return _TOK


def ooms(jobs, prefix="bsw"):
    """(depth, arm) -> sorted OOM rungs from the job logs."""
    out = {}
    for j in jobs:
        for log in glob.glob(f"{LOGDIR}/*-{j}.out"):
            depth = None
            for ln in open(log, errors="replace"):
                m = re.match(r"=== BSW d(\d+) ", ln)
                if m:
                    depth = int(m.group(1))
                m = re.match(r"=== SWEEP OOM (teacher|floor7)_b(\d+) B=(\d+)", ln)
                if m and depth is not None:
                    out.setdefault((depth, ARM[m.group(1)]), []).append(int(m.group(3)))
                m = re.match(rf"!! {prefix} d(\d+) ours OOM at B=(\d+)", ln)
                if m:
                    out.setdefault((int(m.group(1)), "ours"), []).append(int(m.group(2)))
    return {k: sorted(v) for k, v in out.items()}


def main():
    # --prefix bsw4 (2026-09-11): the four-depth one-node run -> results/timing/batch_sweep_fkv_4depth.json;
    # the default bsw (two depths, gh014) -> batch_sweep_fkv.json. Two stores, never merged.
    prefix = sys.argv[sys.argv.index("--prefix") + 1] if "--prefix" in sys.argv else "bsw"
    out_path = {"bsw": "results/timing/batch_sweep_fkv.json",
                "bsw4": "results/timing/batch_sweep_fkv_4depth.json"}.get(prefix, f"results/timing/batch_sweep_{prefix}.json")
    ctx = {}
    if os.path.exists("results/timing/context_axis.json"):
        X = json.load(open("results/timing/context_axis.json"))
        ctx = {int(k[1:]): X[k]["ctx_ktok"] for k in X if k.startswith("d")}
    store, jobs, nodes, answers, bad = {}, set(), set(), set(), []
    for f in sorted(glob.glob(f"results/fusionft/{prefix}_d*_*_b*.jsonl")):
        m = re.match(rf"{prefix}_d(\d+)_(teacher|ours|floor7)_b(\d+)\.jsonl", os.path.basename(f))
        if not m or not os.path.getsize(f):
            continue
        d, arm, b = int(m.group(1)), ARM[m.group(2)], int(m.group(3))
        rows, pv = load(f)
        rows = [r for r in rows if "acc_f1" in r]
        if len(rows) != 96:
            bad.append(f"{os.path.basename(f)}: {len(rows)} rows, not 96 — not scored"); continue
        if pv.get("axis_batch_size") != b:
            bad.append(f"{os.path.basename(f)}: file says B={b}, provenance says {pv.get('axis_batch_size')}")
        mm = measure(rows, tok())
        jobs.add(pv.get("slurm_job_id")); nodes.add(pv.get("slurm_node")); answers.add(mm["rows"])
        e = store.setdefault(f"d{d}", {"docs": d, "ctx_ktok": ctx.get(d)})
        e.setdefault(arm, {})[str(b)] = dict(
            batch=b, ans_s=round(mm["ans_s"], 4), ans_s_full=round(mm["ans_s_full"] or mm["ans_s"], 4),
            tail_rows=mm["tail_rows"], n_batches=mm["n_batches"], n_full_batches=mm["n_full_batches"],
            e2e_s=round(mm["e2e"], 1), prefill_s=round(mm["prefill"], 1),
            tok_s_decode=round(mm["tok_s"], 2), answers=mm["rows"], node=pv.get("slurm_node"),
            job=pv.get("slurm_job_id"), file=os.path.basename(f),
            # this rung's own 96 questions: mean token-F1 and the share that clears a threshold
            # (2026-09-14, user: a throughput counted only on the questions that came out right).
            f1_mean=round(sum(r["acc_f1"] for r in rows) / len(rows), 4),
            correct_share={str(t): round(sum(1 for r in rows if r["acc_f1"] >= t) / len(rows), 4)
                           for t in (1.0, 0.8, 0.5)})
    oo = ooms(jobs, prefix)
    for k, e in store.items():
        e["oom"] = {arm: (oo.get((e["docs"], arm)) or [None])[0] for arm in ("teacher", "ours", "floor")}
        # ★ ONE SHARE PER (depth, arm), POOLED OVER ITS RUNGS — not one per rung (2026-09-14). Every rung of
        # an arm answers the SAME 96 questions, and the per-rung shares differ by a few points because
        # padding changes the floating-point accumulation order and greedy argmax flips on a handful of
        # questions. Batch is not a treatment in this project (CLAUDE.md, a closed question), so a goodput
        # curve must not carry per-rung accuracy: it is the throughput curve scaled by ONE constant per arm.
        # The per-rung shares stay in the store for the record and the spread is printed below.
        e["correct_share_pooled"] = {}
        for arm in ("teacher", "ours", "floor"):
            rungs = [v for kk, v in (e.get(arm) or {}).items() if kk.isdigit()]
            if not rungs:
                continue
            tot = sum(r["answers"] for r in rungs)
            e["correct_share_pooled"][arm] = {
                t: round(sum(r["correct_share"][t] * r["answers"] for r in rungs) / tot, 4)
                for t in ("1.0", "0.8", "0.5")}
            e["correct_share_pooled"][arm]["rungs"] = len(rungs)
            e["correct_share_pooled"][arm]["spread_1.0"] = round(
                max(r["correct_share"]["1.0"] for r in rungs) - min(r["correct_share"]["1.0"] for r in rungs), 4)
    store["_source"] = ("scripts/score_batch_sweep.py — throughput_eval.measure() over every "
                        f"{prefix}_d*_{{teacher,ours,floor7}}_b<B>.jsonl of job(s) " + ", ".join(sorted(map(str, jobs)))
                        + "; hotpotQA d40 s42, the first 96 questions; production FKV path; one file per batch; "
                        "complete-batch rate beside the all-batch rate; OOM rung per arm from the job's stdout.")
    os.makedirs("results/timing", exist_ok=True)
    json.dump(store, open(out_path, "w"), indent=1)
    print(f"wrote {out_path}")
    for k in sorted((k for k in store if k.startswith("d")), key=lambda k: int(k[1:])):
        e = store[k]
        print(f"  {k} ({e['ctx_ktok']}k tokens)  OOM rungs: {e['oom']}")
        print("     B    teacher ans/s (all)     ours ans/s (all)        floor ans/s (all)     ours/teacher")
        bs = sorted({int(b) for arm in ("teacher", "ours", "floor") for b in e.get(arm, {})})
        for b in bs:
            cells = []
            for arm in ("teacher", "ours", "floor"):
                a = e.get(arm, {}).get(str(b))
                cells.append("%-6.3f (%-6.3f)" % (a["ans_s_full"], a["ans_s"]) if a else "   —            ")
            t, o = e.get("teacher", {}).get(str(b)), e.get("ours", {}).get(str(b))
            ratio = f"x{o['ans_s_full'] / t['ans_s_full']:.2f}" if t and o else "—"
            print(f"     {b:<4d} {cells[0]}   {cells[1]}   {cells[2]}   {ratio}")
    # one node PER DEPTH (2026-09-12): d40/d160 are the gh014 job 3129219, d80/d120 one job each (3134478/3134479)
    # after the one-node four-depth job (3131342, 5 h 30 wall) could not be scheduled; every curve within a panel
    # is same-node, and the gh014 top rungs matched ctpf's gh089 within 1% (RESULTS_MASTER 2026-09-11b).
    per_depth = {k: sorted({a["node"] for arm in ("teacher", "ours", "floor") for a in e.get(arm, {}).values()}, key=str)
                 for k, e in store.items() if k.startswith("d")}
    mixed = [k for k, v in per_depth.items() if len(v) > 1]
    print(f"  nodes: {sorted(nodes, key=str)} -> " + ("one node" if len(nodes) == 1 else
          (f"one node PER DEPTH ({', '.join(f'{k}:{v[0]}' for k, v in sorted(per_depth.items(), key=lambda kv: int(kv[0][1:])))}) — every curve in a panel is same-node"
           if not mixed else f"MIXED NODES WITHIN A DEPTH {mixed}, NOT A TABLE")))
    print(f"  answers per rung: {sorted(answers)} -> {'one workload' if len(answers) == 1 else 'DIFFERENT WORKLOADS'}")
    if bad:
        print("  not scored / mismatched:\n   - " + "\n   - ".join(bad))


if __name__ == "__main__":
    main()
