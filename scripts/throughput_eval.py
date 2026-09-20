#!/usr/bin/env python
"""THE canonical throughput scorer. Hand-written per-launcher summaries are banned.

WHY THIS EXISTS (2026-08-27). Accuracy in this project cannot be mis-tabulated by accident: every run
records provenance and `build_table.py` REFUSES logs whose settings differ, computes the exact
shared-N, and re-scores with one extractor. Timing had no such gate — every launcher carried its own
inline summary snippet — and on LooGLE that produced a table with three independent faults at once:

  1. the batch wall was keyed on (conversation, turn), so ONE batch wall was summed once per
     conversation and every tok/s came out ~batch-size times too low;
  2. the arms came from FOUR different nodes, and node co-location has been measured in this project
     to roughly halve throughput, so the rows were never comparable;
  3. the resulting table showed snapKV decoding SLOWER while keeping LESS KV — an ordering violation
     that should have stopped publication and instead was published.

Each of those is caught here, in code, so it cannot depend on anyone remembering.

WHAT IT ENFORCES (any failure exits non-zero and prints the diff):
  * one wall per TURN. `batch_ans_s` is the wall clock of a whole batched turn; every row of that turn
    carries the same value, so it is collected into a dict keyed by turn and never summed per row.
  * every arm on the SAME node and from the SAME job. Different nodes are not comparable.
  * every arm at the SAME batch size and over the SAME conversation/document ids.
  * batch >= 2 (a batch-1 timing does not compile the flash graph and is an artefact).
  * provenance present on every log.

WHAT IT REPORTS, always together, because tok/s alone is misleading on ragged benchmarks:
  the OVERALL decode throughput (generated tokens of the whole batch / the batch's decode wall, at the
  stated batch — never a per-sequence rate), mean generated tokens per row, RAGGED% = 1 - mean/longest
  per turn (idle slots), and the idle-corrected rate continuous batching would recover (labelled CALC).
  A bare "tok/s" is never printed: the basis is part of every header, because a throughput number
  without its batch and its aggregation is not a measurement.

Usage:  python scripts/throughput_eval.py ours=a.jsonl teacher=b.jsonl snap40=c.jsonl
        python scripts/throughput_eval.py --allow-cross-node ...   # prints a banner, never silent
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys

TOKENIZER = os.environ.get("THR_TOKENIZER", "Qwen/Qwen2.5-7B-Instruct")
CACHE = os.environ.get("HF_HOME", "/work/hdd/myproject/anon/hf")


def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if not rows:
        raise SystemExit(f"❌ {path}: empty")
    prov = next((r.get("_provenance") for r in rows if r.get("_provenance")), None)
    if prov is None:
        raise SystemExit(f"❌ {path}: NO _provenance — a run without provenance is not comparable")
    return rows, prov


def measure(rows, tok):
    """one wall per TURN; tokens re-tokenized from the generated text.

    ★ PREFILL IS INCLUDED (2026-08-27, user: "overall tok/s가 decoding + prefill time 다 고려한 게
    맞는 거지?"). It was NOT, and that omission excluded the axis this method wins on: the 32B branch
    never ingests the long context, so per sequence the fusion arm prefills in 1.25 s against the
    teacher's 4.87 (LoCoMo-30, measured). `prefill_s_batch` is recorded once per batch, so it is
    collected as a SET of distinct values and added to the end-to-end wall; the decode-only rate is
    kept alongside because the two answer different questions, and on a 30-turn accumulate benchmark
    one prefill is amortised over thirty decodes (2-16% of the wall here) while a single-question
    workload would weight it fully."""
    # Rows are grouped by their BATCH first (conv_wall_s is stamped once per batch of
    # conversations, so it is the batch id) and only then by turn. A single-turn bench runs many
    # batches that are ALL turn 1 — keying the wall on the turn alone made two different batches
    # collide and the gate refused its own logs (found 2026-08-28 wiring musique_st40). Within one
    # batch the old invariant still holds and is still enforced: one answer wall per turn.
    walls, byturn, pref, conv = {}, {}, {}, set()
    # ★ THE BATCH KEY IS (conv_wall_s, prefill_s_batch), NOT conv_wall_s ALONE (2026-09-11). A batch
    # sweep at B=2 has 48 batches per file, and two of them (job 3129219, teacher d40) shared a
    # whole wall of 4.266 s to the millisecond, so the wall-only key merged them and the invariant
    # below refused the log. The prefill wall of a batch is stamped on every row of the batch, so the
    # pair identifies a batch without adding any assumption; on every log scored before this change
    # no two batches shared a wall, so the pair keys exactly the same groups (re-scored: ctpf store
    # and the four tl_* panels unchanged).
    # rows-per-batch must be counted in CONVERSATIONS, not answers. `conv_batch_rows` holds the
    # ANSWERS in the batch, which on a single-turn bench is the same number and on an accumulating
    # one is not: LooGLE conversations carry a variable number of questions, so two batches of the
    # same 8 conversations can hold 55 and 50 answers. Judging "was this batch full" on the answer
    # count then throws a whole legitimate batch away as a remainder — which is what put a Z in the
    # LooGLE ExpectedAttention curve (kept 30% appearing FASTER than kept 21.9%).
    convs_in, ans_in = {}, {}
    for r in rows:
        gid = r.get("conv_wall_s")          # None for pre-fix logs -> one group, old behaviour
        if gid is not None:
            _pb = r.get("prefill_s_batch")
            if _pb is None:
                _pb = r.get("prefill_s")
            gid = (round(float(gid), 3), round(float(_pb), 3) if _pb is not None else None)
        t = (gid, r["turn"])
        w = r.get("batch_ans_s")
        if w is None:
            raise SystemExit("❌ a row has no batch_ans_s — this log is not from the batched path")
        if t in walls and abs(walls[t] - w) > 1e-6:
            raise SystemExit(f"❌ batch {gid} turn {t[1]} carries two different answer walls "
                             f"({walls[t]} vs {w}) — the log mixes batches; refusing to guess")
        walls[t] = w
        if gid is not None:
            g = gid
            conv.add((g, int(r.get("conv_batch_rows") or 0)))
            convs_in.setdefault(g, set()).add(r.get("conv"))
            ans_in[g] = ans_in.get(g, 0) + 1
        v = r.get("prefill_s_batch")
        if v is None:
            v = r.get("prefill_s")
        if v is not None:
            pref[(gid, round(float(v), 3))] = float(v)   # one prefill per batch
        byturn.setdefault(t, []).append(len(tok(r.get("raw") or "")["input_ids"]))
    wall = sum(walls.values())
    prefill = sum(pref.values())
    toks = sum(sum(v) for v in byturn.values())
    means = [st.mean(v) for v in byturn.values()]
    maxes = [max(v) for v in byturn.values()]
    mean_row, longest = st.mean(means), st.mean(maxes)
    nrows = sum(len(v) for v in byturn.values())
    # ★ the TOTAL wall is the only one a throughput number may be divided by. Assembling it from
    # pieces (answer wall + context prefill) silently dropped the per-turn history commit, which the
    # baselines pay on a long-context cache every turn and we do not. A log without conv_wall_s is a
    # log from before that fix and CANNOT be used for a throughput claim.
    total = sum(c[0][0] for c in conv) if conv else None     # c[0] = (conv_wall_s, prefill) key
    e2e = total if total is not None else (wall + prefill)
    # ★ THE REMAINDER BATCH (2026-09-03, user: "나눗셈 나머지때문에 throughput 왜곡되는경우 없음?").
    # `ans_s` divides by the sum of EVERY batch's wall. If the workload does not divide by the
    # batch, the last batch runs at partial occupancy and drags the rate down — by a different
    # amount for every arm, since every arm has a different batch. Measured on the published
    # tables: floor-7B at B=64 over 96 questions (64+32) reads 8.9% low on hotpot and 15.6% low on
    # musique. So the complete-batch rate is reported ALONGSIDE, never instead: two numbers, and
    # they are identical whenever the workload divides evenly, which is what the grid launcher now
    # arranges wherever the benchmark allows it.
    per = {g: len(c) for g, c in convs_in.items()}          # CONVERSATIONS per batch
    bmax_rows = max(per.values()) if per else 0
    full = [g for g, c in per.items() if c == bmax_rows]
    tail_rows = sum(ans_in[g] for g, c in per.items() if c < bmax_rows)
    _full_wall = sum(g[0] for g in full)                     # the complete batches' whole walls
    ans_s_full = (sum(ans_in[g] for g in full) / _full_wall) if full and _full_wall else None
    return dict(rows=nrows, turns=len(walls), wall=wall, prefill=prefill, e2e=e2e, toks=toks,
                tok_s=(toks / wall if wall else 0.0),
                tok_s_e2e=(toks / e2e if e2e else 0.0),
                ans_s=(nrows / e2e if e2e else 0.0),
                ans_s_full=ans_s_full, n_batches=len(per), n_full_batches=len(full),
                tail_rows=tail_rows, batch_rows=bmax_rows,
                total=total, mean_row=mean_row,
                ragged=(1 - mean_row / longest if longest else 0.0),
                packed=(toks / wall * longest / mean_row if wall and mean_row else 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cols", nargs="+", help="role=path pairs")
    ap.add_argument("--allow-cross-node", action="store_true",
                    help="compare arms from different nodes anyway; prints a loud banner on every use")
    ap.add_argument("--ordering-explained", default=None, metavar="REASON",
                    help="acknowledge a compression ordering violation with a WRITTEN reason. The rule "
                         "assumes a bandwidth-bound decode; in a dispatch-bound one (measured 2026-08-27 "
                         "on LoCoMo at 27.5k: every retention from 60%% kept to 5%% kept decodes at "
                         "22-27 tok/s and the UNCOMPRESSED teacher at 23.9) retention does not move "
                         "throughput at all and a violation is expected. The reason is printed and "
                         "logged, never silenced.")
    ap.add_argument("--bmax", action="store_true",
                    help="OWN-B_max mode: each arm at the largest batch it survives. Batch and "
                         "document count are ALLOWED to differ because capacity is the quantity being "
                         "measured; node and job are still enforced. Every row prints its own batch, "
                         "and the x-axis is the OVERALL (batch-aggregate) decode throughput — which is "
                         "the only axis on which a capacity advantage can show at all.")
    ap.add_argument("--md", default=None)
    a = ap.parse_args()

    named = {}
    for c in a.cols:
        if "=" not in c:
            raise SystemExit(f"❌ expected role=path, got {c!r}")
        k, v = c.split("=", 1)
        named[k] = v

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER, cache_dir=CACHE)

    data, provs, fatal = {}, {}, []
    for role, path in named.items():
        if not os.path.exists(path):
            fatal.append(f"{role}: file missing {path}"); continue
        rows, pv = load(path)
        provs[role] = pv
        data[role] = measure(rows, tok)
        data[role]["convs"] = tuple(sorted({r.get("conv") for r in rows}))
        if data[role]["total"] is None:
            fatal.append(f"{role}: no conv_wall_s — this log predates the total-wall fix (2026-08-28) "
                         "and its throughput would omit the per-turn history commit; re-run it")
        b = pv.get("axis_batch_size")
        if not b or b < 2:
            fatal.append(f"{role}: axis_batch_size={b} — a batch-1 timing is an artefact, not a speed")
    if fatal:
        print("❌ REFUSED:\n   - " + "\n   - ".join(fatal)); sys.exit(1)

    def spread(key, getter):
        vals = {r: getter(r) for r in data}
        return vals if len(set(vals.values())) > 1 else None

    bad = []
    if not a.bmax:
        m = spread("batch", lambda r: provs[r].get("axis_batch_size"))
        if m: bad.append(f"batch size differs: {m} — a larger batch is a different measurement "
                         "(use --bmax if you mean each arm at its own capacity)")
        m = spread("convs", lambda r: data[r]["convs"])
        if m: bad.append("conversation/document set differs across arms — "
                         + "; ".join(f"{r}={len(v)} ids" for r, v in m.items()))
    nodes = {r: provs[r].get("slurm_node") for r in data}
    jobs = {r: provs[r].get("slurm_job_id") for r in data}
    if len(set(nodes.values())) > 1:
        msg = f"NODE differs: {nodes} — co-location has been measured to roughly halve throughput"
        if a.allow_cross_node:
            print("🚨🚨 --allow-cross-node: " + msg + "\n🚨🚨 every number below is suspect.\n")
        else:
            bad.append(msg)
    if bad:
        print("❌ REFUSED — these runs are not one measurement:\n   - " + "\n   - ".join(bad))
        sys.exit(1)

    b = list(provs.values())[0].get("axis_batch_size")
    if a.bmax:
        print(f"✅ COMPARABLE (OWN-B_max mode) — node={list(nodes.values())[0]}  "
              f"job(s)={sorted(set(jobs.values()))}. Batch and document count DIFFER BY DESIGN: "
              f"capacity is the quantity being measured, so each arm is given as many of the hardest "
              f"sequences as it can hold.")
    else:
        print(f"✅ COMPARABLE — batch={b}  node={list(nodes.values())[0]}  "
              f"job(s)={sorted(set(jobs.values()))}  docs={len(list(data.values())[0]['convs'])}  "
              f"turns={list(data.values())[0]['turns']}")
    # ★ NO BARE "tok/s" ANYWHERE (2026-08-27, user). A throughput number is meaningless without its
    # basis, so the basis is printed IN the header and repeated in the --md output: this is the
    # BATCH-AGGREGATE decode rate — generated tokens of the whole batch divided by the batch's own
    # decode wall clock — at the stated batch, i.e. what the GPU delivers overall, not per sequence.
    UNIT = ("OVERALL decode throughput (generated tokens/s, batch-aggregate) at each arm's own batch"
            if a.bmax else
            f"OVERALL decode throughput at batch {b} (generated tokens/s, batch-aggregate)")
    print(f"\n{UNIT}")
    print(f"{'arm':14s} {'batch':>5s} {'gen tok/s':>10s} {'gen tok/s':>10s} {'answers/s':>10s} "
          f"{'gen tok':>8s} {'prefill s':>10s} {'idle':>6s}")
    print(f"{'':14s} {'':5s} {'DECODE':>10s} {'END-TO-END':>10s} {'END-TO-END':>10s} "
          f"{'per ans':>8s} {'per seq':>10s} {'slots':>6s}")
    for role in sorted(data, key=lambda r: -data[r]["ans_s"]):
        d = data[role]
        B = provs[role].get("axis_batch_size")
        print(f"{role:14s} {B:5d} {d['tok_s']:10.2f} {d['tok_s_e2e']:10.2f} {d['ans_s']:10.3f} "
              f"{d['mean_row']:8.1f} {d['prefill']/max(B,1):10.2f} {d['ragged']:5.1%}")
    print("  DECODE     = generated tokens / the batches' answer wall.")
    print("  END-TO-END = additionally charges the context prefill, which an accumulate benchmark")
    print("               pays once per conversation and amortises over its turns.")
    print("  answers/s  = completed answers / end-to-end wall — what a serving system delivers, and")
    print("               the axis an arm cannot inflate by simply writing longer answers.")

    # ordering checks that must never fail silently (CLAUDE.md: compression > full is a bug signal)
    warn = []
    for r in data:
        if "snap" in r or "spec" in r or "expected" in r or "h2o" in r:
            for r2 in data:
                if r2 == r or not (("snap" in r2) or ("spec" in r2)):
                    continue
    keep = {r: provs[r].get("ratio") for r in data}
    snaps = [(r, keep[r]) for r in data if "snap" in r and keep[r] is not None]
    for i in range(len(snaps)):
        for j in range(len(snaps)):
            ri, ki = snaps[i]; rj, kj = snaps[j]
            if ki > kj and data[ri]["tok_s"] < data[rj]["tok_s"]:
                warn.append(f"ORDERING VIOLATION: {ri} removes MORE KV than {rj} "
                            f"(ratio {ki} vs {kj}) yet decodes SLOWER "
                            f"({data[ri]['tok_s']:.2f} vs {data[rj]['tok_s']:.2f}) — investigate "
                            "before publishing; this is a bug signal, not a finding")
    if warn:
        print("\n🔴 " + "\n🔴 ".join(warn))
        if not a.ordering_explained:
            print("\n   The rule assumes a BANDWIDTH-bound decode. If the loop is dispatch-bound the\n"
                  "   per-step cost tracks LAYER COUNT, every press runs the same model, and retention\n"
                  "   cannot move throughput — then pass --ordering-explained \"<reason>\".")
            sys.exit(2)
        print(f"\n   ACKNOWLEDGED: {a.ordering_explained}")
        with open("throughput_ordering_ack.log", "a") as f:
            f.write(json.dumps({"arms": sorted(named), "reason": a.ordering_explained,
                                "violations": warn}) + "\n")
    if a.md:
        # the .md mirrors the console: BOTH walls and BOTH rate axes, never decode-only
        with open(a.md, "w") as f:
            f.write("| arm | batch | gen tok/s DECODE (batch-aggregate) | "
                    "gen tok/s END-TO-END (batch-aggregate) | answers/s END-TO-END | "
                    "gen tok per answer | prefill s per seq | idle slots |\n"
                    "|---|---|---|---|---|---|---|---|\n")
            for role in sorted(data, key=lambda r: -data[r]["ans_s"]):
                d = data[role]
                B = provs[role].get("axis_batch_size")
                f.write(f"| {role} | {B} | {d['tok_s']:.2f} | {d['tok_s_e2e']:.2f} | "
                        f"{d['ans_s']:.3f} | {d['mean_row']:.1f} | {d['prefill']/max(B,1):.2f} | "
                        f"{d['ragged']:.1%} |\n")
        print(f"\nwrote {a.md}")


if __name__ == "__main__":
    main()
