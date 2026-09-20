#!/bin/bash
#SBATCH -A my-slurm-account
#SBATCH -p ghx4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -o /work/hdd/myproject/anon/slurm_logs/%x-%j.out
# THE UNIFIED THROUGHPUT GRID, EACH ARM AT ITS OWN B_max (2026-09-03). Replaces
# run_timing_grid.slurm, whose output (tg_*) is NOT publishable — see CLAUDE.md 사고 #10.
#
# WHAT WENT WRONG THERE. That job exported BATCH_SIZE for the whole sweep, so all 21 arms ran at
# one batch and the per-item ladders never descended. A press holds a fraction of the teacher's KV,
# so running a larger batch IS its advantage: laddered per arm, hotpot snapKV at kept 15% reaches
# B=48 and the 7B floor B=64, against B=12 pinned. Pinning takes the baseline's own benefit away
# and makes our figure look better than it is. Here EVERY arm carries its own descending ladder
# whose TOP rung is scaled to its kept level, so the surviving rung is decided by memory.
# BATCH_SIZE is still exported because `_batchable` needs it > 1 to take the batched path at all —
# it is immediately overridden by each sweep item's own "batch" list, which is the only batch
# source. (Check it afterwards: axis_batch_size must DIFFER across arms. If every arm shows the
# same number, this is a fixed-batch table and not a throughput table.)
#
# THE REMAINDER BATCH — REPORTED, NOT DESIGNED AROUND (2026-09-03, corrected the same day).
# `ans_s` divides the answers by the sum of EVERY batch's wall, so a workload that does not divide
# by the batch ends in a partial batch whose low occupancy drags that arm's rate down. Measured on
# the published tables it touches exactly one arm — floor-7B at B=64 over 96 questions (64+32) —
# reading 8.9% low on hotpot and 15.6% low on musique. Every other arm's batch divides 96.
#
# ★ MY FIRST FIX FOR THIS WAS WRONG AND IS WITHDRAWN. I made the ladder keep only rungs that DIVIDE
# the workload, and raised the single-turn workload from 96 questions to 192 to give that rule a
# finer grid. Both are backwards. B_max is the quantity being measured; letting the question count
# decide which batches an arm may be measured at changes what the number means, to remove a
# distortion that is small and, unlike the substitution, fully visible. And the question count is
# the experiment's CONTENT, not an execution parameter — it was not mine to change (CLAUDE.md
# 사고 #6). Restored: 96 questions, and every arm's ladder is its honest one.
#
# What stays: `measure()` reports `ans_s` (all batches) beside `ans_s_full` (complete batches only)
# with `tail_rows`, and the ledger prints both. A remainder is then a number the reader can see
# rather than a thing the design pretended away.
#
# THE SECOND DEFECT THIS FIXES — the accumulating benches. In lct/lgt every arm ran only as many
# CONVERSATIONS as its batch (`papers_pool`: the B longest), so the teacher was timed on the 3
# longest LoCoMo conversations and a press on all 10. Different workloads, and the direction
# favours the arm with the larger batch — us. Here --max-conv is FIXED per benchmark and every arm
# processes the same conversations; the ladder only changes how many run concurrently.
#   LoCoMo has 10 conversations in total, so its batch axis is capped by the DATASET at 10, not by
#   memory. That is a property of the benchmark and is recorded, not worked around.
#
# New prefix `tl_` so tg_* and the older lct/lgt/mur/ho3 stay exactly as they are.
# BENCH=hotpot|musique|loogle|locomo
# ★ --resume ON EVERY PLAIN CALL (2026-09-03, user). The harness has had conversation-level
# resume since 2026-08-12 and CLAUDE.md's scheduling section says to rely on it; these
# launchers simply never passed the flag, so an arm cut off by the wall clock was re-run
# from zero — and its partial file was overwritten, which is why the size of the loss can
# no longer be recovered. --sweep resumes on its own; a plain call does not.
set -u
cd /u/anon/SLM_LM
BASE=/work/hdd/myproject/$USER; export HF_HOME=$BASE/hf
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /u/anon/.venvs/slm_lm/bin/activate
CKPT=$BASE/kvreuse_ckpts
BENCH=${BENCH:?}
case "$BENCH" in
  # MAXCONV is the timing workload, unchanged from the canonical set: 96 questions on the
  # single-turn benches (CANONICAL_RUNS 2d), and a fixed conversation count on the accumulating
  # ones so every arm processes the same work whatever batch it runs at.
  hotpot)  MAXCONV=96; PFX=tl_ho; SCALE=single; LAM=0.7;  LORA=lam07
           BA="--bench hotpotqa_st40 --max-conv $MAXCONV" ;;
  musique) MAXCONV=96; PFX=tl_mu; SCALE=single; LAM=0.7;  LORA=lam07
           BA="--bench musique_st40 --max-conv $MAXCONV" ;;
  # ★ TWENTY-FOUR, not sixteen (2026-09-04, under the user's 10 GPU-h grant for throughput
  # re-measurement). At 16 the arms that matter left partial last batches — ours at B=6 ran
  # 6+6+4 and its 4-row tail cost 143 s against 178-198 s for a full batch, i.e. 9% of its rate;
  # the teacher at B=3 ran 3x5+1 and its 1-row tail was the shortest conversation and cost it
  # nothing. That asymmetry is the remainder penalty landing on the arm with the bigger batch,
  # which is ours, and the complete-batch rate could not repair it because the complete batches
  # then covered DIFFERENT conversations per arm (ours the 12 longest, the teacher the 15
  # longest) on a benchmark whose conversations carry 1 to 14 questions each. At 24, every
  # frontier arm's batch divides the workload exactly — teacher 3, ours 6, the presses at kept
  # 21.9/30/40% (8, 6, 4, 3, 2) — so every one of them runs full batches over the identical 24
  # conversations and the two rates coincide. Only the kept 5/10/15% arms (B=16/14/10 last time)
  # can still leave a remainder, and their ladder tops rise to the new cap below. 24 = the first
  # 24 conversations of the reference in file order (128 answers); the pool holds 70.
  # MAXCONV_LG (2026-09-04, evening): 48 divides every batch the 24-grid measured except 10
  # (teacher 3, ours 6, presses 2/4/8/12/16/24), doubles the pool, and leaves only the two
  # B=10 arms with a tail (40+8). Default stays 24 until the user sets the budget for 48.
  loogle)  MAXCONV=${MAXCONV_LG:-24}; PFX=tl_lg; SCALE=accum;  LAM=0.7;  LORA=lam07
           BA="--bench loogle --max-conv $MAXCONV" ;;
  # ★ EIGHT, not ten (user, 2026-09-04). The benchmark has ten conversations, and at ten the arms
  # that matter all leave a partial last batch — ours at B=8 runs 8+2, the teacher at B=4 runs
  # 4+4+2 — so a fifth of the workload sits in a quarter-full batch and the rate depends on
  # 10 mod B. At eight, 18 of the 21 arms come out exact: everything that was at the dataset cap
  # of 10 drops to 8 (one batch), ours' 8 is one batch, the teacher's 4 is two, and only the three
  # press arms that top out at B=6 keep a 2-conversation remainder. Nothing is lost by the drop
  # from 10 to 8: the arms that ran at 10 were at the DATASET cap, so their batch was a lower
  # bound either way.
  locomo)  MAXCONV=8; PFX=tl_lc; SCALE=accum;  LAM=0.85; LORA=lam085
           BA="--bench locomo --max-conv $MAXCONV" ;;
esac
# Ladder tops per arm. The single-turn numbers come from the LAST job that laddered per arm
# (ho3/mur, job 3045588: kept 15% survived B=48, kept 21.9% B=32, kept 30% B=24, kept 40% B=16,
# floor B=64, ours B=16, teacher B=12), extended upward for the two new levels. The accumulating
# tops are capped by --max-conv above, because a rung with fewer conversations than rows available
# is not a rung (CANONICAL_RUNS 2u-loogle).
ARMS=$(BENCH=$BENCH SCALE=$SCALE MAXCONV=$MAXCONV python - <<'PY'
import json, os
single = os.environ["SCALE"] == "single"
CAP = int(os.environ["MAXCONV"])
# kept % -> descending ladder. ★ CAPPED AT THE NUMBER OF CONVERSATIONS THE RUN IS GIVEN: on an
# accumulating bench one row IS one conversation, so a rung above CAP cannot be exercised -- it
# "passes" while running CAP rows and reports a batch that never happened (CANONICAL_RUNS 2u-loogle,
# where BATCH_SIZE 16/18/26 all ran 12 rows). LoCoMo has only 10 conversations in total, so its
# ladder tops out at 10 for a reason that is the dataset's and not memory's.
# ladder tops per kept level, BEFORE the divisor filter. Sources: the last per-arm-laddered job
# (ho3/mur, 3045588) measured kept 15% at B=48, 21.9% at 32, 30% at 24, 40% at 16, floor at 64,
# ours at 16, teacher at 12 over 96 questions; the two new levels are extrapolated upward.
# Ladder tops per arm, from the last job that laddered per arm (ho3/mur, job 3045588, 96 questions:
# kept 15% survived B=48, 21.9% B=32, 30% B=24, 40% B=16, floor B=64, ours B=16, teacher B=12); the
# two new levels are extrapolated upward. These are HONEST tops — what an arm might survive — and
# nothing about the question count is allowed to edit them.
PRESS = ({5: [96, 72, 56, 48], 10: [80, 64, 48, 40], 15: [64, 48, 40, 32],
          21.875: [48, 40, 32, 24], 30: [32, 24, 20, 16], 40: [24, 20, 16, 12]} if single else
         # accum tops for kept 5/10/15% raised to 24 on 2026-09-04: at 16 conversations those
         # arms survived their top rung (16, 16, 14) without OOM, so their edge was the old
         # workload's cap and not memory. lad() still caps at MAXCONV, so LoCoMo (8) is unchanged.
         # kept 5% top raised to 48 on 2026-09-04 (evening): both kept-5% presses survived 24 on
         # the 24-conversation grid without OOM, so 24 was the workload's cap and not memory's.
         {5: [48, 24, 16, 12, 10, 8], 10: [24, 16, 12, 10, 8], 15: [24, 16, 14, 10, 8],
          21.875: [12, 10, 8, 6], 30: [10, 8, 6, 4], 40: [8, 6, 4, 3]})
TEACH = [16, 12, 10, 8] if single else [4, 3, 2]


def lad(x):
    """cap at the conversations available, dedupe, descend.

    The only edit the workload is allowed to make is the CAP: a rung above the number of items the
    run is given cannot be exercised — it "passes" while running fewer rows and reports a batch
    that never happened (CANONICAL_RUNS 2u-loogle). Nothing else about the workload touches the
    ladder. An earlier version of this file also dropped every rung that did not DIVIDE the
    workload, so that no batch would be partial; that is backwards — B_max is the quantity being
    measured — and it is withdrawn. A partial last batch is reported by the scorer instead.
    """
    return sorted({min(b, CAP) for b in x}, reverse=True)


A = [dict(tag="teacher", method="teacher", batch=lad(TEACH))]
for k, l in PRESS.items():
    r = round(1 - k / 100, 5)
    # ★ THE FILENAME CARRIES THE RATIO x 1000, NOT THE KEPT PERCENT (CANONICAL_RUNS 1z).
    # parse_tag reads snap<d> as int(d)/10**len(d) and takes kept = 1 - that, so kept 5% must be
    # written snap950. Writing "snap05" here would put the point at kept 95% on every figure.
    A += [dict(tag=f"snap{int(round(r * 1000)):03d}", method="snapkv_frozen", ratio=r, batch=lad(l)),
          dict(tag=f"expected{int(round(r * 1000)):03d}", method="h2o", ratio=r, batch=lad(l))]
print(json.dumps(A))
PY
)
test -n "$ARMS" || { echo "EMPTY ARMS"; exit 2; }
SPEC_ONLY=${SPEC_ONLY:-0}
if [ "$SPEC_ONLY" != 1 ]; then
echo "=== BMAX GRID $BENCH: teacher + 12 press arms, one load, ladder per arm ==="
BATCH_SIZE=4 python scripts/mtrag_accum.py $BA --model Qwen/Qwen2.5-32B-Instruct --max-new 200 \
  --method teacher --sweep "$ARMS" --out "results/fusionft/${PFX}_{tag}.jsonl"

# ★ A THROUGHPUT rung never resumes into a partial file (2026-09-05, job 3085983). When a rung OOMs
# after finishing some conversations, --resume on the next rung keeps them (at the old batch) and
# the harness then filled its --max-conv slice with conversations from beyond it, so the arm ran a
# DIFFERENT workload. The partial is moved to results/quarantine/ with its rung and job in the name;
# the next rung starts clean. Never rm (CLAUDE.md).
_quarantine_partial() {  # $1 = result path, $2 = rung
  [ -f "$1" ] || return 0
  mkdir -p results/quarantine
  mv "$1" "results/quarantine/$(basename "$1" .jsonl).OOM_partial_B$2_job${SLURM_JOB_ID:-0}.jsonl"
  echo "   (partial of rung B=$2 moved to results/quarantine/, next rung starts clean)"
}
echo "=== BMAX GRID $BENCH: floor-7B ==="
for B in $(python -c "
c,s_=$MAXCONV,'$SCALE'; L=[96,80,64,48] if s_=='single' else [24,16,12,10,8]
print(' '.join(str(b) for b in sorted({min(b,c) for b in L}, reverse=True)))"); do
  BATCH_SIZE=$B LONGEST_FIRST=1 python scripts/mtrag_accum.py $BA \
    --model Qwen/Qwen2.5-7B-Instruct --max-new 200 --method teacher \
    --resume --out "results/fusionft/${PFX}_floor7.jsonl" && break
  echo "!! floor7 OOM at B=$B, dropping a rung"; _quarantine_partial "results/fusionft/${PFX}_floor7.jsonl" $B
done

echo "=== BMAX GRID $BENCH: ours ==="
for B in $(python -c "
c,s_=$MAXCONV,'$SCALE'; L=[24,20,16,12] if s_=='single' else [10,8,6,4]
print(' '.join(str(b) for b in sorted({min(b,c) for b in L}, reverse=True)))"); do
  MERGE_LORA=1 BATCH_SIZE=$B LONGEST_FIRST=1 python scripts/mtrag_accum.py $BA \
    --model Qwen/Qwen2.5-32B-Instruct --slm-model Qwen/Qwen2.5-7B-Instruct --max-new 200 \
    --method ours --lam $LAM --lm-no-accum \
    --slm-lora $CKPT/reader_binding_v5distill_r16_s900 \
    --lm-lora $CKPT/stage2_on_v5reader_${LORA}_r16_s600 \
    --resume --out "results/fusionft/${PFX}_ours.jsonl" && break
  echo "!! ours OOM at B=$B, dropping a rung"; _quarantine_partial "results/fusionft/${PFX}_ours.jsonl" $B
done
fi
# SpecPrefill needs the 7B loaded as the speculator, so it cannot share the sweep's model load and
# gets its ladder here. ★ --keep is a CLI FLAG; SPEC_KEEP in the environment is NEVER READ
# (incident #8: 34 files ran at the default 0.10 under filenames claiming otherwise).
# The tag is %05d: parse_tag reads spec<d> as int(d)/10**len(d), so keep 0.05 is "spec05000".
for K in 0.05 0.10 0.15 0.21875 0.30 0.40; do
  T=$(python -c "print('%05d' % round($K*100000))")
  LAD=$(python - "$K" "$SCALE" "$MAXCONV" <<'PY'
import sys
k, scale, cap = float(sys.argv[1]), sys.argv[2], int(sys.argv[3])
S = {0.05: [48, 40, 32, 24], 0.10: [40, 32, 24, 16], 0.15: [40, 32, 24, 16],
     0.21875: [24, 20, 16, 12], 0.30: [24, 16, 12, 8], 0.40: [16, 12, 8, 6]}
# accum keep 0.05 top raised to 24 (2026-09-04): it survived 12 at 16 conversations without OOM.
A = {0.05: [24, 16, 12, 10, 8, 6], 0.10: [12, 10, 8, 6], 0.15: [10, 8, 6, 4],
     0.21875: [10, 8, 6, 4], 0.30: [8, 6, 4, 3], 0.40: [6, 4, 3, 2]}
print(" ".join(str(b) for b in sorted({min(b, cap) for b in (S if scale == "single" else A)[k]},
                                      reverse=True)))
PY
)
  for B in $LAD; do
    echo "=== BMAX GRID $BENCH: specprefill keep=$K B=$B ==="
    BATCH_SIZE=$B LONGEST_FIRST=1 python scripts/mtrag_accum.py $BA --keep "$K" \
      --model Qwen/Qwen2.5-32B-Instruct --slm-model Qwen/Qwen2.5-7B-Instruct --max-new 200 \
      --method specprefill --resume --out "results/fusionft/${PFX}_spec${T}.jsonl" && break
    echo "!! spec keep=$K OOM at B=$B, dropping a rung"; _quarantine_partial "results/fusionft/${PFX}_spec${T}.jsonl" $B
  done
done
echo "=== BMAX GRID $BENCH DONE — check that axis_batch_size DIFFERS across arms ==="
python - "$PFX" "${SLURM_JOB_NAME:-x}-${SLURM_JOB_ID:-0}.out" <<'PYEOF'
import glob, json, os, re, sys
p, logname = sys.argv[1], sys.argv[2]

# ── which batches were actually TRIED, so the table can say what B_max means per arm ────────────
# The ladder descends and stops at the first rung that does not OOM. That gives two different
# things and they must not be printed as one: an arm that OOM'd higher has its true edge BRACKETED
# between the rung that failed and the one that ran, while an arm that succeeded at the TOP rung
# was never pushed and its edge is only a LOWER BOUND. Nothing in the result rows records this —
# it is in the job's own stdout — so it is collected here, next to the numbers it qualifies.
log = os.path.join("/work/hdd/myproject/anon/slurm_logs", logname)
tried = {}
if os.path.exists(log):
    for ln in open(log, errors="replace"):
        m = re.match(r"=== SWEEP (OK|OOM) (\S+) B=(\d+)", ln)
        if m:
            tried.setdefault(m.group(2), {}).setdefault(m.group(1).lower(), []).append(int(m.group(3)))
        m = re.match(r"!! (\S+) OOM at B=(\d+)", ln)
        if m:
            tried.setdefault(m.group(1), {}).setdefault("oom", []).append(int(m.group(2)))
        m = re.match(r"!! spec keep=(\S+) OOM at B=(\d+)", ln)
        if m:
            tried.setdefault("spec%05d" % round(float(m.group(1)) * 100000), {}) \
                 .setdefault("oom", []).append(int(m.group(2)))

rows = []
for f in sorted(glob.glob(f"results/fusionft/{p}_*.jsonl")):
    try:
        t = [json.loads(l) for l in open(f) if l.strip()]
    except Exception:
        continue
    t = [r for r in t if "acc_f1" in r]
    if not t:
        continue
    tag = f.split(p + "_")[1][:-6]
    pv = t[0]["_provenance"]
    oom = sorted(tried.get(tag, {}).get("oom", []))
    rows.append(dict(tag=tag, batch=pv.get("axis_batch_size"), answers=len(t),
                     convs=len({r["conv"] for r in t}), node=pv.get("slurm_node"),
                     oom_at=oom[0] if oom else None,
                     bound="bracketed" if oom else "lower bound - never OOM'd"))
for r in sorted(rows, key=lambda x: -(x["batch"] or 0)):
    print("  %-12s B=%-4s %-26s answers=%-5d convs=%-4d node=%s"
          % (r["tag"], r["batch"],
             (f"(OOM at {r['oom_at']}, so < {r['oom_at']})" if r["oom_at"] else "(top rung: B_max >= this)"),
             r["answers"], r["convs"], r["node"]))
bs = {r["batch"] for r in rows}
ans = {r["answers"] for r in rows}
nodes = {r["node"] for r in rows}
print(f"  distinct batches: {sorted(bs)} -> {'OK' if len(bs) > 1 else 'FIXED-BATCH, NOT A THROUGHPUT TABLE'}")
print(f"  answers per arm:  {sorted(ans)} -> {'OK, one workload' if len(ans) == 1 else 'ARMS RAN DIFFERENT WORKLOADS'}")
print(f"  nodes:            {sorted(nodes)} -> {'OK, one node' if len(nodes) == 1 else 'MIXED NODES, NOT A TABLE'}")
json.dump({r["tag"]: r for r in rows}, open(f"results/timing/bmax_ladder_{p}.json", "w"), indent=1)
print(f"  wrote results/timing/bmax_ladder_{p}.json")
PYEOF
