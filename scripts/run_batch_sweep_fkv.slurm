#!/bin/bash
#SBATCH -A my-slurm-account
#SBATCH -p ghx4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -o /work/hdd/myproject/anon/slurm_logs/%x-%j.out
# BATCH-SIZE SWEEP: throughput at EVERY batch from 2 up to the OOM rung, teacher-32B / ours 32B+7B /
# floor-7B, hotpotQA d40 s42 at two context lengths (40 docs = 6.1k tokens, 160 docs = 23.3k), the SAME
# 96 questions everywhere, production FKV decode path, ONE job ONE node (user, 2026-09-11: "batch size
# vs. throughput을 추가로 보여주는 실험 … 10GPU hr이내로 자유롭게").
#
# WHAT THIS ANSWERS. The published throughput figure puts every arm at its own B_max, and a reader sees
# ours between the teacher and the floor as if it were a mixture of the two. This job measures the
# CURVE behind those points: answers/s as a function of the batch, for each arm, until it runs out of
# memory. Three readings come out of it and none can be read off the B_max points alone:
#   (i)  at a MATCHED batch (the same B for teacher and ours), how the two compare — this is what a
#        "same concurrency" comparison would show;
#   (ii) where each arm's curve STOPS (the OOM rung), which is the memory ceiling the method moves;
#   (iii) how (i) and (ii) shift from 6.1k to 23.3k tokens of context.
#
# WHAT IS NOT NEW HERE. The rung each arm survives at was already laddered on this path by job 3102806
# (results/timing/context_throughput_fkv.json: teacher 10/3, ours >=16/6, floor >=64/24 at d40/d160).
# Those tops are used to bound the sweeps below: every arm gets exactly ONE rung above its known top
# (the rung that OOM'd there) so the ceiling is re-demonstrated on THIS node, and nothing higher —
# except ours at d40, which never OOM'd at 16 on FKV and is pushed to 20 then 24.
#
# HOW EACH RUNG IS MEASURED. One result file per (depth, arm, batch): bsw_d<D>_<arm>_b<B>.jsonl, so no
# file ever mixes batches. Rate = throughput_eval.measure() over the file (whole wall, complete-batch
# rate beside the all-batch rate). teacher and floor are single-model arms, so their rungs run under
# ONE model load via --sweep (one item per rung, batch=[B], longest_first so an infeasible rung fails in
# its first batch). ours needs two models and cannot be swept (validate_specs refuses it), so its rungs
# are plain calls, ascending, stopping at the first OOM. Batch 1 is never run (CLAUDE.md: the flash
# graph does not compile at batch 1, so a batch-1 wall is an artefact).
#
# RESUME CONTRACT. A rung whose file already holds 96 scored rows is skipped. A rung whose file is
# PARTIAL (job died mid-rung) is moved to results/quarantine/ and re-run whole — a throughput arm never
# resumes into a partial (CANONICAL_RUNS 2-resume). Every plain call still passes --resume, per the
# launcher contract; with the partial moved aside it starts clean. Nothing under results/ is deleted.
set -u
cd /u/anon/SLM_LM
BASE=/work/hdd/myproject/$USER; export HF_HOME=$BASE/hf
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /u/anon/.venvs/slm_lm/bin/activate
CKPT=$BASE/kvreuse_ckpts
export FKV_DECODE=1 FKV_GRAPH=1
DEPTHS=${DEPTHS:-"40 160"}
# PFX (2026-09-11): the four-depth run (user: "doc=80, doc=120도 있으면") writes bsw4_ so the two-depth gh014
# files (bsw_) stay untouched as the other-node reproduction; gh014 was held by two-day jobs, so a
# four-panel figure — one node per table — is one new job.
PFX=${PFX:-bsw}; export PFX     # exported: the end-of-job summary (a python child) reads it
OUT=results/fusionft

_n_rows() {  # $1 = path -> number of scored rows (0 if absent)
  [ -f "$1" ] || { echo 0; return; }
  python - "$1" <<'PY'
import json, sys
n = 0
for l in open(sys.argv[1]):
    if l.strip():
        try:
            n += 1 if "acc_f1" in json.loads(l) else 0
        except Exception:
            pass
print(n)
PY
}

_quarantine_partial() {  # $1 = result path, $2 = rung  (never rm — CLAUDE.md)
  [ -f "$1" ] || return 0
  mkdir -p results/quarantine
  mv "$1" "results/quarantine/$(basename "$1" .jsonl).PARTIAL_B$2_job${SLURM_JOB_ID:-0}.jsonl"
  echo "   (partial rung B=$2 moved to results/quarantine/, the rung re-runs whole)"
}

# rungs still to run for a single-model arm: skips complete files, quarantines partial ones,
# prints the JSON --sweep list (one item per rung, its own file, longest-first).
_pending_items() {  # $1 = depth  $2 = arm (teacher|floor7)  $3 = "B B B"
  local D=$1 A=$2 items=""
  for B in $3; do
    local f="$OUT/${PFX}_d${D}_${A}_b${B}.jsonl"
    local n; n=$(_n_rows "$f")
    if [ "$n" = "96" ]; then echo "   skip d$D $A B=$B (complete)" >&2; continue; fi
    if [ "$n" != "0" ] || [ -f "$f" ]; then _quarantine_partial "$f" "$B" >&2; fi
    items="$items $B"
  done
  python - "$A" "$items" <<'PY'
import json, sys
arm, bs = sys.argv[1], [int(b) for b in sys.argv[2].split()]
print(json.dumps([dict(tag=f"{arm}_b{b}", method="teacher", batch=[b], longest_first=True) for b in bs]))
PY
}

for D in $DEPTHS; do
  case $D in
    # one rung above the FKV top of job 3102806 on every arm (teacher 10 -> 12, floor >=64 -> 96,
    # teacher 3 -> 4, floor 24 -> 32, ours 6 -> 8); ours at d40 never OOM'd at 16, so 20 and 24 follow.
    40)  LT="2 4 6 8 10 12";  LF="2 4 8 16 32 48 64 96"; LO="2 4 6 8 12 16 20 24" ;;
    # d80 / d120 (2026-09-11, user): one rung above each arm's FKV top of job 3102806 (teacher 6 -> 8 and
    # 4 -> 6; ours 10 -> 12 and 6 -> 8; floor 32 -> 40 and 24 -> 32)
    80)  LT="2 3 4 6 8";      LF="2 4 8 16 24 32 40";    LO="2 4 6 8 10 12" ;;
    120) LT="2 3 4 6";        LF="2 4 8 16 24 32";       LO="2 3 4 6 8" ;;
    160) LT="2 3 4";          LF="2 4 8 12 16 24 32";    LO="2 3 4 6 8" ;;
    *) echo "unknown depth $D"; exit 2 ;;
  esac
  BA="--bench hotpotqa_st${D}_full --max-conv 96"

  echo "=== BSW d$D teacher-32B sweep over B in {$LT} (one load) ==="
  ITEMS=$(_pending_items $D teacher "$LT")
  if [ "$ITEMS" != "[]" ]; then
    BATCH_SIZE=4 python scripts/mtrag_accum.py $BA --model Qwen/Qwen2.5-32B-Instruct --max-new 200 \
      --method teacher --sweep "$ITEMS" --out "$OUT/${PFX}_d${D}_{tag}.jsonl"
  fi

  echo "=== BSW d$D floor-7B sweep over B in {$LF} (one load) ==="
  ITEMS=$(_pending_items $D floor7 "$LF")
  if [ "$ITEMS" != "[]" ]; then
    BATCH_SIZE=4 python scripts/mtrag_accum.py $BA --model Qwen/Qwen2.5-7B-Instruct --max-new 200 \
      --method teacher --sweep "$ITEMS" --out "$OUT/${PFX}_d${D}_{tag}.jsonl"
  fi

  echo "=== BSW d$D ours 32B+7B, one call per rung, ascending, stop at the first OOM ==="
  for B in $LO; do
    f="$OUT/${PFX}_d${D}_ours_b${B}.jsonl"
    n=$(_n_rows "$f")
    if [ "$n" = "96" ]; then echo "   skip d$D ours B=$B (complete)"; continue; fi
    if [ "$n" != "0" ] || [ -f "$f" ]; then _quarantine_partial "$f" "$B"; fi
    echo "=== BSW d$D ours B=$B ==="
    if ! MERGE_LORA=1 BATCH_SIZE=$B LONGEST_FIRST=1 python scripts/mtrag_accum.py $BA \
        --model Qwen/Qwen2.5-32B-Instruct --slm-model Qwen/Qwen2.5-7B-Instruct --max-new 200 \
        --method ours --lam 0.7 --lm-no-accum \
        --slm-lora $CKPT/reader_binding_v5distill_r16_s900 \
        --lm-lora $CKPT/stage2_on_v5reader_lam07_r16_s600 \
        --resume --out "$f"; then
      echo "!! $PFX d$D ours OOM at B=$B"; _quarantine_partial "$f" "$B"; break
    fi
  done
done

echo "=== BSW DONE — every rung, its batch, answers and node ==="
python - <<'PYEOF'
import glob, json, os, re
rows = []
for f in sorted(glob.glob(f"results/fusionft/{os.environ.get('PFX', 'bsw')}_d*_*_b*.jsonl")):
    t = [json.loads(l) for l in open(f) if l.strip()]
    t = [r for r in t if "acc_f1" in r]
    if not t:
        continue
    pv = t[0]["_provenance"]
    m = re.match(r"\w+?_d(\d+)_(\w+?)_b(\d+)", os.path.basename(f))
    rows.append((int(m.group(1)), m.group(2), int(m.group(3)), pv.get("axis_batch_size"), len(t),
                 pv.get("slurm_node"), pv.get("slurm_job_id")))
for r in sorted(rows):
    print("  d%-4d %-8s rung B=%-3d ran B=%-3s answers=%-3d node=%s job=%s" % r)
nodes = {r[5] for r in rows}
per_depth = {}
for r in rows:
    per_depth.setdefault(r[0], set()).add(r[5])
mixed = [d for d, v in per_depth.items() if len(v) > 1]
print("  nodes: %s -> %s" % (sorted(nodes), "OK, one node" if len(nodes) == 1 else
      ("OK, one node per depth (per-depth jobs since 2026-09-12)" if not mixed else "MIXED NODES WITHIN A DEPTH %s, NOT A TABLE" % mixed)))
print("  answers per rung: %s -> %s" % (sorted({r[4] for r in rows}),
                                        "OK, one workload" if len({r[4] for r in rows}) == 1 else "DIFFERENT WORKLOADS"))
bad = [r for r in rows if r[2] != r[3]]
print("  rung == axis_batch_size on every file: %s" % ("OK" if not bad else f"NO: {bad}"))
PYEOF
