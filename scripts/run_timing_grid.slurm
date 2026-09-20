#!/bin/bash
#SBATCH -A my-slurm-account
#SBATCH -p ghx4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -o /work/hdd/myproject/anon/slurm_logs/%x-%j.out
# THE UNIFIED THROUGHPUT GRID — one job per benchmark, one node, every arm (2026-09-03).
#
# WHY NOT BACKFILL INDIVIDUAL POINTS. build_throughput_report.collect() keeps only the MAJORITY
# NODE and silently drops every arm measured elsewhere, because a throughput curve mixing nodes is
# not a curve. So a missing point cannot be added to an existing table by running it later: the
# whole table has to be one job on one node. That is also cheaper here -- a bench's entire grid is
# ~18 arms at 1.7-4.9 min each -- than adding points piecemeal would have been.
#
# It writes under a NEW prefix so the existing tables stay exactly as published; nothing under
# results/ is overwritten. The unified grid is kept = {5, 10, 15, 21.9, 30, 40}% for all three
# families, replacing the old split where the presses ran {15, 21.9, 30, 40, 50, 60} and
# SpecPrefill ran {10, 15, 21.9, 30, 40} and only {15, 21.9, 30, 40} overlapped.
# The teacher and floor anchors are re-measured in the SAME job, because a curve is read against
# them and they must share the node too.
# BENCH=hotpot|musique|loogle|locomo
set -u
cd /u/anon/SLM_LM
BASE=/work/hdd/myproject/$USER; export HF_HOME=$BASE/hf
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /u/anon/.venvs/slm_lm/bin/activate
CKPT=$BASE/kvreuse_ckpts
BENCH=${BENCH:?}
case "$BENCH" in
  hotpot)  BA="--bench hotpotqa_st40 --max-conv 96"; PFX=tg_ho; BL="[12,8,6]"; B0=12 ;;
  musique) BA="--bench musique_st40 --max-conv 96";  PFX=tg_mu; BL="[12,8,6]"; B0=12 ;;
  loogle)  BA="--bench loogle --max-conv 15";        PFX=tg_lg; BL="[3,2]";    B0=3  ;;
  locomo)  BA="--bench locomo --max-conv 4";         PFX=tg_lc; BL="[3,2]";    B0=3  ;;
esac
# presses take a compression RATIO = 1 - kept; SpecPrefill takes the keep fraction directly.
SW='[{"tag":"teacher","method":"teacher","batch":'$BL'}'
for K in 05 10 15 219 30 40; do
  case $K in 05) R=0.95;; 10) R=0.90;; 15) R=0.85;; 219) R=0.78125;; 30) R=0.70;; 40) R=0.60;; esac
  SW="$SW"',{"tag":"snap'$K'","method":"snapkv_frozen","ratio":'$R',"batch":'$BL'}'
  SW="$SW"',{"tag":"expected'$K'","method":"h2o","ratio":'$R',"batch":'$BL'}'
done
SW="$SW]"
SPEC_ONLY=${SPEC_ONLY:-0}
# SPEC_ONLY=1 redoes ONLY the SpecPrefill arms. It exists because the press/teacher/floor/ours arms
# in this job are already correct, and their plain (non-sweep) runs open --out with 'w': re-running
# them would overwrite good measurements and re-spend the GPU that produced them.
if [ "$SPEC_ONLY" != 1 ]; then
echo "=== TIMING GRID $BENCH: teacher + 12 press arms, one load ==="
BATCH_SIZE=$B0 python scripts/mtrag_accum.py $BA --model Qwen/Qwen2.5-32B-Instruct --max-new 200 \
  --method teacher --sweep "$SW" --out "results/fusionft/${PFX}_{tag}.jsonl"
echo "=== TIMING GRID $BENCH: floor-7B ==="
BATCH_SIZE=$B0 python scripts/mtrag_accum.py $BA --model Qwen/Qwen2.5-7B-Instruct --max-new 200 \
  --method teacher --out "results/fusionft/${PFX}_floor7.jsonl" || echo "!! floor failed"
echo "=== TIMING GRID $BENCH: ours ==="
MERGE_LORA=1 BATCH_SIZE=$B0 python scripts/mtrag_accum.py $BA \
  --model Qwen/Qwen2.5-32B-Instruct --slm-model Qwen/Qwen2.5-7B-Instruct --max-new 200 \
  --method ours --lam 0.7 --lm-no-accum \
  --slm-lora $CKPT/reader_binding_v5distill_r16_s900 \
  --lm-lora $CKPT/stage2_on_v5reader_lam07_r16_s600 \
  --out "results/fusionft/${PFX}_ours.jsonl" || echo "!! ours failed"
fi
# SpecPrefill needs the small model loaded as the speculator, so it cannot share the sweep's load.
for K in 0.05 0.10 0.15 0.21875 0.30 0.40; do
  # %05d, not %d: parse_tag reads spec<d> as int(d)/10**len(d), so keep 0.05 must be
  # "spec05000" (5000/1e5) and never "spec5000", which reads as 0.5. Renamed 2026-09-03
  # after the LoCoMo grid came back with its 5% arm plotted at 50%.
  T=$(python -c "print('%05d' % int($K*100000))")
  echo "=== TIMING GRID $BENCH: specprefill keep=$K ==="
  # ★ --keep, NOT SPEC_KEEP in the environment. mtrag_accum reads the SpecPrefill keep from
# args.keep (a CLI flag, default 0.1) and never looks at the environment, so every run
# launched with SPEC_KEEP=... silently used keep=0.10 whatever its filename said. That is
# what happened on 2026-09-02/03 to 30 files, now in results/quarantine/ under their real
# setting, and it also produced a false conclusion ("the keep knob is inert on gemma and
# OLMo") that had to be retracted -- two runs at different labels were the same run.
BATCH_SIZE=$B0 python scripts/mtrag_accum.py $BA --keep "$K" \
    --model Qwen/Qwen2.5-32B-Instruct --slm-model Qwen/Qwen2.5-7B-Instruct --max-new 200 \
    --method specprefill --out "results/fusionft/${PFX}_spec${T}.jsonl" || echo "!! spec $K failed"
done
echo "=== TIMING GRID $BENCH DONE ==="
