#!/bin/bash
#SBATCH -J mtacc
#SBATCH -p ghx4
#SBATCH -A my-slurm-account
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 3:00:00
#SBATCH -o /work/hdd/myproject/%u/slurm_logs/%x-%j.out
set -euo pipefail
# ★ default 3, not 1. batch 1 is banned project-wide (flash-attention's graph does not compile at batch 1,
#   so it is a guaranteed throughput loss), and a launcher whose DEFAULT violates the rule means every run
#   that forgets to pass BATCH_SIZE silently violates it. 3 is the value every current comparison uses.
export BATCH_SIZE="${BATCH_SIZE:-3}"
export PREFILL_CHUNK="${PREFILL_CHUNK:-4096}"   # column slice per forward; 32B+7B leaves ~19GiB after weights
export TOKLOG="${TOKLOG:-}"
export FUSE_RESTRICT="${FUSE_RESTRICT:-0}"
export HF_HOME=/work/hdd/myproject/$USER/hf HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
cd /u/anon/SLM_LM
source /u/anon/.venvs/slm_lm/bin/activate
python scripts/mtrag_accum.py --method "${METHOD}" ${BENCH:+--bench $BENCH} --out "${OUT}" --max-conv "${MAXCONV:-110}" \
  --max-new "${MAXNEW:-128}" ${FULLANS:+--fullans} \
  ${RESUME:+--resume} \
  --model "${MODEL:-Qwen/Qwen2.5-14B-Instruct}" --slm-model "${SLM_MODEL:-Qwen/Qwen2.5-3B-Instruct}" \
  --lam "${LAM:-0.7}" --ratio "${RATIO:-0.8125}" ${KEEP:+--keep $KEEP} ${LM_NO_ACCUM:+--lm-no-accum} ${NO_ACCUM:+--no-accum} ${LAM_SCHEDULE:+--lam-schedule $LAM_SCHEDULE} \
  --ref "${REF:-/work/hdd/myproject/anon/mtrag/reference.jsonl}" ${INSTRUCTION:+--instruction $INSTRUCTION} \
  ${REF_OVERRIDE:+--ref-override $REF_OVERRIDE} \
  ${LM_LORA:+--lm-lora $LM_LORA} ${SLM_LORA:+--slm-lora $SLM_LORA} \
  ${PAPERS:+--papers $PAPERS}
