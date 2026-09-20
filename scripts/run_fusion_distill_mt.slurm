#!/bin/bash
#SBATCH -J fdistmt
#SBATCH -p ghx4
#SBATCH -A my-slurm-account
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 4:00:00
#SBATCH -o /work/hdd/myproject/%u/slurm_logs/%x-%j.out
#
# MULTI-TURN (accumulate) fusion distillation — trains the LoRA IN the same KV-reuse setting §7.0-V evaluates,
# on the same benchmark, at the same λ (removes the setting/task/λ shift that killed single-turn transfer).
# Params via --export=ALL,...: TEACHERLOG (req, mtrag_accum teacher run over the TRAIN ref), REF (req, TRAIN ref),
#   LAM, INSTRUCTION, EPOCHS, GRAD_ACCUM, LR, F1_THRESH, MAXCONV, TRAIN_SLM(1), MAXSTEPS, OUT (req).
set -euo pipefail
BASE=/work/hdd/myproject/$USER; export HF_HOME=$BASE/hf
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export REASON_FIX="${REASON_FIX:-1}" REASON_HIST="${REASON_HIST:-ref}"
cd /u/anon/SLM_LM
source /u/anon/.venvs/slm_lm/bin/activate

TEACHERLOG="${TEACHERLOG:?set TEACHERLOG}"; REF="${REF:?set REF}"; OUT="${OUT:?set OUT}"
LAM="${LAM:-0.85}"; INSTRUCTION="${INSTRUCTION:-QA_REASON_V3}"; EPOCHS="${EPOCHS:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"; LR="${LR:-1e-4}"; F1_THRESH="${F1_THRESH:-0.5}"; MAXCONV="${MAXCONV:-250}"
MAXSTEPS="${MAXSTEPS:-0}"; MODEL="${MODEL:-Qwen/Qwen2.5-14B-Instruct}"; SLM_MODEL="${SLM_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
TS=(); [ "${TRAIN_SLM:-0}" = "1" ] && TS=(--train-slm)

echo "[Info] $(date -Is) MT-distill ref=$REF lam=$LAM instr=$INSTRUCTION epochs=$EPOCHS train_slm=${TRAIN_SLM:-0} -> $OUT"
python scripts/fusion_distill_train_mt.py \
  --teacher-log "$TEACHERLOG" --ref "$REF" \
  --model "$MODEL" --slm-model "$SLM_MODEL" \
  --instruction "$INSTRUCTION" --lam "$LAM" --f1-threshold "$F1_THRESH" \
  --max-conv "$MAXCONV" --epochs "$EPOCHS" --grad-accum "$GRAD_ACCUM" --lr "$LR" \
  --max-steps "$MAXSTEPS" "${TS[@]}" --cache-dir "$HF_HOME" --device cuda:0 --out "$OUT"
echo "[Done] $(date -Is)"
