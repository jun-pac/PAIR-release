#!/bin/bash
#SBATCH -J stage2sft
#SBATCH -p ghx4
#SBATCH -A my-slurm-account
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH -t 8:00:00
#SBATCH -o /work/hdd/myproject/%u/slurm_logs/%x-%j.out
# STAGE-2 fusion LM SFT on top of a FROZEN stage-1 reader (fusion_stage2_lm_sft.py).
# 2 GPUs: 32B+7B = 76G weights + LM grad activations OOM one 95GB card (measured).
set -euo pipefail
BASE=/work/hdd/myproject/$USER; export HF_HOME=$BASE/hf
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
cd /u/anon/SLM_LM
source /u/anon/.venvs/slm_lm/bin/activate
python scripts/fusion_stage2_lm_sft.py \
  --corpus "${CORPUS:?}" --slm-lora "${SLM_LORA:?}" --out "${OUT:?}" \
  --train-branch "${TRAIN_BRANCH:-lm}" ${LM_LORA_INIT:+--lm-lora $LM_LORA_INIT} \
  --slm-model "${SLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}" --lm-model "${LM_MODEL:-Qwen/Qwen2.5-32B-Instruct}" \
  --lam "${LAM:-0.7}" --lr "${LR:-1e-4}" --max-steps "${STEPS:-600}" --cache-dir "$HF_HOME"
echo "[Done] $(date -Is)"
