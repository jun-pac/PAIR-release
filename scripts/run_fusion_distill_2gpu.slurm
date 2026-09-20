#!/bin/bash
#SBATCH -J fdistill
#SBATCH -p ghx4
#SBATCH -A my-slurm-account
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH -t 3:00:00
#SBATCH -o /work/hdd/myproject/%u/slurm_logs/%x-%j.out
#
# 2-GPU variant (2026-08-07): 32B+7B = 76G weights alone -> OOM on one 95GB GH200 (measured: 10MiB free
# at first LM grad-forward). Both models device_map=auto-sharded across 2 GPUs. ONLY for pairs that cannot
# fit one card; the canonical 14B+3B recipe stays on run_fusion_distill.slurm (1 GPU).
# END-TO-END FUSION DISTILLATION — LoRA-FT the 14B LM so the FUSED SLM+LM reproduces the full-context teacher.
# See LORA_FUSION_FT_PLAN.md. 1 GPU (14B bf16 + 3B frozen + LoRA fit one GH200; the LM grad-forward is over the
# short query+completion). Params via --export=ALL,...: TEACHERLOG (req), SAMPLE, DOC, MAXLEN, LAM, LR, EPOCHS,
# GRAD_ACCUM, F1_THRESH, LORA_R, LORA_ALPHA, OUT (req), MAX_STEPS(0=full).
set -euo pipefail
BASE=/work/hdd/myproject/$USER; export HF_HOME=$BASE/hf
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
# ★ 2026-07-29: canonical QA_REASON_V3 (c1a237e0) — MUST equal the variant used for teacher-target generation
# AND for eval. Pass ANSWER_PROMPT_VARIANT=reason_then_answer to reproduce the old (f92f6e17) adapters.
export ANSWER_PROMPT_VARIANT="${ANSWER_PROMPT_VARIANT:-reason_v3}"
cd /u/anon/SLM_LM
source /u/anon/.venvs/slm_lm/bin/activate

MIX_CORPUS="${MIX_CORPUS:-}"     # benchmark-agnostic held-out corpus from ft_mix_teacher_gen.py
if [ -z "$MIX_CORPUS" ]; then TEACHERLOG="${TEACHERLOG:?set TEACHERLOG (single-source) or MIX_CORPUS (held-out mix)}"; else TEACHERLOG=""; fi
OUT="${OUT:?set OUT to the adapter output dir}"
SAMPLE="${SAMPLE:-3000}"; DOC="${DOC:-40}"; MAXLEN="${MAXLEN:-30000}"; LAM="${LAM:-0.7}"
# ★ F1_THRESH defaults to 0: gold must not appear anywhere in the pipeline (behaviour cloning). >0 = a
# clearly-labelled oracle-filtered ABLATION only.  MAX_STEPS should be FIXED across adapters that will be
# compared, or in-domain-vs-held-out is confounded with training amount (the 2026-07 runs were 238-865 steps).
LR="${LR:-1e-4}"; EPOCHS="${EPOCHS:-2}"; GRAD_ACCUM="${GRAD_ACCUM:-8}"; F1_THRESH="${F1_THRESH:-0}"
REQ_REASON="${REQ_REASON:-0}"; LOSS_NORM="${LOSS_NORM:-token}"; TERMINATOR="${TERMINATOR:-both}"
KL_W="${KL_W:-1.0}"; KL_T="${KL_T:-1.0}"; ONP="${ONP:-0}"; ONP_MAX="${ONP_MAX:-24}"   # soft-target distillation (1.0 = pure KL); KL_W=0 = old hard-token CE
LORA_R="${LORA_R:-16}"; LORA_ALPHA="${LORA_ALPHA:-32}"; MAX_STEPS="${MAX_STEPS:-0}"
LM_MODEL="${LM_MODEL:-Qwen/Qwen2.5-14B-Instruct}"; SLM_MODEL="${SLM_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
TS=(); [ "${TRAIN_SLM:-0}" = "1" ] && TS=(--train-slm)   # joint SLM+LM LoRA (else LM-only)

SRC=(); if [ -n "$MIX_CORPUS" ]; then SRC=(--mix-corpus "$MIX_CORPUS"); else SRC=(--teacher-log "$TEACHERLOG"); fi
echo "[Info] $(date -Is) fusion-distill src=${MIX_CORPUS:-$TEACHERLOG} prompt=$ANSWER_PROMPT_VARIANT SAMPLE=$SAMPLE DOC=$DOC LAM=$LAM LR=$LR EPOCHS=$EPOCHS MAX_STEPS=$MAX_STEPS gold_gate=$F1_THRESH req_reason=$REQ_REASON loss_norm=$LOSS_NORM terminator=$TERMINATOR kl_w=$KL_W kl_T=$KL_T onpolicy=$ONP train_slm=${TRAIN_SLM:-0} -> $OUT"
python scripts/fusion_distill_train.py \
  "${SRC[@]}" --require-reasoning "$REQ_REASON" --loss-norm "$LOSS_NORM" --target-terminator "$TERMINATOR" \
  --kl-weight "$KL_W" --kl-temperature "$KL_T" --onpolicy-frac "$ONP" --onpolicy-max-tokens "$ONP_MAX" \
  --lm-model "$LM_MODEL" --slm-model "$SLM_MODEL" \
  --sample "$SAMPLE" --doc-number "$DOC" --max-length "$MAXLEN" --lam "$LAM" \
  --f1-threshold "$F1_THRESH" --lora-r "$LORA_R" --lora-alpha "$LORA_ALPHA" \
  --lr "$LR" --epochs "$EPOCHS" --grad-accum "$GRAD_ACCUM" --max-steps "$MAX_STEPS" "${TS[@]}" \
  --cache-dir "$HF_HOME" --device auto --out "$OUT"
echo "[Done] $(date -Is)"
