#!/bin/bash
#SBATCH -J qax
#SBATCH -p ghx4
#SBATCH -A my-slurm-account
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 1:30:00
#SBATCH -o /work/hdd/myproject/%u/slurm_logs/%x-%j.out
#
# CROSS-FAMILY QA sandwich (gemma-3 / olmo-3 / llama-2 / ...) — mirrors run_qa.slurm's EXACT fair recipe
# (no-ES + reason_then_answer + manual + stop-markers, max_new 256, maxlen 24000) so cross-family results
# are directly comparable to the firm Qwen 14B+3B numbers. Only the model paths (and, for OLMo-3, the
# chat-template flag) differ. Goal: confirm the fusion phenomenon is OBSERVED across model families at
# small N (breadth over depth), NOT to finalize. Params via --export=ALL,...:
#   LM_MODEL (req, full HF path), SLM_MODEL (req for slm_lm), MODE(teacher|slm_lm), BENCH, DOC, LAMBDA,
#   SAMPLE, MAX_NEW, MAXLEN, USE_CHAT(1 for OLMo-3), REVERT(1 to add --revert-previous-prompt),
#   SPLIT_ARG(e.g. test for CLUTRR), GPN(2 => 2-GPU auto_disjoint for a big LM like olmo-32b), TAG.
set -euo pipefail
BASE=/work/hdd/myproject/$USER; export HF_HOME=$BASE/hf
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export ANSWER_PROMPT_VARIANT="${ANSWER_PROMPT_VARIANT:-reason_then_answer}"
cd /u/anon/SLM_LM
source /u/anon/.venvs/slm_lm/bin/activate

LM_MODEL="${LM_MODEL:?set LM_MODEL to a full HF path}"; MODE="${MODE:-teacher}"
BENCH="${BENCH:-musique}"; DOC="${DOC:-120}"; LAMBDA="${LAMBDA:-0.7}"
SAMPLE="${SAMPLE:-300}"; PASS_NUMBER="${PASS_NUMBER:-0}"
MAX_NEW="${MAX_NEW:-256}"; MAXLEN="${MAXLEN:-24000}"
TAG="${TAG:-xfam}"
if [ "$MODE" = "teacher" ]; then SLM_MODEL="$LM_MODEL"; else SLM_MODEL="${SLM_MODEL:?set SLM_MODEL for slm_lm}"; fi
LMt=$(basename "$LM_MODEL"); SLMt=$(basename "$SLM_MODEL")
SAMPLE_SEED="${SAMPLE_SEED:-}"; SEEDTAG=""; [ -n "$SAMPLE_SEED" ] && SEEDTAG="s${SAMPLE_SEED}"
if [ "$MODE" = "teacher" ]; then RESULT="results/${TAG}_${BENCH}d${DOC}_t_${LMt}_n${SAMPLE}${SEEDTAG}.jsonl"
else RESULT="results/${TAG}_${BENCH}d${DOC}_ours_${LMt}-${SLMt}_l${LAMBDA/./}_n${SAMPLE}${SEEDTAG}.jsonl"; fi

CHAT=(); [ "${USE_CHAT:-0}" = "1" ] && CHAT=(--use-chat-template)
REV=(); [ "${REVERT:-0}" = "1" ] && REV=(--revert-previous-prompt)
SPL=(); [ -n "${SPLIT_ARG:-}" ] && SPL=(--split "$SPLIT_ARG")
if [ "${GPN:-1}" = "2" ]; then
  PLACE=(--placement-strategy auto_disjoint --lm-device-map balanced_low_0 --slm-device-map cuda:0 --teacher-device-map balanced_low_0)
else
  PLACE=(--placement-strategy manual --lm-device-map cuda:0 --slm-device-map cuda:0)
fi
echo "[Info] $(date -Is) XFAM $BENCH doc=$DOC MODE=$MODE LM=$LM_MODEL SLM=$SLM_MODEL lambda=$LAMBDA chat=${USE_CHAT:-0} gpn=${GPN:-1} -> $RESULT"
python scripts/run_evidence_sketch_experiments.py \
  --dataset "$BENCH" --doc-number "$DOC" \
  --lm-model "$LM_MODEL" --slm-model "$SLM_MODEL" \
  --sample "$SAMPLE" --pass-number "$PASS_NUMBER" \
  $([ -n "$SAMPLE_SEED" ] && echo --sample-seed "$SAMPLE_SEED") \
  "${SPL[@]}" \
  --max-new-tokens "$MAX_NEW" --sketch-max-new-tokens 0 --max-length "$MAXLEN" \
  --fusion-mode "${FUSION_MODE:-weighted_sum}" --lambda-weight "$LAMBDA" --entropy-scale 1.0 \
  --decode-sync-interval 1 \
  "${PLACE[@]}" "${CHAT[@]}" "${REV[@]}" \
  --stop-on-repeat "Final Answer:" --stop-after-answer "Final Answer:" \
  --cache-dir "$HF_HOME" --modes "$MODE" \
  --single-model-decoding manual \
  $([ "${RESUME:-0}" = "1" ] && echo --resume) \
  --output "$RESULT"
