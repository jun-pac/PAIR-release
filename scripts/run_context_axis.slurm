#!/bin/bash
#SBATCH -A my-slurm-account
#SBATCH -p ghx4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -o /work/hdd/myproject/anon/slurm_logs/%x-%j.out
# THE CONTEXT-LENGTH AXIS (user, 2026-09-02): "context가 길어지면 batch도 바뀌고 throughput gain도
# 바뀔 수 있음 ... 40(이미 있고), 80, 160, 320 ... baseline은 다 할 필요 없고 teacher에 대해서는
# 결과를 봐야할듯. throughput gain도."
#
# One run yields BOTH panels: token-F1 for accuracy and conv_wall_s for end-to-end answers/s, each
# arm at its own largest surviving batch -- which is the point, since the batch an arm can hold is
# exactly what changes with context.
#
# WHAT IS HELD FIXED: the same 600 questions, the same golds, the same prompt, the same metric, and
# the same 128,000-character truncation. Only the number of retrieved documents varies.
# WHAT THE X AXIS IS: MEASURED median context tokens, not the document count. At d320 the raw
# context is 169k characters and the truncation binds, so d320 is 31.0k tokens rather than twice
# d160's 21.5k. Measured medians: d40 5.4k, d80 10.6k, d160 21.5k, d320 31.0k -- a 5.7x span.
# Plotting the nominal doc count would imply a spacing the data does not have.
#
# ARM=teacher|ours|floor  DOCS=80|160|320
set -u
cd /u/anon/SLM_LM
BASE=/work/hdd/myproject/$USER; export HF_HOME=$BASE/hf
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /u/anon/.venvs/slm_lm/bin/activate
CKPT=$BASE/kvreuse_ckpts
ARM=${ARM:?}; DOCS=${DOCS:?}
# Ladders from the measured token counts and a 32B KV of 256 KiB/token against ~34 GiB free after
# weights; the 7B reader is 56 KiB/token, which is why ours and floor start far higher.
# ★ Every ARM-DOCS pair must appear. Jobs 3074905/06/07 died in 5 SECONDS because DOCS=120 had no
# branch here, `set -u` hit the unset L, and the script exited before loading anything -- the axis
# silently lost its third point. d320 was withdrawn (past the models' 32,768-token window), so the
# axis is 40 / 80 / 120 / 160 and 40 already exists.
case "$ARM-$DOCS" in
  teacher-80)  L="10 8 6 4" ;;   ours-80)  L="16 12 8 6" ;;   floor-80)  L="40 32 24 16" ;;
  teacher-120) L="8 6 4 3"  ;;   ours-120) L="14 10 8 6" ;;   floor-120) L="32 24 16 12" ;;
  teacher-160) L="6 4 3 2"  ;;   ours-160) L="12 8 6 4"  ;;   floor-160) L="24 16 12 8" ;;
  *) echo "no batch ladder for ARM=$ARM DOCS=$DOCS"; exit 2 ;;
esac
case "$ARM" in
  teacher) A="--model Qwen/Qwen2.5-32B-Instruct --method teacher"; P=t ;;
  floor)   A="--model Qwen/Qwen2.5-7B-Instruct --method teacher";  P=f ;;
  ours)    A="--model Qwen/Qwen2.5-32B-Instruct --slm-model Qwen/Qwen2.5-7B-Instruct --method ours \
              --lam 0.7 --lm-no-accum --slm-lora $CKPT/reader_binding_v5distill_r16_s900 \
              --lm-lora $CKPT/stage2_on_v5reader_lam07_r16_s600"; P=o; export MERGE_LORA=1 ;;
esac
for B in $L; do
  echo "=== CTX $ARM d$DOCS B=$B ==="
  if BATCH_SIZE=$B LONGEST_FIRST=1 python scripts/mtrag_accum.py \
       --bench hotpotqa_st${DOCS}_full --max-conv 600 --max-new 200 $A \
       --out "results/fusionft/ctx_ho${DOCS}_${P}.jsonl"; then
    echo "=== CTX OK $ARM d$DOCS B=$B ==="; break
  fi
  echo "=== CTX FAILED $ARM d$DOCS B=$B ==="
done
echo "=== CONTEXT AXIS $ARM d$DOCS DONE ==="
