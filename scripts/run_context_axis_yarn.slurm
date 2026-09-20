#!/bin/bash
#SBATCH -A my-slurm-account
#SBATCH -p ghx4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -o /work/hdd/myproject/anon/slurm_logs/%x-%j.out
# THE CONTEXT-LENGTH AXIS PAST THE NATIVE WINDOW — ACCURACY (user, 2026-09-11): hotpotQA d40 s42 at
# 200 / 240 / 280 / 320 documents (median context 28.0k / 33.8k / 39.7k / 45.5k tokens, max 41k / 49k /
# 57k / 67k), the SAME 600 questions, golds, prompt, metric and extractor as the four native-window
# points (scripts/run_context_axis.slurm: d80/d120/d160; d40 = hofa_*). Both models carry Qwen's
# documented YaRN setting (ROPE_YARN_FACTOR=4 -> rope_scaling yarn, factor 4 over 32768; stamped in
# provenance as axis_rope_yarn_factor). That makes these points a different model from the native four
# — the user's decision is one axis of eight points, with the YaRN boundary stated on the figure.
#
# Decode path: EAGER, like every accuracy arm of the axis (FKV is a timing backend and never supplies an
# accuracy column — CANONICAL_RUNS §1). Files: ctx_ho<D>_{t,o,f}.jsonl, the existing naming, scored by
# scripts/score_context_axis.py under the axis's uniform answer boundary.
#
# CARDS. Memory decides, not preference (CLAUDE.md 0원칙 / 2-card rule). Eager KV for the 32B is
# 256 KiB per token: at d320 the longest context (67.4k tokens) needs 16.9 GiB per row, so B=2 is
# 33.7 + 61 GiB of weights > 95 — the teacher at d320 runs on TWO cards, layer-sharded
# (FUSION_LM_DEV=balanced; the probe measured the one-card B=2 OOM). d280 (57.4k -> 14.4 GiB per row,
# B=2 = 89.7 GiB) is at the edge: it is submitted on one card first and moved to two if the probe's
# one-card attempt did not fit. ours (reader 7B at 56 KiB/token) and floor fit one card everywhere.
# Batch 1 is never a rung.
#
# ★ --resume ON EVERY CALL. These are accuracy arms: a rung that OOMs after writing rows keeps them and
# the next rung continues (batch is not a treatment, CANONICAL_RUNS 2e-note); a job cut by the wall is
# resubmitted and continues.
#   ARM=teacher|ours|floor   DOCS=200|240|280|320
set -u
cd /u/anon/SLM_LM
BASE=/work/hdd/myproject/$USER; export HF_HOME=$BASE/hf
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /u/anon/.venvs/slm_lm/bin/activate
CKPT=$BASE/kvreuse_ckpts
ARM=${ARM:?teacher, ours or floor}; DOCS=${DOCS:?200, 240, 280 or 320 (or 160 = the rope CONTROL)}
export ROPE_YARN_FACTOR=4
case "$ARM-$DOCS" in
  # 160 = the ROPE CONTROL (user, 2026-09-11 "어 해봐"): the native d160 cell re-run with YaRN x4 on both models,
  # same 600 questions and contexts, only the rope changed. Batches = the native cell's measured ones
  # (ctx_ho160_*: teacher 3, ours 8, floor 24; YaRN changes no memory), one rung below each as the ladder.
  teacher-160) L="3 2" ;;   ours-160) L="8 6 4" ;;   floor-160) L="24 16 12" ;;
  teacher-200) L="3 2" ;;   ours-200) L="8 6 4 3" ;;  floor-200) L="32 24 16" ;;
  teacher-240) L="2"   ;;   ours-240) L="6 4 3 2" ;;  floor-240) L="24 16 12" ;;
  teacher-280) L="2"   ;;   ours-280) L="6 4 3 2" ;;  floor-280) L="24 16 12" ;;
  teacher-320) L="2"   ;;   ours-320) L="4 3 2"   ;;  floor-320) L="16 12 8"  ;;
  *) echo "no batch ladder for ARM=$ARM DOCS=$DOCS"; exit 2 ;;
esac
case "$ARM" in
  teacher) A="--model Qwen/Qwen2.5-32B-Instruct --method teacher"; P=t ;;
  floor)   A="--model Qwen/Qwen2.5-7B-Instruct --method teacher";  P=f ;;
  ours)    A="--model Qwen/Qwen2.5-32B-Instruct --slm-model Qwen/Qwen2.5-7B-Instruct --method ours \
              --lam 0.7 --lm-no-accum --slm-lora $CKPT/reader_binding_v5distill_r16_s900 \
              --lm-lora $CKPT/stage2_on_v5reader_lam07_r16_s600"; P=o; export MERGE_LORA=1 ;;
esac
# two cards (sbatch --gpus-per-node=2): the 32B is layer-sharded over both and the ladder starts higher
if [ "$ARM" = teacher ] && [ "${SLURM_GPUS_ON_NODE:-1}" -ge 2 ]; then
  export FUSION_LM_DEV=balanced; L="6 4 3 2"
  echo "=== CTXY teacher d$DOCS on TWO cards (device_map=balanced, layer-sharded), ladder $L ==="
fi
# The control's files carry a "y" (ctx_ho160y_*) so they can never --resume into, or be scored as, the native
# canonical cell ctx_ho160_* (score_context_axis.py reads fixed names per depth; the control has its own scorer).
if [ "$DOCS" -le 160 ]; then OUTF="results/fusionft/ctx_ho${DOCS}y_${P}.jsonl"; else OUTF="results/fusionft/ctx_ho${DOCS}_${P}.jsonl"; fi
for B in $L; do
  echo "=== CTXY $ARM d$DOCS B=$B (YaRN x4, eager) -> $OUTF ==="
  if BATCH_SIZE=$B LONGEST_FIRST=1 python scripts/mtrag_accum.py \
       --bench hotpotqa_st${DOCS}_full --max-conv 600 --max-new 200 $A \
       --resume --out "$OUTF"; then
    echo "=== CTXY OK $ARM d$DOCS B=$B ==="; break
  fi
  echo "!! ctxy $ARM d$DOCS OOM or failure at B=$B, dropping a rung (finished rows kept)"
done
python - "$OUTF" <<'PYEOF'
import json, statistics as st, sys
rows = [r for r in (json.loads(l) for l in open(sys.argv[1]) if l.strip()) if "acc_f1" in r]
if not rows:
    print("[ctxy] NO ROWS"); raise SystemExit(1)
pv = rows[0]["_provenance"]
bs = sorted({r["_provenance"].get("axis_batch_size") for r in rows})
print(f"[ctxy] n={len(rows)}/600 inline-F1={st.mean(r['acc_f1'] for r in rows):.4f} batches={bs} "
      f"yarn={pv.get('axis_rope_yarn_factor')} max_ctx_tok={max(r['acc_ctx_tok'] for r in rows)} "
      f"node={pv.get('slurm_node')} — the canonical score is scripts/score_context_axis.py's re-score")
if len(rows) < 600:
    print("[ctxy] INCOMPLETE — resubmit the same ARM/DOCS; --resume continues it"); raise SystemExit(3)
PYEOF
echo "=== CTXY $ARM d$DOCS DONE ==="
