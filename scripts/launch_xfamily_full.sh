#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# Cross-family doc-grid EXTENSION to 240/280/320 — musique + hotpot, byte-identical to the old 40-200 grid.
# Recipe verified 2026-07-21 by doc200 canary (shared_prompt_sha256 + em/f1 vs old logs).
#   ★ max_length: MUSIQUE=32000 (Qwen-3B 32K window), HOTPOT=24000.  (canary-proven; do NOT swap)
#   no-ES (sketch_max_new=0), reason_then_answer, manual greedy, lambda 0.7, max_new 256, NO --revert.
#   musique = deterministic first-N (SAMPLE 2500 qwen / 2417 gemma,olmo). hotpot = SAMPLE 600 + SAMPLE_SEED=42.
#   Launchers: run_qa.slurm (qwen musique) / run_qa_xfamily.slurm (gemma,olmo musique + ALL hotpot). Both --output.
#   Output names are the recovery_vs_context.py globs (see STATUS_260721_xfamily.md).
# Usage: bash launch_xfamily_full.sh <musique|hotpot> <DOC>
# ─────────────────────────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd /u/anon/SLM_LM
BENCH="$1"; DOC="$2"
MDC="MUSIQUE_DOCS_CACHE_DIR=results/_docs_cache"; HDC="HOTPOT_DOCS_CACHE_DIR=results/_docs_cache"

if [ "$BENCH" = "musique" ]; then
  ML=32000; QSAMP=2500; XSAMP=2417; SEED=""
  # qwen (run_qa.slurm): teacher / ours / floor
  sbatch -t "${TLIMIT:-4:00:00}" --export=ALL,RESUME=${RESUME:-0},BENCH=musique,DOC=$DOC,SAMPLE=$QSAMP,MAXLEN=$ML,MAX_NEW=256,LAMBDA=0.7,LM_TAG=14B,MODE=teacher,$MDC              scripts/run_qa.slurm
  sbatch -t "${TLIMIT:-4:00:00}" --export=ALL,RESUME=${RESUME:-0},BENCH=musique,DOC=$DOC,SAMPLE=$QSAMP,MAXLEN=$ML,MAX_NEW=256,LAMBDA=0.7,LM_TAG=14B,SLM_TAG=3B,MODE=slm_lm,$MDC    scripts/run_qa.slurm
  sbatch -t "${TLIMIT:-4:00:00}" --export=ALL,RESUME=${RESUME:-0},BENCH=musique,DOC=$DOC,SAMPLE=$QSAMP,MAXLEN=$ML,MAX_NEW=256,LAMBDA=0.7,LM_TAG=3B,MODE=teacher,$MDC               scripts/run_qa.slurm
  # gemma (run_qa_xfamily.slurm, TAG=gemma)
  G="TAG=gemma,BENCH=musique,DOC=$DOC,SAMPLE=$XSAMP,MAXLEN=$ML,MAX_NEW=256,LAMBDA=0.7,$MDC"
  sbatch -t "${TLIMIT:-4:00:00}" --export=ALL,RESUME=${RESUME:-0},$G,LM_MODEL=google/gemma-3-12b-it,MODE=teacher                                scripts/run_qa_xfamily.slurm
  sbatch -t "${TLIMIT:-4:00:00}" --export=ALL,RESUME=${RESUME:-0},$G,LM_MODEL=google/gemma-3-12b-it,SLM_MODEL=google/gemma-3-4b-it,MODE=slm_lm   scripts/run_qa_xfamily.slurm
  sbatch -t "${TLIMIT:-4:00:00}" --export=ALL,RESUME=${RESUME:-0},$G,LM_MODEL=google/gemma-3-4b-it,MODE=teacher                                  scripts/run_qa_xfamily.slurm
  # olmo (TAG=olmo3, USE_CHAT=1; 32B GPN=2, 7B floor GPN=1)
  O="TAG=olmo3,BENCH=musique,DOC=$DOC,SAMPLE=$XSAMP,MAXLEN=$ML,MAX_NEW=256,LAMBDA=0.7,USE_CHAT=1,$MDC"
  sbatch -t "${TLIMIT:-4:00:00}" --gpus-per-node=2 --export=ALL,RESUME=${RESUME:-0},$O,GPN=2,LM_MODEL=allenai/Olmo-3.1-32B-Instruct,MODE=teacher                                            scripts/run_qa_xfamily.slurm
  sbatch -t "${TLIMIT:-4:00:00}" --gpus-per-node=2 --export=ALL,RESUME=${RESUME:-0},$O,GPN=2,LM_MODEL=allenai/Olmo-3.1-32B-Instruct,SLM_MODEL=allenai/OLMo-3-7B-Instruct,MODE=slm_lm         scripts/run_qa_xfamily.slurm
  sbatch -t "${TLIMIT:-4:00:00}" --export=ALL,RESUME=${RESUME:-0},$O,GPN=1,LM_MODEL=allenai/OLMo-3-7B-Instruct,MODE=teacher                                               scripts/run_qa_xfamily.slurm

elif [ "$BENCH" = "hotpot" ]; then
  ML=24000; XSAMP=600
  H="BENCH=hotpotqa,DOC=$DOC,SAMPLE=$XSAMP,SAMPLE_SEED=42,MAXLEN=$ML,MAX_NEW=256,LAMBDA=0.7,$HDC"
  # qwen (TAG=xf, GPN=1)
  sbatch -t "${TLIMIT:-4:00:00}" --export=ALL,RESUME=${RESUME:-0},TAG=xf,$H,LM_MODEL=Qwen/Qwen2.5-14B-Instruct,MODE=teacher                                     scripts/run_qa_xfamily.slurm
  sbatch -t "${TLIMIT:-4:00:00}" --export=ALL,RESUME=${RESUME:-0},TAG=xf,$H,LM_MODEL=Qwen/Qwen2.5-14B-Instruct,SLM_MODEL=Qwen/Qwen2.5-3B-Instruct,MODE=slm_lm    scripts/run_qa_xfamily.slurm
  sbatch -t "${TLIMIT:-4:00:00}" --export=ALL,RESUME=${RESUME:-0},TAG=xf,$H,LM_MODEL=Qwen/Qwen2.5-3B-Instruct,MODE=teacher                                      scripts/run_qa_xfamily.slurm
  # gemma
  sbatch -t "${TLIMIT:-4:00:00}" --export=ALL,RESUME=${RESUME:-0},TAG=xf,$H,LM_MODEL=google/gemma-3-12b-it,MODE=teacher                                         scripts/run_qa_xfamily.slurm
  sbatch -t "${TLIMIT:-4:00:00}" --export=ALL,RESUME=${RESUME:-0},TAG=xf,$H,LM_MODEL=google/gemma-3-12b-it,SLM_MODEL=google/gemma-3-4b-it,MODE=slm_lm           scripts/run_qa_xfamily.slurm
  sbatch -t "${TLIMIT:-4:00:00}" --export=ALL,RESUME=${RESUME:-0},TAG=xf,$H,LM_MODEL=google/gemma-3-4b-it,MODE=teacher                                          scripts/run_qa_xfamily.slurm
  # olmo (USE_CHAT=1; 32B GPN=2, 7B floor GPN=1)
  sbatch -t "${TLIMIT:-4:00:00}" --gpus-per-node=2 --export=ALL,RESUME=${RESUME:-0},TAG=xf,$H,GPN=2,USE_CHAT=1,LM_MODEL=allenai/Olmo-3.1-32B-Instruct,MODE=teacher                                     scripts/run_qa_xfamily.slurm
  sbatch -t "${TLIMIT:-4:00:00}" --gpus-per-node=2 --export=ALL,RESUME=${RESUME:-0},TAG=xf,$H,GPN=2,USE_CHAT=1,LM_MODEL=allenai/Olmo-3.1-32B-Instruct,SLM_MODEL=allenai/OLMo-3-7B-Instruct,MODE=slm_lm  scripts/run_qa_xfamily.slurm
  sbatch -t "${TLIMIT:-4:00:00}" --export=ALL,RESUME=${RESUME:-0},TAG=xf,$H,GPN=1,USE_CHAT=1,LM_MODEL=allenai/OLMo-3-7B-Instruct,MODE=teacher                                        scripts/run_qa_xfamily.slurm
else echo "usage: $0 <musique|hotpot> <DOC>"; exit 1; fi
echo "[launched] $BENCH doc=$DOC (9 jobs: 3 families x teacher/ours/floor)"
