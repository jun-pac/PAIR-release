#!/bin/bash
#SBATCH -A my-slurm-account
#SBATCH -p ghx4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -o /work/hdd/myproject/anon/slurm_logs/%x-%j.out
# D_fused on the PAPER's 240-example basis (user, 2026-09-18: "나머지는 뒤로 미루고 최우선으로 240개 저걸
# 얻어야함"). Every setting below is copied field for field from run_branch_kl_grid.slurm, the launcher that
# produced the published cell (job 3107089, 13:18 on one card for strictly MORE work: the teacher pass plus
# five reader passes). This job re-runs only the 32B no-context forward and the 7B full-context forward and
# reads the teacher trajectory + logits from lm32B_Tpass.npz, so the published numbers' basis is untouched.
# ANALYSIS PROBE: B=1 forced forwards, no timing, no benchmark score (preflight override recorded).
set -u
cd /u/anon/SLM_LM
BASE=/work/hdd/myproject/$USER; export HF_HOME=$BASE/hf
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# the docs cache the original dump used; without it the loader re-runs BM25 over the whole retrieval split
export HOTPOT_DOCS_CACHE_DIR=results/_docs_cache ANSWER_PROMPT_VARIANT=reason_v3
source /u/anon/.venvs/slm_lm/bin/activate
python scripts/branch_kl_fused.py --lm "${LM:-32B}" --reader "${READER:-7B}" \
       --sample "${SAMPLE:-240}" --lam "${LAM:-0.7}" ${FT:+--ft} 2>&1 | grep -vE "^Loading"
rc=${PIPESTATUS[0]}
echo "=== BRANCH-KL FUSED DONE (exit $rc) ==="
exit $rc
