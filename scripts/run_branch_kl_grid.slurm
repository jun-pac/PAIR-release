#!/bin/bash
#SBATCH -A my-slurm-account
#SBATCH -p ghx4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -o /work/hdd/myproject/anon/slurm_logs/%x-%j.out
# Branch-KL grid, one LM size per job (user, 2026-09-07: "어 이건 다 해도 되겠다"). LM=32B|14B|7B|3B (72B
# separately, two cards, after the disk decision) and READERS = every smaller size. The 32B job carries
# the canary (32B+7B must reproduce the published 0.3826 / 0.7477 / 0.2208 on 4,362 TASK positions) and
# exits 3 if it does not, so the afterok-chained blocks never run on a basis that failed to reproduce.
# ANALYSIS PROBE: B=1 forced forwards, no timing, no benchmark score (preflight override recorded).
set -u
cd /u/anon/SLM_LM
BASE=/work/hdd/myproject/$USER; export HF_HOME=$BASE/hf
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# the docs cache the original dump used (run_branch_logits.slurm); without it the loader re-runs BM25
export HOTPOT_DOCS_CACHE_DIR=results/_docs_cache ANSWER_PROMPT_VARIANT=reason_v3
source /u/anon/.venvs/slm_lm/bin/activate
LM=${LM:?32B|14B|7B|3B|72B}; READERS=${READERS:?comma list}
DM=${LM_DEVICE_MAP:-cuda:0}
python scripts/branch_kl_grid.py --lm "$LM" --readers "$READERS" --sample "${SAMPLE:-60}" --dataset hotpotqa --lm-device-map "$DM" 2>&1 | grep -vE "^Loading"
rc=${PIPESTATUS[0]}    # captured BEFORE any other command: the first submission read the echo's status (0) and released the afterok chain on a failed block
echo "=== BRANCH-KL GRID $LM DONE (exit $rc) ==="
exit $rc
