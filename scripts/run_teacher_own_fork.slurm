#!/bin/bash
#SBATCH -A my-slurm-account
#SBATCH -p ghx4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 20:00
#SBATCH -J towfork
#SBATCH -o /work/hdd/myproject/anon/slurm_logs/%x-%j.out
# One forced forward of the full-context 32B along its own published answer, to read its distribution at
# its own decisive token. Settings copied from run_decisive_token_probe.slurm.
set -u
cd /u/anon/SLM_LM
BASE=/work/hdd/myproject/$USER; export HF_HOME=$BASE/hf
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HOTPOT_DOCS_CACHE_DIR=results/_docs_cache ANSWER_PROMPT_VARIANT=reason_v3
source /u/anon/.venvs/slm_lm/bin/activate
python scripts/teacher_own_fork.py 2>&1 | grep -vE "^Loading"
rc=${PIPESTATUS[0]}
echo "=== TEACHER OWN FORK DONE (exit $rc) ==="
exit $rc
