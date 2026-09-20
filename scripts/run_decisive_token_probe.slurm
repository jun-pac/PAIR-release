#!/bin/bash
#SBATCH -A my-slurm-account
#SBATCH -p ghx4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 0:35:00
#SBATCH -J dectok
#SBATCH -o /work/hdd/myproject/anon/slurm_logs/%x-%j.out
# The decisive token of the motivation example, in numbers (user, 2026-09-18). ONE conversation:
# hotpotQA ho-5a794119554299029c4b5f3c, the reader-alone text and ours forced through the reader
# (7B + reader adapter), the query LM (32B + lam07, no context) and the teacher (32B, adapter off,
# full context); the fork is position 45, "... 17 September 1962. Therefore, |" -> " Baz" against " N".
# Also decodes the query LM's own free-running no-context answer to this question, which is the arm the
# example is missing while the 600-question closed-book job (3167579) waits in the queue.
# ANALYSIS PROBE: B=2 forced forwards + one 200-token greedy decode. No timing, no benchmark score.
set -u
cd /u/anon/SLM_LM
BASE=/work/hdd/myproject/$USER; export HF_HOME=$BASE/hf
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /u/anon/.venvs/slm_lm/bin/activate
python scripts/decisive_token_probe.py 2>&1 | grep -vE "^Loading"
echo "=== DECISIVE-TOKEN PROBE DONE (exit ${PIPESTATUS[0]}) ==="
