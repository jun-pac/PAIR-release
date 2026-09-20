#!/bin/bash
#SBATCH -A my-slurm-account
#SBATCH -p ghx4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 1:30:00
#SBATCH -J hotA
#SBATCH -o /work/hdd/myproject/anon/slurm_logs/%x-%j.out
# hotpot option A (user-approved 2026-08-30): the lambda-free arms (teacher/floor/press) are
# reinstated from the quarantine; ONLY the ours arm is re-run, at hotpot's CANONICAL lambda 0.7
# (the quarantined ours ran 0.85 — void). Ledger campaign: hotpot-A (quote 0.75 GPU-h).
set -u
cd /u/anon/SLM_LM
BASE=/work/hdd/myproject/$USER; export HF_HOME=$BASE/hf
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /u/anon/.venvs/slm_lm/bin/activate
CKPT=$BASE/kvreuse_ckpts
BATCH_SIZE=16 MERGE_LORA=1 python scripts/mtrag_accum.py --bench hotpotqa_st40_full --max-conv 3000 \
  --model Qwen/Qwen2.5-32B-Instruct --max-new 200 --method ours --lam 0.7 --lm-no-accum \
  --slm-model Qwen/Qwen2.5-7B-Instruct --slm-lora $CKPT/reader_binding_v5distill_r16_s900 \
  --lm-lora $CKPT/stage2_on_v5reader_lam07_r16_s600 \
  --out results/fusionft/hofa_ours.jsonl 2>&1 | tail -3
python - <<'PYEOF'
import json, glob, statistics as st
out = {}
for f in sorted(glob.glob("results/fusionft/hofa_*.jsonl")):
    rows = [json.loads(l) for l in open(f) if l.strip()]
    if not rows: continue
    tag = f.split("hofa_")[1][:-6]
    out[tag] = dict(f1=round(st.mean(r["acc_f1"] for r in rows), 4),
                    em=round(st.mean(r["acc_em"] for r in rows), 4), n=len(rows))
    print(tag, out[tag])
json.dump(out, open("results/timing/hotpotqa_st40_acc_full.json", "w"), indent=2)
PYEOF
python scripts/job_efficiency_check.py "$SLURM_JOB_ID" 'results/fusionft/hofa_ours.jsonl' 2>&1 | tail -3
echo "=== HOTA DONE ==="
