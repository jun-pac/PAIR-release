#!/bin/bash
#SBATCH -A my-slurm-account
#SBATCH -p ghx4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 1:10:00
#SBATCH -J hocb32
#SBATCH -o /work/hdd/myproject/anon/slurm_logs/%x-%j.out
# THE QUERY LM ALONE, ON HOTPOT (user, 2026-09-17: "hotpot에도 query LM 단독 돌려").
#
# WHAT THIS ANSWERS. The motivation example needs all four arms on the paper's main benchmark: the
# teacher, PAIR, the reader alone (hofa_floor7) and the QUERY LM ALONE. The last one did not exist for
# hotpot — the only closed-book run in the project is LooGLE's lg_closedbook32 (job 3012402). This is
# that same arm on hotpotQA d40's 600 questions: the 32B with the DOCUMENTS STRIPPED, so it answers
# from the question alone, which is exactly what the query branch of the pair sees.
#
# HOW THE CONTEXT IS REMOVED. CTX_CAP=16 truncates the CONTEXT BLOCK ONLY to ~16 tokens (mtrag_accum
# fmt(); the instruction and the question are untouched) and is recorded in provenance as
# ctx_cap_tokens=16. Same switch, same value as the LooGLE closed-book arm, so the two are comparable.
#
# EVERYTHING ELSE IS THE hofa_ FAMILY'S SETTING, checked field by field against hofa_teacher's
# provenance: bench hotpotqa_st40_full, --max-new 200, prompt QA_REASON_V3 (sha c1a237e09583a6a5),
# reason_fix on, reason_hist ref, seed 42, order random, ratio 0.8125. Only the context differs.
#
# BATCH. The prompts are ~600 tokens instead of ~6,900, so the 32B fits a far larger batch than the
# 8 that hofa_teacher used; the ladder descends only in case of OOM. Batch is not a treatment
# (CLAUDE.md), and every arm of this bench already ran at its own batch.
set -u
cd /u/anon/SLM_LM
BASE=/work/hdd/myproject/$USER; export HF_HOME=$BASE/hf
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /u/anon/.venvs/slm_lm/bin/activate
export CTX_CAP=16
OUT=results/fusionft/hofa_closedbook32.jsonl
for B in 32 24 16 8; do
  echo "=== hotpot closed-book 32B, batch $B ==="
  BATCH_SIZE=$B LONGEST_FIRST=1 python scripts/mtrag_accum.py --bench hotpotqa_st40_full --max-conv 3000 \
    --model Qwen/Qwen2.5-32B-Instruct --max-new 200 --method teacher --resume --out "$OUT" 2>&1 | tail -4 && break
  echo "!! closed-book OOM at B=$B"
done
echo "=== DONE ==="
python - <<'PYEOF'
import json, statistics as st, sys
sys.path.insert(0, ".")
from src.eval import compute_best_em_f1
def load(t):
    d = {}
    for l in open(f"results/fusionft/{t}.jsonl"):
        if l.strip():
            r = json.loads(l)
            if "acc_f1" in r: d[(r["conv"], int(r["turn"]))] = r
    return d
C = load("hofa_closedbook32")
pv = list(C.values())[0]["_provenance"]
print(f"  rows {len(C)} | batch {pv.get('axis_batch_size')} | ctx_cap {pv.get('ctx_cap_tokens')} "
      f"| ctx tok median {int(st.median(r['acc_ctx_tok'] for r in C.values()))} | node {pv.get('slurm_node')}")
print("  first three answers:", [r["acc_pred"][:60] for r in list(C.values())[:3]])
T, O, F = load("hofa_teacher"), load("hofa_ours"), load("hofa_floor7")
ks = sorted(set(T) & set(O) & set(F) & set(C))
f1 = lambda r: compute_best_em_f1(r["acc_pred"], [r["gold"]])[1]
print(f"  shared-N {len(ks)}: teacher {st.fmean(f1(T[k]) for k in ks):.4f} | ours {st.fmean(f1(O[k]) for k in ks):.4f} "
      f"| reader alone {st.fmean(f1(F[k]) for k in ks):.4f} | query LM alone {st.fmean(f1(C[k]) for k in ks):.4f}")
cand = [k for k in ks if f1(T[k]) >= .99 and f1(O[k]) >= .99 and f1(F[k]) <= .2 and f1(C[k]) <= .2]
print(f"  teacher+ours right, BOTH branches alone wrong: {len(cand)} questions")
for k in cand[:5]:
    print(f"   -- {T[k]['q'][:110]} | gold {T[k]['gold'][:40]}")
    print(f"      reader   : {F[k]['acc_pred'][:70]!r}")
    print(f"      query LM : {C[k]['acc_pred'][:70]!r}")
PYEOF
python scripts/job_efficiency_check.py "$SLURM_JOB_ID" 'results/fusionft/hofa_closedbook32.jsonl' 2>&1 | tail -3
