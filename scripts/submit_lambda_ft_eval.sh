#!/bin/bash
# Fine-tuned L-fusion at λ 0.75 / 0.80 / 0.85 on the other four benchmarks (user, 2026-09-10: "locomo만 했음? 나머지는?").
# The reader adapter is fixed; the LM adapter is the one TRAINED at the evaluation λ (stage2_on_v5reader_lam{075,080,085}).
# Existing matched points: hotpot 0.70 (hofa_ours), musique 0.70 (mufull_ours), LooGLE 0.70 (lg_SsoloLfus_l07) and 0.85
# (lg_SsoloLfus), CLUTRR-sq 0.70 (clsq_ours). Batches are each benchmark's canonical ours batch; --resume on.
# musique: a seeded (42) 600 of the 2,417 conversations — a λ comparison, not a ledger cell; the 0.70 full log is
# scored on the same 600 ids.
set -euo pipefail
cd /u/anon/SLM_LM
source /u/anon/.venvs/slm_lm/bin/activate
CKPT=/work/hdd/myproject/anon/kvreuse_ckpts
ids_mu=$(python - <<'PY'
import json,collections,random
by=collections.OrderedDict()
for l in open('/work/hdd/myproject/anon/singleturn/musique_st40_full_ref.jsonl'):
    if l.strip(): by.setdefault(json.loads(l)["conversation_id"],1)
print("+".join(random.Random(42).sample(list(by),600)))
PY
)
echo "musique subset: $(echo $ids_mu | tr '+' '\n' | wc -l) conversations"
COMMON="METHOD=ours,MODEL=Qwen/Qwen2.5-32B-Instruct,SLM_MODEL=Qwen/Qwen2.5-7B-Instruct,MAXNEW=200,LM_NO_ACCUM=1,RESUME=1,REASON_HIST=ref,SLM_LORA=$CKPT/reader_binding_v5distill_r16_s900"
JOBS=""
sub() {  # tag bench batch maxconv lam wall extra_export
  local tag=$1 bench=$2 batch=$3 maxconv=$4 lam=$5 wall=$6 extra=${7:-}
  local lt=$(echo $lam | tr -d '.'); [ "$lt" = "08" ] && lt=080
  local j=$(sbatch --parsable -A my-slurm-account -J lfe-$tag$lt -t $wall \
      --export=ALL,$COMMON,BENCH=$bench,LAM=$lam,BATCH_SIZE=$batch,MAXCONV=$maxconv,LM_LORA=$CKPT/stage2_on_v5reader_lam${lt}_r16_s600,OUT=results/fusionft/lfe_${tag}_SsoloLfus${lt}_lam${lt}.jsonl${extra} \
      scripts/run_mtrag_accum.slurm)
  echo "$tag λ$lam -> $j"; JOBS="$JOBS $j"
}
for lam in 0.75 0.80 0.85; do sub ho hotpotqa_st40_full 16 600 $lam 0:50:00; done
for lam in 0.75 0.80 0.85; do sub mu musique_st40_full 16 2417 $lam 0:50:00 ",PAPERS=$ids_mu"; done
for lam in 0.75 0.80;      do sub lg loogle 3 70 $lam 1:45:00; done
for lam in 0.75 0.80 0.85; do sub cl clutrr_sq 24 254 $lam 0:30:00; done
echo "$JOBS" > /tmp/claude-92137/-u-anon-SLM-LM/baedb6b9-44fd-4769-9306-fe52f5ea3631/scratchpad/lfe_jobs
python scripts/gpu_budget.py add lambda-ft-260910 $JOBS | tail -1
