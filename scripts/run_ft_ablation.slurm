#!/bin/bash
#SBATCH -A my-slurm-account
#SBATCH -p ghx4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -o /work/hdd/myproject/anon/slurm_logs/%x-%j.out
# THE FINE-TUNING ABLATION, ON THE CURRENT BASIS (user-approved 2026-09-02).
#
# WHY THESE FIVE RUNS EXIST. The plain / reader-only / reader+LM-branch progression is published in
# MAIN_RESULTS §1, but those musique and hotpot rows come from an older single-turn launcher whose
# logs carry NO provenance fingerprint and whose context construction differs (40 docs retrieved and
# cut at 30k characters vs this bench's pre-built reference at 128k). They therefore cannot be read
# beside §2. On the current basis the progression exists for LoCoMo-30 alone. These runs put the
# other three benchmarks on it.
#
# EVERY SETTING IS COPIED FROM THAT BENCH'S OWN CANONICAL `ours` RUN so the ONLY thing that varies
# is which adapters are loaded:
#   hotpot   hofa_ours.jsonl        lam 0.7  B=16  MERGE_LORA=1  lm_no_accum
#   musique  mufull_ours.jsonl      lam 0.7  B=16  MERGE_LORA=1  lm_no_accum
#   loogle   lg_SsoloLfus_l07.jsonl lam 0.7  B=3   MERGE_LORA=0  lm_no_accum
# plain   = neither adapter        (MERGE_LORA is moot: there is nothing to merge)
# S-solo  = the reader adapter only, no LM-branch adapter
# LooGLE's S-solo already exists (lg_Ssolo_l07) so only its plain arm is run here.
# BENCH=hotpot|musique|loogle
set -u
cd /u/anon/SLM_LM
BASE=/work/hdd/myproject/$USER; export HF_HOME=$BASE/hf
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /u/anon/.venvs/slm_lm/bin/activate
CKPT=$BASE/kvreuse_ckpts; READER=$CKPT/reader_binding_v5distill_r16_s900
BENCH=${BENCH:?hotpot|musique|loogle}
case "$BENCH" in
  hotpot)  BA="--bench hotpotqa_st40_full --max-conv 600";  B=16; MG=1; PFX=abl_ho; ARMS="plain ssolo" ;;
  musique) BA="--bench musique_st40_full --max-conv 2417";  B=16; MG=1; PFX=abl_mu; ARMS="plain ssolo" ;;
  loogle)  BA="--bench loogle";                             B=3;  MG=0; PFX=abl_lg; ARMS="plain" ;;
esac
COMMON="$BA --model Qwen/Qwen2.5-32B-Instruct --slm-model Qwen/Qwen2.5-7B-Instruct \
 --max-new 200 --method ours --lam 0.7 --lm-no-accum"
for ARM in $ARMS; do
  EXTRA=""; [ "$ARM" = ssolo ] && EXTRA="--slm-lora $READER"
  echo "=== FT-ABLATION $BENCH $ARM  B=$B MERGE_LORA=$MG ==="
  MERGE_LORA=$MG BATCH_SIZE=$B LONGEST_FIRST=1 python scripts/mtrag_accum.py $COMMON $EXTRA \
    --out "results/fusionft/${PFX}_${ARM}.jsonl" || echo "!!! $BENCH $ARM FAILED"
done
python - "$PFX" <<'PYEOF'
import glob, json, sys, statistics as st
for f in sorted(glob.glob(f"results/fusionft/{sys.argv[1]}_*.jsonl")):
    rows=[json.loads(l) for l in open(f) if l.strip()]
    if not rows: continue
    pv=rows[0]['_provenance']
    print(f"{f.split('/')[-1]:22s} n={len(rows):5d} F1={st.mean(r['acc_f1'] for r in rows):.4f} "
          f"EM={st.mean(r['acc_em'] for r in rows):.4f} B={pv['axis_batch_size']} "
          f"reader={pv.get('slm_lora')} lm={pv.get('lm_lora')}")
PYEOF
echo "=== FT-ABLATION $BENCH DONE ==="
