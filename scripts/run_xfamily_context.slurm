#!/bin/bash
#SBATCH -A my-slurm-account
#SBATCH -p ghx4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -o /work/hdd/myproject/anon/slurm_logs/%x-%j.out
# THE CONTEXT AXIS FOR THE OTHER MODEL FAMILIES (2026-09-03, user: "다 해 전부다").
#
# WHY. The context-length result — ours' closeness RISES with context (80% -> 87% on Qwen) while
# every press falls — is currently one family's finding. Gemma-3 and OLMo-3 already have the d40
# point measured (ledger §8), so ONE more context length per family turns a point into a line, and
# it needs no new training: the adapters exist and this is the same eval.
#
# 23.3k tokens fits both families (OLMo-3 max_position_embeddings=65536; Gemma-3's config omits the
# field and takes the 131072 class default), so d160 is a longer run of the same experiment and not
# a different one — which is exactly why d320 was dropped for Qwen.
#
# ACCURACY ONLY. No throughput or memory number is taken from these runs: OLMo-3-7B is full-MHA at
# 512 KiB per token, twice its own 32B teacher's, so a memory claim for that pair is impossible and
# a latency one meaningless. That constraint does not touch accuracy.
#
# One arm per job (short -t schedules; a long one waits). FAM=gemma|olmo  ARM=teacher|floor|ours
#
# TWO CARDS, only for the case that cannot use one (user, 2026-09-03). OLMo's pair is 78 GiB of
# weights and its 7B reader holds 512 KiB/token, so at 23.3k tokens one sequence needs ~11.9 GiB:
# on a single 95 GiB card even B=2 OOMs (job 3079276 died at 4, 3 and 2). Splitting puts the 32B
# alone on card 1 and leaves the reader ~81 GiB, which is the batch this arm needs to exist at all.
# What the second card buys here is CAPACITY, not overlap: on the eager path the two branch
# forwards do NOT overlap usefully (measured, fused/max 1.42-1.46 — the host serialises the kernel
# launches), and FUSION_PAR's gain was measured on the FKV path, which an accuracy run does not
# use. So FUSION_PAR stays off and the gain is the larger batch.
# Submit with: sbatch --gpus-per-node=2 ...
set -u
cd /u/anon/SLM_LM
BASE=/work/hdd/myproject/$USER; export HF_HOME=$BASE/hf
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /u/anon/.venvs/slm_lm/bin/activate
CKPT=$BASE/kvreuse_ckpts
FAM=${FAM:?gemma, gemmacw, olmo, olmocw or qwen3b}; ARM=${ARM:?teacher, floor or ours}
# D = document count (2026-09-04): 80 / 120 fill the middle of the axis that the 2026-09-03
# campaign left empty (it ran d160 only, on top of the existing d40). Default 160 = unchanged.
D=${D:-160}
BA="--bench hotpotqa_st${D}_full --max-conv 600"    # the same 600 questions as d40 at every depth
case "$FAM" in
  gemma) LM=google/gemma-3-12b-it;         SM=google/gemma-3-4b-it;       P=gmho${D}
         LT="12 8 6 4"; LF="32 24 16 12"; LO="10 8 6 4"; TT=teacher12; TF=floor4 ;;
  # gemmacw (2026-09-05): Gemma-3 fed its own chat format (<bos> + turn markers, CHAT_WRAP=gemma).
  # The raw-prompt gemma cells above never reasoned (plain 31%, FT 0%) and are withdrawn; this is
  # the re-run. New file names (gmcwho*) and its own adapters (reader_binding_gemmacw / stage2_gemmacw).
  gemmacw) LM=google/gemma-3-12b-it;       SM=google/gemma-3-4b-it;       P=gmcwho${D}
         export CHAT_WRAP=gemma
         LT="12 8 6 4"; LF="32 24 16 12"; LO="10 8 6 4"; TT=teacher12; TF=floor4 ;;
  # Qwen2.5 32B+3B (user, 2026-09-05: "FT된 32B+3B도 저렇게 실험해보자"): the pair's adapters exist
  # (reader_binding_v5distill_3b_r16_s900, stage2_on_v5reader3b_lam07_r16_s600); the teacher at
  # every depth is the Qwen axis's own. Ladders per depth; the 3B reader holds ~36 KiB/token.
  qwen3b) LM=Qwen/Qwen2.5-32B-Instruct;    SM=Qwen/Qwen2.5-3B-Instruct;    P=q3ho${D}
         RB=reader_binding_v5distill_3b_r16_s900; S2=stage2_on_v5reader3b_lam07_r16_s600
         TT=teacher32; TF=floor3
         case "$D" in
           40)  LT="12 8 6";  LF="64 48 32"; LO="24 16 12" ;;
           80)  LT="8 6 4";   LF="48 32 24"; LO="16 12 8" ;;
           120) LT="6 4 3";   LF="40 32 24"; LO="12 10 8" ;;
           *)   LT="4 3 2";   LF="32 24 16"; LO="10 8 6" ;;
         esac ;;
  # olmocw (2026-09-06): the OLMo pair with adapters RE-TRAINED under the ChatML rendering the evals use
  # (run_xfamily_ft.slurm, family olmocw); the 09-03 adapters were trained un-wrapped. Files omcwho<D>_*
  # at every depth including 40; teacher/floor arms are unchanged and stay omho*.
  olmocw) LM=allenai/Olmo-3.1-32B-Instruct; SM=allenai/OLMo-3-7B-Instruct; P=omcwho${D}
         export CHAT_WRAP=1
         RB=reader_binding_olmocw_r16_s900; S2=stage2_olmocw_lam07_r16_s600
         LT="6 4 3 2"; LF="8 6 4 3"; LO="4 3 2"; TT=teacher32; TF=floor7
         # two cards as the 09-03/09-04 OLMo fusion arms (d40 ran B=8, d80/d120 6, d160 4)
         if [ "${SLURM_GPUS_ON_NODE:-1}" -ge 2 ]; then LO="6 4 3 2"; [ "$D" = 40 ] && LO="8 6 4 3"; fi ;;
  olmo)  LM=allenai/Olmo-3.1-32B-Instruct; SM=allenai/OLMo-3-7B-Instruct; P=omho${D}
         export CHAT_WRAP=1
         # OLMo's 7B reader holds 512 KiB/token, so at 23.3k tokens a single sequence needs ~11.9
         # GiB of KV. The d40 batches (8-12) cannot survive a 3.8x longer context; the ladder is
         # scaled accordingly and still never reaches 1, which is forbidden.
         LT="6 4 3 2"; LF="8 6 4 3"; LO="4 3 2"; TT=teacher32; TF=floor7
         # with the 32B moved to its own card the reader has ~81 GiB, i.e. ~6 sequences
         [ "${SLURM_GPUS_ON_NODE:-1}" -ge 2 ] && LO="6 4 3 2" ;;
esac
case "$ARM" in
  teacher) LAD=$LT; TAG=$TT; EXTRA="--method teacher --model $LM" ;;
  floor)   LAD=$LF; TAG=$TF; EXTRA="--method teacher --model $SM" ;;
  ours)    LAD=$LO; TAG=ft_ours
           # the FT arm, because that is the deployed configuration and the one the ledger's §8
           # cross-family row reports; the un-adapted arm is a separate question already answered
           # at d40 for both families.
           RB=${RB:-reader_binding_${FAM}_r16_s900}; S2=${S2:-stage2_${FAM}_lam07_r16_s600}
           EXTRA="--method ours --model $LM --slm-model $SM --lam 0.7 --lm-no-accum \
                  --slm-lora $CKPT/$RB --lm-lora $CKPT/$S2"
           export MERGE_LORA=1 ;;
esac
# the 32B goes on the second card when there is one; a no-op on a single-card allocation.
if [ "${SLURM_GPUS_ON_NODE:-1}" -ge 2 ]; then
  export FUSION_LM_DEV=cuda:1
  echo "=== XCTX $FAM $ARM: 2 cards — 32B on cuda:1, reader + its KV on cuda:0 ==="
fi
OUT="results/fusionft/${P}_${TAG}.jsonl"
for B in $LAD; do
  echo "=== XCTX $FAM d$D $ARM B=$B -> $OUT ==="
  # --resume: skip the conversations already in $OUT and append. Without it a job that hits the
  # wall clock, or a rung that OOMs after writing some conversations, throws away everything it
  # finished and starts over. Batch may differ between the resumed halves and that is fine here —
  # these are ACCURACY arms, and batch is not a treatment (CANONICAL_RUNS 2e-note).
  if BATCH_SIZE=$B LONGEST_FIRST=1 python scripts/mtrag_accum.py $BA --max-new 200 $EXTRA \
       --resume --out "$OUT"; then
    echo "=== XCTX OK $FAM d$D $ARM B=$B ==="; break
  fi
  echo "!! XCTX $FAM d$D $ARM OOM or failure at B=$B, dropping a rung"
done
python - "$OUT" <<'PYEOF'
import json, statistics as st, sys
rows = [r for r in (json.loads(l) for l in open(sys.argv[1]) if l.strip()) if "acc_f1" in r]
if not rows:
    print("[xctx] NO ROWS"); raise SystemExit(1)
pv = rows[0]["_provenance"]
print(f"[xctx] n={len(rows)} inline-F1={st.mean(r['acc_f1'] for r in rows):.4f} "
      f"EM={st.mean(r['acc_em'] for r in rows):.4f} B={pv['axis_batch_size']} "
      f"model={pv['model']} reader={pv.get('slm_lora')} lm={pv.get('lm_lora')} "
      f"node={pv.get('slurm_node')}")
print("[xctx] the canonical score is the RE-SCORE, not this inline mean — "
      "scripts/score_context_axis.py applies the uniform answer boundary and the canonical "
      "extractor to every arm of the axis together.")
PYEOF
echo "=== XCTX $FAM d$D $ARM DONE ==="
