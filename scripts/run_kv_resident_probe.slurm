#!/bin/bash
#SBATCH -A my-slurm-account   # myproject is at 0 h (checked 2026-09-03); this account has 1915 h
#SBATCH -p ghx4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 2:30:00
#SBATCH -J kvprobe
#SBATCH -o /work/hdd/myproject/anon/slurm_logs/%x-%j.out
# RESIDENT-KV measurement wave (2026-08-30, user priority-high): every figure arm's retained
# KV-cache is MEASURED as tensor bytes (sum of numel x element_size over each layer's key/value
# tensors, plus packed+meta tensors for quantized caches), recorded at answer time AFTER the
# question/generation rollback — i.e. exactly the committed context+history the arm retains.
# This replaces (a) the tokens-x-architecture-constant X of kv_tradeoff, (b) the stale/partial
# kv_peak figure (whose LooGLE SpecPrefill column was PEAK-during-selection, keep-independent —
# a real number measuring the wrong quantity). One model load serves every arm (--kv-probe).
# BENCH=musique|hotpot|loogle|locomo. Ledger: kv-resident-probe.
set -u
cd /u/anon/SLM_LM
BASE=/work/hdd/myproject/$USER; export HF_HOME=$BASE/hf
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export QUANT_CACHE=append_only     # the int2 arm must be on the canonical cache, not HF's
source /u/anon/.venvs/slm_lm/bin/activate
CKPT=$BASE/kvreuse_ckpts
BENCH=${BENCH:?}
case "$BENCH" in
  musique) BARG="--bench musique_st40_full"; MC=24; PFX=kvpmu; LAM=0.7;  LORA=lam07 ;;
  hotpot)  BARG="--bench hotpotqa_st40_full"; MC=24; PFX=kvpho; LAM=0.7; LORA=lam07 ;;
  loogle)  BARG="--bench loogle"; MC=3; PFX=kvplg; LAM=0.7; LORA=lam07 ;;
  locomo)  BARG="--bench locomo"; MC=2; PFX=kvplc; LAM=0.85; LORA=lam085 ;;
esac
QUANT_ONLY=${QUANT_ONLY:-0}
FILL=${FILL:-0}
ARMS=$(BENCH=$BENCH QUANT_ONLY=$QUANT_ONLY FILL=$FILL python - <<'PY'
import json, os
b = os.environ["BENCH"]
if os.environ.get("FILL") == "1":
    # ★ FILL (2026-09-03, user: "다 해 전부다"). The unified retention grid
    # kept = {5, 10, 15, 21.9, 30, 40}% is COMPLETE on accuracy (72/72 cells) and incomplete on
    # BYTES, so those points cannot be drawn on the KV figure. These are exactly the arms whose
    # byte coordinate is missing, read off results/timing/kv_resident.json against the grid --
    # 26 arms in total -- plus int2 where the accuracy exists and the byte coordinate does not
    # (int2 is the point a quantization curve exists to show, so its absence drops the whole
    # curve on three benchmarks). Nothing here re-measures an arm that already has a coordinate.
    KEPT = {                       # kept % -> the arms this bench still owes
        "locomo":  [(15, "press"), (30, "press"), (10, "spec"), (15, "spec"), (30, "spec")],
        "loogle":  [(5, "press"), (10, "press"), (5, "spec"), (40, "spec")],
        "musique": [(5, "press"), (10, "press"), (5, "spec")],
        "hotpot":  [(5, "press"), (10, "press"), (5, "spec")],
    }[b]
    A = []
    for k, kind in KEPT:
        if kind == "press":
            r = round(1 - k / 100, 5)
            A += [dict(tag=f"snap{int(round(r * 1000)):03d}", method="snapkv_frozen", ratio=r),
                  dict(tag=f"expected{int(round(r * 1000)):03d}", method="h2o", ratio=r)]
        else:
            A.append(dict(tag=f"spec{int(round(k * 1000)):05d}", method="specprefill",
                          keep=k / 100))
    if b != "loogle":                      # loogle already has a measured int2
        A.append(dict(tag="quant_int2", method="quant_int2"))
    print(json.dumps(A)); raise SystemExit
if os.environ.get("PRESS_ONLY") == "1":
    A = []
    rs = (0.40, 0.50, 0.60, 0.70, 0.78125, 0.85) if b in ("musique", "hotpot") else \
         ((0.50, 0.60, 0.70, 0.78125, 0.85) if b == "loogle" else (0.40, 0.60, 0.78125, 0.90, 0.95))
    for r in rs:
        A += [dict(tag=f"snap{int(r*1000):03d}", method="snapkv_frozen", ratio=r),
              dict(tag=f"expected{int(r*1000):03d}", method="h2o", ratio=r)]
    print(json.dumps(A)); raise SystemExit
if os.environ.get("QUANT_ONLY") == "1":
    A = [dict(tag="teacher", method="teacher"),
         dict(tag="quant_int8", method="quant_int8"), dict(tag="quant_int4", method="quant_int4")]
    if b == "loogle":
        A += [dict(tag="quant_int3", method="quant_int3"), dict(tag="quant_int2", method="quant_int2")]
    print(json.dumps(A)); raise SystemExit
A = [dict(tag="teacher", method="teacher"), dict(tag="floor7", method="teacher", floor=True),
     dict(tag="ours", method="ours"),
     dict(tag="quant_int8", method="quant_int8"), dict(tag="quant_int4", method="quant_int4")]
if b in ("musique", "hotpot"):
    for r in (0.40, 0.50, 0.60, 0.70, 0.78125, 0.85):
        A += [dict(tag=f"snap{int(r*1000):03d}", method="snapkv_frozen", ratio=r),
              dict(tag=f"expected{int(r*1000):03d}", method="h2o", ratio=r)]
    A += [dict(tag=f"spec{int(k*100000):05d}", method="specprefill", keep=k)
          for k in (0.40, 0.30, 0.21875, 0.15, 0.10)]
elif b == "loogle":
    for r in (0.50, 0.60, 0.70, 0.78125, 0.85):
        A += [dict(tag=f"snap{int(r*1000):03d}", method="snapkv_frozen", ratio=r),
              dict(tag=f"expected{int(r*1000):03d}", method="h2o", ratio=r)]
    A += [dict(tag=f"spec{int(k*100000):05d}", method="specprefill", keep=k)
          for k in (0.30, 0.21875, 0.15, 0.10)]
    A += [dict(tag="quant_int3", method="quant_int3"), dict(tag="quant_int2", method="quant_int2"),
          dict(tag="oursq_int8", method="ours", reader_quant=8),
          dict(tag="oursq_int4", method="ours", reader_quant=4)]
else:  # locomo
    for r in (0.40, 0.60, 0.78125, 0.90, 0.95):
        A += [dict(tag=f"snap{int(r*1000):03d}", method="snapkv_frozen", ratio=r),
              dict(tag=f"expected{int(r*1000):03d}", method="h2o", ratio=r)]
    A += [dict(tag=f"spec{int(k*100000):05d}", method="specprefill", keep=k)
          for k in (0.40, 0.21875, 0.05)]
print(json.dumps(A))
PY
)
test -n "$ARMS" || { echo "EMPTY ARMS"; exit 2; }
BATCH_SIZE=1 python scripts/mtrag_accum.py $BARG --max-conv $MC \
  --method ours --model Qwen/Qwen2.5-32B-Instruct --slm-model Qwen/Qwen2.5-7B-Instruct \
  --lam $LAM --lm-no-accum --slm-lora $CKPT/reader_binding_v5distill_r16_s900 \
  --lm-lora $CKPT/stage2_on_v5reader_${LORA}_r16_s600 --max-new 8 \
  --kv-probe "$ARMS" --out "results/fusionft/${PFX}_{tag}.jsonl"
echo "=== KVPROBE $BENCH DONE ==="
