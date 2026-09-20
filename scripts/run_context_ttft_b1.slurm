#!/bin/bash
#SBATCH -A my-slurm-account
#SBATCH -p ghx4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -o /work/hdd/myproject/anon/slurm_logs/%x-%j.out
# TIME TO FIRST TOKEN ALONG THE CONTEXT AXIS, BATCH 1 (user, 2026-09-14: "context prefill 말고 진짜 TTFT를
# 재는건 어떰? B=1로 고정하고 … hotpot 96개를 다 할 필요도 없어보이긴 하는데"). One depth per job, the three
# arms in sequence on the same card: teacher-32B / floor-7B / PAIR 32B+7B, the first 24 questions of the
# 96-question throughput set (--max-conv 24 slices the same first conversations at every depth), the
# production FKV decode path, YaRN x4 on both models from 200 documents on (as the throughput and accuracy
# cells). BATCH_SIZE=1 with FORCE_BATCHED=1 on every arm: the batched class at B=1 is the same code path and
# graph as B>=2 (CLAUDE.md, batch=1 rule scoped 2026-09-12); B=1 is the point of this probe, not a ladder.
#
# What is measured: `ttft_from_prefill_s` per question — ONE wall from the synchronized instant before the
# context prefill to the first generated token arriving on the host (prefill of every branch + question
# forward + decode step 0), stamped by mtrag_accum on turn 0. Also kept per row: prefill_s_batch (the
# prefill alone), ttft_s (question forward + step 0 alone), conv_wall_s. The job still decodes the full
# answer (the harness is unchanged), so accuracy rows exist but are NOT a cell — N=24.
#
# Sentence for the job: "this job answers how long a single question waits for its first token at each
# context depth on one card, teacher against PAIR against the 7B floor" — the panel that replaces the
# prefill-latency panel of figures/paper_context_yarn8.
set -u
cd /u/anon/SLM_LM
BASE=/work/hdd/myproject/$USER; export HF_HOME=$BASE/hf
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /u/anon/.venvs/slm_lm/bin/activate
CKPT=$BASE/kvreuse_ckpts
export FKV_DECODE=1 FKV_GRAPH=1
export FORCE_BATCHED=1 BATCH_SIZE=1
D=${DEPTH:?set DEPTH=40|80|120|160|200|240|280|320}
N=${NQ:-24}
if [ "$D" -ge 200 ]; then export ROPE_YARN_FACTOR=4; else unset ROPE_YARN_FACTOR; fi
BA="--bench hotpotqa_st${D}_full --max-conv $N"
_done() { [ -f "$1" ] && [ "$(grep -c '"acc_f1"' "$1")" = "$N" ]; }
echo "=== CTTF d$D  B=1  N=$N  (YaRN factor: ${ROPE_YARN_FACTOR:-none})  node=${SLURMD_NODENAME:-?} ==="
f="results/fusionft/cttf_d${D}_teacher.jsonl"
if _done "$f"; then echo "   skip teacher (complete)"; else
  echo "=== CTTF d$D teacher ==="
  python scripts/mtrag_accum.py $BA --model Qwen/Qwen2.5-32B-Instruct --max-new 200 --method teacher --resume --out "$f" \
    || echo "!! cttf d$D teacher FAILED (exit $?)"
fi
f="results/fusionft/cttf_d${D}_floor7.jsonl"
if _done "$f"; then echo "   skip floor7 (complete)"; else
  echo "=== CTTF d$D floor7 ==="
  python scripts/mtrag_accum.py $BA --model Qwen/Qwen2.5-7B-Instruct --max-new 200 --method teacher --resume --out "$f" \
    || echo "!! cttf d$D floor7 FAILED (exit $?)"
fi
f="results/fusionft/cttf_d${D}_ours.jsonl"
if _done "$f"; then echo "   skip ours (complete)"; else
  echo "=== CTTF d$D ours ==="
  MERGE_LORA=1 python scripts/mtrag_accum.py $BA \
    --model Qwen/Qwen2.5-32B-Instruct --slm-model Qwen/Qwen2.5-7B-Instruct --max-new 200 \
    --method ours --lam 0.7 --lm-no-accum \
    --slm-lora $CKPT/reader_binding_v5distill_r16_s900 \
    --lm-lora $CKPT/stage2_on_v5reader_lam07_r16_s600 \
    --resume --out "$f" || echo "!! cttf d$D ours FAILED (exit $?)"
fi
echo "=== CTTF d$D DONE ==="
python - "$D" <<'PYEOF'
import json, os, sys, statistics as st
D = sys.argv[1]
for arm in ("teacher", "floor7", "ours"):
    f = f"results/fusionft/cttf_d{D}_{arm}.jsonl"
    if not os.path.exists(f):
        print(f"  {arm:8s} MISSING"); continue
    t = [json.loads(l) for l in open(f) if l.strip()]
    t = [r for r in t if "acc_f1" in r]
    v = [r["ttft_from_prefill_s"] for r in t if r.get("ttft_from_prefill_s") is not None]
    pv = t[0]["_provenance"] if t else {}
    br = sorted({r.get("batch_rows") for r in t})
    print(f"  {arm:8s} rows={len(t):3d} batch_rows={br} axis_batch={pv.get('axis_batch_size')} path={pv.get('axis_decode_path')} "
          f"node={pv.get('slurm_node')} ttft n={len(v)} median={st.median(v) if v else float('nan'):.3f} "
          f"min={min(v) if v else float('nan'):.3f} max={max(v) if v else float('nan'):.3f}")
PYEOF
