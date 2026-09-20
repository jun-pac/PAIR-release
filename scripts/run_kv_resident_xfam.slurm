#!/bin/bash
#SBATCH -A my-slurm-account
#SBATCH -p ghx4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 0:50:00
#SBATCH -J kvpxf
#SBATCH -o /work/hdd/myproject/anon/slurm_logs/%x-%j.out
# PERSISTENT KV for the four pairs of figures/paper_xfam_context, AT EVERY CONTEXT DEPTH THE FIGURE
# DRAWS (user, 2026-09-18: "이 그래프에 ours점에 각각 persistent KV % 가 얼마인지 적어줄 수 있나" +
# "다른 model없으면 측정해봐. probe돌려서" — "각각" is every ours point, i.e. every depth; the first run of
# this launcher did d40 only, which narrowed the instruction, and the share is depth-dependent anyway:
# ours holds the reader's context cache PLUS the query-only LM's cache, and that second term does not
# grow with the context, so the share falls toward the floor's fixed per-token ratio as context grows).
# DEPTH=40|80|120|160 -> bench hotpotqa_st{DEPTH}_full; the store is keyed [FAM]["d{DEPTH}"].
# d40 is already measured for all four pairs (the three transfer families here, the main Qwen 32B+7B pair
# in results/timing/kv_resident.json) and is NOT re-run.
#
# WHY. figures/paper_xfam_context annotates nothing about memory because the persistent-KV probe
# (results/timing/kv_resident.json) only ever ran the Qwen 32B+7B pair. Writing the other three panels'
# shares from an architecture constant is what figures/README records as a past defect, so they are
# measured instead — the same probe, the same definition (CANONICAL_RUNS 2e-bytes: the retained key/value
# tensor bytes at answer time, after the question/generation rollback), batch 1, 24 conversations.
#
# ARM SETTINGS ARE COPIED FIELD BY FIELD from the provenance of the transfer figure's own ours cells
# (q3ho40_ft_ours / omcwho40_ft_ours / gmcwho_ft_ours): the family's models, its two adapters, lam 0.7,
# lm_no_accum, QA_REASON_V3, and the family's chat rendering (CHAT_WRAP=1 for OLMo, =gemma for Gemma) —
# the arm the figure draws, not a near neighbour of it.
#
# Three arms per family: teacher (the big model, full context), ours, floor (the small model alone), so
# the share has a measured denominator. One model load serves all three (--kv-probe).
# FAM=qwen3b|olmocw|gemmacw. Campaign: kv-resident-xfam.
set -u
cd /u/anon/SLM_LM
BASE=/work/hdd/myproject/$USER; export HF_HOME=$BASE/hf
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /u/anon/.venvs/slm_lm/bin/activate
CKPT=$BASE/kvreuse_ckpts
FAM=${FAM:?qwen7b, qwen3b, olmocw or gemmacw}
DEPTHS=${DEPTHS:-${DEPTH:-40}}   # space list: one model load per depth, all in one job
case "$FAM" in
  qwen7b)  LM=Qwen/Qwen2.5-32B-Instruct;     SM=Qwen/Qwen2.5-7B-Instruct
           RB=reader_binding_v5distill_r16_s900; S2=stage2_on_v5reader_lam07_r16_s600; PFX=kvxq7 ;;
  qwen3b)  LM=Qwen/Qwen2.5-32B-Instruct;     SM=Qwen/Qwen2.5-3B-Instruct
           RB=reader_binding_v5distill_3b_r16_s900; S2=stage2_on_v5reader3b_lam07_r16_s600; PFX=kvxq3 ;;
  olmocw)  LM=allenai/Olmo-3.1-32B-Instruct; SM=allenai/OLMo-3-7B-Instruct
           export CHAT_WRAP=1
           RB=reader_binding_olmocw_r16_s900; S2=stage2_olmocw_lam07_r16_s600; PFX=kvxom ;;
  gemmacw) LM=google/gemma-3-12b-it;         SM=google/gemma-3-4b-it
           export CHAT_WRAP=gemma
           RB=reader_binding_gemmacw_r16_s900; S2=stage2_gemmacw_lam07_r16_s600; PFX=kvxgm ;;
esac
ARMS='[{"tag":"teacher","method":"teacher"},{"tag":"ours","method":"ours"},{"tag":"floor","method":"teacher","floor":true}]'
echo "=== KVXFAM $FAM: $LM + $SM, adapters $RB / $S2 ==="
for DEPTH in $DEPTHS; do
echo "=== depth d${DEPTH} (bench hotpotqa_st${DEPTH}_full) ==="
BATCH_SIZE=1 MERGE_LORA=1 python scripts/mtrag_accum.py --bench "hotpotqa_st${DEPTH}_full" --max-conv 24 \
  --method ours --model "$LM" --slm-model "$SM" --lam 0.7 --lm-no-accum \
  --slm-lora "$CKPT/$RB" --lm-lora "$CKPT/$S2" --max-new 8 \
  --kv-probe "$ARMS" --out "results/fusionft/${PFX}d${DEPTH}_{tag}.jsonl"
echo "=== KVXFAM $FAM d$DEPTH DONE ==="
FAM=$FAM PFX=$PFX DEPTH=$DEPTH python - <<'PYEOF'
import json, os, statistics as st
fam, pfx, depth = os.environ["FAM"], os.environ["PFX"], os.environ["DEPTH"]
out = "results/timing/kv_resident_xfam.json"
store = json.load(open(out)) if os.path.exists(out) else {}
e = {}
for tag in ("teacher", "ours", "floor"):
    f = f"results/fusionft/{pfx}d{depth}_{tag}.jsonl"
    if not os.path.exists(f):
        print(f"  {tag}: MISSING {f}"); continue
    rows = [json.loads(l) for l in open(f) if l.strip()]
    rows = [r for r in rows if r.get("kv_gib")]
    if not rows:
        print(f"  {tag}: no kv_gib rows"); continue
    pv = rows[0]["_provenance"]
    e[tag] = dict(n=len(rows), kv_gib=round(st.fmean(r["kv_gib"] for r in rows), 4),
                  kv_gib_reader=(round(st.fmean(r["kv_gib_reader"] for r in rows), 4)
                                 if rows[0].get("kv_gib_reader") else None),
                  kv_gib_lm=(round(st.fmean(r["kv_gib_lm"] for r in rows), 4)
                             if rows[0].get("kv_gib_lm") else None),
                  ctx_tok=int(st.median(r["acc_ctx_tok"] for r in rows)),
                  model=pv.get("model"), slm_model=pv.get("slm_model"), lam=pv.get("lam"),
                  node=pv.get("slurm_node"), job=pv.get("slurm_job_id"), file=os.path.basename(f))
    print(f"  {tag}: n={e[tag]['n']} kv_gib={e[tag]['kv_gib']} ctx_tok={e[tag]['ctx_tok']}")
if "teacher" in e and "ours" in e:
    e["ours_pct_of_teacher"] = round(100 * e["ours"]["kv_gib"] / e["teacher"]["kv_gib"], 2)
    if "floor" in e:
        e["floor_pct_of_teacher"] = round(100 * e["floor"]["kv_gib"] / e["teacher"]["kv_gib"], 2)
    print(f"  => ours {e['ours_pct_of_teacher']}% of the teacher's persistent KV"
          + (f", floor {e['floor_pct_of_teacher']}%" if "floor" in e else ""))
e["_basis"] = (f"hotpotQA d{depth}, 24 conversations, batch 1, retained key/value tensor bytes at answer "
               "time (CANONICAL_RUNS 2e-bytes); arm settings copied from the transfer figure's own ours cell")
store.setdefault(fam, {})[f"d{depth}"] = e
json.dump(store, open(out + ".tmp", "w"), indent=1)
os.replace(out + ".tmp", out)      # never rewrite the store in place (accident #17)
print(f"wrote {out} [{fam}][d{depth}]")
PYEOF
done
