#!/bin/bash
#SBATCH -J v5seed777
#SBATCH -p ghx4
#SBATCH -A my-slurm-account
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 4:00:00
#SBATCH -o /work/hdd/myproject/%u/slurm_logs/%x-%j.out
# v5 = FAIR reader distillation: merge teacher-32B target shards + replay -> train 7B on TEACHER TARGETS ONLY.
set -euo pipefail
BASE=/work/hdd/myproject/$USER; export HF_HOME=$BASE/hf HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
cd /u/anon/SLM_LM
source /u/anon/.venvs/slm_lm/bin/activate
python - <<'PY'
import json
out = open('results/fusionft/binding_corpus_v5.jsonl','w'); n=g=0
for s in (0,1,2):
    for l in open(f'results/fusionft/binding_v4_teacher_shard{s}.jsonl'):
        r = json.loads(l)
        tt = (r.get('teacher_text') or '').strip()
        if not tt or r.get('hit_max_new'):    # usability gates ONLY (non-empty, terminated) — no gold gate
            g += 1; continue
        r['target'] = tt
        out.write(json.dumps(r)+'\n'); n += 1
for l in open('results/fusionft/binding_corpus_v4.jsonl'):
    r = json.loads(l)
    if r['meta'].get('fmt') == 'replay':
        out.write(json.dumps(r)+'\n'); n += 1
out.close(); print(f'[v5 merge] {n} rows ({g} gated out)')
PY
python scripts/reader_binding_sft.py --corpus results/fusionft/binding_corpus_v5.jsonl \
  --out /work/hdd/myproject/anon/kvreuse_ckpts/reader_binding_v5distill_SEED777_r16_s900 \
  --seed 777 --max-steps 900 --cache-dir "$HF_HOME"
echo "[Done] $(date -Is)"
