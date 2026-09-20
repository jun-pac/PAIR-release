#!/usr/bin/env python
"""v4 corpus = v2 (3,000 DEEP dialogue-temporal — the LoCoMo-weakness dose) + v3 non-dialogue synthetic
(2,400 — the genericity that bought CLUTRR) + replay (2,000). Rationale: v2 hit LoCoMo 78% but regressed
CLUTRR; v3 fixed genericity (CLUTRR +0.055) but diluted the LoCoMo dose 5x and lost it (fusion 0.536->0.477
best). v4 pre-registered prediction: LoCoMo fusion back to >=0.53 AND CLUTRR >=0.48 AND musique no-loss."""
import json
out = open('results/fusionft/binding_corpus_v4.jsonl','w'); n=0
for l in open('results/fusionft/binding_corpus_v2.jsonl'):
    r = json.loads(l); r['instruction']='QA_REASON_V3_LOCOMO'; r['example_id']='v4d-'+r['example_id']
    out.write(json.dumps(r)+'\n'); n+=1
for l in open('results/fusionft/binding_corpus_v3.jsonl'):
    r = json.loads(l)
    if r['meta']['fmt'] == 'dialogue': continue
    r['example_id']='v4-'+r['example_id']; out.write(json.dumps(r)+'\n'); n+=1
out.close(); print('v4 corpus:', n)
