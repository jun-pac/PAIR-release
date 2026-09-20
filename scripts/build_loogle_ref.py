#!/usr/bin/env python
"""Build the LooGLE-accum reference (2026-08-23; the NarrativeQA lesson applied).

Selection filter, IN ORDER (the NarrativeQA mistake was screening on format only):
1. REASONING-BOUND candidate: LooGLE longdep_qa was chosen over DetectiveQA (MC), InfiniteBench
   En.QA (~95k ctx), ConditionalQA (~1.9k ctx), TimeQA (~2.5k ctx), NovelQA (gated, >100k),
   QuALITY (MC) — full dissection in RESULTS_MASTER 2026-08-23. Free-form question mix measured:
   26% compute/count, 8% why/causal, plus multihop what/who (longdep design). The GPU gate still
   verifies: teacher vs floor vs CLOSED-BOOK 32B before any fusion arm.
2. Accum shape: one document = one conversation, full text at turn 1, official per-doc questions.
3. Row hygiene: drop MC-contaminated rows (embedded numbered options / bare-digit answers, 216/1101);
   drop golds > 12 words (median gold is 3 words; the tail is explanation-style prose that poisons
   token-F1 — recorded filter, not silent); docs with 8k <= ~tok <= 30k; <= 30 Q/doc, <= 600 total.

Row format mirrors narrativeqa/qasper refs (single gold -> targets/golds length 1; evidence kept).
"""
import json
import os
import re

os.environ.pop("HF_DATASETS_OFFLINE", None)
os.environ.pop("HF_HUB_OFFLINE", None)
from datasets import load_dataset

OUT = "/work/hdd/myproject/anon/loogle/loogle_reuse_ref.jsonl"
LO, HI, MAXQ, MAXTOT = 8000, 30000, 30, 600


def is_mc(q, a):
    return bool(re.search(r"\n\s*[1-4][\.\)]", q)) or (a.strip().isdigit() and len(a.strip()) <= 2)


def main():
    ds = load_dataset("bigai-nlco/LooGLE", "longdep_qa", split="test")
    by = {}
    for ex in ds:
        by.setdefault(ex["doc_id"], []).append(ex)
    rows, kept_docs, dropped = [], 0, dict(mc=0, longgold=0, len_window=0)
    for doc_id in sorted(by):
        exs = by[doc_id]
        toks = len(exs[0]["context"]) // 4
        if not (LO <= toks <= HI):
            dropped["len_window"] += len(exs)
            continue
        good = []
        for ex in exs:
            if is_mc(ex["question"], ex["answer"]):
                dropped["mc"] += 1
                continue
            if len(ex["answer"].split()) > 12:
                dropped["longgold"] += 1
                continue
            good.append(ex)
        if not good:
            continue
        if len(rows) + min(len(good), MAXQ) > MAXTOT:
            good = good[: MAXTOT - len(rows)]
            if not good:
                break
        kept_docs += 1
        for i, ex in enumerate(good[:MAXQ], 1):
            rows.append(dict(
                conversation_id=f"lg-{doc_id}"[:24], turn=i,
                input=[{"speaker": "user", "text": ex["question"].strip()}],
                contexts=([{"document_id": doc_id, "text": ex["context"], "title": ex["title"]}]
                          if i == 1 else []),
                targets=[{"text": ex["answer"].strip()}],
                golds=[ex["answer"].strip()], Answerability=["ANSWERABLE"],
                qid=ex["id"], evidence=ex.get("evidence"),
            ))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {OUT}: {len(rows)} Q over {kept_docs} docs; dropped {dropped}")


if __name__ == "__main__":
    main()
