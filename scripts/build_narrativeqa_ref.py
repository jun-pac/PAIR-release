#!/usr/bin/env python
"""Build the NarrativeQA-accum reference (2026-08-20, user-directed new accum bench).

Why NarrativeQA over QuALITY (both analyzed from raw data this session): QuALITY is 4-way
multiple-choice (gold_label = option index) — the eventqa lesson is that MC selection is
reader-bound and fusion value shrinks there, and the metric would leave the token-F1/EM framework;
its contexts are also the shortest of any of our benches (~6k tok). NarrativeQA is free-form short
answers with TWO references per question (the QASPER-accum multi-gold framework verbatim), has a
natural ~30 questions per story, and one shared narrative per episode — exactly the accum setting.

Selection (deterministic): stream the OFFICIAL test split in dataset order; keep documents whose
raw text is 8k-25k approx-tokens (chars/4; fits the 95GB accum harness incl. batch-3 like
LoCoMo-30); take the FIRST 20 such documents, first 30 questions each (user caps: <=30 Q/episode,
<=600 total). Movie scripts and short gutenberg texts both qualify; light HTML-tag strip only
(some movie scripts are scraped pages), recorded per doc.

Row format mirrors qasper_reuse_ref_full.jsonl exactly: conversation_id / turn / input(user msg) /
contexts (FULL story at turn 1 only) / targets (both refs; harness scores inline vs targets[0]) /
golds (both, for offline multi-gold rescoring) / Answerability / qid.
"""
import json
import os
import re
import sys

os.environ.pop("HF_DATASETS_OFFLINE", None)
os.environ.pop("HF_HUB_OFFLINE", None)
from datasets import load_dataset

OUT = "/work/hdd/myproject/anon/narrativeqa/narrativeqa_reuse_ref.jsonl"
LO_TOK, HI_TOK, N_DOCS, MAX_Q = 8000, 25000, 20, 30

TAG_RE = re.compile(r"<[^>]{1,80}>")
WS_RE = re.compile(r"\n{4,}")


def clean(text):
    stripped = TAG_RE.sub(" ", text)
    frac = 1 - len(stripped) / max(len(text), 1)
    return WS_RE.sub("\n\n\n", stripped), round(frac, 4)


def main():
    ds = load_dataset("deepmind/narrativeqa", split="test", streaming=True)
    docs, rows, order = {}, [], []
    for ex in ds:
        d = ex["document"]
        did = d["id"]
        if did not in docs:
            toks = len(d["text"]) // 4
            keep = LO_TOK <= toks <= HI_TOK and len(order) < N_DOCS
            docs[did] = dict(keep=keep, kind=d["kind"], toks=toks, nq=0,
                             text=(d["text"] if keep else None))
            if keep:
                order.append(did)
        info = docs[did]
        if not info["keep"] or info["nq"] >= MAX_Q:
            # stop condition: every kept doc saturated AND we already have N_DOCS
            if len(order) == N_DOCS and all(docs[x]["nq"] >= MAX_Q for x in order):
                break
            continue
        info["nq"] += 1
        answers = [a["text"].strip() for a in ex["answers"]][:2]
        rows.append(dict(
            conversation_id=f"nqa-{did[:12]}", turn=info["nq"],
            input=[{"speaker": "user", "text": ex["question"]["text"].strip()}],
            contexts=([{"document_id": did, "text": None}] if info["nq"] == 1 else []),
            targets=[{"text": a} for a in answers],
            golds=answers, Answerability=["ANSWERABLE"],
            qid=f"{did[:12]}_{info['nq']:02d}",
        ))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        for r in rows:
            if r["contexts"]:
                did = r["contexts"][0]["document_id"]
                text, frac = clean(docs[did]["text"])
                r["contexts"][0]["text"] = text
                r["contexts"][0]["html_stripped_frac"] = frac
            f.write(json.dumps(r) + "\n")
    print(f"wrote {OUT}: {len(rows)} questions over {len(order)} docs")
    for did in order:
        d = docs[did]
        print(f"  {did[:12]} kind={d['kind']:9s} ~tok={d['toks']:6d} nq={d['nq']}")


if __name__ == "__main__":
    main()
