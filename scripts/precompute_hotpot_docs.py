#!/usr/bin/env python
"""
GPU-FREE precompute of HotpotQA retrieval docs -> a docs-cache, so generation jobs SKIP the ~15-30 min BM25 corpus
build (train split ~90K examples) which otherwise runs on CPU with the GPU idle, re-done per job. Builds the BM25
corpus ONCE and retrieves for every requested doc-number over ALL validation examples (so any --sample-seed subset
is covered), writing results/_docs_cache/hotpotqa_<split>_d<doc>.json = {example_id: [ordered docs...]}.
load_hotpotqa_split picks these up via HOTPOT_DOCS_CACHE_DIR and skips BM25. Run on CPU (no --gpus).
  DOCS=40,80,120,160,200 SPLIT=validation python scripts/precompute_hotpot_docs.py
"""
import json, os, sys, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.pop("HOTPOT_DOCS_CACHE_DIR", None)  # force real retrieval here
from datasets import load_dataset  # noqa: E402
from src.data import (  # noqa: E402
    _build_hotpot_corpus, _format_documents, BM25Retriever, _bm25_scores, _order_by_scores,
)

HF = os.environ.get("HF_HOME", "/work/hdd/myproject/anon/hf")
SPLIT = os.environ.get("SPLIT", "validation")
DOCS = [int(x) for x in os.environ.get("DOCS", "40,80,120,160,200").split(",")]
RETR_SPLIT = os.environ.get("RETRIEVAL_SPLIT", "train")
OUT = os.environ.get("DOCS_CACHE_OUT", "results/_docs_cache")
# retrieve ONLY the seed-subset the runner will generate on (loader restricts to these ids), so BM25 over the
# 483K-passage corpus runs 600x not 7405x. Replicates run_evidence_sketch_experiments._maybe_shuffle_and_sample.
SAMPLE = int(os.environ.get("SAMPLE", "600"))
SEED = int(os.environ.get("SAMPLE_SEED", "42"))
os.makedirs(OUT, exist_ok=True)

print(f"[precompute] loading hotpotqa {SPLIT}...", flush=True)
ds = load_dataset("hotpot_qa", "distractor", split=SPLIT, cache_dir=HF)
_ids = [str(r["id"]) for r in ds]
_idx = list(range(len(ds))); random.Random(SEED).shuffle(_idx)
_sel = set(_ids[i] for i in _idx[:min(SAMPLE, len(ds))])
ds = ds.filter(lambda r: str(r["id"]) in _sel)
print(f"[precompute] selected {len(ds)}/{len(_ids)} examples (seed={SEED}, sample={SAMPLE}). "
      f"Building BM25 corpus from {RETR_SPLIT} ONCE (slow part)...", flush=True)
t0 = time.perf_counter()
corpus = _build_hotpot_corpus(RETR_SPLIT, cache_dir=HF, max_examples=None)
retriever = BM25Retriever(corpus)
print(f"[precompute] corpus built in {time.perf_counter()-t0:.0f}s ({len(corpus)} passages). Retrieving per doc-number.", flush=True)

for d in DOCS:
    outp = os.path.join(OUT, f"hotpotqa_{SPLIT}_d{d}.json")
    if os.path.exists(outp):
        print(f"[precompute] d{d}: cache exists, skip", flush=True); continue
    t1 = time.perf_counter(); cache = {}
    for row in ds:
        documents = _format_documents(row["context"])
        if d > len(documents):
            extras = retriever.search(row["question"], k=d - len(documents), exclude=documents)
            documents = documents + extras
        else:
            documents = documents[:d]
        scores = _bm25_scores(documents, row["question"])
        documents, _ = _order_by_scores(documents, scores)
        cache[str(row["id"])] = documents
    json.dump(cache, open(outp, "w"))
    print(f"[precompute] wrote {outp} ({len(cache)} examples, {time.perf_counter()-t1:.0f}s)", flush=True)
print("[precompute] DONE", flush=True)
