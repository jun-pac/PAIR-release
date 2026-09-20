#!/usr/bin/env python
"""
GPU-FREE precompute of the musique retrieval docs -> a docs-cache, so generation jobs SKIP the ~50-min BM25 corpus
build (which otherwise runs on CPU with the GPU idle, re-done per job). Builds the BM25 corpus ONCE and retrieves for
every requested doc-number, writing results/_docs_cache/musique_<split>_d<doc>.json = {example_id: [docs...]}.
load_musique_split picks these up via MUSIQUE_DOCS_CACHE_DIR and skips BM25. Run on CPU (no --gpus needed).
  DOCS=40,80,120,160,200 SPLIT=validation python scripts/precompute_musique_docs.py
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.pop("MUSIQUE_DOCS_CACHE_DIR", None)  # force real retrieval here, never read a (partial) cache

from src.data import (  # noqa: E402
    _load_musique_dataset, _build_musique_corpus, BM25Retriever,
    _prioritize_musique_assigned_documents, _format_musique_documents,
    _build_assigned_plus_retrieved_docs,
)

HF = os.environ.get("HF_HOME", "/work/hdd/myproject/anon/hf")
SPLIT = os.environ.get("SPLIT", "validation")
DOCS = [int(x) for x in os.environ.get("DOCS", "40,80,120,160,200").split(",")]
OUT = os.environ.get("DOCS_CACHE_OUT", "results/_docs_cache")
os.makedirs(OUT, exist_ok=True)

print(f"[precompute] loading musique {SPLIT} + building BM25 corpus ONCE (this is the slow part)...", flush=True)
ds = _load_musique_dataset(SPLIT, cache_dir=HF)
t0 = time.perf_counter()
corpus = _build_musique_corpus("train", cache_dir=HF, max_examples=None)
retriever = BM25Retriever(corpus) if corpus else None
print(f"[precompute] corpus built in {time.perf_counter()-t0:.0f}s ({len(corpus)} passages). Retrieving per doc-number.", flush=True)

for d in DOCS:
    outp = os.path.join(OUT, f"musique_{SPLIT}_d{d}.json")
    if os.path.exists(outp):
        print(f"[precompute] d{d}: cache exists, skip", flush=True); continue
    t1 = time.perf_counter(); cache = {}
    for idx, row in enumerate(ds):
        eid = str(row.get("id") or row.get("_id") or idx)
        q = row.get("question") or row.get("query") or ""
        assigned = _prioritize_musique_assigned_documents(row, _format_musique_documents(row))
        docs, _ = _build_assigned_plus_retrieved_docs(
            question=q, assigned_docs=assigned, corpus=corpus, bm25_retriever=retriever, doc_number=d)
        cache[eid] = docs
    json.dump(cache, open(outp, "w"))
    print(f"[precompute] wrote {outp}  ({len(cache)} examples, {time.perf_counter()-t1:.0f}s)", flush=True)
print("[precompute] DONE.", flush=True)
