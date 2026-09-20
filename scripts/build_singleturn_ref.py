#!/usr/bin/env python
"""Build 1-turn-conversation refs for musique d40 / hotpotQA d40 (task A4.3, HANDOFF_260827).

WHY. The single-turn RAG benchmarks are the workloads where the method's prefill advantage is NOT
amortised over later turns — but their only harness (run_evidence_sketch_experiments.py) is
sequential (batch 1: timing is an artefact by the hard rule) and its old logs carry no provenance
and no timing fields at all. mtrag_accum.py already has batched decode, the single total-wall timer
(conv_wall_s — nothing outside a timer by construction), provenance, and every baseline arm. So a
single-turn question becomes a 1-turn conversation: contexts = its 40 retrieved docs, one question,
done. Nothing about the accumulate machinery activates (there is no turn 2).

BASIS (bench names carry their basis): these refs define NEW benches `musique_st40` /
`hotpotqa_st40`. The passage formatting is mtrag_accum's accumulate convention ("[Evidence
Passage]…"), NOT the old single-turn harness's — so scores from these refs are comparable ONLY
within this basis (the runs carry their own teacher/floor sandwich) and never against the old
q327*_musiqued40 logs.

Docs come from the SAME deterministic docs-caches the canonical harness uses
(results/_docs_cache/{musique,hotpotqa}_validation_d40.json):
  * musique — cache insertion order == dataset order == the canonical first-N order.
  * hotpotqa — the canonical subset is random.Random(42).shuffle over dataset indices, first 600
    (replicated from scripts/precompute_hotpot_docs.py); the ref keeps that shuffled order.
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HF = os.environ.get("HF_HOME", "/work/hdd/myproject/anon/hf")
K = int(os.environ.get("K", "96"))          # questions per bench = the data cap on any arm's batch
# DOCS selects WHICH docs-cache to read, i.e. the context length of the resulting bench. The
# caches at 40/80/120/160/200/240/280/320 documents already exist for both benches and hold every
# question (hotpot 600, musique 2417), so a longer-context bench costs no retrieval -- only this
# file being rebuilt. Added 2026-09-02 for the context-length axis: the same questions and the
# same gold answers at four context lengths, so accuracy and throughput can both be read against
# context with nothing else varying.
DOCS = int(os.environ.get("DOCS", "40"))
CACHE_DIR = "results/_docs_cache"
OUT_DIR = "/work/hdd/myproject/anon/singleturn"


def rows_for(bench, ex_ids, cache, qa):
    rows = []
    for eid in ex_ids:
        q, golds = qa[eid]
        rows.append(dict(
            conversation_id=f"{bench[:2]}-{eid}"[:40],
            turn="1",
            input=[dict(speaker="user", text=q)],
            contexts=[dict(document_id=f"{eid}-d{j}", text=doc)
                      for j, doc in enumerate(cache[eid])],
            targets=[dict(text=golds[0])],
            golds=list(golds),
            Answerability=["ANSWERABLE"],
            qid=str(eid),
        ))
    return rows


def musique(shuffle_seed=None):
    from src.data import _load_musique_dataset, _row_answers
    cache = json.load(open(f"{CACHE_DIR}/musique_validation_d{DOCS}.json"))
    ds = _load_musique_dataset("validation", cache_dir=HF)
    qa = {}
    for i, r in enumerate(ds):
        eid = str(r.get("id") or r.get("_id") or i)
        if eid in cache:
            a, answers = _row_answers(r)
            qa[eid] = (r.get("question") or r.get("query") or "", answers or [a])
    ids = [i for i in cache if i in qa]
    if shuffle_seed is None:
        ids = ids[:K]                                # cache order == canonical first-N
    else:
        # ★ 2026-08-29: musique's validation is HOP-ORDERED, so first-N = the 2-hop easy band —
        # exactly the subset where the distilled arm edges the zero-shot teacher (the _st40
        # sandwich inversion, RESULTS_MASTER 28e addendum v2). A SEEDED RANDOM sample mixes the
        # hops, matching how hotpotqa_st40 was built (whose sandwich is normal).
        random.Random(shuffle_seed).shuffle(ids)
        ids = ids[:K]
    return rows_for("musique", ids, cache, qa)


def hotpot():
    from datasets import load_dataset
    cache = json.load(open(f"{CACHE_DIR}/hotpotqa_validation_d{DOCS}.json"))
    ds = load_dataset("hotpot_qa", "distractor", split="validation", cache_dir=HF)
    all_ids = [str(r["id"]) for r in ds]
    qa = {str(r["id"]): (r["question"], [r["answer"]]) for r in ds}
    idx = list(range(len(ds)))
    random.Random(42).shuffle(idx)                   # replicates precompute_hotpot_docs.py exactly
    ids = [all_ids[i] for i in idx if all_ids[i] in cache][:K]
    return rows_for("hotpotqa", ids, cache, qa)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    global K
    if DOCS != 40:                      # context-length axis: only the FULL bases are built
        K = 10**9
        for name, fn in ((f"musique_st{DOCS}_full", musique),
                         (f"hotpotqa_st{DOCS}_full", hotpot)):
            if not os.path.exists(f"{CACHE_DIR}/{'musique' if 'musique' in name else 'hotpotqa'}"
                                  f"_validation_d{DOCS}.json"):
                print(f"skip {name}: no docs cache at d{DOCS}"); continue
            _write(name, fn())
        return
    for name, fn in (("musique_st40", musique), ("hotpotqa_st40", hotpot),
                     ("musique_st40r", lambda: musique(shuffle_seed=42))):
        rows = fn()
        _write(name, rows)
    # ★ FULL accuracy bases (2026-08-29, user: accuracy comes from the WHOLE set, only throughput
    # may approximate). Every question the docs-cache holds: musique 2417, hotpot 600.
    K = 10**9
    for name, fn in (("musique_st40_full", musique), ("hotpotqa_st40_full", hotpot)):
        rows = fn()
        _write(name, rows)
    return


def _write(name, rows):
    import json as _json
    out = f"{OUT_DIR}/{name}_ref.jsonl"
    with open(out, "w") as fh:
        for r in rows:
            fh.write(_json.dumps(r) + "\n")
    ndocs = [len(r["contexts"]) for r in rows]
    print(f"{name}: {len(rows)} conversations, docs/conv min={min(ndocs)} max={max(ndocs)} -> {out}")


def _dead(rows):
        out = f"{OUT_DIR}/{name}_ref.jsonl"
        with open(out, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        ndocs = [len(r["contexts"]) for r in rows]
        print(f"{name}: {len(rows)} conversations (1 turn each), docs/conv min={min(ndocs)} "
              f"max={max(ndocs)} -> {out}")


if __name__ == "__main__":
    main()
