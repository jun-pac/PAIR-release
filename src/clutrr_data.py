"""CLUTRR multi-hop kinship reasoning, long-context-ified with distractor stories.

CLUTRR (Sinha et al. 2019, https://arxiv.org/abs/1908.06177): a short story states some
family relations among named people; the task is to INFER the unstated relation between two
queried people by COMPOSING 2-10 hops of relations. This is genuine multi-step *reasoning*
(relation composition), NOT span extraction -- the regime the SLM+LM thesis targets: the SLM
reads the long context (relevant story + distractors) and surfaces the relevant relations; the
query-only LM reasons the kinship chain. We pad the relevant story with DISTRACTOR stories
(other CLUTRR examples whose entities are DISJOINT from the query, so they cannot inject false
relations) to a target token length -> a clean long-context reasoning test where simple KV
compression should drop the dispersed premises. Answer = one kinship word -> EM/F1
(compute_best_em_f1, same path as BABILong).

Data = the public CLUTRR CSVs (kliang5/CLUTRR_huggingface_dataset), downloaded to
$CLUTRR_DATA_DIR (default /work/hdd/myproject/$USER/clutrr_data) to AVOID the HF `datasets`
FileLock, which fails (`OSError: [Errno 5]` / `BrokenPipeError: [Errno 108]`) on the /work
Lustre mount. Each example is a ``ShortAnswerExample`` (the QA loop's container) routed through
the reason_then_answer QA prompt + EM scoring (set ANSWER_PROMPT_VARIANT=reason_then_answer).
"""
from __future__ import annotations

import ast
import csv
import os
import random
import re
from typing import List, Optional

from .data import ShortAnswerExample

_DEFAULT_DIR = f"/work/hdd/myproject/{os.environ.get('USER', 'anon')}/clutrr_data"
CLUTRR_DATA_DIR = os.environ.get("CLUTRR_DATA_DIR", _DEFAULT_DIR)
# gen_train234_test2to10 = the standard generalization split (test stories span 2-10 hops).
CLUTRR_TASK = os.environ.get("CLUTRR_TASK", "gen_train234_test2to10")


def _entities(story: str) -> set:
    """Bracketed entity names, e.g. '[Clarence]'s grandson [Jeff]' -> {Clarence, Jeff}."""
    return set(re.findall(r"\[([^\]]+)\]", story))


def _strip(s: str) -> str:
    return s.replace("[", "").replace("]", "")


def _hops(r: dict) -> int:
    try:
        return len(ast.literal_eval(r["edge_types"]))
    except Exception:  # noqa: BLE001
        return 0


def load_clutrr_split(
    target_tokens: int = 4000,
    split: str = "test",
    sample: Optional[int] = None,
    min_hops: Optional[int] = None,
    seed: int = 0,
    cache_dir: Optional[str] = None,  # unused; CSVs are local
) -> List[ShortAnswerExample]:
    """Load CLUTRR as long-context reasoning examples.

    target_tokens: pad the relevant story with disjoint-entity distractor stories until the
      whole context reaches ~this many whitespace tokens (the long-context stress knob).
    min_hops: keep only examples whose reasoning chain is >= this many hops (genuine reasoning;
      default 4, override via CLUTRR_MIN_HOPS). Higher hops => bigger big-LM advantage.
    """
    if min_hops is None:
        min_hops = int(os.environ.get("CLUTRR_MIN_HOPS", "4"))
    path = os.path.join(CLUTRR_DATA_DIR, CLUTRR_TASK, f"{split}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"CLUTRR csv not found: {path}. Download from "
            f"https://raw.githubusercontent.com/kliang5/CLUTRR_huggingface_dataset/main/"
            f"{CLUTRR_TASK}/{split}.csv into {CLUTRR_DATA_DIR}/{CLUTRR_TASK}/."
        )
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    pool = [r for r in rows if (r.get("clean_story") or "").strip()]
    cand = [r for r in pool if _hops(r) >= min_hops]
    rng = random.Random(seed)
    rng.shuffle(cand)

    examples: List[ShortAnswerExample] = []
    for idx, r in enumerate(cand):
        if sample is not None and len(examples) >= sample:
            break
        story = r["clean_story"].strip()
        try:
            e1, e2 = ast.literal_eval(r["query"])
        except Exception:  # noqa: BLE001
            continue
        target = (r.get("target_text") or "").strip()
        if not target:
            continue
        qent = _entities(story) | {e1, e2}

        # Distractor stories with DISJOINT entities (cannot inject a false relation about the
        # queried people), padded until the context reaches target_tokens.
        docs = [story]
        toks = len(story.split())
        order = list(range(len(pool)))
        rng.shuffle(order)
        for j in order:
            dr = pool[j]
            if dr is r:
                continue
            ds = (dr.get("clean_story") or "").strip()
            if not ds or (_entities(ds) & qent):
                continue
            docs.append(ds)
            toks += len(ds.split())
            if toks >= target_tokens:
                break
        rng.shuffle(docs)  # place the relevant story at a random position among distractors
        docs = [_strip(d) for d in docs]

        question = (
            f"How is {e2} related to {e1}? In other words, {e2} is {e1}'s what? "
            f"Answer with a SINGLE family-relationship word (one of: son, daughter, mother, "
            f"father, brother, sister, grandfather, grandmother, grandson, granddaughter, "
            f"uncle, aunt, nephew, niece, father-in-law, mother-in-law, son-in-law, "
            f"daughter-in-law)."
        )
        examples.append(
            ShortAnswerExample(
                example_id=(r.get("id") or f"clutrr-{idx}"),
                question=question,
                documents=docs,
                answer=target,
                answers=[target],
                metadata={
                    "task": "clutrr",
                    "hops": _hops(r),
                    "n_docs": len(docs),
                    "context_words": toks,
                    "query": [e1, e2],
                    "target_tokens": target_tokens,
                    "f_comb": r.get("f_comb"),
                },
            )
        )
    return examples


# ------------------------- MULTI-QUESTION KV-REUSE variant (QASPER replacement, 2026-07-05) -------------------------
def build_clutrr_question(b: str, a: str) -> str:
    """The EXACT standard-CLUTRR question phrasing (shared by the fusion loader and the snapKV harness so the prompt
    is identical). Fact (A,r,B)=="B is A's r" -> ask how b is related to a. Uses the standard reason_then_answer
    prompt (NOT a custom variant — the custom one made the fusion answer-first / skip reasoning)."""
    return (f"How is {b} related to {a}? In other words, {b} is {a}'s what? "
            f"Answer with a SINGLE family-relationship word (one of: son, daughter, mother, "
            f"father, brother, sister, grandfather, grandmother, grandson, granddaughter, "
            f"uncle, aunt, nephew, niece, father-in-law, mother-in-law, son-in-law, "
            f"daughter-in-law).")


def load_clutrr_multiq_split(
    sample: Optional[int] = None,
    split: str = "test",
    cache_dir: Optional[str] = None,
    seed: int = 0,
    **_ignore,
) -> List[ShortAnswerExample]:
    """Multi-question CLUTRR: ONE story = shared context for several DERIVED (multi-hop) kinship questions
    harvested from `proof_state` (built by scripts/clutrr_multiq_build.py). Each question's answer is a single
    relation word (clean EM, no length pathology). Story count via env CLUTRR_MULTIQ_STORIES (default 100); the
    flat example order is story-major so example_id `<story_id>_<qi>` aligns with the snapKV-reuse harness.
    Fact convention (A, r, B) == "B is A's r"; question is phrased "In one word, B is A's what?" -> answer r."""
    import json as _json
    n_stories = int(os.environ.get("CLUTRR_MULTIQ_STORIES", "100"))
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "results/clutrr_multiq/clutrr_multiq.json")
    data = _json.load(open(path))[:n_stories]
    out: List[ShortAnswerExample] = []
    for item in data:
        story = item["context"]  # padded context (target story + disjoint-entity distractors)
        for qi, q in enumerate(item["questions"]):
            a, b, rel = q["a"], q["b"], q["rel"]
            out.append(ShortAnswerExample(
                example_id=f"{item['story_id']}_{qi}",
                question=build_clutrr_question(b, a),
                documents=[story],
                answer=rel,
                answers=[rel],
                num_supporting_facts=item.get("n_edges"),
                metadata={"dataset": "clutrr_multiq", "story_id": item["story_id"],
                          "q_index": qi, "n_questions": item["n_questions"]},
            ))
    if sample:
        out = out[:sample]
    return out
