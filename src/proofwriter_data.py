"""ProofWriter deductive-reasoning, long-context-ified with distractor theories.

ProofWriter (Tafjord et al. 2021): a small "theory" = a set of FACTS + RULES about a few
entities; the task is to decide whether a QUERY statement is TRUE or FALSE (or Unknown under
the open-world assumption) by applying the rules to the facts over QDep reasoning hops. This is
genuine multi-step deductive *reasoning* (rule application), NOT span extraction -- the regime
the SLM+LM thesis targets: the SLM reads the long context (relevant theory + distractors) and
surfaces the relevant facts/rules; the (query-only) LM is meant to do the deduction.

We pad the relevant theory with DISTRACTOR theories whose subject-entities are DISJOINT from the
query's entities (so they cannot inject a conflicting fact/rule about the queried entities) to a
target token length -> a clean long-context reasoning needle-in-haystack, exactly parallel to
src/clutrr_data.py. Answer = True/False (binary, QDep>=2 -> 50% majority baseline; set
PROOFWRITER_INCLUDE_UNKNOWN=1 to keep the 3-way Unknown class). Scored with EM/F1 over the word.

Data = the public tasksource/proofwriter HF dataset, pre-dumped to jsonl under
$PROOFWRITER_DATA_DIR (default /work/hdd/myproject/$USER/proofwriter_data) to AVOID the HF `datasets`
FileLock that fails on the /work Lustre mount (same workaround as CLUTRR). Each example becomes a
``ShortAnswerExample`` routed through the reason_then_answer QA prompt + EM scoring.
"""
from __future__ import annotations

import json
import os
import random
import re
from typing import List, Optional

from .data import ShortAnswerExample

_DEFAULT_DIR = f"/work/hdd/myproject/{os.environ.get('USER', 'anon')}/proofwriter_data"
PROOFWRITER_DATA_DIR = os.environ.get("PROOFWRITER_DATA_DIR", _DEFAULT_DIR)

_STOP = {
    "The", "If", "Is", "Are", "Based", "Statement", "Answer", "True", "False",
    "Unknown", "Does", "Do", "Can", "Someone", "Something", "All", "Then", "They",
}


def _query_tokens(question: str) -> set:
    """Subject tokens of the query statement: proper names (people) + the noun after 'the'
    (animals). Used to find distractor theories that say NOTHING about the queried entities."""
    toks = set()
    for w in re.findall(r"\b([A-Z][a-z]+)\b", question):
        if w not in _STOP:
            toks.add(w.lower())
    for w in re.findall(r"\bthe ([a-z]+)\b", question.lower()):
        toks.add(w)
    return toks


def _mentions_any(theory: str, tokens: set) -> bool:
    low = theory.lower()
    return any(re.search(r"\b" + re.escape(t) + r"\b", low) for t in tokens)


def _flavor(example_id: str) -> str:
    """ProofWriter worlds split cleanly by entity vocabulary: Att* = people names
    (Anne/Bob/...), Rel* = animals (the dog/bear/...). The two are disjoint."""
    return "att" if str(example_id).startswith("Att") else "rel"


_RULE_MARKERS = re.compile(r"\b(if|all|then|someone|something|are)\b", re.IGNORECASE)


def _facts_only(theory: str) -> str:
    """Keep only ground FACTS, drop RULES. ProofWriter rules are UNIVERSAL ('All furry things
    are white', 'If someone is rough then they chase the bald eagle'), so a distractor's rules
    would leak into the gold entity's logical closure and corrupt the True/False label. Ground
    facts ('Charlie is cold') about disjoint (cross-flavor) entities cannot change the query
    subject's derivation -> label-preserving distractors."""
    sents = re.split(r"(?<=\.)\s+", theory.strip())
    facts = [s for s in sents if s and not _RULE_MARKERS.search(s)]
    return " ".join(facts)


def load_proofwriter_split(
    target_tokens: int = 4000,
    split: str = "validation",
    sample: Optional[int] = None,
    min_depth: Optional[int] = None,
    include_unknown: Optional[bool] = None,
    seed: int = 0,
    cache_dir: Optional[str] = None,  # unused; jsonl is local
) -> List[ShortAnswerExample]:
    """Load ProofWriter as long-context deductive-reasoning examples.

    target_tokens: pad the relevant theory with disjoint-entity distractor theories until the
      whole context reaches ~this many whitespace tokens (the long-context stress knob).
    min_depth: keep only QDep >= this (genuine multi-hop deduction; default 2, env
      PROOFWRITER_MIN_DEPTH). include_unknown: keep the OWA 'Unknown' class (default False ->
      binary True/False, a clean 50% majority baseline; env PROOFWRITER_INCLUDE_UNKNOWN=1).
    """
    if min_depth is None:
        min_depth = int(os.environ.get("PROOFWRITER_MIN_DEPTH", "2"))
    if include_unknown is None:
        include_unknown = os.environ.get("PROOFWRITER_INCLUDE_UNKNOWN", "0") == "1"
    path = os.path.join(PROOFWRITER_DATA_DIR, f"{split}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"ProofWriter jsonl not found: {path}. Dump tasksource/proofwriter to "
            f"{PROOFWRITER_DATA_DIR}/{{validation,test}}.jsonl (fields id,theory,question,answer,QDep)."
        )
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    keep_ans = {"True", "False"} | ({"Unknown"} if include_unknown else set())
    pool = [r for r in rows if (r.get("theory") or "").strip()]
    cand = [
        r for r in pool
        if int(r.get("QDep", 0)) >= min_depth and str(r.get("answer", "")).strip() in keep_ans
    ]
    rng = random.Random(seed)
    rng.shuffle(cand)
    # Distractor pool = OPPOSITE-flavor theories (disjoint entity vocabulary), rules stripped.
    opp_facts = {"att": [], "rel": []}
    for r in pool:
        opp_facts[_flavor(r.get("id", ""))].append(r)

    examples: List[ShortAnswerExample] = []
    for idx, r in enumerate(cand):
        if sample is not None and len(examples) >= sample:
            break
        theory = r["theory"].strip()
        stmt = r["question"].strip()
        answer = str(r["answer"]).strip()
        gold_flavor = _flavor(r.get("id", ""))
        dpool = opp_facts["rel" if gold_flavor == "att" else "att"]

        # Pad with OPPOSITE-flavor distractor theories, RULES STRIPPED (ground facts only) -> the
        # disjoint cross-flavor entities + absence of universal rules guarantee the gold deduction
        # (and thus the True/False label) is unchanged. The gold theory is the only one with rules
        # about the queried entities -> a findable-but-buried reasoning needle.
        docs = [theory]
        toks = len(theory.split())
        order = list(range(len(dpool)))
        rng.shuffle(order)
        for j in order:
            df = _facts_only((dpool[j].get("theory") or ""))
            if not df:
                continue
            docs.append(df)
            toks += len(df.split())
            if toks >= target_tokens:
                break
        rng.shuffle(docs)  # place the relevant theory at a random position among distractors

        ans_options = "True or False" if not include_unknown else "True, False, or Unknown"
        question = (
            f"Using ONLY the facts and rules stated in the documents, determine whether the "
            f"following statement is {ans_options}.\n"
            f"Statement: {stmt}\n"
            f"Answer with a single word ({ans_options})."
        )
        examples.append(
            ShortAnswerExample(
                example_id=(r.get("id") or f"proofwriter-{idx}"),
                question=question,
                documents=docs,
                answer=answer,
                answers=[answer],
                metadata={
                    "task": "proofwriter",
                    "qdep": int(r.get("QDep", 0)),
                    "n_docs": len(docs),
                    "context_words": toks,
                    "flavor": "att" if str(r.get("id", "")).startswith("Att") else "rel",
                    "target_tokens": target_tokens,
                },
            )
        )
    return examples
