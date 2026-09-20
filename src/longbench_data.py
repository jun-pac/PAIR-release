"""LongBench summarization loaders (gov_report, multi_news).

Long-document summarization tasks used to stress-test the SLM+LM logit-fusion
method on long-form output that needs broad context (where KV-eviction
baselines like SnapKV lose dispersed content). Unlike the extractive QA
datasets here, the WHOLE context is summarized -- there is NO BM25 retrieval;
each example carries a single long context document.

Examples are returned as ``ShortAnswerExample`` (the same container the QA
loop consumes), so the run harness routes them through the ASQA-style prompt +
inline ``rougeLsum`` scoring path (see ASQA_STYLE_DATASETS in
run_evidence_sketch_experiments.py).
"""
from __future__ import annotations

import os
from typing import List, Optional

from datasets import load_dataset

from .data import ShortAnswerExample

# dataset-name suffix -> (THUDM/LongBench config name, summarization instruction).
# The instruction is used as the "question" of the shared QA prompt; LongBench
# summarization rows have an empty ``input`` field, so the instruction carries
# the task description.
LONGBENCH_SUMM_TASKS = {
    "gov_report": "You are given a government report. Write a one-page summary of the report.",
    "multi_news": "You are given several news articles. Write a summary of them.",
    # QMSum: query-based meeting summarization. Unlike gov_report/multi_news, the per-example
    # `input` field carries a QUERY (appended to this instruction by load_longbench_split), so the
    # answer is content the query points at in the transcript -> a query DOES enter the LM here.
    "qmsum": "You are given a meeting transcript. Answer the query below by summarizing the relevant content of the meeting.",
}

# ★ Length-constrained QMSum variant (2026-07-01): the default prompt lets every model over-generate
# (gold ≈ 71 words but generations run 179-346w), so rougeLsum-F is length-dominated (verbosity artifact,
# not summary quality). Select at runtime with SUMM_PROMPT_VARIANT=concise. Original NEVER overwritten
# (reproducibility). Gold QMSum answers average ~70 words → constrain to ~70.
QMSUM_CONCISE_INSTRUCTION = (
    "You are given a meeting transcript. Answer the query below by summarizing the relevant content "
    "of the meeting. Be concise: write about 60-80 words, no preamble, just the summary."
)

# WHOLE-CONTEXT, EM-scored LongBench tasks (NOT summarization). passage_count requires processing
# the ENTIRE context (count unique paragraphs after dedup) → a clean test of whether KV compression
# (snapkv evicting 90%) breaks when the answer truly needs all of the context. Answer = a number → EM.
LONGBENCH_COUNT_TASKS = {
    "passage_count": (
        "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. "
        "Read ALL the paragraphs carefully and determine how many UNIQUE paragraphs there are after "
        "removing the duplicates. The final answer must be a single integer (the count) only."
    ),
    "passage_retrieval_en": (
        "Here are several numbered paragraphs, followed by an abstract. Read ALL the paragraphs, then "
        "determine which one paragraph the abstract is summarizing. The final answer must be exactly in "
        "the form 'Paragraph X' where X is the paragraph number."
    ),
}
_ALL_LONGBENCH_TASKS = {**LONGBENCH_SUMM_TASKS, **LONGBENCH_COUNT_TASKS}


def load_longbench_split(
    task: str,
    split: str = "test",
    sample: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> List[ShortAnswerExample]:
    """Load a LongBench summarization split as ShortAnswerExample objects.

    Each LongBench row has fields ``input`` (query, empty for these summ tasks),
    ``context`` (the long document), ``answers`` (list[str] reference summary),
    and ``length``. We set ``question`` to a summarization instruction,
    ``documents`` to the single ``context`` document (no retrieval), and
    ``answer``/``answers`` to the reference summary.
    """
    if task not in _ALL_LONGBENCH_TASKS:
        raise ValueError(
            f"Unsupported LongBench task {task!r}; expected one of {sorted(_ALL_LONGBENCH_TASKS)}."
        )
    instruction = _ALL_LONGBENCH_TASKS[task]
    # Length-constrained QMSum variant (env-selected; task stays "qmsum" so the SAME HF config loads).
    if task == "qmsum" and os.environ.get("SUMM_PROMPT_VARIANT", "").lower() == "concise":
        instruction = QMSUM_CONCISE_INSTRUCTION
    # LongBench summarization configs only ship a `test` split.
    load_split = split or "test"

    def _load(split_name: str):
        # THUDM/LongBench ships a custom builder script -> needs trust_remote_code.
        # Some datasets versions don't accept the kwarg; fall back gracefully.
        try:
            return load_dataset(
                "THUDM/LongBench", task, split=split_name, cache_dir=cache_dir, trust_remote_code=True
            )
        except TypeError:
            return load_dataset("THUDM/LongBench", task, split=split_name, cache_dir=cache_dir)

    try:
        rows = _load(load_split)
    except Exception:  # noqa: BLE001
        if load_split != "test":
            print(
                f"[Warn] LongBench/{task} only ships a test split; "
                f"falling back from split={load_split!r} to test."
            )
            rows = _load("test")
        else:
            raise

    examples: List[ShortAnswerExample] = []
    for idx, row in enumerate(rows):
        if sample is not None and len(examples) >= sample:
            break
        context = (row.get("context") or "").strip()
        if not context:
            continue
        answers = [a for a in (str(x).strip() for x in (row.get("answers") or [])) if a]
        if not answers:
            continue
        query = (row.get("input") or "").strip()
        question = instruction if not query else f"{instruction}\n{query}"
        example_id = str(row.get("_id") or row.get("id") or f"{task}-{idx}")
        examples.append(
            ShortAnswerExample(
                example_id=example_id,
                question=question,
                documents=[context],
                answer=answers[0],
                answers=answers,
                metadata={
                    "task": task,
                    "longbench_length": row.get("length"),
                    "dataset": row.get("dataset") or task,
                },
            )
        )
    return examples
