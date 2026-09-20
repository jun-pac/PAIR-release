"""BABILong loader (long-context bAbI multi-fact reasoning QA).

BABILong (RMT-team/babilong) embeds classic bAbI reasoning tasks (qa1=single-
supporting-fact, qa2=two-fact, qa3=three-fact, qa5=three-arg-relations, ...) into
long natural-language distractor text drawn from PG19 books. As the context grows
(1k -> 128k tokens) the relevant facts get scattered among ever more irrelevant
text, so a model must LOCATE + COMBINE several dispersed facts. This is the hard
long-context regime that KV-eviction baselines (SnapKV) should drop facts on, and
where SLM+LM logit fusion is meant to help: the small SLM reads the full long
context, the big LM reasons over query+sketch.

Each row carries a SHORT entity answer (e.g. "kitchen", "John", "yes") so the
harness scores it on the EM/F1 path (like musique), NOT the ASQA rouge path.

The HuggingFace dataset is laid out as: each LENGTH is a config name ("1k", "2k",
"4k", "8k", "16k", "32k", "64k", "128k", ...), and within a length config the
SPLITS are the bAbI tasks ("qa1".."qa10"). A row has columns:
  - ``input``    : the long context (PG19 narrative with bAbI facts interleaved)
  - ``question`` : the bAbI question (e.g. "Where is Mary?")
  - ``target``   : the short gold answer (e.g. "bathroom")

We return ``ShortAnswerExample`` objects (the container the QA loop consumes):
``question`` = the bAbI question, ``documents`` = [the single long context]
(no BM25 retrieval; the doc_number arg is ignored, like qasper/longbench), and
``answer``/``answers`` = the short target. The question is kept OUT of the
documents so the harness' query-preserving truncation never drops it.
"""
from __future__ import annotations

import re
from typing import List, Optional, Sequence, Union

from datasets import load_dataset

from .data import ShortAnswerExample

# Primary repo: 100 samples / task / length, lengths up to 10M. Fallback repo:
# 1000 samples / task / length (lengths up to 128k). We prefer the primary and
# fall back only if it can't be reached.
BABILONG_REPOS: Sequence[str] = ("RMT-team/babilong", "RMT-team/babilong-1k-samples")

# bAbI tasks shipped inside every BABILong length config (verified: the length
# configs expose splits qa1..qa10). Used as the candidate set when mixing all
# tasks. We load each split directly (NOT get_dataset_split_names, which needs
# hub/metadata access and breaks on offline compute nodes).
BABILONG_ALL_TASKS: Sequence[str] = (
    "qa1", "qa2", "qa3", "qa4", "qa5", "qa6", "qa7", "qa8", "qa9", "qa10",
)

# bAbI task -> number of supporting facts that must be located + combined. Used
# only for metadata/analysis (break down accuracy-vs-length by #facts).
BABILONG_TASK_SUPPORTING_FACTS = {
    "qa1": 1,  # single supporting fact
    "qa2": 2,  # two supporting facts
    "qa3": 3,  # three supporting facts
    "qa4": 1,  # two-argument relations
    "qa5": 1,  # three-argument relations
    "qa6": 1,  # yes/no questions
    "qa7": 1,  # counting
    "qa8": 1,  # lists/sets
    "qa9": 1,  # simple negation
    "qa10": 1,  # indefinite knowledge
}

_TASK_RE = re.compile(r"^qa\d+$", re.IGNORECASE)


def _normalize_length(length: Optional[str]) -> str:
    """Normalize a length config name (e.g. '16K' -> '16k'). Defaults to '16k'."""
    if length is None or str(length).strip() == "":
        return "16k"
    return str(length).strip().lower()


def _requested_tasks(task, split) -> Optional[List[str]]:
    """Resolve the requested task list. None means 'all available tasks (mix)'."""
    raw: Optional[Sequence] = None
    if task is not None and task != "":
        raw = re.split(r"[,\s]+", task) if isinstance(task, str) else list(task)
    elif split is not None and _TASK_RE.match(str(split).strip()):
        # Allow the HF-style split ("qa2") to select a single task when no
        # explicit task is given. A generic split like "test"/"train" => mix all.
        raw = [str(split).strip()]
    if raw is None:
        return None
    tasks = [str(t).strip().lower() for t in raw if str(t).strip()]
    return tasks or None


def _build_example(row, *, length: str, task: str, idx: int) -> Optional[ShortAnswerExample]:
    context = (row.get("input") or row.get("context") or "").strip()
    question = (row.get("question") or "").strip()
    target = row.get("target")
    if target is None:
        target = row.get("answer")
    if isinstance(target, (list, tuple)):
        answer = ", ".join(str(x).strip() for x in target if str(x).strip())
    else:
        answer = str(target).strip() if target is not None else ""
    if not context or not question or not answer:
        return None
    return ShortAnswerExample(
        example_id=f"babilong-{length}-{task}-{idx}",
        question=question,
        documents=[context],
        answer=answer,
        answers=[answer],
        num_supporting_facts=BABILONG_TASK_SUPPORTING_FACTS.get(task),
        metadata={
            "dataset": "babilong",
            "babi_task": task,
            "babilong_length": length,
        },
    )


def load_babilong_split(
    length: str = "16k",
    task: Optional[Union[str, Sequence[str]]] = None,
    split: Optional[str] = None,
    sample: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> List[ShortAnswerExample]:
    """Load a BABILong length config as ShortAnswerExample objects.

    Args:
        length: BABILong length config, e.g. "1k","2k","4k","8k","16k","32k",
            "64k","128k". Encodes how much distractor text surrounds the facts.
        task: bAbI task selector. ``None`` => mix ALL available tasks
            (round-robin interleaved so a small ``sample`` stays balanced across
            reasoning types). A single task ("qa2") or a list/comma-string
            (["qa1","qa2"] / "qa1,qa2") restricts to those tasks.
        split: HF split. For BABILong the split IS the task; a generic value
            like "test"/"train" is ignored (=> mix), but a "qaN" split selects
            that task when ``task`` is not given.
        sample: cap on total returned examples (across the task mix).
        cache_dir: HF cache dir (compute nodes use the shared offline cache).

    The single long context is placed in ``documents`` and the bAbI question in
    ``question`` so the harness' query-preserving prompt never truncates the
    question. doc_number is irrelevant here (no retrieval).
    """
    norm_len = _normalize_length(length)
    requested = _requested_tasks(task, split)
    # The candidate task splits to attempt. When an explicit task list is given we
    # require all of them to load; when mixing (None) we try qa1..qa10 and keep
    # whichever splits are present in the cache (compute nodes load from cache).
    candidate_tasks = list(requested) if requested is not None else list(BABILONG_ALL_TASKS)

    last_err: Optional[Exception] = None
    for repo in BABILONG_REPOS:
        # Load each candidate split directly from cache (no metadata/hub call, so
        # this works on offline compute nodes once the shards are pre-downloaded).
        loaded = []
        per_task_err: Optional[Exception] = None
        for t in candidate_tasks:
            try:
                ds = load_dataset(repo, norm_len, split=t, cache_dir=cache_dir)
            except Exception as err:  # noqa: BLE001
                per_task_err = err
                last_err = err
                continue
            loaded.append((t, ds))
        if not loaded:
            # Nothing for this repo (e.g. config not cached); try the next repo.
            continue
        if requested is not None and len(loaded) != len(candidate_tasks):
            missing = [t for t in candidate_tasks if t not in {tt for tt, _ in loaded}]
            raise ValueError(
                f"Requested BABILong tasks {missing} could not be loaded for "
                f"length={norm_len!r} from {repo}. Last error: {per_task_err}"
            )

        # Round-robin interleave across tasks so a small sample is balanced.
        iters = [(t, iter(ds)) for t, ds in loaded]
        counters = {t: 0 for t, _ in loaded}
        exhausted = set()
        examples: List[ShortAnswerExample] = []
        while len(exhausted) < len(iters):
            for t, it in iters:
                if t in exhausted:
                    continue
                try:
                    row = next(it)
                except StopIteration:
                    exhausted.add(t)
                    continue
                ex = _build_example(row, length=norm_len, task=t, idx=counters[t])
                counters[t] += 1
                if ex is not None:
                    examples.append(ex)
                    if sample is not None and len(examples) >= sample:
                        return examples
        if examples:
            return examples
        last_err = RuntimeError(f"{repo}/{norm_len} produced no usable BABILong examples.")

    raise RuntimeError(
        f"Failed to load BABILong length={norm_len!r} (task={task!r}). "
        f"Tried repos {list(BABILONG_REPOS)}. Last error: {last_err}"
    ) from last_err
