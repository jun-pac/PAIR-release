from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import json
import os
import random
import time

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from transformers import AutoModel, AutoTokenizer

from datasets import load_dataset

from .counterfactual_qa import (
    build_counterfactual_gold_passage,
)
from .qa_prompts import build_full_context_qa_prompt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COUNTERFACTUAL_DATA_PATH = PROJECT_ROOT / "data" / "counterfactual_qa.jsonl"
DEFAULT_COUNTERFACTUAL_CORPUS_PATH = PROJECT_ROOT / "data" / "counterfactual_qa_corpus.jsonl"


@dataclass
class HotpotExample:
    """Container for a single HotpotQA distractor example."""

    example_id: str
    question: str
    documents: List[str]
    answer: str
    retrieval_scores: Optional[List[float]] = None
    num_supporting_facts: Optional[int] = None

    @property
    def context_text(self) -> str:
        return "\n\n".join(self.documents)

    def prompt(self, include_docs: bool = True, *, include_instruction: bool = True) -> str:
        """Return the text prompt used for generation."""
        if include_docs:
            return build_full_context_qa_prompt(self.question, self.documents)
        if include_instruction:
            return build_full_context_qa_prompt(self.question, [])
        return f"Context:\n\nQuestion: {self.question}\n"


@dataclass
class QasperExample:
    """Container for a single QASPER example."""

    example_id: str
    question: str
    documents: List[str]
    answer: str
    answers: Optional[List[str]] = None
    retrieval_scores: Optional[List[float]] = None

    @property
    def context_text(self) -> str:
        return "\n\n".join(self.documents)

    def prompt(self, include_docs: bool = True, *, include_instruction: bool = True) -> str:
        if include_docs:
            return build_full_context_qa_prompt(self.question, self.documents)
        if include_instruction:
            return build_full_context_qa_prompt(self.question, [])
        return f"Context:\n\nQuestion: {self.question}\n"


@dataclass
class ShortAnswerExample:
    """Container for short-answer/label QA benchmarks normalized to the QA prompt."""

    example_id: str
    question: str
    documents: List[str]
    answer: str
    answers: Optional[List[str]] = None
    retrieval_scores: Optional[List[float]] = None
    num_supporting_facts: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

    @property
    def context_text(self) -> str:
        return "\n\n".join(self.documents)

    def prompt(self, include_docs: bool = True, *, include_instruction: bool = True) -> str:
        if include_docs:
            return build_full_context_qa_prompt(self.question, self.documents)
        if include_instruction:
            return build_full_context_qa_prompt(self.question, [])
        return f"Context:\n\nQuestion: {self.question}\n"


def _simple_tokenize(text: str) -> List[str]:
    """Whitespace + lowercase tokenizer used for BM25."""
    return text.lower().split()


class BM25Retriever:
    """Lightweight BM25 wrapper with exclusion support."""

    def __init__(self, corpus: Sequence[str]):
        self.corpus = list(corpus)
        tokenized = [_simple_tokenize(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, k: int, *, exclude: Optional[Sequence[str]] = None) -> List[str]:
        exclude_set = set(exclude or [])
        tokens = _simple_tokenize(query)
        scores = self.bm25.get_scores(tokens)
        ranked_idx = np.argsort(scores)[::-1]
        results: List[str] = []
        for idx in ranked_idx:
            doc = self.corpus[int(idx)]
            if doc in exclude_set:
                continue
            results.append(doc)
            if len(results) >= k:
                break
        return results


def _bm25_top_k(docs: Sequence[str], query: str, k: int) -> List[str]:
    """Select top-k docs from a local list using BM25 to the query."""
    bm25 = BM25Okapi([_simple_tokenize(d) for d in docs])
    scores = bm25.get_scores(_simple_tokenize(query))
    ranked = np.argsort(scores)[::-1]
    return [docs[i] for i in ranked[:k]]


def _bm25_scores(docs: Sequence[str], query: str) -> List[float]:
    if not docs:
        return []
    bm25 = BM25Okapi([_simple_tokenize(d) for d in docs])
    scores = bm25.get_scores(_simple_tokenize(query))
    return scores.tolist()


def _load_bge_m3(cache_dir: Optional[str] = None):
    model_name = "BAAI/bge-m3"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
    model.eval()
    return model, tokenizer


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


def _encode_bge(
    texts: Sequence[str],
    *,
    model,
    tokenizer,
    batch_size: int = 16,
) -> torch.Tensor:
    embeddings: List[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            emb = _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            emb = emb / emb.norm(dim=1, keepdim=True)
        embeddings.append(emb.cpu())
    if not embeddings:
        return torch.empty((0, 0))
    return torch.cat(embeddings, dim=0)


def _cosine_scores(
    question: str,
    docs: Sequence[str],
    *,
    cache_dir: Optional[str] = None,
    model=None,
    tokenizer=None,
) -> List[float]:
    """Compute dense retrieval scores using BGE-M3 (dense mode); cosine similarity in [-1,1]."""
    if not docs:
        return []
    if model is None or tokenizer is None:
        model, tokenizer = _load_bge_m3(cache_dir)
    d_emb = _encode_bge(list(docs), model=model, tokenizer=tokenizer)
    with torch.no_grad():
        q_inputs = tokenizer(question, return_tensors="pt", truncation=True)
        q_emb = _mean_pool(model(**q_inputs).last_hidden_state, q_inputs["attention_mask"])
        q_emb = q_emb / q_emb.norm(dim=1, keepdim=True)
        scores = torch.matmul(d_emb, q_emb.T).squeeze(1)
    return scores.cpu().tolist()


def _format_documents(raw_context: Iterable) -> List[str]:
    """Flatten HotpotQA context into readable paragraphs."""
    documents: List[str] = []

    def _join_text_units(units) -> str:
        if isinstance(units, str):
            return units
        if units is None:
            return ""
        return " ".join(str(unit) for unit in units)

    if isinstance(raw_context, str):
        return [raw_context]

    if isinstance(raw_context, dict):
        titles = raw_context.get("title") or raw_context.get("titles") or []
        sentences_list = raw_context.get("sentences") or raw_context.get("content") or raw_context.get("contents") or []
        pairs = zip(titles, sentences_list)
    else:
        pairs = []
        for entry in raw_context:
            if isinstance(entry, dict):
                text_units = entry.get("sentences")
                if text_units is None:
                    text_units = entry.get("content")
                if text_units is None:
                    text_units = entry.get("contents", [])
                pairs.append((entry.get("title", ""), text_units))
            else:
                # Assume list/tuple with [title, sentences, ...].
                pairs.append((entry[0], entry[1]))

    for title, sentences in pairs:
        text = _join_text_units(sentences)
        if title:
            documents.append(f"{title}: {text}")
        else:
            documents.append(text)
    return documents


def _count_supporting_facts(row: Dict[str, Any]) -> int:
    """
    Count supporting facts in a HotpotQA row.
    The field is typically a dict with parallel lists: {"title": [...], "sent_id": [...]}.
    """
    sf = row.get("supporting_facts")
    if isinstance(sf, dict):
        titles = sf.get("title")
        sent_ids = sf.get("sent_id")
        if isinstance(titles, list):
            return len(titles)
        if isinstance(sent_ids, list):
            return len(sent_ids)
    if isinstance(sf, list):
        return len(sf)
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        return _count_supporting_facts(metadata)
    return 0


def _build_hotpot_corpus(
    split: str,
    *,
    cache_dir: Optional[str],
    max_examples: Optional[int] = None,
) -> List[str]:
    """Collect a deduplicated corpus of documents from the HotpotQA split for dense retrieval expansion."""
    dataset = None
    last_err: Optional[Exception] = None
    for split_name in _eval_split_aliases(split):
        try:
            dataset = load_dataset("hotpot_qa", "distractor", split=split_name, cache_dir=cache_dir)
            break
        except Exception as err:  # noqa: BLE001
            last_err = err
    if dataset is None:
        raise RuntimeError(f"Failed to load HotpotQA corpus split={split!r}. Last error: {last_err}") from last_err
    corpus: List[str] = []
    seen = set()
    for idx, row in enumerate(dataset):
        docs = _format_documents(row["context"])
        for doc in docs:
            if doc not in seen:
                seen.add(doc)
                corpus.append(doc)
        if max_examples and idx + 1 >= max_examples:
            break
    return corpus


def _eval_split_aliases(split: str) -> List[str]:
    if split in {"valid", "validation", "val", "dev"}:
        candidates = [split, "validation", "valid", "val", "dev", "test"]
    else:
        candidates = [split]
    aliases: List[str] = []
    for candidate in candidates:
        if candidate not in aliases:
            aliases.append(candidate)
    return aliases


def _load_first_available_dataset(
    candidates: Sequence[Tuple[str, Optional[str]]],
    *,
    split: str,
    cache_dir: Optional[str],
):
    last_err: Optional[Exception] = None
    for name, config in candidates:
        for split_name in _eval_split_aliases(split):
            try:
                if config is None:
                    return load_dataset(name, split=split_name, cache_dir=cache_dir)
                return load_dataset(name, config, split=split_name, cache_dir=cache_dir)
            except Exception as err:  # noqa: BLE001
                last_err = err
    tried = ", ".join(name if config is None else f"{name}/{config}" for name, config in candidates)
    raise RuntimeError(f"Failed to load dataset. Tried: {tried}. Last error: {last_err}") from last_err


def _load_2wiki_dataset(split: str, *, cache_dir: Optional[str]):
    try:
        return _load_first_available_dataset(
            [
                ("cmriat/2wikimultihopqa", None),
                ("framolfese/2WikiMultihopQA", None),
                ("2wikimultihopqa", None),
                ("voidful/2WikiMultihopQA", None),
            ],
            split=split,
            cache_dir=cache_dir,
        )
    except Exception as first_err:  # noqa: BLE001
        # LongBench exposes 2WikiMultiHopQA as the 2wikimqa config with only a test split.
        try:
            print(f"[Warn] Falling back to THUDM/LongBench/2wikimqa test split for requested split={split!r}.")
            return load_dataset("THUDM/LongBench", "2wikimqa", split="test", cache_dir=cache_dir)
        except Exception as longbench_err:  # noqa: BLE001
            raise RuntimeError(
                "Failed to load 2WikiMultiHopQA. Tried cmriat/2wikimultihopqa, "
                "framolfese/2WikiMultihopQA, 2wikimultihopqa, voidful/2WikiMultihopQA, "
                "and THUDM/LongBench/2wikimqa:test. "
                f"First error: {first_err}; LongBench error: {longbench_err}"
            ) from longbench_err


def _load_musique_dataset(split: str, *, cache_dir: Optional[str]):
    try:
        return _load_first_available_dataset(
            [
                ("dgslibisey/MuSiQue", None),
                ("bdsaglam/musique", None),
            ],
            split=split,
            cache_dir=cache_dir,
        )
    except Exception as first_err:  # noqa: BLE001
        # LongBench exposes MuSiQue with only a test split.
        try:
            print(f"[Warn] Falling back to THUDM/LongBench/musique test split for requested split={split!r}.")
            return load_dataset("THUDM/LongBench", "musique", split="test", cache_dir=cache_dir)
        except Exception as longbench_err:  # noqa: BLE001
            raise RuntimeError(
                "Failed to load MuSiQue. Tried dgslibisey/MuSiQue, bdsaglam/musique, "
                f"and THUDM/LongBench/musique:test. First error: {first_err}; "
                f"LongBench error: {longbench_err}"
            ) from longbench_err


def _format_musique_documents(row: Dict[str, Any]) -> List[str]:
    paragraphs = row.get("paragraphs") or row.get("contexts") or row.get("context") or []
    docs: List[str] = []
    if isinstance(paragraphs, dict):
        keys = paragraphs.keys()
        n = max((len(v) for v in paragraphs.values() if isinstance(v, list)), default=0)
        rebuilt = []
        for idx in range(n):
            item = {}
            for key in keys:
                values = paragraphs.get(key)
                if isinstance(values, list) and idx < len(values):
                    item[key] = values[idx]
            rebuilt.append(item)
        paragraphs = rebuilt
    for idx, paragraph in enumerate(paragraphs):
        if isinstance(paragraph, str):
            text = paragraph.strip()
            if text:
                docs.append(text)
            continue
        if not isinstance(paragraph, dict):
            continue
        title = paragraph.get("title") or paragraph.get("paragraph_title") or paragraph.get("doc_title") or ""
        text = (
            paragraph.get("paragraph_text")
            or paragraph.get("text")
            or paragraph.get("contents")
            or paragraph.get("passage")
            or ""
        )
        if isinstance(text, list):
            text = " ".join(str(x) for x in text)
        if not isinstance(text, str) or not text.strip():
            continue
        prefix = f"{title}: " if title else ""
        para_idx = paragraph.get("idx", idx)
        docs.append(f"[{para_idx}] {prefix}{text.strip()}")
    return docs


def _dedupe_preserve_order(docs: Iterable[str]) -> List[str]:
    deduped: List[str] = []
    seen = set()
    for doc in docs:
        if not doc or doc in seen:
            continue
        seen.add(doc)
        deduped.append(doc)
    return deduped


def _formatted_doc_title(doc: str) -> str:
    if doc.startswith("[") and "] " in doc:
        doc = doc.split("] ", 1)[1]
    if ": " not in doc:
        return ""
    return doc.split(": ", 1)[0].strip()


def _prioritize_2wiki_assigned_documents(row: Dict[str, Any], assigned_docs: Sequence[str]) -> List[str]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    supporting = metadata.get("supporting_facts") or row.get("supporting_facts") or {}
    support_titles = supporting.get("title") if isinstance(supporting, dict) else []
    if isinstance(support_titles, str):
        support_titles = [support_titles]
    support_title_set = {str(title).strip() for title in support_titles if str(title).strip()}
    if not support_title_set:
        return list(assigned_docs)
    gold_docs = [doc for doc in assigned_docs if _formatted_doc_title(doc) in support_title_set]
    distractors = [doc for doc in assigned_docs if _formatted_doc_title(doc) not in support_title_set]
    return _dedupe_preserve_order([*gold_docs, *distractors])


def _prioritize_musique_assigned_documents(row: Dict[str, Any], assigned_docs: Sequence[str]) -> List[str]:
    paragraphs = row.get("paragraphs") or []
    support_indices = {
        int(p.get("idx"))
        for p in paragraphs
        if isinstance(p, dict) and p.get("is_supporting") and p.get("idx") is not None
    }
    if not support_indices:
        return list(assigned_docs)
    support_prefixes = {f"[{idx}] " for idx in support_indices}
    gold_docs = [doc for doc in assigned_docs if any(doc.startswith(prefix) for prefix in support_prefixes)]
    distractors = [doc for doc in assigned_docs if not any(doc.startswith(prefix) for prefix in support_prefixes)]
    return _dedupe_preserve_order([*gold_docs, *distractors])


def _count_musique_supporting_facts(row: Dict[str, Any]) -> int:
    paragraphs = row.get("paragraphs") or []
    if isinstance(paragraphs, list):
        return sum(1 for p in paragraphs if isinstance(p, dict) and p.get("is_supporting"))
    return 0


def _row_answers(row: Dict[str, Any], *, answer_key: str = "answer") -> Tuple[str, List[str]]:
    answer = (
        row.get(answer_key)
        or row.get("answers")
        or row.get("golden_answers")
        or row.get("gold_answer")
        or row.get("final_answer")
        or ""
    )
    answers: List[str] = []
    if isinstance(answer, str) and answer.strip():
        answers.append(answer.strip())
    elif isinstance(answer, list):
        answers.extend(str(x).strip() for x in answer if str(x).strip())
    aliases = row.get("answer_aliases") or row.get("aliases") or row.get("answer_alias") or []
    if isinstance(aliases, str):
        aliases = [aliases]
    if isinstance(aliases, list):
        answers.extend(str(x).strip() for x in aliases if str(x).strip())
    deduped: List[str] = []
    seen = set()
    for item in answers:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return (deduped[0] if deduped else ""), deduped


def _build_assigned_plus_retrieved_docs(
    *,
    question: str,
    assigned_docs: Sequence[str],
    corpus: Optional[Sequence[str]],
    bm25_retriever: Optional[BM25Retriever],
    doc_number: int,
) -> Tuple[List[str], List[float]]:
    if doc_number < 0:
        raise ValueError("doc_number must be non-negative.")
    if doc_number == 0:
        return [], []  # closed-book: no context passages (parametric-knowledge / no-context control)
    docs = list(assigned_docs[:doc_number])
    if doc_number > len(docs):
        if bm25_retriever is None and corpus:
            bm25_retriever = BM25Retriever(corpus)
        if bm25_retriever is not None:
            docs.extend(bm25_retriever.search(question, k=doc_number - len(docs), exclude=docs))
    scores = _bm25_scores(docs, question)
    return _order_by_scores(docs, scores)


def _build_2wiki_corpus(
    split: str,
    *,
    cache_dir: Optional[str],
    max_examples: Optional[int],
) -> List[str]:
    dataset = _load_2wiki_dataset(split, cache_dir=cache_dir)
    corpus: List[str] = []
    seen = set()
    for idx, row in enumerate(dataset):
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        raw_context = row.get("context") or row.get("contexts") or metadata.get("context") or []
        for doc in _format_documents(raw_context):
            if doc and doc not in seen:
                seen.add(doc)
                corpus.append(doc)
        if max_examples and idx + 1 >= max_examples:
            break
    return corpus


def _build_musique_corpus(
    split: str,
    *,
    cache_dir: Optional[str],
    max_examples: Optional[int],
) -> List[str]:
    dataset = _load_musique_dataset(split, cache_dir=cache_dir)
    corpus: List[str] = []
    seen = set()
    for idx, row in enumerate(dataset):
        for doc in _format_musique_documents(row):
            if doc and doc not in seen:
                seen.add(doc)
                corpus.append(doc)
        if max_examples and idx + 1 >= max_examples:
            break
    return corpus


def _normalize_qasper_doc(entry: Any) -> List[str]:
    """Turn a QASPER doc entry into paragraph-level text blocks."""
    if isinstance(entry, str):
        return [entry]
    if not isinstance(entry, dict):
        return []
    pieces: List[str] = []
    title = entry.get("title") or ""
    if title:
        pieces.append(title)
    abstract = entry.get("abstract") or entry.get("abstract_text") or []
    if isinstance(abstract, str):
        abstract = [abstract]
    pieces.extend([p for p in abstract if isinstance(p, str)])
    if "paragraphs" in entry and isinstance(entry["paragraphs"], list):
        pieces.extend([p for p in entry["paragraphs"] if isinstance(p, str)])
    if "sections" in entry and isinstance(entry["sections"], list):
        for sec in entry["sections"]:
            if isinstance(sec, dict) and isinstance(sec.get("paragraphs"), list):
                pieces.extend([p for p in sec["paragraphs"] if isinstance(p, str)])
    return [p.strip() for p in pieces if isinstance(p, str) and p.strip()]


def _format_qasper_documents(example: Dict[str, Any]) -> List[str]:
    """Flatten QASPER paper structure into paragraph-level documents."""
    docs: List[str] = []
    full_text = example.get("full_text") or {}
    if isinstance(full_text, dict):
        section_names = full_text.get("section_name") or []
        paragraph_groups = full_text.get("paragraphs") or []
        for idx, paragraphs in enumerate(paragraph_groups):
            section_name = section_names[idx] if idx < len(section_names) else ""
            if not isinstance(paragraphs, list):
                continue
            for paragraph in paragraphs:
                if not isinstance(paragraph, str):
                    continue
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                docs.append(f"{section_name}: {paragraph}" if section_name else paragraph)
    if docs:
        return docs

    contexts = example.get("contexts") or example.get("context") or []
    if contexts:
        for ctx in contexts:
            docs.extend(_normalize_qasper_doc(ctx))

    if not docs:
        # Fallback to abstract/sections format
        abstract = example.get("abstract") or []
        if isinstance(abstract, str):
            abstract = [abstract]
        sections = example.get("sections") or []
        section_paras: List[str] = []
        for sec in sections:
            if isinstance(sec, dict) and isinstance(sec.get("paragraphs"), list):
                section_paras.extend([p for p in sec["paragraphs"] if isinstance(p, str)])
        docs.extend([p for p in abstract if isinstance(p, str)])
        docs.extend([p.strip() for p in section_paras if isinstance(p, str) and p.strip()])
    return docs


def _extract_qasper_questions(example: Dict[str, Any]) -> List[Tuple[str, str, str, List[str]]]:
    """
    Return list of (example_id, question, answer, answers) tuples.
    `answer` keeps the first valid answer string for compatibility.
    `answers` keeps all valid answer annotations for evaluation.
    """
    qas = example.get("qas") or example.get("questions") or []
    results: List[Tuple[str, str, str, List[str]]] = []
    if qas:
        if isinstance(qas, dict):
            # Some cached arrow rows store lists as columns inside a dict-like QA bundle.
            if isinstance(qas.get("question"), list) and isinstance(qas.get("question_id"), list):
                questions = qas.get("question") or []
                qids = qas.get("question_id") or []
                answers_list = qas.get("answers") or qas.get("answer") or []
                n = min(len(questions), len(qids), len(answers_list))
                for idx in range(n):
                    q_text = questions[idx] if isinstance(questions[idx], str) else ""
                    qid = qids[idx] if isinstance(qids[idx], str) else ""
                    all_answers = _extract_all_answer_texts(answers_list[idx])
                    ans_text = all_answers[0] if all_answers else ""
                    results.append((qid or example.get("id", ""), q_text, ans_text, all_answers))
                return results
            qas = [qas]
        for qa in qas:
            if isinstance(qa, dict):
                q_text = qa.get("question") or qa.get("question_text") or qa.get("query") or ""
                if isinstance(q_text, list):
                    q_text = q_text[0] if q_text else ""
                qid = qa.get("question_id") or qa.get("id") or qa.get("questionId") or ""
                answers = qa.get("answers") or qa.get("answer") or []
                all_answers = _extract_all_answer_texts(answers)
                ans_text = all_answers[0] if all_answers else ""
                results.append((qid or example.get("id", ""), q_text, ans_text, all_answers))
            elif isinstance(qa, str):
                results.append((example.get("id", ""), qa, "", []))
    else:
        # Single-question fallback
        q_text = example.get("question") or ""
        ans = example.get("answer") or ""
        ex_id = example.get("id", "")
        if q_text:
            all_answers = [ans] if isinstance(ans, str) and ans else []
            results.append((ex_id, q_text, ans if isinstance(ans, str) else "", all_answers))
    return results


def _extract_qasper_answer_text(answer_obj: Dict[str, Any]) -> str:
    if answer_obj.get("unanswerable") is True:
        return "Unanswerable"
    if answer_obj.get("yes_no") is not None:
        return "Yes" if answer_obj.get("yes_no") else "No"
    spans = answer_obj.get("extractive_spans") or []
    if isinstance(spans, list) and spans:
        span_texts = [s.strip() for s in spans if isinstance(s, str) and s.strip()]
        if span_texts:
            return ", ".join(span_texts)
    free_form = answer_obj.get("free_form_answer")
    if isinstance(free_form, str) and free_form.strip():
        return free_form.strip()
    return ""


def _iter_qasper_answer_objs(answers: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(answers, dict):
        if isinstance(answers.get("answer"), list):
            for item in answers["answer"]:
                if isinstance(item, dict):
                    yield item
            return
        if isinstance(answers.get("answer"), dict):
            yield answers["answer"]
            return
        yield answers
        return
    if isinstance(answers, list):
        for item in answers:
            if isinstance(item, dict):
                if isinstance(item.get("answer"), list):
                    for sub in item["answer"]:
                        if isinstance(sub, dict):
                            yield sub
                    continue
                if isinstance(item.get("answer"), dict):
                    yield item["answer"]
                else:
                    yield item


def _extract_answer_any_format(answers: Any) -> str:
    if isinstance(answers, str):
        return answers.strip()
    if isinstance(answers, list):
        for item in answers:
            if isinstance(item, str):
                s = item.strip()
                if s:
                    return s
    for ans_obj in _iter_qasper_answer_objs(answers):
        s = _extract_qasper_answer_text(ans_obj)
        if s:
            return s
    return ""


def _extract_all_answer_texts(answers: Any) -> List[str]:
    results: List[str] = []
    if isinstance(answers, str):
        text = answers.strip()
        return [text] if text else []
    if isinstance(answers, list):
        for item in answers:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    results.append(text)
    for ans_obj in _iter_qasper_answer_objs(answers):
        text = _extract_qasper_answer_text(ans_obj)
        if text:
            results.append(text)
    deduped: List[str] = []
    seen = set()
    for text in results:
        if text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped


def _get_qasper_paper_id(example: Dict[str, Any], *, fallback: str) -> str:
    return (
        example.get("id")
        or example.get("paper_id")
        or example.get("paperId")
        or example.get("doc_id")
        or fallback
    )


def _dense_search_indices(
    corpus: Sequence[str],
    corpus_embs: torch.Tensor,
    query: str,
    *,
    model,
    tokenizer,
    k: int,
    exclude_texts: Optional[Sequence[str]] = None,
    exclude_paper_id: Optional[str] = None,
    paper_ids: Optional[Sequence[str]] = None,
) -> List[int]:
    if not corpus:
        return []
    with torch.no_grad():
        q_inputs = tokenizer(query, return_tensors="pt", truncation=True)
        q_emb = _mean_pool(model(**q_inputs).last_hidden_state, q_inputs["attention_mask"])
        q_emb = q_emb / q_emb.norm(dim=1, keepdim=True)
        scores = torch.matmul(corpus_embs, q_emb.T).squeeze(1)
    ranked_idx = torch.argsort(scores, descending=True).tolist()
    results: List[int] = []
    exclude_set = set(exclude_texts or [])
    for idx in ranked_idx:
        if exclude_set and corpus[int(idx)] in exclude_set:
            continue
        if exclude_paper_id and paper_ids and paper_ids[int(idx)] == exclude_paper_id:
            continue
        results.append(int(idx))
        if len(results) >= k:
            break
    return results


def _order_by_scores(
    docs: Sequence[str],
    scores: Sequence[float],
    *,
    k: Optional[int] = None,
) -> Tuple[List[str], List[float]]:
    if not docs:
        return [], []
    ranked = np.argsort(scores)[::-1]
    # ranked = np.random.permutation(len(scores)) # random order
    if k is not None:
        ranked = ranked[:k]
    ordered_docs = [docs[i] for i in ranked]
    ordered_scores = [scores[i] for i in ranked]
    return ordered_docs, ordered_scores


def _load_qasper_dataset(split: str, *, cache_dir: Optional[str]):
    candidates = ["allenai/qasper", "qasper"]
    last_err: Optional[Exception] = None
    for name in candidates:
        for split_name in _eval_split_aliases(split):
            try:
                return load_dataset(name, split=split_name, cache_dir=cache_dir)
            except Exception as err:  # noqa: BLE001 - surface the most actionable error below
                last_err = err
                if "Dataset scripts are no longer supported" in str(err):
                    continue
    if last_err is not None:
        message = (
            "Failed to load QASPER dataset. Tried: "
            + ", ".join(candidates)
            + ". Last error: "
            + str(last_err)
        )
        if "Dataset scripts are no longer supported" in str(last_err):
            message += (
                " | QASPER still relies on a dataset script; please use datasets<3.0.0 "
                "(e.g., pip install 'datasets<3.0.0')."
            )
        raise RuntimeError(message) from last_err
    raise RuntimeError("Failed to load QASPER dataset (no candidates succeeded).")


def _build_qasper_corpus(
    split: str,
    *,
    cache_dir: Optional[str],
    max_examples: Optional[int] = None,
) -> List[str]:
    corpus, _ = _build_qasper_corpus_with_meta(split, cache_dir=cache_dir, max_examples=max_examples)
    return corpus


def _build_qasper_corpus_with_meta(
    split: str,
    *,
    cache_dir: Optional[str],
    max_examples: Optional[int] = None,
) -> Tuple[List[str], List[str]]:
    dataset = _load_qasper_dataset(split, cache_dir=cache_dir)
    corpus: List[str] = []
    paper_ids: List[str] = []
    for idx, row in enumerate(dataset):
        paper_id = _get_qasper_paper_id(row, fallback=str(idx))
        docs = _format_qasper_documents(row)
        for doc in docs:
            corpus.append(doc)
            paper_ids.append(paper_id)
        if max_examples and idx + 1 >= max_examples:
            break
    return corpus, paper_ids


def _load_scifact_claims(split: str, *, cache_dir: Optional[str]):
    last_err: Optional[Exception] = None
    for config in ("claims", None):
        for split_name in _eval_split_aliases(split):
            try:
                if config is None:
                    return load_dataset(
                        "allenai/scifact",
                        split=split_name,
                        cache_dir=cache_dir,
                        trust_remote_code=True,
                    )
                return load_dataset(
                    "allenai/scifact",
                    config,
                    split=split_name,
                    cache_dir=cache_dir,
                    trust_remote_code=True,
                )
            except Exception as err:  # noqa: BLE001
                last_err = err
    raise RuntimeError(f"Failed to load SciFact claims split={split!r}. Last error: {last_err}") from last_err


def _load_scifact_corpus(*, cache_dir: Optional[str]):
    for split in ("train", "validation", "test"):
        try:
            return load_dataset(
                "allenai/scifact",
                "corpus",
                split=split,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )
        except Exception:  # noqa: BLE001
            continue
    try:
        return load_dataset("allenai/scifact", "corpus", cache_dir=cache_dir, trust_remote_code=True)
    except Exception as err:  # noqa: BLE001
        raise RuntimeError(f"Failed to load SciFact corpus from allenai/scifact/corpus: {err}") from err


def _format_scifact_doc(row: Dict[str, Any]) -> Tuple[str, str]:
    doc_id = str(row.get("doc_id") or row.get("id") or row.get("corpusid") or row.get("corpus_id") or "")
    title = row.get("title") or ""
    abstract = row.get("abstract") or row.get("sentences") or row.get("text") or ""
    if isinstance(abstract, list):
        abstract_text = " ".join(str(x) for x in abstract)
    else:
        abstract_text = str(abstract)
    if title and abstract_text:
        return doc_id, f"{title}: {abstract_text}"
    return doc_id, abstract_text or str(title)


def _extract_scifact_gold_label(row: Dict[str, Any]) -> str:
    label = row.get("label") or row.get("evidence_label") or row.get("gold_label")
    if isinstance(label, str) and label:
        return label
    if isinstance(label, list):
        for item in label:
            if isinstance(item, str) and item:
                return item
    evidence = row.get("evidence") or {}
    labels: List[str] = []
    if isinstance(evidence, dict):
        iterable = evidence.values()
    elif isinstance(evidence, list):
        iterable = evidence
    else:
        iterable = []
    for value in iterable:
        if isinstance(value, dict):
            maybe = value.get("label")
            if isinstance(maybe, str):
                labels.append(maybe)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and isinstance(item.get("label"), str):
                    labels.append(item["label"])
    if labels:
        return labels[0]
    return "NOT_ENOUGH_INFO"


def _format_scifact_question(claim: str) -> str:
    return (
        "Classify the scientific claim using exactly one label: "
        "SUPPORT, CONTRADICT, or NOT_ENOUGH_INFO.\n"
        f"Claim: {claim}"
    )


def _iter_dataset_rows(dataset) -> Iterable[Dict[str, Any]]:
    if isinstance(dataset, dict):
        for split_rows in dataset.values():
            for row in split_rows:
                yield row
        return
    for row in dataset:
        yield row


def _load_counterfactual_hotpotqa_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if path.suffix.lower() == ".jsonl":
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path} at line {line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object rows in {path}; got {type(row).__name__} at line {line_no}.")
            rows.append(row)
        return rows
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(loaded, dict):
        loaded = loaded.get("data") or loaded.get("examples") or loaded.get("rows")
    if not isinstance(loaded, list):
        raise ValueError(f"Expected JSON list or JSONL rows in {path}.")
    for idx, row in enumerate(loaded):
        if not isinstance(row, dict):
            raise ValueError(f"Expected object rows in {path}; got {type(row).__name__} at index {idx}.")
        rows.append(row)
    return rows


def _load_counterfactual_corpus(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Counterfactual distractor corpus not found: {path}. "
            "Build it once with scripts/prepare_counterfactual_qa_benchmark.py."
        )
    docs: List[str] = []
    if path.suffix.lower() == ".jsonl":
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path} at line {line_no}: {exc}") from exc
            if isinstance(row, str):
                text = row
            elif isinstance(row, dict):
                text = str(row.get("text") or row.get("passage") or row.get("document") or "").strip()
            else:
                raise ValueError(f"Expected string or object rows in {path}; got {type(row).__name__}.")
            if text:
                docs.append(text)
        return docs
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(loaded, dict):
        loaded = loaded.get("corpus") or loaded.get("documents") or loaded.get("passages")
    if not isinstance(loaded, list):
        raise ValueError(f"Expected JSON list or JSONL rows in {path}.")
    for row in loaded:
        if isinstance(row, str):
            text = row
        elif isinstance(row, dict):
            text = str(row.get("text") or row.get("passage") or row.get("document") or "").strip()
        else:
            continue
        if text:
            docs.append(text)
    return docs


def _normalize_counterfactual_row(row: Dict[str, Any], *, idx: int) -> Dict[str, Any]:
    person = str(row.get("person") or "").strip()
    event = str(row.get("event") or "").strip()
    attribute_type = str(row.get("attribute_type") or "").strip()
    answer = str(row.get("answer") or row.get("gold") or "").strip()
    reason = str(row.get("reason") or "").strip()
    question = str(row.get("question") or "").strip()
    if not question and person and attribute_type:
        question = f"What is {person}'s favorite {attribute_type}?"
    gold_passage = str(row.get("gold_passage") or row.get("document") or row.get("context") or "").strip()
    if not gold_passage:
        required_fields = (
            ("person", person),
            ("event", event),
            ("attribute_type", attribute_type),
            ("answer", answer),
            ("reason", reason),
        )
        missing = [name for name, value in required_fields if not value]
        if missing:
            raise ValueError(f"Counterfactual row {idx} is missing fields needed to build gold_passage: {missing}")
        gold_passage = build_counterfactual_gold_passage(
            person=person,
            event=event,
            attribute_type=attribute_type,
            answer=answer,
            reason=reason,
        )
    if not question:
        raise ValueError(f"Counterfactual row {idx} is missing question.")
    if not answer:
        raise ValueError(f"Counterfactual row {idx} is missing answer.")
    if answer.lower() not in gold_passage.lower():
        raise ValueError(f"Counterfactual row {idx} gold_passage does not contain answer={answer!r}.")
    normalized = dict(row)
    normalized["id"] = str(row.get("id") or row.get("example_id") or f"cf_qa_{idx:04d}")
    normalized["question"] = question
    normalized["answer"] = answer
    normalized["gold_passage"] = gold_passage
    return normalized


def load_counterfactual_hotpotqa_split(
    split: str,
    cache_dir: Optional[str] = None,
    sample: Optional[int] = None,
    *,
    doc_number: int = 40,
    retrieval_split: str = "train",
    max_corpus_examples: Optional[int] = None,
    return_stats: bool = False,
    retriever: str = "BM25",
    data_path: Optional[str] = None,
    corpus_path: Optional[str] = None,
    count: Optional[int] = None,
    seed: int = 13,
) -> List[ShortAnswerExample]:
    """
    Load the standalone counterfactual QA benchmark.

    `doc_number` is the total number of passages placed in context. `doc_number=0`
    is the no-context / LM-only baseline: neither the synthetic gold passage nor
    distractors are included. For positive values, the synthetic gold passage is
    always included, so `doc_number=40` gives one gold passage plus 39 fixed-corpus
    distractors.
    """
    del split
    del count
    del retrieval_split
    if doc_number < 0:
        raise ValueError("Counterfactual benchmark requires doc_number >= 0.")
    if retriever not in {"BM25", "BGE"}:
        raise ValueError("Counterfactual benchmark supports BM25 or BGE retrieval score computation.")

    resolved_data_path = Path(data_path) if data_path else DEFAULT_COUNTERFACTUAL_DATA_PATH
    if not resolved_data_path.exists():
        raise FileNotFoundError(
            f"Counterfactual examples file not found: {resolved_data_path}. "
            "This loader never generates synthetic examples during evaluation; pass --data explicitly "
            "or create data/counterfactual_qa.jsonl."
        )
    raw_rows = _load_counterfactual_hotpotqa_rows(resolved_data_path)
    if sample:
        raw_rows = raw_rows[: min(sample, len(raw_rows))]
    rows = [_normalize_counterfactual_row(row, idx=idx) for idx, row in enumerate(raw_rows)]

    retrieval_start = time.perf_counter()
    distractor_count = max(doc_number - 1, 0)
    resolved_corpus_path = Path(corpus_path) if corpus_path else DEFAULT_COUNTERFACTUAL_CORPUS_PATH
    corpus = _load_counterfactual_corpus(resolved_corpus_path) if distractor_count else []
    if max_corpus_examples is not None:
        corpus = corpus[:max_corpus_examples]
    bm25_time_s = time.perf_counter() - retrieval_start
    if distractor_count and not corpus:
        raise RuntimeError("Counterfactual distractor corpus is empty; cannot build contexts.")

    dense_model = dense_tokenizer = None
    if retriever == "BGE" and doc_number > 0:
        dense_model, dense_tokenizer = _load_bge_m3(cache_dir)

    examples: List[ShortAnswerExample] = []
    for idx, row in enumerate(rows):
        rng = random.Random(seed * 1000003 + idx)
        if doc_number == 0:
            docs = []
            gold_index = None
            retrieval_scores = []
        else:
            gold_doc = row["gold_passage"]
            if distractor_count <= len(corpus):
                distractors = rng.sample(corpus, k=distractor_count)
            else:
                distractors = [rng.choice(corpus) for _ in range(distractor_count)]
            docs = [gold_doc, *distractors]
            # Place the guaranteed gold passage at a deterministic non-fixed position to exercise long-context retrieval.
            rng.shuffle(docs)
            gold_index = docs.index(gold_doc)
            if retriever == "BGE":
                retrieval_scores = _cosine_scores(
                    row["question"],
                    docs,
                    cache_dir=cache_dir,
                    model=dense_model,
                    tokenizer=dense_tokenizer,
                )
            else:
                retrieval_scores = _bm25_scores(docs, row["question"])
        examples.append(
            ShortAnswerExample(
                example_id=row["id"],
                question=row["question"],
                documents=docs,
                answer=row["answer"],
                answers=[row["answer"]],
                retrieval_scores=retrieval_scores,
                num_supporting_facts=0 if doc_number == 0 else 1,
                metadata={
                    "dataset": "counterfactual_qa",
                    "gold_doc_index": gold_index,
                    "no_context_baseline": doc_number == 0,
                    "person": row.get("person"),
                    "event": row.get("event"),
                    "attribute_type": row.get("attribute_type"),
                    "base_gold_passage": row.get("base_gold_passage"),
                },
            )
        )
    if return_stats:
        return examples, {"bm25_time_s": bm25_time_s}
    return examples


def load_counterfactual_qa_split(*args, **kwargs):
    return load_counterfactual_hotpotqa_split(*args, **kwargs)


def load_hotpotqa_split(
    split: str,
    cache_dir: Optional[str] = None,
    sample: Optional[int] = None,
    *,
    doc_number: int = 10,
    retrieval_split: str = "train",
    max_corpus_examples: Optional[int] = None,
    return_stats: bool = False,
    retriever: str = "BM25",
) -> List[HotpotExample]:
    """
    Load HotpotQA distractor split and optionally subsample.

    Args:
        split: "train" or "validation".
        cache_dir: Optional HF cache dir override.
        sample: If set, truncate to the first `sample` examples for quick debugging.
        doc_number: how many documents to include in the context (must be >=10).
        retrieval_split: split used to build retrieval corpus when doc_number>10.
        max_corpus_examples: optional limit when building the retrieval corpus for speed.
        return_stats: if True, return (examples, stats) where stats includes retrieval_time_s.
        retriever: "BM25" or "BGE".
    """
    if doc_number < 10:
        raise ValueError("HotpotQA requires at least 10 documents in context (doc_number >= 10).")

    dataset = None
    last_err: Optional[Exception] = None
    for split_name in _eval_split_aliases(split):
        try:
            dataset = load_dataset("hotpot_qa", "distractor", split=split_name, cache_dir=cache_dir)
            break
        except Exception as err:  # noqa: BLE001
            last_err = err
    if dataset is None:
        raise RuntimeError(f"Failed to load HotpotQA split={split!r}. Last error: {last_err}") from last_err
    if sample:
        dataset = dataset.select(range(min(sample, len(dataset))))

    corpus: Optional[List[str]] = None
    corpus_embs: Optional[torch.Tensor] = None
    bm25_retriever: Optional[BM25Retriever] = None
    bm25_time_s = 0.0
    dense_model = dense_tokenizer = None
    # HOTPOT_DOCS_CACHE_DIR: precomputed {example_id: [ordered docs]} (GPU-free, scripts/precompute_hotpot_docs.py).
    # If it covers all example_ids for this doc_number, load docs from it and SKIP the ~15-30 min BM25 corpus build.
    import json as _json
    _hcache = None
    _hdir = os.environ.get("HOTPOT_DOCS_CACHE_DIR")
    if _hdir:
        _hp = os.path.join(_hdir, f"hotpotqa_{split}_d{doc_number}.json")
        if os.path.exists(_hp):
            _hcache = _json.load(open(_hp))
    if _hcache is not None:
        # The cache holds a precomputed seed-subset (scripts/precompute_hotpot_docs.py). RESTRICT the dataset to
        # exactly the cached example_ids -> the run generates on that subset and NO BM25 is needed for anyone.
        _keys = set(_hcache.keys())
        dataset = dataset.filter(lambda r: str(r["id"]) in _keys)
    _hids = [str(r["id"]) for r in dataset]
    _use_hcache = _hcache is not None and len(_hids) > 0 and all(i in _hcache for i in _hids)
    if _use_hcache:
        print(f"[Info] HOTPOT docs-cache HIT ({_hp}) — {len(_hids)} examples, BM25 skipped (GPU-idle avoided)", flush=True)
    if (not _use_hcache) and doc_number >= 10:
        retrieval_start = time.perf_counter()
        corpus = _build_hotpot_corpus(retrieval_split, cache_dir=cache_dir, max_examples=max_corpus_examples)
        if retriever == "BGE":
            dense_model, dense_tokenizer = _load_bge_m3(cache_dir)
            corpus_embs = _encode_bge(corpus, model=dense_model, tokenizer=dense_tokenizer)
        else:
            bm25_retriever = BM25Retriever(corpus)
        bm25_time_s = time.perf_counter() - retrieval_start
    elif retriever == "BGE":
        dense_model, dense_tokenizer = _load_bge_m3(cache_dir)

    examples: List[HotpotExample] = []
    for row in dataset:
        if _use_hcache:
            examples.append(
                HotpotExample(
                    example_id=row["id"],
                    question=row["question"],
                    documents=_hcache[str(row["id"])],
                    answer=row["answer"],
                    retrieval_scores=None,
                    num_supporting_facts=_count_supporting_facts(row),
                )
            )
            continue
        documents = _format_documents(row["context"])
        if doc_number > len(documents):
            if corpus is None:
                raise ValueError("Retriever corpus was not initialized for HotpotQA expansion.")
            needed = doc_number - len(documents)
            if retriever == "BGE":
                if corpus_embs is None or dense_model is None or dense_tokenizer is None:
                    raise ValueError("Dense retriever was not initialized for HotpotQA expansion.")
                extra_indices = _dense_search_indices(
                    corpus,
                    corpus_embs,
                    row["question"],
                    model=dense_model,
                    tokenizer=dense_tokenizer,
                    k=needed,
                    exclude_texts=documents,
                )
                extras = [corpus[i] for i in extra_indices]
            else:
                if bm25_retriever is None:
                    raise ValueError("BM25 retriever was not initialized for HotpotQA expansion.")
                extras = bm25_retriever.search(row["question"], k=needed, exclude=documents)
            documents = documents + extras
        else:
            documents = documents[:doc_number]
        if retriever == "BGE":
            retrieval_scores = _cosine_scores(
                row["question"], documents, cache_dir=cache_dir, model=dense_model, tokenizer=dense_tokenizer
            )
        else:
            retrieval_scores = _bm25_scores(documents, row["question"])
        documents, retrieval_scores = _order_by_scores(documents, retrieval_scores)
        examples.append(
            HotpotExample(
                example_id=row["id"],
                question=row["question"],
                documents=documents,
                answer=row["answer"],
                retrieval_scores=retrieval_scores,
                num_supporting_facts=_count_supporting_facts(row),
            )
        )
    if return_stats:
        return examples, {"bm25_time_s": bm25_time_s}
    return examples


def load_qasper_split(
    split: str,
    cache_dir: Optional[str] = None,
    sample: Optional[int] = None,
    *,
    doc_number: int = 1,
    retrieval_split: str = "train",
    max_corpus_examples: Optional[int] = None,
    return_stats: bool = False,
    retriever: str = "BM25",
) -> List[QasperExample]:
    """
    Load QASPER split using the full paper context for every question.

    Args:
        split: dataset split ("train", "validation", or "test").
        cache_dir: Optional HF cache dir override.
        sample: If set, truncate to the first `sample` papers (pre-flattening).
        doc_number: ignored for QASPER. The full paper is always used.
        retrieval_split: ignored for QASPER.
        max_corpus_examples: ignored for QASPER.
        return_stats: if True, return (examples, stats) where stats includes retrieval_time_s.
        retriever: ignored for QASPER corpus expansion. Full-paper paragraphs are
            still ordered by in-paper BM25, matching the historical doc_number=0 behavior.
    """
    if doc_number < 0:
        raise ValueError("QASPER requires doc_number >= 0. The full paper context is always used.")

    dataset = _load_qasper_dataset(split, cache_dir=cache_dir)
    if sample:
        dataset = dataset.select(range(min(sample, len(dataset))))

    bm25_time_s = 0.0

    examples: List[QasperExample] = []
    for paper_idx, paper in enumerate(dataset):
        paper_docs = _format_qasper_documents(paper)
        questions = _extract_qasper_questions(paper)
        for qid, question, answer, answers in questions:
            in_paper = BM25Retriever(paper_docs)
            docs = in_paper.search(question, k=len(paper_docs))
            retrieval_scores = _bm25_scores(docs, question)
            examples.append(
                QasperExample(
                    example_id=qid,
                    question=question,
                    documents=docs,
                    answer=answer,
                    answers=answers,
                    retrieval_scores=retrieval_scores if paper_docs else [],
                )
            )
    if return_stats:
        return examples, {"bm25_time_s": bm25_time_s}
    return examples


def load_2wiki_split(
    split: str,
    cache_dir: Optional[str] = None,
    sample: Optional[int] = None,
    *,
    doc_number: int = 10,
    retrieval_split: str = "train",
    max_corpus_examples: Optional[int] = None,
    return_stats: bool = False,
    retriever: str = "BM25",
) -> List[ShortAnswerExample]:
    if retriever != "BM25":
        raise ValueError("2WikiMultiHopQA currently uses BM25 retrieval for assigned+extra passage ordering.")
    dataset = _load_2wiki_dataset(split, cache_dir=cache_dir)
    if sample:
        dataset = dataset.select(range(min(sample, len(dataset))))

    retrieval_start = time.perf_counter()
    corpus = _build_2wiki_corpus(retrieval_split, cache_dir=cache_dir, max_examples=max_corpus_examples)
    bm25_retriever = BM25Retriever(corpus) if corpus else None
    bm25_time_s = time.perf_counter() - retrieval_start

    examples: List[ShortAnswerExample] = []
    for idx, row in enumerate(dataset):
        question = row.get("question") or row.get("input") or ""
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        raw_context = row.get("context") or row.get("contexts") or metadata.get("context") or []
        assigned_docs = _prioritize_2wiki_assigned_documents(row, _format_documents(raw_context))
        docs, retrieval_scores = _build_assigned_plus_retrieved_docs(
            question=question,
            assigned_docs=assigned_docs,
            corpus=corpus,
            bm25_retriever=bm25_retriever,
            doc_number=doc_number,
        )
        answer, answers = _row_answers(row)
        examples.append(
            ShortAnswerExample(
                example_id=str(row.get("_id") or row.get("id") or idx),
                question=question,
                documents=docs,
                answer=answer,
                answers=answers,
                retrieval_scores=retrieval_scores,
                num_supporting_facts=_count_supporting_facts(row),
                metadata={"dataset": "2wiki"},
            )
        )
    if return_stats:
        return examples, {"bm25_time_s": bm25_time_s}
    return examples


def load_strategyqa_split(
    split: str = "train",
    cache_dir: Optional[str] = None,
    sample: Optional[int] = None,
    *,
    doc_number: int = 0,
    return_stats: bool = False,
    **kwargs,
) -> List[ShortAnswerExample]:
    """StrategyQA: implicit multi-hop yes/no reasoning. Uses the GOLD ``facts`` as the
    context (clean, no BM25 retrieval) — a pure test of whether the query-only LM
    completes the reasoning the small reader surfaces. Answer is yes/no. The test split
    labels are hidden, so we evaluate on the (labeled) train split."""
    # Offline mode can't resolve the Hub builder, so prefer the save_to_disk copy.
    import os
    from datasets import load_from_disk
    local = os.path.join(os.environ.get("HF_HOME", cache_dir or "."), "strategyqa_local")
    if os.path.isdir(local):
        ds = load_from_disk(local)
    else:
        ds = load_dataset("ChilleD/StrategyQA", cache_dir=cache_dir)
    dataset = ds["train"]
    if sample:
        dataset = dataset.select(range(min(sample, len(dataset))))
    examples: List[ShortAnswerExample] = []
    for idx, row in enumerate(dataset):
        facts = row.get("facts") or []
        if isinstance(facts, str):
            facts = [facts]
        docs = [str(f) for f in facts if str(f).strip()]
        ans = "yes" if row.get("answer") else "no"
        examples.append(
            ShortAnswerExample(
                example_id=str(row.get("qid") or idx),
                question=str(row.get("question") or ""),
                documents=docs,
                answer=ans,
                answers=[ans],
                num_supporting_facts=len(docs),
                metadata={"dataset": "strategyqa"},
            )
        )
    if return_stats:
        return examples, {"bm25_time_s": 0.0}
    return examples


def load_musique_split(
    split: str,
    cache_dir: Optional[str] = None,
    sample: Optional[int] = None,
    *,
    doc_number: int = 10,
    retrieval_split: str = "train",
    max_corpus_examples: Optional[int] = None,
    return_stats: bool = False,
    retriever: str = "BM25",
) -> List[ShortAnswerExample]:
    if retriever != "BM25":
        raise ValueError("MuSiQue currently uses BM25 retrieval for assigned+extra passage ordering.")
    dataset = _load_musique_dataset(split, cache_dir=cache_dir)
    if sample:
        dataset = dataset.select(range(min(sample, len(dataset))))

    # ★ DOCS-CACHE (2026-07-06): the BM25 corpus build + retrieval is ~50 min on CPU with the GPU IDLE, and the
    # retrieved docs are deterministic. If MUSIQUE_DOCS_CACHE_DIR holds a precomputed cache covering these example_ids
    # (built GPU-free by scripts/precompute_musique_docs.py), load docs from it and SKIP BM25 entirely.
    import json as _json
    _ids = [str(row.get("id") or row.get("_id") or i) for i, row in enumerate(dataset)]
    _cache = None
    _cdir = os.environ.get("MUSIQUE_DOCS_CACHE_DIR")
    if _cdir:
        _cp = os.path.join(_cdir, f"musique_{split}_d{doc_number}.json")
        if os.path.exists(_cp):
            _cache = _json.load(open(_cp))
    _use_cache = _cache is not None and all(i in _cache for i in _ids)

    retrieval_start = time.perf_counter()
    if _use_cache:
        corpus = None
        bm25_retriever = None
        print(f"[Info] MUSIQUE docs-cache HIT ({_cp}) — skipping BM25 for {len(_ids)} examples (GPU-idle avoided)", flush=True)
    else:
        corpus = _build_musique_corpus(retrieval_split, cache_dir=cache_dir, max_examples=max_corpus_examples)
        bm25_retriever = BM25Retriever(corpus) if corpus else None
    bm25_time_s = time.perf_counter() - retrieval_start

    examples: List[ShortAnswerExample] = []
    for idx, row in enumerate(dataset):
        question = row.get("question") or row.get("query") or ""
        _eid = str(row.get("id") or row.get("_id") or idx)
        if _use_cache:
            docs, retrieval_scores = _cache[_eid], None
        else:
            assigned_docs = _prioritize_musique_assigned_documents(row, _format_musique_documents(row))
            docs, retrieval_scores = _build_assigned_plus_retrieved_docs(
                question=question,
                assigned_docs=assigned_docs,
                corpus=corpus,
                bm25_retriever=bm25_retriever,
                doc_number=doc_number,
            )
        answer, answers = _row_answers(row)
        examples.append(
            ShortAnswerExample(
                example_id=str(row.get("id") or row.get("_id") or idx),
                question=question,
                documents=docs,
                answer=answer,
                answers=answers,
                retrieval_scores=retrieval_scores,
                num_supporting_facts=_count_musique_supporting_facts(row),
                metadata={"dataset": "musique", "answerable": row.get("answerable")},
            )
        )
    if return_stats:
        return examples, {"bm25_time_s": bm25_time_s}
    return examples


def load_scifact_split(
    split: str,
    cache_dir: Optional[str] = None,
    sample: Optional[int] = None,
    *,
    doc_number: int = 10,
    retrieval_split: str = "train",
    max_corpus_examples: Optional[int] = None,
    return_stats: bool = False,
    retriever: str = "BM25",
) -> List[ShortAnswerExample]:
    del retrieval_split
    del max_corpus_examples
    if retriever != "BM25":
        raise ValueError("SciFact is configured as BM25-only open-corpus retrieval.")
    if doc_number <= 0:
        raise ValueError("SciFact requires doc_number > 0.")

    claims = _load_scifact_claims(split, cache_dir=cache_dir)
    if sample:
        claims = claims.select(range(min(sample, len(claims))))

    retrieval_start = time.perf_counter()
    corpus_dataset = _load_scifact_corpus(cache_dir=cache_dir)
    corpus_rows = list(_iter_dataset_rows(corpus_dataset))
    corpus_ids: List[str] = []
    corpus_texts: List[str] = []
    for row in corpus_rows:
        doc_id, text = _format_scifact_doc(row)
        if text:
            corpus_ids.append(doc_id)
            corpus_texts.append(text)
    bm25 = BM25Okapi([_simple_tokenize(text) for text in corpus_texts]) if corpus_texts else None
    bm25_time_s = time.perf_counter() - retrieval_start

    examples: List[ShortAnswerExample] = []
    for idx, row in enumerate(claims):
        claim = row.get("claim") or row.get("question") or row.get("query") or ""
        if bm25 is None:
            docs: List[str] = []
            scores: List[float] = []
            doc_ids: List[str] = []
        else:
            all_scores = bm25.get_scores(_simple_tokenize(claim))
            ranked = np.argsort(all_scores)[::-1][:doc_number]
            docs = [corpus_texts[int(i)] for i in ranked]
            scores = [float(all_scores[int(i)]) for i in ranked]
            doc_ids = [corpus_ids[int(i)] for i in ranked]
        answer = _extract_scifact_gold_label(row)
        examples.append(
            ShortAnswerExample(
                example_id=str(row.get("id") or row.get("claim_id") or idx),
                question=_format_scifact_question(claim),
                documents=docs,
                answer=answer,
                answers=[answer],
                retrieval_scores=scores,
                metadata={"dataset": "scifact", "claim": claim, "doc_ids": doc_ids},
            )
        )
    if return_stats:
        return examples, {"bm25_time_s": bm25_time_s}
    return examples
