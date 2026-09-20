from __future__ import annotations

import collections
import re
import string
from typing import Dict, Sequence, Tuple


_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)


def _normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation/articles/extra whitespace."""
    text = text.lower()
    # Drop academic citation markers (e.g. "BIBREF32"). In QASPER these are noise that
    # inflates token mismatch for verbose answers; applied to both pred and gold so a
    # citation-bearing gold ("Europarl BIBREF31") still matches "Europarl BIBREF31".
    # Harmless on datasets without citations (musique answers contain no BIBREF).
    text = re.sub(r"\bbibref\s*\d+", " ", text)
    text = _ARTICLES.sub(" ", text)
    # Hyphen family becomes a SPACE (not deleted): the blanket punctuation-delete below fused
    # hyphenated compounds into one token, so "Tulip-tree" (a prediction quoting the story's
    # 19th-century orthography) scored F1 0.0 against gold "tulip tree" (NarrativeQA-accum,
    # 2026-08-23; 14/154 fusion-vs-plain losses were pure hyphen artifacts). SQuAD-style deletion
    # is kept for all other punctuation ("U.S." -> "us" unchanged).
    text = re.sub(r"[-–—/]", " ", text)
    # A comma BETWEEN DIGITS is a SEPARATOR, so it becomes a space: "2,4,1,3" and "2, 4, 1, 3"
    # (LooGLE timeline-reorder answers) must agree, and under the bare punctuation-delete they do
    # not ("2413" vs "2 4 1 3").
    # ⚠ 2026-08-24 REGRESSION + FIX: the first version joined instead ("11, 2023" -> "112023"),
    # which GLUED dates written "Month DD, YYYY" into one token and silently cost LoCoMo-30
    # teacher 0.5583 -> 0.5329 (22/300 rows). Measured on real logs, space-substitution changes
    # 0/300 LoCoMo rows and 0/589 NarrativeQA rows while fixing 13/355 LooGLE rows. Thousands
    # separators are the only losers in principle ("1,600" -> "1 600"); zero such rows exist in
    # any current log.
    text = re.sub(r"(?<=\d)\s*,\s*(?=\d)", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = " ".join(text.split())
    return text


_FINAL_ANSWER = re.compile(r"\bfinal\s+answer\s*:\s*", re.IGNORECASE)
_ANSWER_TAG_PAIR = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_ANSWER_TAG = re.compile(r"</?answer>", re.IGNORECASE)
_ANSWER_STOP_MARKERS = (
    "<|endoftext|>",
    "<|im_end|>",
    "</s>",
    "<end_of_turn>",    # Gemma chat end-token (else it glues to the answer and breaks EM/F1)
    "<start_of_turn>",  # Gemma
    "<eos>",            # Gemma
    "<|user|>",         # OLMo-2 chat turn marker (else a new turn glues to the answer)
    "<|assistant|>",    # OLMo-2
    "Human:",
    "Assistant:",
    "User:",
    "\nQuestion:",
    "\nContext:",
    "\nDocument:",
    "\nGold answer:",
    "\nModel output:",
)
_LEADING_ANSWER_PHRASES = re.compile(
    r"^(?:[-*]\s*)?(?:the\s+)?(?:final\s+)?answer\s+(?:is|would\s+be|should\s+be)\s*:?\s+",
    re.IGNORECASE,
)


def _trim_at_stop_markers(text: str) -> str:
    for marker in _ANSWER_STOP_MARKERS:
        if marker in text:
            text = text.split(marker, 1)[0]
    return text.strip()


def _strip_wrapping(text: str) -> str:
    text = text.strip()
    text = text.strip(" \t\r\n`")
    text = text.strip(" \t\r\n\"'")
    return text.strip()


def _collapse_repeated_phrase(text: str) -> str:
    """Collapse exact token-level degeneration like 'A B A B A B' to 'A B'."""
    tokens = text.split()
    if len(tokens) < 4:
        return text
    max_unit = min(len(tokens) // 2, 32)
    for unit_len in range(1, max_unit + 1):
        unit = tokens[:unit_len]
        consumed = 0
        while consumed < len(tokens):
            chunk = tokens[consumed : consumed + unit_len]
            if chunk != unit[: len(chunk)]:
                break
            consumed += len(chunk)
        repeats = consumed / unit_len
        coverage = consumed / len(tokens)
        if repeats >= 2 and coverage >= 0.8:
            return " ".join(unit)
    return text


# 2026-08-10: our own reason_then_answer scaffold, re-emitted AFTER the answer. Qwen-32B does this on
# 14.5% of hotpot / 8.0% of musique turns ("Final Answer: USS Monitor Step 1 (Reasoning): The Orleans
# County Monitor was named after ..."), on ONE line, so the existing newline truncation misses it and the
# whole restarted block was scored as the answer. The floor-7B does it on only 2.2%, so the artifact is
# ARM-ASYMMETRIC — it was a main contributor to hotpot's inverted ladder (floor-7B EM-strict 0.525 >
# teacher-32B 0.483). A restarted scaffold is definitionally not part of the committed answer.
# Applied ONLY to text that follows a 'Final Answer:' marker (see extract_final_answer), so a log with no
# marker at all keeps its previous behaviour and no historical number changes silently for that reason.
# 2026-08-11: an LM-side adapter also re-emits the INSTRUCTION PREAMBLE after the answer
# ("Final Answer: 2020You are answering a question about a long multi-session conversation ..."),
# on 15/300 turns of the LM-FT+reader-FT arm at lambda 0.7. Same class as the restarted scaffold:
# definitionally not part of the committed answer, and it penalises whichever arm degenerates.
_RESTARTED_SCAFFOLD = re.compile(
    r"\s*(?:Step\s*\d+\s*[\(:]|\bReasoning\s*:|You are answering|You are given context|"
    r"\bRules:)", re.IGNORECASE)


def _cut_restarted_scaffold(text: str) -> str:
    m = _RESTARTED_SCAFFOLD.search(text)
    return text[: m.start()] if m and m.start() > 0 else text


def _clean_answer_candidate(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _ANSWER_TAG.sub("", text)
    text = _trim_at_stop_markers(text)
    text = text.split("\n", 1)[0].strip()
    text = _LEADING_ANSWER_PHRASES.sub("", text).strip()
    text = _strip_wrapping(text)
    text = _collapse_repeated_phrase(text)
    return _strip_wrapping(text)


def extract_final_answer(text: str) -> str:
    """Extract and clean the answer line following 'Final Answer:' when present."""
    text = str(text or "").strip()
    if not text:
        return ""

    tag_match = _ANSWER_TAG_PAIR.search(text)
    if tag_match is not None:
        text = tag_match.group(1).strip()

    # 2026-08-11: our own prompt emits a REDUNDANT pair of markers — "Step 2 (Answer):" on one line and
    # "Final Answer:" on the next — and a LoRA-tuned reader drops the second one on 25-36% of examples in
    # renderings it did not train on ("Step 2 (Answer):\nSerie B<|im_end|>"). The answer is right there; the
    # extractor was returning the whole reasoning and scoring it 0, which is a scoring failure, not a model
    # failure. Accept "Step 2 (Answer):" as an equivalent marker when the canonical one is absent.
    if not _FINAL_ANSWER.search(text):
        m = re.search(r"Step\s*2\s*\(Answer\)\s*:", text, re.IGNORECASE)
        if m:
            return _clean_answer_candidate(_cut_restarted_scaffold(text[m.end():]))
    parts = _FINAL_ANSWER.split(text)
    if len(parts) > 1:
        candidates = [_clean_answer_candidate(_cut_restarted_scaffold(part)) for part in parts[1:]]
        candidates = [candidate for candidate in candidates if candidate]
        if candidates:
            return candidates[0]
        return _clean_answer_candidate(parts[0])
    return _clean_answer_candidate(text)


def extract_final_answer_full(text: str) -> str:
    """Everything after the LAST 'Final Answer:' marker, preserving newlines, multiple
    sentences and bracketed citations [1][2]. Unlike extract_final_answer (which truncates
    to the first line and is used for short-factoid EM), this keeps the full cited long-form
    answer — needed for ASQA reason+citation, where the answer after 'Final Answer:' is a
    multi-sentence attributed paragraph fed to ALCE. If no marker is present, returns the
    whole (stop-marker-trimmed) text so non-reason logs are unaffected."""
    text = str(text or "").strip()
    if not text:
        return ""
    tag_match = _ANSWER_TAG_PAIR.search(text)
    if tag_match is not None:
        text = tag_match.group(1).strip()
    parts = _FINAL_ANSWER.split(text)
    tail = parts[-1] if len(parts) > 1 else text
    return _trim_at_stop_markers(tail).strip()


def compute_em_f1(pred: str, gold: str) -> Tuple[int, float]:
    """Return (EM, F1) for a single prediction."""
    norm_pred = _normalize_answer(pred)
    norm_gold = _normalize_answer(gold)
    em = int(norm_pred == norm_gold)

    pred_tokens = norm_pred.split()
    gold_tokens = norm_gold.split()
    if not pred_tokens and not gold_tokens:
        return em, 1.0
    if not pred_tokens or not gold_tokens:
        return em, 0.0

    # Count overlap
    common = {}
    for tok in pred_tokens:
        common[tok] = common.get(tok, 0) + 1
    overlap = 0
    for tok in gold_tokens:
        if common.get(tok, 0) > 0:
            overlap += 1
            common[tok] -= 1
    precision = overlap / len(pred_tokens) if pred_tokens else 0.0
    recall = overlap / len(gold_tokens) if gold_tokens else 0.0
    f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
    # Robust extractor rule: if the gold answer is fully contained in the prediction
    # (every gold token present, recall==1.0), count it as an exact match. Fixes
    # named-entity over-pruning, e.g. gold "Bombardier" vs pred "Bombardier Aerospace".
    # Wrong answers are unaffected: "Columbia Records" vs "Sundazed Records" has recall<1.0.
    # (No length guard: it over-penalized verbose-but-correct answers and was the wrong fix
    # for QASPER — the real issue there was BIBREF citation noise, handled in
    # _normalize_answer. QASPER's main metric is token-F1, not EM, anyway.)
    if not em and gold_tokens and recall >= 1.0:
        em = 1
    return em, f1


def compute_best_em_f1(pred: str, golds: Sequence[str]) -> Tuple[int, float]:
    """Return best (EM, F1) over multiple gold references."""
    if not golds:
        return compute_em_f1(pred, "")
    best_em = 0
    best_f1 = 0.0
    for gold in golds:
        em, f1 = compute_em_f1(pred, gold)
        if f1 > best_f1 or (f1 == best_f1 and em > best_em):
            best_em = em
            best_f1 = f1
    return best_em, best_f1


# --- QASPER answer canonicalization (robust EM/F1) ---
# Shared by the inline scorer (run_evidence_sketch_experiments.py) and the offline
# log evaluator (run_robust_em_f1_log_eval.py) so both report identical QASPER scores.
_LEADING_BOOLEAN_LABEL = re.compile(r"^\s*(yes|no)\b", re.IGNORECASE)
_UNANSWERABLE_PATTERNS = (
    re.compile(r"^\s*unanswerable\b", re.IGNORECASE),
    re.compile(r"^\s*(?:not|cannot be|can not be)\s+answer(?:ed|able)\b", re.IGNORECASE),
    re.compile(
        r"^\s*(?:the\s+)?(?:context|document|paper|passage)\s+(?:does\s+not|doesn't)\s+"
        r"(?:provide|state|mention|contain|say)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:the\s+)?(?:answer|information)\s+is\s+not\s+(?:provided|stated|mentioned|specified)\b",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*there\s+is\s+no\s+(?:information|evidence|mention)\b", re.IGNORECASE),
)


def canonicalize_qasper_prediction(pred: str, golds: Sequence[str]) -> str:
    """Canonicalize a QASPER prediction toward yes/no/unanswerable golds.

    QASPER golds are frequently the boolean labels "Yes"/"No" or "Unanswerable".
    Free-form generations like "Yes, the paper does." or "The context does not
    mention this." should score as exact matches. This collapses such phrasings to
    the canonical label *only when the gold set contains that label*, so other golds
    are left untouched.
    """
    normalized_golds = {_normalize_answer(gold) for gold in golds}
    if normalized_golds & {"yes", "no"}:
        match = _LEADING_BOOLEAN_LABEL.match(pred)
        if match is not None:
            return match.group(1).capitalize()
    if "unanswerable" in normalized_golds:
        for pattern in _UNANSWERABLE_PATTERNS:
            if pattern.search(pred):
                return "Unanswerable"
    return pred


# --- CLUTRR relation-word exact match (clean single-word metric, NO length pathology) ---
# The prediction is scanned for the FIRST relation-vocabulary word (LONGEST-first so "grandmother"/"mother-in-law"
# match before "mother"); score = 1.0 iff that word equals the gold relation. This is the identical rule the
# non-accumulation CLUTRR harness uses (scripts/clutrr_multiq_snapkv.py:rel_em) so the accumulate metric matches.
_CLUTRR_RELVOCAB = (
    "mother-in-law", "father-in-law", "daughter-in-law", "son-in-law", "sister-in-law", "brother-in-law",
    "grandmother", "grandfather", "granddaughter", "grandson", "mother", "father", "daughter", "son",
    "sister", "brother", "aunt", "uncle", "niece", "nephew", "wife", "husband",
)


def clutrr_relation_em(pred: str, golds: Sequence[str]) -> float:
    """1.0 iff the first relation word in `pred` equals the gold relation (golds[0]); else 0.0."""
    p = (pred or "").lower()
    gold = str((golds or [""])[0]).lower()
    for rel in _CLUTRR_RELVOCAB:
        if rel in p:
            return float(rel == gold)
    return 0.0


def compute_2wiki_em_f1(pred: str, golds: Sequence[str]) -> Tuple[int, float]:
    """
    2WikiMultiHopQA answer EM/F1, adapted from the official
    https://github.com/Alab-NII/2wikimultihop/blob/main/2wikimultihop_evaluate_v1.1.py.

    The important difference from the generic QA helper is the Hotpot-style special
    handling for yes/no/noanswer labels: a mismatched label receives zero token F1.
    """
    if not golds:
        golds = [""]
    norm_pred = _normalize_answer(pred)
    best_em = 0
    best_f1 = 0.0
    for gold in golds:
        norm_gold = _normalize_answer(gold)
        em = int(norm_pred == norm_gold)
        if norm_pred in {"yes", "no", "noanswer"} and norm_pred != norm_gold:
            f1 = 0.0
        elif norm_gold in {"yes", "no", "noanswer"} and norm_pred != norm_gold:
            f1 = 0.0
        else:
            pred_tokens = norm_pred.split()
            gold_tokens = norm_gold.split()
            common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
            overlap = sum(common.values())
            if overlap == 0:
                f1 = 0.0
            else:
                precision = overlap / len(pred_tokens)
                recall = overlap / len(gold_tokens)
                f1 = (2 * precision * recall) / (precision + recall)
        if f1 > best_f1 or (f1 == best_f1 and em > best_em):
            best_em = em
            best_f1 = f1
    return best_em, best_f1


def compute_musique_em_f1(pred: str, golds: Sequence[str]) -> Tuple[int, float]:
    """
    MuSiQue official answer metric: SQuAD/AllenNLP-style max EM/F1 over aliases.
    See https://github.com/StonyBrookNLP/musique/blob/main/metrics/answer.py.
    """
    return compute_best_em_f1(pred, golds)


_SCIFACT_LABEL_ALIASES = {
    "SUPPORT": "SUPPORT",
    "SUPPORTS": "SUPPORT",
    "SUPPORTED": "SUPPORT",
    "TRUE": "SUPPORT",
    "YES": "SUPPORT",
    "CONTRADICT": "CONTRADICT",
    "CONTRADICTS": "CONTRADICT",
    "CONTRADICTED": "CONTRADICT",
    "REFUTE": "CONTRADICT",
    "REFUTES": "CONTRADICT",
    "REFUTED": "CONTRADICT",
    "FALSE": "CONTRADICT",
    "NO": "CONTRADICT",
    "NOT_ENOUGH_INFO": "NOT_ENOUGH_INFO",
    "NOT ENOUGH INFO": "NOT_ENOUGH_INFO",
    "NEI": "NOT_ENOUGH_INFO",
    "UNKNOWN": "NOT_ENOUGH_INFO",
    "INSUFFICIENT": "NOT_ENOUGH_INFO",
}


def normalize_scifact_label(text: str) -> str:
    """Normalize generated text to one of SciFact's official label names when possible."""
    upper = text.upper().strip()
    upper = re.sub(r"[^A-Z_ ]+", " ", upper)
    upper = " ".join(upper.split())
    if upper in _SCIFACT_LABEL_ALIASES:
        return _SCIFACT_LABEL_ALIASES[upper]
    for candidate in ("NOT_ENOUGH_INFO", "CONTRADICT", "SUPPORT"):
        if candidate.replace("_", " ") in upper or candidate in upper:
            return candidate
    for phrase, label in (
        ("NOT ENOUGH", "NOT_ENOUGH_INFO"),
        ("INSUFFICIENT", "NOT_ENOUGH_INFO"),
        ("REFUTE", "CONTRADICT"),
        ("FALSE", "CONTRADICT"),
        ("TRUE", "SUPPORT"),
        ("SUPPORT", "SUPPORT"),
    ):
        if phrase in upper:
            return label
    return upper


def compute_scifact_label_metrics(preds: Sequence[str], golds: Sequence[str]) -> Dict[str, float]:
    """
    SciFact label-prediction metrics, adapted from the official
    https://github.com/allenai/scifact/blob/master/verisci/evaluate/label_prediction.py:
    accuracy, macro F1 over all labels, and macro F1 without NEI.
    """
    labels = ["CONTRADICT", "NOT_ENOUGH_INFO", "SUPPORT"]
    pred_norm = [normalize_scifact_label(p) for p in preds]
    gold_norm = [normalize_scifact_label(g) for g in golds]
    if not gold_norm:
        return {"accuracy": 0.0, "macro_f1": 0.0, "macro_f1_wo_nei": 0.0}
    accuracy = sum(int(p == g) for p, g in zip(pred_norm, gold_norm)) / len(gold_norm)

    def f1_for(label: str) -> float:
        tp = sum(1 for p, g in zip(pred_norm, gold_norm) if p == label and g == label)
        fp = sum(1 for p, g in zip(pred_norm, gold_norm) if p == label and g != label)
        fn = sum(1 for p, g in zip(pred_norm, gold_norm) if p != label and g == label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        return (2 * precision * recall / (precision + recall)) if precision + recall else 0.0

    macro_f1 = sum(f1_for(label) for label in labels) / len(labels)
    macro_f1_wo_nei = (f1_for("CONTRADICT") + f1_for("SUPPORT")) / 2.0
    return {"accuracy": accuracy, "macro_f1": macro_f1, "macro_f1_wo_nei": macro_f1_wo_nei}


def aggregate_metrics(counts: Dict[str, Dict[str, float]], total: int) -> Dict[str, float]:
    """Average metric dict given metric sums and total examples."""
    return {name: vals["sum"] / total for name, vals in counts.items()}
