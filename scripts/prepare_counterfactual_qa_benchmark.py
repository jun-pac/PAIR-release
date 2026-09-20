#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _join_text_units(units: Any) -> str:
    if isinstance(units, str):
        return units
    if units is None:
        return ""
    return " ".join(str(unit) for unit in units)


def _format_documents(raw_context: Any) -> List[str]:
    documents: List[str] = []
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
                pairs.append((entry[0], entry[1]))

    for title, sentences in pairs:
        text = _join_text_units(sentences).strip()
        if not text:
            continue
        documents.append(f"{title}: {text}" if title else text)
    return documents


def _iter_hotpot_documents(*, split: str, cache_dir: Optional[str], max_examples: Optional[int]) -> Iterable[str]:
    dataset = load_dataset("hotpot_qa", "distractor", split=split, cache_dir=cache_dir)
    seen = set()
    for idx, row in enumerate(dataset):
        for doc in _format_documents(row["context"]):
            if doc in seen:
                continue
            seen.add(doc)
            yield doc
        if max_examples is not None and idx + 1 >= max_examples:
            break


def _load_examples(path: Path) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_no}")
        for key in ("id", "question", "answer", "gold_passage"):
            if not str(row.get(key) or "").strip():
                raise ValueError(f"Missing required key {key!r} at {path}:{line_no}")
        if str(row["answer"]).lower() not in str(row["gold_passage"]).lower():
            raise ValueError(f"Gold passage for {row['id']} does not contain answer={row['answer']!r}")
        examples.append(row)
    return examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze a standalone counterfactual benchmark by pairing fixed examples with a fixed distractor corpus."
    )
    parser.add_argument(
        "--examples",
        default=str(PROJECT_ROOT / "data" / "counterfactual_qa.jsonl"),
        help="Existing 400-example counterfactual JSONL file.",
    )
    parser.add_argument(
        "--corpus-output",
        default=str(PROJECT_ROOT / "data" / "counterfactual_qa_corpus.jsonl"),
        help="Output JSONL file containing fixed distractor passages.",
    )
    parser.add_argument(
        "--metadata-output",
        default=str(PROJECT_ROOT / "data" / "counterfactual_qa_metadata.json"),
        help="Output metadata file describing the frozen benchmark.",
    )
    parser.add_argument("--corpus-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--hotpot-split", default="train")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--max-hotpot-examples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples_path = Path(args.examples)
    corpus_path = Path(args.corpus_output)
    metadata_path = Path(args.metadata_output)

    examples = _load_examples(examples_path)
    docs = list(
        _iter_hotpot_documents(
            split=args.hotpot_split,
            cache_dir=args.cache_dir,
            max_examples=args.max_hotpot_examples,
        )
    )
    if len(docs) < args.corpus_size:
        raise RuntimeError(f"Requested {args.corpus_size} corpus passages, but only found {len(docs)} unique passages.")
    rng = random.Random(args.seed)
    selected = rng.sample(docs, k=args.corpus_size)

    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with corpus_path.open("w", encoding="utf-8") as handle:
        for idx, text in enumerate(selected):
            row = {
                "id": f"hotpot_distractor_{args.seed}_{idx:05d}",
                "source": "hotpot_qa/distractor",
                "source_split": args.hotpot_split,
                "text": text,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    metadata = {
        "name": "counterfactual_qa",
        "examples_file": str(examples_path),
        "num_examples": len(examples),
        "corpus_file": str(corpus_path),
        "num_corpus_passages": len(selected),
        "corpus_source": "hotpot_qa/distractor",
        "corpus_source_split": args.hotpot_split,
        "seed": args.seed,
        "notes": (
            "Examples are fixed counterfactual QA rows. The corpus is a fixed distractor pool sampled once "
            "from HotpotQA; evaluation code reads this file and does not load HotpotQA."
        ),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Validated {len(examples)} examples from {examples_path}")
    print(f"Wrote {len(selected)} fixed distractor passages to {corpus_path}")
    print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
