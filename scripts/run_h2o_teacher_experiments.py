#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_evidence_sketch_experiments import (  # noqa: E402
    ASQA_STYLE_DATASETS,
    BABILONG_DATASET_NAMES,
    CLUTRR_DATASET_NAMES,
    _CLUTRR_TARGET_TOKENS,
    LONGBENCH_SUMM_DATASETS,
    _ALL_LONGBENCH_DATASETS,
    _build_asqa_examples,
    _build_teacher_prompt_with_audit,
    _context_stats,
    _extract_final_answer_for_dataset,
    _format_score_map,
    _load_alce_data,
    _maybe_shuffle_and_sample_examples,
    _resolve_alce_data_path,
    _score_asqa_prediction,
    _score_qa_prediction,
    _strip_generation_artifacts,
)
from src.data import (  # noqa: E402
    load_2wiki_split,
    load_hotpotqa_split,
    load_musique_split,
    load_qasper_split,
    load_scifact_split,
)
from src.longbench_data import load_longbench_split  # noqa: E402
from src.babilong_data import load_babilong_split  # noqa: E402
from src.clutrr_data import load_clutrr_split  # noqa: E402
from src.h2o_cache import generate_with_h2o_teacher  # noqa: E402
from src.memory import (  # noqa: E402
    cuda_memory_extra,
    format_cuda_memory_extra,
    format_cuda_memory_summary,
    reset_cuda_peak_memory,
    snapshot_cuda_memory,
)
from src.models import load_causal_lm, normalize_device_map  # noqa: E402
from src.qa_prompts import ASQA_PROMPT_VERSION_ANSWER_FIRST, ASQA_PROMPT_VERSIONS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teacher-only H2O KV cache eviction baseline.")
    parser.add_argument(
        "--dataset",
        choices=["hotpotqa", "qasper", "asqa", "2wiki", "2wikimultihopqa", "musique", "scifact", "longbench_govreport", "longbench_multinews", "longbench_passagecount", "longbench_passageretrieval", *BABILONG_DATASET_NAMES, *CLUTRR_DATASET_NAMES],
        default="hotpotqa",
    )
    parser.add_argument(
        "--babilong-task",
        default=None,
        help="BABILong task selector for babilong_* datasets (default None mixes qa1..qa10).",
    )
    parser.add_argument("--data", default=None, help="Optional ASQA/ALCE data path.")
    parser.add_argument(
        "--asqa-prompt-version",
        choices=sorted(ASQA_PROMPT_VERSIONS),
        default=ASQA_PROMPT_VERSION_ANSWER_FIRST,
    )
    parser.add_argument("--doc-number", type=int, default=10)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=None)
    parser.add_argument("--pass-number", type=int, default=0)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--retrieval-split", default="train")
    parser.add_argument("--max-corpus-examples", type=int, default=None)
    parser.add_argument("--retriever", choices=["BM25", "BGE"], default="BM25")
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=50000)
    parser.add_argument("--h2o-heavy-size", type=int, default=256)
    parser.add_argument("--h2o-recent-size", type=int, default=256)
    parser.add_argument("--h2o-prefill-chunk-size", type=int, default=256)
    parser.add_argument(
        "--h2o-keep-fraction",
        type=float,
        default=None,
        help=(
            "If set (e.g. 0.5), kept budget scales PER EXAMPLE with context: "
            "heavy=recent=keep_fraction/2 * prompt_len → fixed compression RATE instead of a "
            "fixed 256+256 token count. Overrides --h2o-heavy-size/--h2o-recent-size."
        ),
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--log-memory", action="store_true")
    parser.add_argument(
        "--revert-previous-prompt",
        action="store_true",
        help="Restore historical QA prompt behavior.",
    )
    return parser.parse_args()


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _load_examples(args: argparse.Namespace) -> Tuple[str, List[Any], Optional[Path]]:
    dataset_name = "2wiki" if args.dataset == "2wikimultihopqa" else args.dataset
    babilong_length: Optional[str] = None
    if dataset_name.startswith("babilong_"):
        babilong_length = dataset_name.split("babilong_", 1)[1]
        dataset_name = "babilong"
    clutrr_target_tokens: Optional[int] = None
    if dataset_name in _CLUTRR_TARGET_TOKENS:
        clutrr_target_tokens = _CLUTRR_TARGET_TOKENS[dataset_name]
        dataset_name = "clutrr"
    loader_sample = None if args.sample_seed is not None else args.sample
    data_path = None
    if dataset_name == "hotpotqa":
        examples, _ = load_hotpotqa_split(
            args.split,
            cache_dir=args.cache_dir,
            sample=loader_sample,
            doc_number=args.doc_number,
            retrieval_split=args.retrieval_split,
            max_corpus_examples=args.max_corpus_examples,
            return_stats=True,
            retriever=args.retriever,
        )
    elif dataset_name == "qasper":
        examples, _ = load_qasper_split(
            args.split,
            cache_dir=args.cache_dir,
            sample=loader_sample,
            doc_number=args.doc_number,
            retrieval_split=args.retrieval_split,
            max_corpus_examples=args.max_corpus_examples,
            return_stats=True,
            retriever=args.retriever,
        )
    elif dataset_name == "2wiki":
        examples, _ = load_2wiki_split(
            args.split,
            cache_dir=args.cache_dir,
            sample=loader_sample,
            doc_number=args.doc_number,
            retrieval_split=args.retrieval_split,
            max_corpus_examples=args.max_corpus_examples,
            return_stats=True,
            retriever=args.retriever,
        )
    elif dataset_name == "musique":
        examples, _ = load_musique_split(
            args.split,
            cache_dir=args.cache_dir,
            sample=loader_sample,
            doc_number=args.doc_number,
            retrieval_split=args.retrieval_split,
            max_corpus_examples=args.max_corpus_examples,
            return_stats=True,
            retriever=args.retriever,
        )
    elif dataset_name == "scifact":
        examples, _ = load_scifact_split(
            args.split,
            cache_dir=args.cache_dir,
            sample=loader_sample,
            doc_number=args.doc_number,
            retrieval_split=args.retrieval_split,
            max_corpus_examples=args.max_corpus_examples,
            return_stats=True,
            retriever=args.retriever,
        )
    elif dataset_name == "asqa":
        data_path = _resolve_alce_data_path(args)
        rows = _load_alce_data(data_path)
        if args.sample_seed is not None:
            random.Random(args.sample_seed).shuffle(rows)
            print(f"[SampleSeed] Shuffled {len(rows)} ASQA rows with seed={args.sample_seed}.")
        if args.sample:
            rows = rows[: args.sample]
        examples = _build_asqa_examples(rows, doc_number=args.doc_number, max_corpus_examples=args.max_corpus_examples)
    elif dataset_name in _ALL_LONGBENCH_DATASETS:
        examples = load_longbench_split(
            _ALL_LONGBENCH_DATASETS[dataset_name],
            split=args.split,
            sample=loader_sample,
            cache_dir=args.cache_dir,
        )
    elif dataset_name == "babilong":
        examples = load_babilong_split(
            length=babilong_length,
            task=getattr(args, "babilong_task", None),
            split=args.split,
            sample=loader_sample,
            cache_dir=args.cache_dir,
        )
        print(
            f"[Info] BABILong length={babilong_length} task={getattr(args, 'babilong_task', None) or 'ALL'}: "
            f"single long context, EM/F1 scoring."
        )
    elif dataset_name == "clutrr":
        examples = load_clutrr_split(
            target_tokens=clutrr_target_tokens or 4000,
            split=args.split,
            sample=loader_sample,
            cache_dir=args.cache_dir,
        )
        print(
            f"[Info] CLUTRR multi-hop kinship reasoning (target_tokens={clutrr_target_tokens or 4000}): "
            f"relevant story + disjoint-entity distractors, EM/F1 over the single kinship-word target."
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    if dataset_name != "asqa":
        examples = _maybe_shuffle_and_sample_examples(examples, sample=args.sample, sample_seed=args.sample_seed)
    return dataset_name, examples, data_path


def _update_running(running: Dict[str, float], score_map: Dict[str, float]) -> Dict[str, float]:
    running["count"] = running.get("count", 0.0) + 1.0
    for key, value in score_map.items():
        running[f"{key}_sum"] = running.get(f"{key}_sum", 0.0) + float(value)
    count = running["count"]
    return {key: running[f"{key}_sum"] / count for key in score_map}


def _attach_memory_extra(snapshot: Dict[str, Any], baseline: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    extra = cuda_memory_extra(snapshot, baseline)
    if extra:
        snapshot["extra_from_after_load"] = extra
    return snapshot


def _print_memory(label: str, snapshot: Dict[str, Any]) -> None:
    extra = snapshot.get("extra_from_after_load", {})
    extra_text = f" | {format_cuda_memory_extra(extra)}" if extra else ""
    print(f"[Memory][{label}] {format_cuda_memory_summary(snapshot)}{extra_text}")


def main() -> None:
    args = parse_args()
    if args.pass_number < 0:
        raise ValueError("--pass-number must be >= 0.")
    args.device_map = normalize_device_map(args.device_map, torch.cuda.device_count() if torch.cuda.is_available() else 0)

    dataset_name, examples, data_path = _load_examples(args)
    total_loaded = len(examples)
    if args.pass_number:
        skipped = min(args.pass_number, total_loaded)
        examples = examples[skipped:]
        print(f"[PassNumber] Skipping first {skipped} of {total_loaded} loaded examples; remaining={len(examples)}.")

    model, tokenizer = load_causal_lm(args.model, cache_dir=args.cache_dir, device_map=args.device_map)
    setattr(tokenizer, "_codex_revert_previous_prompt", bool(args.revert_previous_prompt))
    load_memory = None
    if args.log_memory:
        load_memory = snapshot_cuda_memory()
        print(f"[Memory][after_load] {format_cuda_memory_summary(load_memory)}")

    out_path = Path(args.output) if args.output else None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("", encoding="utf-8")

    print(
        f"[Info] Running H2O teacher baseline dataset={dataset_name} split={args.split} "
        f"doc_number={args.doc_number} sample={args.sample} sample_seed={args.sample_seed} "
        f"heavy_size={args.h2o_heavy_size} recent_size={args.h2o_recent_size} "
        f"prefill_chunk_size={args.h2o_prefill_chunk_size}"
    )

    running: Dict[str, float] = {}
    records = []
    progress_start = min(args.pass_number, total_loaded) + 1
    for idx, ex in enumerate(examples, start=progress_start):
        stats = _context_stats(ex.documents, tokenizer)
        print(f"[{idx}/{total_loaded}] docs={stats['doc_count']} tokens={stats['context_tokens']} words={stats['context_words']}")
        prompt, audit = _build_teacher_prompt_with_audit(
            ex,
            dataset=dataset_name,
            tokenizer=tokenizer,
            max_length=args.max_length,
            asqa_prompt_version=args.asqa_prompt_version,
        )
        if args.log_memory:
            reset_cuda_peak_memory()
        heavy_size, recent_size = args.h2o_heavy_size, args.h2o_recent_size
        if args.h2o_keep_fraction is not None:
            prompt_len = int(tokenizer(prompt, return_tensors="pt").input_ids.shape[1])
            half = max(1, int(args.h2o_keep_fraction / 2.0 * prompt_len))
            heavy_size = recent_size = half
            print(
                f"  [H2O] keep_fraction={args.h2o_keep_fraction} prompt_len={prompt_len} "
                f"-> heavy=recent={half} (kept≈{2*half}/{prompt_len}={2*half/max(1,prompt_len):.3f})"
            )
        result = generate_with_h2o_teacher(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            max_length=args.max_length,
            heavy_hitter_size=heavy_size,
            recent_size=recent_size,
            prefill_chunk_size=args.h2o_prefill_chunk_size,
            measure_memory=args.log_memory,
        )
        memory_snapshot = _attach_memory_extra(snapshot_cuda_memory(), load_memory) if args.log_memory else None
        raw_phase_memory = result.stats.pop("phase_memory", {})
        phase_memory = {
            phase: _attach_memory_extra(snapshot, load_memory)
            for phase, snapshot in raw_phase_memory.items()
            if isinstance(snapshot, dict)
        }
        if memory_snapshot is not None:
            _print_memory("teacher_h2o", memory_snapshot)
        for phase, snapshot in phase_memory.items():
            _print_memory(f"teacher_h2o.{phase}", snapshot)
        text = result.text
        if dataset_name in ASQA_STYLE_DATASETS:
            text = _strip_generation_artifacts(text)
            extracted, score_map = _score_asqa_prediction(text, ex)
        else:
            extracted = _extract_final_answer_for_dataset(text, dataset_name)
            em, f1 = _score_qa_prediction(ex, extracted, dataset_name)
            score_map = {"em": em, "f1": f1}
        avg = _update_running(running, score_map)
        print(
            f"[Eval][teacher_h2o] {_format_score_map(score_map)} {_format_score_map(avg, prefix='avg_')} "
            f"| prefill={result.timing.prefill_s:.2f}s decode={result.timing.decode_s:.2f}s "
            f"| cache_final={result.stats.get('final_cache_len')} cache_budget={result.stats.get('cache_size')}"
        )
        if args.debug:
            print(f"\n<<<Teacher question>>>\n{ex.question}")
            print(f"<<<Teacher raw>>>\n{text}")
            print(f"<<<Teacher extracted>>>\n{extracted}")
            print(f"<<<Teacher gold>>>\n{ex.answer}")
        record = {
            "example_id": ex.example_id,
            "question": ex.question,
            "answer": ex.answer,
            "text": text,
            "extracted": extracted,
            "metrics": score_map,
            "timing": {
                "prefill_s": result.timing.prefill_s,
                "decode_s": result.timing.decode_s,
                "total_s": result.timing.total_s,
            },
            "h2o": result.stats,
            "prompt_audit": audit,
        }
        if memory_snapshot is not None:
            record["memory"] = memory_snapshot
        if phase_memory:
            record["phase_memory"] = phase_memory
        records.append(record)
        if out_path is not None:
            _append_jsonl(out_path, record)

    summary = {
        "dataset": dataset_name,
        "examples": len(records),
        "total_loaded_examples": total_loaded,
        "sample": args.sample,
        "sample_seed": args.sample_seed,
        "pass_number": args.pass_number,
        "model": args.model,
        "asqa_prompt_version": args.asqa_prompt_version if dataset_name == "asqa" else None,
        "h2o": {
            "heavy_hitter_size": args.h2o_heavy_size,
            "recent_size": args.h2o_recent_size,
            "cache_size": args.h2o_heavy_size + args.h2o_recent_size,
            "prefill_chunk_size": args.h2o_prefill_chunk_size,
        },
        "metrics": {
            key[: -len("_sum")]: value / running["count"]
            for key, value in running.items()
            if key.endswith("_sum") and running.get("count", 0) > 0
        },
        "data_path": str(data_path) if data_path is not None else None,
    }
    if load_memory is not None:
        summary["memory_after_load"] = load_memory
    if out_path is not None:
        _append_jsonl(out_path, {"summary": summary})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
