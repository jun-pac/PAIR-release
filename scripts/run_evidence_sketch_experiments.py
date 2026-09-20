import argparse
import os
import copy
import hashlib
import json
import random
import re
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import sys
import torch
from rank_bm25 import BM25Okapi
from rouge_score import rouge_scorer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.attention_gate import AttentionLinearCalibrator, AttentionRatioMLPGate, AttentionStatsGateMLP, infer_context_token_span
from src.attention_gate import attention_ctx_ratio, build_attention_stat_features, extract_attention_head_masses
from src.data import (
    HotpotExample,
    QasperExample,
    load_counterfactual_qa_split,
    load_2wiki_split,
    load_hotpotqa_split,
    load_musique_split,
    load_qasper_split,
    load_scifact_split,
    load_strategyqa_split,
)
from src.longbench_data import load_longbench_split
from src.babilong_data import load_babilong_split
from src.clutrr_data import load_clutrr_split, load_clutrr_multiq_split
from src.proofwriter_data import load_proofwriter_split
from src.eval import (
    compute_2wiki_em_f1,
    compute_best_em_f1,
    compute_em_f1,
    compute_musique_em_f1,
    compute_scifact_label_metrics,
    canonicalize_qasper_prediction,
    extract_final_answer,
    extract_final_answer_full,
    normalize_scifact_label,
)
from src.block_attention import (
    build_block_attention_decode_mask,
    build_block_attention_mask,
    build_pcw_decode_position_builder,
    build_pcw_position_ids,
)
from src.fusion import (
    FixedLambdaFusionDecoder,
    GenerationTiming,
    generate_with_single_model,
    get_eos_token_ids,
    normalize_learned_gate_type,
)
from src.gate_kl import KLStepSample, SampleBuildStats, TokenAgnosticGateMLP, build_attn_ratio_and_logit_topk_features, build_gate_features
from src.memory import (
    cuda_memory_extra,
    format_cuda_memory_extra,
    format_cuda_memory_summary,
    reset_cuda_peak_memory,
    snapshot_cuda_memory,
)
from src.models import (
    get_model_input_device,
    get_model_output_device,
    load_causal_lm,
    normalize_device_map,
    prepare_inputs,
    set_use_chat_template,
    synchronize_model,
)
from src.pced import PCEDContrastiveDecoder
from src.qa_prompts import (
    ASQA_PROMPT_VERSION_ANSWER_FIRST,
    ASQA_PROMPT_VERSION_SUPPORTED_ONLY,
    ASQA_PROMPT_VERSIONS,
    ASQA_EVIDENCE_SKETCH_INSTRUCTION,
    QA_FULL_CONTEXT_INSTRUCTION,
    QA_FULL_CONTEXT_INSTRUCTION_LEGACY,
    QA_OUTPUT_FORMAT,
    build_asqa_citation_answer_prompt,
    build_longbench_summary_prompt,
    build_asqa_citation_sketch_prompt,
    build_full_context_qa_prompt,
    build_query_preserving_full_context_qa_prompt,
    build_query_preserving_full_context_qa_prompt_with_token_segments,
    build_sketch_conditioned_qa_prompt,
)

DEFAULT_ALCE_DIR = Path("/work/hdd/myproject/anon/ALCE")
ALCE_TAR_URL = (
    "https://huggingface.co/datasets/princeton-nlp/ALCE-data/resolve/main/ALCE-data.tar?download=true"
)
_LEARNED_GATE_TYPES = {
    "logit_topk",
    "attn_ratio_and_logit_topk",
    "attn_linear_weighted",
    "attn_stats_mlp_unweighted",
    "attn_ratio_mlp_weighted",
}
_EXPORT_GATE_FEATURE_TYPES = {
    "logit_topk",
    "attn_ratio_and_logit_topk",
    "attn_linear_weighted",
    "attn_stats_mlp_unweighted",
    "attn_ratio_mlp_weighted",
}
_HEADWISE_EXPORT_GATE_TYPES = {"attn_linear_weighted", "attn_ratio_mlp_weighted"}
_NEG_INF = -1e9
_NEW_QA_DATASETS = {"2wiki", "2wikimultihopqa", "musique", "scifact"}
# LongBench summarization tasks reuse the ASQA machinery: the ASQA-style answer
# prompt (coverage-oriented, reason_then_answer) AND the inline rougeLsum
# scoring path. dataset-name -> THUDM/LongBench config name.
LONGBENCH_SUMM_DATASETS = {
    "longbench_govreport": "gov_report",
    "longbench_multinews": "multi_news",
    "longbench_qmsum": "qmsum",
}
# WHOLE-CONTEXT, EM-scored LongBench tasks (NOT ASQA-style): passage_count needs the entire context
# (count unique paragraphs) → answer is a number, scored with EM/F1 like QA. Routed through the QA
# path (reason_then_answer prompt + EM scoring), NOT the rougeLsum/ASQA path.
LONGBENCH_QA_DATASETS = {
    "longbench_passagecount": "passage_count",
    "longbench_passageretrieval": "passage_retrieval_en",
}
_ALL_LONGBENCH_DATASETS = {**LONGBENCH_SUMM_DATASETS, **LONGBENCH_QA_DATASETS}
# Datasets routed through the ASQA prompt + rougeLsum scoring path (free-form,
# reference-summary/long-answer tasks scored with ROUGE rather than EM/F1).
ASQA_STYLE_DATASETS = frozenset({"asqa", *LONGBENCH_SUMM_DATASETS.keys()})
# BABILong: long-context bAbI multi-fact reasoning QA. The dataset NAME encodes
# the context length (babilong_<len>) so length can be swept by name. The length
# config is parsed out in main(); scoring uses the EM/F1 path (like musique).
# Each name maps to a HF length config of RMT-team/babilong.
BABILONG_LENGTHS = ("0k", "1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k")
BABILONG_DATASET_NAMES = tuple(f"babilong_{length}" for length in BABILONG_LENGTHS)

# CLUTRR multi-hop kinship reasoning, long-context-ified with distractor stories. The name
# carries the target context length (clutrr_<len>) so length can be swept by name; bare "clutrr"
# defaults to 4k. min-hops filter via CLUTRR_MIN_HOPS env (default 4). See src/clutrr_data.py.
CLUTRR_LENGTHS = ("1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k")
CLUTRR_DATASET_NAMES = ("clutrr",) + tuple(f"clutrr_{length}" for length in CLUTRR_LENGTHS)
_CLUTRR_TARGET_TOKENS = {f"clutrr_{l}": int(l[:-1]) * 1000 for l in CLUTRR_LENGTHS}
_CLUTRR_TARGET_TOKENS["clutrr"] = 4000
# ProofWriter (deductive reasoning, long-context-ified) — same length-by-name + distractor scheme.
PROOFWRITER_LENGTHS = ("1k", "2k", "4k", "8k", "16k", "32k")
PROOFWRITER_DATASET_NAMES = ("proofwriter",) + tuple(f"proofwriter_{length}" for length in PROOFWRITER_LENGTHS)
_PROOFWRITER_TARGET_TOKENS = {f"proofwriter_{l}": int(l[:-1]) * 1000 for l in PROOFWRITER_LENGTHS}
_PROOFWRITER_TARGET_TOKENS["proofwriter"] = 4000
_ANSWER_TAG_PAIR = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE)
_ANSWER_TAG = re.compile(r"</?answer>", re.IGNORECASE)

_QA_SHARED_LEAD = (
    "You are given context passages and a question.\n"
    "Answer using ONLY the context.\n\n"
)
_QA_ANSWER_SUFFIX = (
    "Output format (STRICT):\n"
    "Final Answer: <answer>\n\n"
    "Rules:\n"
    "- Keep the answer short.\n"
    "- Output ONLY one line.\n"
    "- Start the line with `Final Answer:`.\n"
    "- Do NOT include explanations.\n"
    "- Do NOT include any other text.\n"
)
_QA_SKETCH_SUFFIX = (
    "Create a compact evidence sketch for answer generation.\n\n"
    "Rules:\n"
    "- Do NOT write the final answer.\n"
    "- Keep it short.\n"
    "- Output ONLY short bullet points.\n"
    "- Include only information relevant to the question.\n"
    "- Mention document numbers or locations when possible.\n"
    "- If the evidence is weak or incomplete, say so explicitly.\n\n"
    "Evidence Sketch:\n"
)

# Strengthened sketch prompt (2026-06-16). ADDED, not overwriting _QA_SKETCH_SUFFIX, so prior
# experiments stay reproducible. Selected at runtime via env SKETCH_PROMPT_VARIANT=strong.
# Goal: raise sketch recall (currently ~21% of gold appears in sketch) by demanding specific
# entities + multi-hop bridges and removing the brevity/"evidence is weak" hedging.
_QA_SKETCH_SUFFIX_STRONG = (
    "Create a detailed evidence sketch that captures every fact needed to answer the question.\n\n"
    "Rules:\n"
    "- Do NOT write the final answer.\n"
    "- Output bullet points; include as many as needed (do not force brevity).\n"
    "- Include the SPECIFIC entities, names, numbers, and dates from the context, in their exact surface form.\n"
    "- For multi-hop questions, include EVERY intermediate/bridge entity that links the question to the answer.\n"
    "- Mention document numbers or locations when possible.\n"
    "- Commit to the relevant facts; do NOT say the evidence is weak, missing, or incomplete.\n\n"
    "Evidence Sketch:\n"
)


def _get_sketch_suffix() -> str:
    import os

    return (
        _QA_SKETCH_SUFFIX_STRONG
        if os.environ.get("SKETCH_PROMPT_VARIANT", "default") == "strong"
        else _QA_SKETCH_SUFFIX
    )


def _get_answer_prompt_variant() -> str:
    """Answer-prompt variant, env-selected (mirrors SKETCH_PROMPT_VARIANT). Default OFF so
    queued/legacy jobs are unaffected. Set ANSWER_PROMPT_VARIANT=reason_then_answer (or
    no_refuse) to switch. Applies identically to teacher / SLM-teacher / ours (fairness)."""
    import os

    return os.environ.get("ANSWER_PROMPT_VARIANT", "default")


_EVIDENCE_SKETCH_PROMPT_PATH = PROJECT_ROOT / "Evidence_sketch_prompt.md"


def _ensure_eager_attention(model, *, reason: str) -> None:
    if hasattr(model, "set_attn_implementation"):
        try:
            model.set_attn_implementation("eager")
            print(f"[Info] Set SLM attn_implementation='eager' ({reason}).")
            return
        except Exception as exc:  # noqa: BLE001
            print(f"[Warn] Failed to set eager attention via set_attn_implementation: {exc}")
    try:
        if hasattr(model, "config"):
            setattr(model.config, "attn_implementation", "eager")
            setattr(model.config, "_attn_implementation", "eager")
            print(f"[Info] Forced SLM config attn_implementation='eager' ({reason}).")
    except Exception as exc:  # noqa: BLE001
        print(f"[Warn] Failed to force eager attention in config: {exc}")


def _ensure_sdpa_attention(model, *, reason: str) -> None:
    if hasattr(model, "set_attn_implementation"):
        try:
            model.set_attn_implementation("sdpa")
            print(f"[Info] Set model attn_implementation='sdpa' ({reason}).")
            return
        except Exception as exc:  # noqa: BLE001
            print(f"[Warn] Failed to set sdpa attention via set_attn_implementation: {exc}")
    try:
        if hasattr(model, "config"):
            setattr(model.config, "attn_implementation", "sdpa")
            setattr(model.config, "_attn_implementation", "sdpa")
            print(f"[Info] Forced model config attn_implementation='sdpa' ({reason}).")
    except Exception as exc:  # noqa: BLE001
        print(f"[Warn] Failed to force sdpa attention in config: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evidence-sketch experiments for SLM+LM fusion.")
    parser.add_argument(
        "--dataset",
        choices=[
            "hotpotqa",
            "counterfactual_qa",
            "counterfactual_hotpotqa",
            "cf_hotpotqa",
            "cf_qa",
            "qasper",
            "asqa",
            "2wiki",
            "2wikimultihopqa",
            "musique",
            "scifact",
            "strategyqa",
            "longbench_govreport",
            "longbench_multinews",
            "longbench_qmsum",
            "longbench_passagecount", "longbench_passageretrieval",
            *BABILONG_DATASET_NAMES,
            *CLUTRR_DATASET_NAMES,
            "clutrr_multiq",
            *PROOFWRITER_DATASET_NAMES,
        ],
        default="hotpotqa",
    )
    parser.add_argument(
        "--babilong-task",
        default=None,
        help=(
            "BABILong bAbI task selector (only used for babilong_* datasets). "
            "Default None mixes ALL available tasks (qa1..qa10) round-robin. "
            "Pass a single task (e.g. qa2) or a comma list (qa1,qa2,qa3) to restrict."
        ),
    )
    parser.add_argument(
        "--data",
        default=None,
        help=(
            "Path to ASQA/ALCE dataset file with retrieval results, or counterfactual_qa JSON/JSONL rows. "
            f"If omitted for ASQA, will look under {DEFAULT_ALCE_DIR} and download if missing. "
            "If omitted for counterfactual_qa, uses data/counterfactual_qa.jsonl."
        ),
    )
    parser.add_argument(
        "--asqa-prompt-version",
        choices=sorted(ASQA_PROMPT_VERSIONS),
        default=ASQA_PROMPT_VERSION_ANSWER_FIRST,
        help=(
            "ASQA citation prompt version. "
            f"Use {ASQA_PROMPT_VERSION_SUPPORTED_ONLY!r} to reproduce the previous refusal-on-unsupported prompt; "
            f"default {ASQA_PROMPT_VERSION_ANSWER_FIRST!r} starts with a direct answer and discourages unsupported refusals."
        ),
    )
    parser.add_argument("--doc-number", type=int, default=10)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument(
        "--counterfactual-count",
        type=int,
        default=400,
        help=(
            "Deprecated for counterfactual_qa. Evaluation now reads fixed data from --data "
            "or data/counterfactual_qa.jsonl and never generates examples."
        ),
    )
    parser.add_argument(
        "--counterfactual-corpus",
        default=None,
        help=(
            "Fixed distractor corpus JSON/JSONL for counterfactual_qa. "
            "Defaults to data/counterfactual_qa_corpus.jsonl."
        ),
    )
    parser.add_argument(
        "--counterfactual-seed",
        type=int,
        default=13,
        help="Seed for deterministic counterfactual distractor placement.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help=(
            "If set, shuffle the loaded evaluation examples with this seed before applying --sample. "
            "Without this flag, --sample keeps the historical first-N behavior."
        ),
    )
    parser.add_argument(
        "--decode-temperature",
        type=float,
        default=0.0,
        help=(
            "Manual-decode sampling temperature. Default 0.0 keeps pure greedy (argmax, "
            "byte-identical to prior runs). >0 enables nucleus sampling on the manual decode paths."
        ),
    )
    parser.add_argument(
        "--decode-top-p",
        type=float,
        default=1.0,
        help="Nucleus (top-p) cutoff for manual-decode sampling; only used when --decode-temperature>0.",
    )
    parser.add_argument(
        "--decode-seed",
        type=int,
        default=0,
        help="Seed for the manual-decode sampling RNG; only used when --decode-temperature>0.",
    )
    parser.add_argument(
        "--decode-sync-interval",
        type=int,
        default=1,
        help="Fusion decode: batch the GPU->CPU .item() sync + consume every K tokens (K>1 pipelines "
             "the forwards; output is byte-identical to K=1). Default 1 = exact current behavior.",
    )
    parser.add_argument(
        "--stop-on-repeat",
        type=str,
        default="",
        help=(
            "Opt-in manual-decode stop: once this marker substring appears a SECOND time in the "
            "generated text, stop and truncate before the 2nd occurrence (kills the late-EOS "
            "'Final Answer: X Final Answer: X ...' loop). Empty default = byte-identical to prior runs."
        ),
    )
    parser.add_argument(
        "--stop-after-answer",
        type=str,
        default="",
        help=(
            "Opt-in fusion-decode SPEEDUP for short-answer tasks: once this marker (e.g. "
            "'Final Answer:') is followed by a newline or --answer-token-budget tokens, stop "
            "decoding. The answer-extractor keeps only the marker line, so this is greedy-EQUIVALENT "
            "in the extracted answer (validated 0/579 mismatch) while cutting ~85%% of decode. "
            "Empty default = off (byte-identical to prior runs)."
        ),
    )
    parser.add_argument("--answer-token-budget", type=int, default=16,
                        help="Tokens to allow after the --stop-after-answer marker before forcing a stop.")
    parser.add_argument(
        "--pass-number",
        type=int,
        default=0,
        help=(
            "Sequentially skip this many loaded examples before evaluation. "
            "Use this to resume a simple debug/evaluation run that stopped mid-file."
        ),
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--retrieval-split", default="train")
    parser.add_argument("--max-corpus-examples", type=int, default=None)
    parser.add_argument("--retriever", choices=["BM25", "BGE"], default="BM25")
    parser.add_argument("--slm-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--lm-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--lm-lora", default=None,
                        help="Optional PEFT/LoRA adapter dir to load onto the LM (the query-only fusion branch). "
                             "Used to eval a fusion-FT'd LM (scripts/fusion_distill_train.py). Recorded in the output.")
    parser.add_argument("--slm-lora", default=None,
                        help="Optional PEFT/LoRA adapter dir to load onto the SLM (the context reader). "
                             "For joint SLM+LM fusion-FT (--train-slm).")
    parser.add_argument(
        "--quantized-lm-model",
        default=None,
        help="Context-reading model for quantized_lm_lm mode. Defaults to --lm-model.",
    )
    parser.add_argument("--lambda-weight", type=float, default=0.5)
    parser.add_argument(
        "--fusion-mode",
        choices=["weighted_sum", "prob_mix", "entropy"],
        default="weighted_sum",
        help="weighted_sum (raw-logit blend, default) or prob_mix (scale-robust probability mixture "
             "lambda*softmax(SLM)+(1-lambda)*softmax(LM); fixes big-magnitude-LM dominance, e.g. 72B) or "
             "entropy (per-token confidence gate: lambda from sigmoid(lm_ent-slm_ent), trust SLM when it is confident).",
    )
    parser.add_argument("--entropy-scale", type=float, default=1.0,
        help="slope for fusion-mode=entropy: lambda = entropy_scale*(sigmoid(lm_ent-slm_ent)-0.5)+0.5 (1.0=full swing).")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=50000)
    # ★ DEFAULT = 0 = NO evidence sketch. ES was abandoned (2026-06-20); the only method is
    # no-ES + reason_then_answer. A caller that omits this flag MUST get no-ES — never ES by
    # accident (a forgotten flag once silently turned ES back on for gov_report). To run the
    # legacy ES pipeline, pass --sketch-max-new-tokens >0 EXPLICITLY.
    parser.add_argument("--sketch-max-new-tokens", type=int, default=0)
    parser.add_argument("--sketch-max-length", type=int, default=None)
    parser.add_argument(
        "--single-model-decoding",
        choices=["generate", "manual"],
        default="generate",
        help="Decoding path for teacher/sketch generation.",
    )
    parser.add_argument("--slm-device-map", default="auto")
    parser.add_argument("--lm-device-map", default="auto")
    parser.add_argument("--quantized-lm-device-map", default="auto")
    parser.add_argument("--teacher-device-map", default="auto")
    parser.add_argument(
        "--quantized-lm-bits",
        type=int,
        choices=[4, 8],
        default=4,
        help="bitsandbytes weight quantization for the contextual LM branch in quantized_lm_lm mode.",
    )
    parser.add_argument(
        "--quantized-lm-4bit-quant-type",
        choices=["nf4", "fp4"],
        default="nf4",
        help="bitsandbytes 4-bit quantization type for quantized_lm_lm.",
    )
    parser.add_argument(
        "--quantized-lm-4bit-compute-dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="bitsandbytes 4-bit compute dtype for quantized_lm_lm.",
    )
    parser.add_argument(
        "--quantized-lm-4bit-use-double-quant",
        action="store_true",
        help="Enable nested bitsandbytes 4-bit quantization for quantized_lm_lm.",
    )
    parser.add_argument(
        "--slm-kv-quant-bits",
        type=int,
        choices=[0, 4, 8],
        default=0,
        help="KV-cache quantization (HF QuantizedCache) for ONLY the SLM/context branch of "
        "slm_lm / quantized_lm_lm fusion. 0=off (default, full-precision KV). 4/8 store the "
        "long-context KV at 4/8-bit so the context-reader gets a KV-memory advantage. The "
        "LM/query branch is never KV-quantized. Requires no-evidence-sketch (sketch-max-new-tokens=0). "
        "NOTE: 8-bit needs --slm-kv-backend hqq (quanto supports only 2/4-bit).",
    )
    parser.add_argument(
        "--slm-kv-backend",
        choices=["quanto", "hqq"],
        default="quanto",
        help="Backend for --slm-kv-quant-bits (quanto: 2/4-bit; hqq: 1/2/3/4/8-bit).",
    )
    parser.add_argument("--slm-kv-group-size", type=int, default=64, help="q_group_size for SLM-branch KV quant.")
    parser.add_argument(
        "--slm-kv-residual-length",
        type=int,
        default=128,
        help="Most-recent KV positions kept at full precision for SLM-branch KV quant (KIVI residual).",
    )
    parser.add_argument(
        "--teacher-quant-bits",
        type=int,
        default=0,
        choices=[0, 4, 8],
        help="WEIGHT-quantize the teacher model (bitsandbytes 4-bit nf4 / 8-bit int8). 0=off. "
        "This is the WEIGHT-quant compression baseline (reduces weight peak mem, unlike KV-quant).",
    )
    parser.add_argument(
        "--placement-strategy",
        choices=["auto_disjoint", "manual"],
        default="auto_disjoint",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--gpu-memory-reserve-gib", type=float, default=8.0)
    parser.add_argument(
        "--use-chat-template",
        action="store_true",
        default=False,
        help=(
            "Opt-in: wrap each prompt with the tokenizer's chat template before "
            "tokenizing (needed for models like OLMo-2 that require chat formatting). "
            "Default off keeps the historical raw-prompt behavior byte-identical."
        ),
    )
    parser.add_argument(
        "--load-weight",
        default=None,
        help="Optional .pt gate checkpoint from run_train_gate_kl.py for weighted_sum fusion.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["teacher", "slm_lm"],
        choices=["teacher", "slm_lm", "quantized_lm_lm", "blockattention", "pcw", "pced"],
        help="Which evaluation modes to run.",
    )
    parser.add_argument(
        "--pced-beta0",
        type=float,
        default=0.75,
        help="Fixed beta_0 coefficient for PCED when --pced-beta-mode fixed.",
    )
    parser.add_argument(
        "--pced-beta-mode",
        choices=["dynamic", "fixed"],
        default="dynamic",
        help="PCED beta strategy. The paper computes beta dynamically from first-token JSD and keeps it fixed.",
    )
    parser.add_argument(
        "--pced-beta-reduce",
        choices=["max", "mean"],
        default="max",
        help="How to reduce per-expert first-token JSD values into one PCED beta when beta mode is dynamic.",
    )
    parser.add_argument(
        "--pced-beta-warmup",
        type=float,
        default=0.0,
        help="Lower bound for dynamic PCED beta. Keep 0.0 for QA-style AdaCAD behavior.",
    )
    parser.add_argument(
        "--pced-gamma",
        type=float,
        default=2.5,
        help="Gamma coefficient for PCED rank prior (used when mode includes pced and retrieval scores are available).",
    )
    parser.add_argument(
        "--pced-relevance-mode",
        choices=["sparse", "dense", "minmax", "softmax", "none"],
        default="sparse",
        help=(
            "How to normalize loader retrieval scores into PCED r_k. Use sparse for BM25, "
            "dense for cosine-like scores in [-1,1]."
        ),
    )
    parser.add_argument(
        "--stop-strings",
        nargs="*",
        default=[],
        help="Optional stop strings for answer generation.",
    )
    parser.add_argument(
        "--sketch-stop-strings",
        nargs="*",
        default=[],
        help="Optional stop strings for sketch generation.",
    )
    parser.add_argument("--output", default=None, help="Optional JSONL output path.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output/export files by skipping already-saved example_ids.",
    )
    parser.add_argument(
        "--export-teacher-trace",
        default=None,
        help="Optional JSONL path for incrementally saving teacher-forcing traces by example_id.",
    )
    parser.add_argument(
        "--teacher-trace-input",
        default=None,
        help="Optional JSONL teacher-trace file previously exported with --export-teacher-trace.",
    )
    parser.add_argument(
        "--export-output-for-training",
        default=None,
        help="Optional .pt prepared-samples cache for gate training; updated incrementally after each example.",
    )
    parser.add_argument(
        "--export-gate-feature-type",
        choices=sorted(_EXPORT_GATE_FEATURE_TYPES),
        default="logit_topk",
        help="Feature type used when exporting prepared KL samples.",
    )
    parser.add_argument(
        "--export-gate-top-k",
        type=int,
        default=10,
        help="Top-k used for exported gate features.",
    )
    parser.add_argument(
        "--export-loss-top-k",
        type=int,
        default=20,
        help="Top-k from each model used to build the exported KL union.",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--revert-previous-prompt",
        action="store_true",
        help="Restore the pre-refactor QA prompt behavior from commit ed8c665.",
    )
    parser.add_argument(
        "--teacher-prompt-audit",
        action="store_true",
        help="Print teacher prompt construction/truncation stats for QA datasets.",
    )
    parser.add_argument(
        "--log-memory",
        action="store_true",
        help="Log CUDA memory after model load and per-mode generation peak memory.",
    )
    parser.add_argument(
        "--run-alce-eval",
        dest="run_alce_eval",
        action="store_true",
        default=True,
        help="Run official ALCE evaluation after ASQA generation.",
    )
    parser.add_argument(
        "--no-alce-eval",
        dest="run_alce_eval",
        action="store_false",
    )
    parser.add_argument("--alce-eval-mode", default=None)
    parser.add_argument(
        "--alce-dir",
        default=str(PROJECT_ROOT / "external" / "ALCE"),
    )
    parser.add_argument("--alce-data-file", default=None)
    parser.add_argument("--skip-autoais", action="store_true")
    parser.add_argument("--skip-mauve", action="store_true")
    return parser.parse_args()


def _parse_model_size_billions(model_name: str) -> Optional[float]:
    # Case-insensitive so Gemma-style lowercase names parse too (gemma-3-12b-it -> 12.0).
    match = re.search(r"(\d+(?:\.\d+)?)b\b", model_name, re.IGNORECASE)
    if match is None:
        return None
    return float(match.group(1))


def _single_gpu_budget_gib(*, gpu_idx: int, utilization: float, reserve_gib: float) -> float:
    total_gib = torch.cuda.get_device_properties(gpu_idx).total_memory / (1024 ** 3)
    usable_gib = min(total_gib * utilization, total_gib - reserve_gib)
    return max(usable_gib, 1.0)


def _fixed_min_required_gpus(model_name: str) -> int:
    size_b = _parse_model_size_billions(model_name)
    # Gemma sizes added: 4b/12b fit one 40GB card; 27b needs 2 on 40GB. OLMo-2 13B added (~14B class).
    if size_b in {0.5, 1.0, 1.5, 3.0, 4.0, 7.0, 8.0, 12.0, 13.0, 14.0, 32.0}:
        return 1
    if size_b in {27.0}:
        return 2
    if size_b in {70.0, 72.0}:
        return 2  # 96GB GH200: 72B bf16 (~145GB) fits on 2 cards (~78GB usable each). (was 5 for old 40GB cards.)
    raise ValueError(f"Unsupported model size in {model_name!r}.")


def _fixed_target_gpu_count(model_name: str, *, is_last: bool, available_now: int) -> int:
    size_b = _parse_model_size_billions(model_name)
    if size_b in {0.5, 1.0, 1.5, 3.0, 4.0, 7.0, 8.0, 12.0}:
        return 1
    if size_b in {13.0, 14.0}:  # OLMo-2 13B in the 14B class
        return available_now if is_last else 2
    if size_b in {27.0, 32.0, 70.0, 72.0}:
        return available_now
    raise ValueError(f"Unsupported model size in {model_name!r}.")


def _auto_assign_device_maps(args: argparse.Namespace, gpu_count: int) -> dict[str, Optional[dict]]:
    if gpu_count <= 0:
        return {}
    auto_like = {"auto", "balanced", "balanced_low_0", "sequential"}
    requests = []

    def add_request(field: str, model_name: str, enabled: bool) -> None:
        if not enabled:
            return
        requested_map = getattr(args, field)
        if requested_map not in auto_like:
            return
        requests.append(
            {
                "field": field,
                "model_name": model_name,
                "requested_map": requested_map,
                "min_required_gpus": _fixed_min_required_gpus(model_name),
            }
        )

    utilization = float(args.gpu_memory_utilization)
    reserve_gib = float(args.gpu_memory_reserve_gib)
    per_gpu_budget_gib = min(
        _single_gpu_budget_gib(gpu_idx=i, utilization=utilization, reserve_gib=reserve_gib)
        for i in range(gpu_count)
    )
    quantized_lm_name = args.quantized_lm_model or args.lm_model
    add_request("slm_device_map", args.slm_model, "slm_lm" in args.modes)
    add_request("quantized_lm_device_map", quantized_lm_name, "quantized_lm_lm" in args.modes)
    add_request("lm_device_map", args.lm_model, any(mode in args.modes for mode in ("slm_lm", "quantized_lm_lm", "blockattention", "pcw", "pced")))
    add_request("teacher_device_map", args.lm_model, "teacher" in args.modes)

    # ★★★ GPUs already claimed by a MANUAL cuda:N device map (e.g. --slm-device-map cuda:0) are
    # NOT in `requests` (add_request skips non-auto maps), so without reserving them the auto-assigned
    # LM would start at GPU 0 and OVERLAP the manually-pinned model. That was the 72B-fusion bug:
    # SLM pinned cuda:0 + 72B auto -> 72B got [0,1] (balanced) sharing GPU 0 with the SLM, corrupting
    # the fused output. Reserve manual-cuda GPUs so the auto pool is DISJOINT from them.
    reserved_gpus = set()
    for field in ("slm_device_map", "quantized_lm_device_map", "lm_device_map", "teacher_device_map"):
        m = getattr(args, field, None)
        if isinstance(m, str) and m.startswith("cuda:"):
            try:
                reserved_gpus.add(int(m.split(":", 1)[1]))
            except (ValueError, IndexError):
                pass
    free_gpus = [g for g in range(gpu_count) if g not in reserved_gpus]

    required_total = sum(item["min_required_gpus"] for item in requests)
    if required_total > len(free_gpus):
        raise RuntimeError(
            f"Auto placement cannot fit requested models on {len(free_gpus)} free GPU(s) "
            f"(of {gpu_count}; {sorted(reserved_gpus)} reserved by manual device maps)."
        )

    assignments = {}
    cursor = 0
    for idx, item in enumerate(requests):
        reserve_for_later = sum(later["min_required_gpus"] for later in requests[idx + 1 :])
        available_now = len(free_gpus) - cursor - reserve_for_later
        assigned_count = _fixed_target_gpu_count(
            item["model_name"],
            is_last=idx == len(requests) - 1,
            available_now=available_now,
        )
        assigned = free_gpus[cursor:cursor + assigned_count]
        cursor += assigned_count
        if len(assigned) == 1:
            resolved_map = f"cuda:{assigned[0]}"
            max_memory = None
        else:
            resolved_map = "balanced"
            max_memory = {gpu_idx: f"{per_gpu_budget_gib:.0f}GiB" for gpu_idx in assigned}
        setattr(args, item["field"], resolved_map)
        assignments[item["field"]] = max_memory
        print(
            f"[Info] Auto placement {item['field']} model={item['model_name']} "
            f"assigned_gpus={assigned} device_map={resolved_map}"
        )
    return assignments


def _prepare_devices(args: argparse.Namespace) -> None:
    args._model_max_memory = {}
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        args.slm_device_map = normalize_device_map(args.slm_device_map, gpu_count)
        args.lm_device_map = normalize_device_map(args.lm_device_map, gpu_count)
        args.quantized_lm_device_map = normalize_device_map(args.quantized_lm_device_map, gpu_count)
        args.teacher_device_map = normalize_device_map(args.teacher_device_map, gpu_count)
        if args.placement_strategy == "auto_disjoint":
            args._model_max_memory = _auto_assign_device_maps(args, gpu_count)
    else:
        for field in ("slm_device_map", "lm_device_map", "quantized_lm_device_map", "teacher_device_map"):
            if getattr(args, field).startswith("cuda"):
                print(f"Warning: CUDA not available. Falling back {field} to CPU.")
                setattr(args, field, "cpu")


def _torch_dtype_from_name(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype name: {name}")


def _build_quantized_lm_config(args: argparse.Namespace):
    try:
        from transformers import BitsAndBytesConfig
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "quantized_lm_lm mode requires transformers BitsAndBytesConfig and bitsandbytes. "
            "Install bitsandbytes in the active environment before running this mode."
        ) from exc

    if args.quantized_lm_bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.quantized_lm_4bit_quant_type,
            bnb_4bit_compute_dtype=_torch_dtype_from_name(args.quantized_lm_4bit_compute_dtype),
            bnb_4bit_use_double_quant=bool(args.quantized_lm_4bit_use_double_quant),
        )
    if args.quantized_lm_bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)
    raise ValueError(f"Unsupported quantized LM bit width: {args.quantized_lm_bits}")


def _context_stats(documents: Sequence[str], tokenizer) -> Dict[str, int]:
    text = "\n\n".join(documents)
    tokens = tokenizer(text, add_special_tokens=False, truncation=False).input_ids
    return {
        "doc_count": len(documents),
        "context_tokens": len(tokens),
        "context_words": len(text.split()),
    }


def _prompt_token_count(prompt: str, tokenizer) -> int:
    return len(tokenizer(prompt, add_special_tokens=False, truncation=False).input_ids)


def _warn_if_prompt_truncated(label: str, prompt: str, tokenizer, max_length: int) -> None:
    token_count = _prompt_token_count(prompt, tokenizer)
    if token_count > max_length:
        print(f"[Warn] {label} prompt tokens={token_count} exceeds max_length={max_length}; inputs will be truncated.")


def _truncate_docs_into_prompt(
    documents: Sequence[str],
    *,
    prefix: str,
    suffix: str,
    tokenizer,
    max_length: int,
) -> str:
    full_prompt = prefix + "\n\n".join(documents) + suffix
    if _prompt_token_count(full_prompt, tokenizer) <= max_length:
        return full_prompt

    fixed_len = _prompt_token_count(prefix + suffix, tokenizer)
    if fixed_len >= max_length:
        fixed_ids = tokenizer(prefix + suffix, add_special_tokens=False, truncation=False).input_ids
        return tokenizer.decode(fixed_ids[-max_length:], skip_special_tokens=True)

    context_budget = max_length - fixed_len
    context_parts: List[str] = []
    used = 0
    for chunk in documents:
        chunk_text = chunk if not context_parts else "\n\n" + chunk
        chunk_ids = tokenizer(chunk_text, add_special_tokens=False, truncation=False).input_ids
        if used + len(chunk_ids) <= context_budget:
            context_parts.append(chunk_text if not context_parts else chunk_text[2:])
            used += len(chunk_ids)
            continue
        remain = context_budget - used
        if remain > 0:
            clipped = tokenizer.decode(chunk_ids[:remain], skip_special_tokens=True)
            context_parts.append(clipped if not context_parts else clipped.lstrip())
        break
    return prefix + "\n\n".join(context_parts) + suffix


def _build_query_preserving_prompt_exact(
    example,
    include_docs: bool,
    tokenizer,
    max_length: int,
    *,
    warn_label: Optional[str] = None,
    warn_on_truncation: bool = False,
) -> str:
    """
    Match run_long_context_experiments.py teacher prompt construction exactly.
    """
    revert_previous_prompt = bool(getattr(tokenizer, "_codex_revert_previous_prompt", False))
    answer_variant = _get_answer_prompt_variant()
    if not include_docs:
        return build_full_context_qa_prompt(
            example.question,
            [],
            revert_previous_prompt=revert_previous_prompt,
            prompt_variant=answer_variant,
        )

    full_prompt = build_full_context_qa_prompt(
        example.question,
        example.documents,
        revert_previous_prompt=revert_previous_prompt,
        prompt_variant=answer_variant,
    )
    full_len = _prompt_token_count(full_prompt, tokenizer)
    prompt = build_query_preserving_full_context_qa_prompt(
        question=example.question,
        documents=example.documents,
        tokenizer=tokenizer,
        max_length=max_length,
        revert_previous_prompt=revert_previous_prompt,
        prompt_variant=answer_variant,
    )
    if warn_on_truncation:
        label = warn_label or "Prompt"
        trunc_len = _prompt_token_count(prompt, tokenizer)
        if trunc_len != full_len:
            print(
                f"[WARN] {label} prompt truncated (query-preserving): full_len={full_len} "
                f"max_length={max_length} trunc_len={trunc_len}"
            )
    return prompt


def _qa_answer_prompt(
    example,
    *,
    include_docs: bool,
    evidence_sketch: Optional[str] = None,
    revert_previous_prompt: bool = False,
) -> str:
    answer_variant = _get_answer_prompt_variant()
    if include_docs:
        return build_full_context_qa_prompt(
            example.question,
            example.documents,
            revert_previous_prompt=revert_previous_prompt,
            prompt_variant=answer_variant,
        )
    return build_sketch_conditioned_qa_prompt(
        example.question,
        evidence_sketch,
        revert_previous_prompt=revert_previous_prompt,
        prompt_variant=answer_variant,
    )


def _qa_sketch_prompt(example) -> str:
    doc_blocks = "\n\n".join(f"### Document {idx}\n{doc}" for idx, doc in enumerate(example.documents, start=1))
    return f"Context:\n{doc_blocks}\n\nQuestion: {example.question}\n\n{_load_evidence_sketch_prompt_text()}\n"


def _load_evidence_sketch_prompt_text() -> str:
    return _EVIDENCE_SKETCH_PROMPT_PATH.read_text(encoding="utf-8").strip()


def _build_qa_answer_prompt_preserving_query(example, *, include_docs: bool, tokenizer, max_length: int, evidence_sketch: Optional[str] = None) -> str:
    if not include_docs:
        return _qa_answer_prompt(
            example,
            include_docs=False,
            evidence_sketch=evidence_sketch,
            revert_previous_prompt=bool(getattr(tokenizer, "_codex_revert_previous_prompt", False)),
        )
    return _build_query_preserving_prompt_exact(
        example,
        include_docs=True,
        tokenizer=tokenizer,
        max_length=max_length,
    )


def _build_qa_sketch_prompt_preserving_query(example, *, tokenizer, max_length: int) -> str:
    prefix = f"{_QA_SHARED_LEAD}Context:\n"
    suffix = f"\n\nQuestion: {example.question}\n\n{_get_sketch_suffix()}"
    docs = [f"### Document {idx}\n{doc}" for idx, doc in enumerate(example.documents, start=1)]
    return _truncate_docs_into_prompt(docs, prefix=prefix, suffix=suffix, tokenizer=tokenizer, max_length=max_length)


def _build_qa_slm_shared_prefix_preserving_query(example, *, tokenizer, max_length: int, reserve_suffix: str) -> str:
    del reserve_suffix
    return _build_query_preserving_prompt_exact(
        example,
        include_docs=True,
        tokenizer=tokenizer,
        max_length=max_length,
    )


def _build_qa_slm_prompt_bundle(example, *, tokenizer, max_length: int, sketch_max_length: int) -> Dict[str, str]:
    shared_prompt = _build_qa_slm_shared_prefix_preserving_query(
        example,
        tokenizer=tokenizer,
        max_length=min(max_length, sketch_max_length),
        reserve_suffix="",
    )
    return {
        "shared_prompt": shared_prompt,
        "answer_prompt": shared_prompt,
        "sketch_prompt": shared_prompt + _get_sketch_suffix(),
    }


def _count_prompt_doc_headers(prompt: str) -> int:
    return len(re.findall(r"(?m)^### Document \d+\s*$", prompt))


def _prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _build_teacher_prompt_with_audit(
    example,
    *,
    dataset: str,
    tokenizer,
    max_length: int,
    asqa_prompt_version: str = ASQA_PROMPT_VERSION_ANSWER_FIRST,
) -> tuple[str, Optional[Dict[str, Any]]]:
    if dataset in ASQA_STYLE_DATASETS:
        prompt = build_answer_prompt(
            example,
            dataset,
            include_docs=True,
            tokenizer=tokenizer,
            asqa_prompt_version=asqa_prompt_version,
        )
        audit = {
            "dataset": dataset,
            "asqa_prompt_version": asqa_prompt_version,
            "full_prompt_tokens": _prompt_token_count(prompt, tokenizer),
            "final_prompt_tokens": _prompt_token_count(prompt, tokenizer),
            "max_length": max_length,
            "truncated": False,
            "doc_count_total": len(example.documents),
            "doc_headers_in_prompt": _count_prompt_doc_headers(prompt),
            "question_present": f"Question:\n{example.question}" in prompt or f"Question: {example.question}" in prompt,
            "prompt_sha256": _prompt_sha256(prompt),
            "prompt_head": prompt[:500],
            "prompt_tail": prompt[-500:],
        }
        return prompt, audit

    full_prompt = _qa_answer_prompt(
        example,
        include_docs=True,
        revert_previous_prompt=bool(getattr(tokenizer, "_codex_revert_previous_prompt", False)),
    )
    prompt = _build_query_preserving_prompt_exact(
        example,
        include_docs=True,
        tokenizer=tokenizer,
        max_length=max_length,
        warn_label="Teacher",
        warn_on_truncation=True,
    )
    doc_headers = re.findall(r"(?m)^### Document (\d+)\s*$", prompt)
    audit = {
        "dataset": dataset,
        "full_prompt_tokens": _prompt_token_count(full_prompt, tokenizer),
        "final_prompt_tokens": _prompt_token_count(prompt, tokenizer),
        "max_length": max_length,
        "truncated": _prompt_token_count(full_prompt, tokenizer) > max_length,
        "doc_count_total": len(example.documents),
        "doc_headers_in_prompt": len(doc_headers),
        "last_doc_index_in_prompt": int(doc_headers[-1]) if doc_headers else None,
        "question_present": f"Question: {example.question}" in prompt,
        "output_format_present": QA_OUTPUT_FORMAT in prompt,
        "instruction_present": (
            prompt.startswith(_QA_SHARED_LEAD)
            or prompt.startswith(QA_FULL_CONTEXT_INSTRUCTION)
            or prompt.startswith(QA_FULL_CONTEXT_INSTRUCTION_LEGACY)
        ),
        "prompt_sha256": _prompt_sha256(prompt),
        "prompt_head": prompt[:500],
        "prompt_tail": prompt[-500:],
    }
    return prompt, audit


class ALCEExample:
    def __init__(
        self,
        example_id: str,
        question: str,
        answer: str,
        documents: List[str],
        doc_meta: List[Dict[str, Any]],
        *,
        qa_pairs: Optional[List[Dict[str, Any]]] = None,
        annotations: Optional[List[Dict[str, Any]]] = None,
        claims: Optional[List[str]] = None,
    ) -> None:
        self.example_id = example_id
        self.question = question
        self.answer = answer
        self.documents = documents
        self.doc_meta = doc_meta
        self.qa_pairs = qa_pairs
        self.annotations = annotations
        self.claims = claims


class SketchConditionedExample:
    def __init__(
        self,
        base_example: Any,
        dataset: str,
        evidence_sketch: str,
        *,
        asqa_prompt_version: str = ASQA_PROMPT_VERSION_ANSWER_FIRST,
    ) -> None:
        self.base = base_example
        self.dataset = dataset
        self.evidence_sketch = evidence_sketch
        self.asqa_prompt_version = asqa_prompt_version
        self.example_id = base_example.example_id
        self.question = base_example.question
        self.answer = base_example.answer
        self.documents = base_example.documents
        self.doc_meta = getattr(base_example, "doc_meta", None)
        self.qa_pairs = getattr(base_example, "qa_pairs", None)
        self.annotations = getattr(base_example, "annotations", None)
        self.claims = getattr(base_example, "claims", None)

    def prompt(self, include_docs: bool = True, tokenizer=None) -> str:
        return build_answer_prompt(
            self.base,
            self.dataset,
            include_docs=include_docs,
            tokenizer=tokenizer,
            evidence_sketch=None if include_docs else self.evidence_sketch,
            asqa_prompt_version=self.asqa_prompt_version,
        )


def build_answer_prompt(
    example,
    dataset: str,
    *,
    include_docs: bool,
    tokenizer=None,
    evidence_sketch: Optional[str] = None,
    asqa_prompt_version: str = ASQA_PROMPT_VERSION_ANSWER_FIRST,
) -> str:
    if dataset in ASQA_STYLE_DATASETS:
        _av = _get_answer_prompt_variant()
        # ★ LongBench summarization (gov_report/multi_news) with ANSWER_PROMPT_VARIANT=summarize uses the
        # benchmark-matched summarization prompt (comprehensive one-page summary, no citations/reasoning)
        # instead of the ASQA citation/QA prompt — fixes the too-short-14B / LM<SLM-reversal prompt bug.
        if _av.startswith("summarize") and dataset in LONGBENCH_SUMM_DATASETS:
            return build_longbench_summary_prompt(
                question=example.question,
                documents=example.documents,
                include_docs=include_docs,
                tokenizer=tokenizer,
                evidence_sketch=evidence_sketch,
                task=(getattr(example, "metadata", None) or {}).get("task"),
            )
        asqa_reason = _av in ("reason_then_answer", "reason_then_answer_asqa", "reason_then_answer_asqa_long", "reason_then_answer_asqa_cite")
        asqa_long = _av == "reason_then_answer_asqa_long"
        asqa_cite = _av == "reason_then_answer_asqa_cite"
        return build_asqa_citation_answer_prompt(
            question=example.question,
            documents=example.documents,
            include_docs=include_docs,
            tokenizer=tokenizer,
            evidence_sketch=evidence_sketch,
            prompt_version=asqa_prompt_version,
            reason=asqa_reason,
            long_answer=asqa_long,
            cite_reason=asqa_cite,
        )
    revert_previous_prompt = bool(getattr(tokenizer, "_codex_revert_previous_prompt", False)) if tokenizer is not None else False
    return _qa_answer_prompt(
        example,
        include_docs=include_docs,
        evidence_sketch=evidence_sketch,
        revert_previous_prompt=revert_previous_prompt,
    )


def build_sketch_prompt(
    example,
    dataset: str,
    *,
    tokenizer=None,
    asqa_prompt_version: str = ASQA_PROMPT_VERSION_ANSWER_FIRST,
) -> str:
    del asqa_prompt_version
    if dataset in ASQA_STYLE_DATASETS:
        return build_asqa_citation_sketch_prompt(
            question=example.question,
            documents=example.documents,
            evidence_sketch_instruction=ASQA_EVIDENCE_SKETCH_INSTRUCTION,
            tokenizer=tokenizer,
        )
    return _qa_sketch_prompt(example)


def _strip_generation_artifacts(text: str) -> str:
    cleaned = text.replace("<|im_end|>", "").strip()
    cleaned = cleaned.replace("</s>", "").strip()
    return cleaned


def _trim_sketch_to_complete_lines(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return stripped
    last_newline = stripped.rfind("\n")
    if last_newline == -1:
        return stripped
    trimmed = stripped[:last_newline].rstrip()
    return trimmed or stripped


def _generate_single(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    max_length: int,
    stop_strings: Optional[List[str]] = None,
    use_generate: bool = True,
    memory_trace: Optional[Dict[str, Any]] = None,
    decode_temperature: float = 0.0,
    decode_top_p: float = 1.0,
    decode_seed: int = 0,
    stop_on_repeat: Optional[str] = None,
) -> Tuple[str, Optional[GenerationTiming]]:
    _warn_if_prompt_truncated("Generation", prompt, tokenizer, max_length)
    return generate_with_single_model(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        min_new_tokens=int(os.environ.get("MIN_NEW_TOKENS", "0")),
        max_length=max_length,
        return_timing=True,
        stop_strings=stop_strings,
        stop_on_repeat=stop_on_repeat,
        use_generate=use_generate,
        memory_trace=memory_trace,
        decode_temperature=decode_temperature,
        decode_top_p=decode_top_p,
        decode_seed=decode_seed,
    )


def _run_blockattention_baseline(
    model,
    tokenizer,
    example,
    *,
    dataset: str,
    max_new_tokens: int,
    max_length: int,
    debug: bool,
    use_generate: bool,
    mode_label: str,
    use_pcw_positions: bool,
    memory_trace: Optional[Dict[str, Any]] = None,
    decode_temperature: float = 0.0,
    decode_top_p: float = 1.0,
    decode_seed: int = 0,
    stop_on_repeat: Optional[str] = None,
) -> Dict[str, Any]:
    if dataset == "asqa":
        raise ValueError("blockattention/pcw baselines are currently implemented for QA-style datasets, not ASQA.")
    base_example = example.base if hasattr(example, "base") else example
    prompt, segment_ids, prompt_token_ids = build_query_preserving_full_context_qa_prompt_with_token_segments(
        question=base_example.question,
        documents=base_example.documents,
        tokenizer=tokenizer,
        max_length=max_length,
        revert_previous_prompt=bool(getattr(tokenizer, "_codex_revert_previous_prompt", False)),
    )
    prompt_token_count = len(prompt_token_ids)
    if prompt_token_count != len(segment_ids):
        raise RuntimeError(
            f"{mode_label} tokenization mismatch: prompt_tokens={prompt_token_count} segment_ids={len(segment_ids)}"
        )

    mask_dtype = next(model.parameters()).dtype
    block_mask = build_block_attention_mask(segment_ids, device=torch.device("cpu"), dtype=mask_dtype)
    input_ids = torch.tensor([prompt_token_ids], dtype=torch.long)
    position_ids = None
    position_builder = None
    if use_pcw_positions:
        position_ids, next_shared_position = build_pcw_position_ids(segment_ids, device=torch.device("cpu"))
        position_builder = build_pcw_decode_position_builder(prompt_token_count, next_shared_position)
    text, timing = generate_with_single_model(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        max_length=max_length,
        use_generate=use_generate,
        attention_mask_override=block_mask,
        incremental_attention_mask_builder=build_block_attention_decode_mask,
        input_ids_override=input_ids,
        position_ids_override=position_ids,
        incremental_position_ids_builder=position_builder,
        return_timing=True,
        memory_trace=memory_trace,
        decode_temperature=decode_temperature,
        decode_top_p=decode_top_p,
        decode_seed=decode_seed,
        stop_on_repeat=stop_on_repeat,
    )
    extracted = _extract_final_answer_for_dataset(text, dataset)
    if debug:
        _print_debug_answer(mode_label, base_example.question, text, extracted, base_example.answer)
    em, f1 = _score_qa_prediction(base_example, extracted, dataset)
    return {
        "text": text,
        "extracted": extracted,
        "em": em,
        "f1": f1,
        "prompt_tokens": prompt_token_count,
        "visible_documents": max(segment_ids) if segment_ids else 0,
        "pcw_positions": bool(use_pcw_positions),
        "timing": {
            "prefill_s": timing.prefill_s if timing else None,
            "decode_s": timing.decode_s if timing else None,
            "total_s": timing.total_s if timing else None,
        },
    }


def _build_pced_prompt_builder(args: argparse.Namespace, dataset_name: str) -> Callable[[Any, bool, Any], str]:
    def prompt_builder(example, include_docs: bool, tokenizer) -> str:
        base_example = example.base if hasattr(example, "base") else example
        if include_docs:
            if dataset_name in ASQA_STYLE_DATASETS:
                return build_answer_prompt(
                    base_example,
                    dataset_name,
                    include_docs=True,
                    tokenizer=tokenizer,
                    asqa_prompt_version=args.asqa_prompt_version,
                )
            return _build_qa_answer_prompt_preserving_query(
                base_example,
                include_docs=True,
                tokenizer=tokenizer,
                max_length=args.max_length,
            )

        if dataset_name in ASQA_STYLE_DATASETS:
            query_only = copy.copy(base_example)
            query_only.documents = []
            return build_answer_prompt(
                query_only,
                dataset_name,
                include_docs=True,
                tokenizer=tokenizer,
                asqa_prompt_version=args.asqa_prompt_version,
            )
        return build_full_context_qa_prompt(
            base_example.question,
            [],
            revert_previous_prompt=bool(getattr(tokenizer, "_codex_revert_previous_prompt", False)),
            prompt_variant=_get_answer_prompt_variant(),
        )

    return prompt_builder


def _run_pced_baseline(
    decoder: PCEDContrastiveDecoder,
    example,
    *,
    dataset: str,
    debug: bool,
) -> Dict[str, Any]:
    base_example = example.base if hasattr(example, "base") else example
    decoded = decoder.decode(base_example, retrieval_scores=getattr(base_example, "retrieval_scores", None))
    text = str(decoded.get("text", ""))
    if dataset in ASQA_STYLE_DATASETS:
        text = _strip_generation_artifacts(text)
        extracted, score_map = _score_asqa_prediction(text, base_example)
        result = {
            "text": text,
            "extracted": extracted,
            "metrics": score_map,
            "timing": decoded.get("timing"),
        }
    else:
        extracted = _extract_final_answer_for_dataset(text, dataset)
        em, f1 = _score_qa_prediction(base_example, extracted, dataset)
        result = {
            "text": text,
            "extracted": extracted,
            "em": em,
            "f1": f1,
            "timing": decoded.get("timing"),
        }
    if debug:
        _print_debug_answer("PCED", base_example.question, text, extracted, base_example.answer)
    return result


def _common_prefix_len(a: Sequence[int], b: Sequence[int]) -> int:
    n = min(len(a), len(b))
    idx = 0
    while idx < n and a[idx] == b[idx]:
        idx += 1
    return idx


def _sync_models(*models) -> None:
    for model in models:
        if model is not None:
            synchronize_model(model)


def _clone_past_key_values(past_key_values):
    if past_key_values is None:
        return None
    try:
        return copy.deepcopy(past_key_values)
    except Exception:  # noqa: BLE001
        pass
    if isinstance(past_key_values, tuple):
        cloned_layers = []
        for layer in past_key_values:
            if isinstance(layer, tuple):
                cloned_layers.append(tuple(t.clone() if torch.is_tensor(t) else copy.deepcopy(t) for t in layer))
            else:
                cloned_layers.append(copy.deepcopy(layer))
        return tuple(cloned_layers)
    return copy.deepcopy(past_key_values)


def _advance_model_with_suffix(
    model,
    suffix_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    past_key_values,
    needs_attention: bool,
):
    if suffix_ids.shape[1] == 0:
        raise ValueError("suffix_ids must contain at least one token when advancing model state.")
    return model(
        input_ids=suffix_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=True,
        output_attentions=needs_attention,
    )


def _generate_from_existing_state(
    model,
    tokenizer,
    *,
    initial_outputs,
    initial_attention_mask: torch.Tensor,
    max_new_tokens: int,
    stop_strings: Optional[List[str]],
    needs_attention: bool = False,
) -> Tuple[str, GenerationTiming, bool]:
    eos_ids = get_eos_token_ids(tokenizer, model)
    attention_mask = initial_attention_mask
    outputs = initial_outputs
    past = outputs.past_key_values
    generated_ids: List[int] = []
    text_parts: List[str] = []

    _sync_models(model)
    decode_start = time.perf_counter()
    for _ in range(max_new_tokens):
        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1)
        token_id = int(next_token[0].item())
        generated_ids.append(token_id)
        text_parts.append(tokenizer.decode([token_id], skip_special_tokens=False))
        current_text = "".join(text_parts)
        if stop_strings:
            for stop in stop_strings:
                if stop and stop in current_text:
                    cutoff = current_text.find(stop)
                    final_text = current_text[:cutoff]
                    _sync_models(model)
                    decode_time = time.perf_counter() - decode_start
                    return final_text, GenerationTiming(prefill_s=0.0, decode_s=decode_time, total_s=decode_time), False
        if token_id in eos_ids:
            break

        next_token_tensor = next_token.unsqueeze(0)
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones((1, 1), device=attention_mask.device, dtype=attention_mask.dtype),
            ],
            dim=1,
        )
        with torch.no_grad():
            outputs = model(
                input_ids=next_token_tensor.to(attention_mask.device),
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
                output_attentions=needs_attention,
            )
        past = outputs.past_key_values

    _sync_models(model)
    decode_time = time.perf_counter() - decode_start
    text = "".join(text_parts) if text_parts else tokenizer.decode(generated_ids, skip_special_tokens=True)
    hit_max_new_tokens = len(generated_ids) >= max_new_tokens and (not generated_ids or generated_ids[-1] not in eos_ids)
    return text, GenerationTiming(prefill_s=0.0, decode_s=decode_time, total_s=decode_time), hit_max_new_tokens


def _apply_stop_strings(text: str, stop_strings: Sequence[str]) -> Tuple[str, bool]:
    for stop in stop_strings:
        if stop and stop in text:
            return text[: text.find(stop)], True
    return text, False


# Chat-role / special-token escapes. Once the model writes one of these it has left the answer and is
# hallucinating a new turn — never valid output, for any benchmark, at any answer length.
_DEGENERATE_LEAKS = ("Human:", "Assistant:", "<|im_start|>", "<|im_end|>", "\nuser\n", "\nassistant\n")


def _apply_degenerate_stops(text: str, decoder: Any) -> Tuple[str, bool]:
    """Stops that only ever remove text the scorer discards: a REPEATED `--stop-on-repeat` marker (the
    'Final Answer: X Final Answer: X ...' greedy loop) and a chat-role leak. Mirrors src/fusion.py's decode()."""
    marker = getattr(decoder, "stop_on_repeat", None)
    if marker and text.count(marker) >= 2:
        return text[: text.find(marker, text.find(marker) + 1)], True
    for leak in _DEGENERATE_LEAKS:
        if leak in text:
            return text[: text.find(leak)], True
    return text, False


def _run_evidence_sketch_fusion(
    decoder: FixedLambdaFusionDecoder,
    example: Any,
    *,
    dataset: str,
    max_length: int,
    sketch_max_length: int,
    sketch_max_new_tokens: int,
    sketch_stop_strings: Optional[List[str]],
    asqa_prompt_version: str = ASQA_PROMPT_VERSION_ANSWER_FIRST,
    training_trace: Optional[Dict[str, Any]] = None,
    export_gate_feature_type: str = "logit_topk",
    export_gate_top_k: int = 10,
    export_loss_top_k: int = 20,
    measure_memory: bool = False,
    memory_baseline: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if decoder.fusion_mode not in {"weighted_sum", "prob_mix", "entropy"}:
        raise ValueError(f"Unsupported fusion mode for evidence sketch runner: {decoder.fusion_mode}")
    need_export_attention = export_gate_feature_type in {
        "attn_ratio_and_logit_topk",
        "attn_linear_weighted",
        "attn_stats_mlp_unweighted",
        "attn_ratio_mlp_weighted",
    }
    need_attention = bool(getattr(decoder, "_needs_attention", False) or (training_trace is not None and need_export_attention))

    if dataset in ASQA_STYLE_DATASETS:
        slm_answer_prompt = build_answer_prompt(
            example,
            dataset,
            include_docs=True,
            tokenizer=decoder.slm_tokenizer,
            asqa_prompt_version=asqa_prompt_version,
        )
        slm_sketch_prompt = build_sketch_prompt(
            example,
            dataset,
            tokenizer=decoder.slm_tokenizer,
            asqa_prompt_version=asqa_prompt_version,
        )
        answer_inputs = prepare_inputs(decoder.slm_tokenizer, slm_answer_prompt, max_length=max_length, return_tensors="pt")
        sketch_inputs = prepare_inputs(
            decoder.slm_tokenizer,
            slm_sketch_prompt,
            max_length=sketch_max_length,
            return_tensors="pt",
        )
        answer_ids_list = answer_inputs["input_ids"][0].tolist()
        sketch_ids_list = sketch_inputs["input_ids"][0].tolist()
        shared_len = _common_prefix_len(answer_ids_list, sketch_ids_list)
        if shared_len <= 0:
            raise RuntimeError("Failed to find a shared SLM prefix between answer and sketch prompts.")

        shared_ids_full = answer_inputs["input_ids"][:, :shared_len]
        shared_mask_full = answer_inputs["attention_mask"][:, :shared_len]
        answer_suffix_full = answer_inputs["input_ids"][:, shared_len:]
        sketch_suffix_full = sketch_inputs["input_ids"][:, shared_len:]
        answer_mask_full = answer_inputs["attention_mask"]
        sketch_mask_full = sketch_inputs["attention_mask"]
    else:
        bundle = _build_qa_slm_prompt_bundle(
            example,
            tokenizer=decoder.slm_tokenizer,
            max_length=max_length,
            sketch_max_length=sketch_max_length,
        )
        shared_prompt = bundle["shared_prompt"]
        slm_answer_prompt = bundle["answer_prompt"]
        slm_sketch_prompt = bundle["sketch_prompt"]
        canonical_teacher_prompt = build_query_preserving_full_context_qa_prompt(
            question=example.question,
            documents=example.documents,
            tokenizer=decoder.slm_tokenizer,
            max_length=max_length,
            revert_previous_prompt=bool(getattr(decoder.slm_tokenizer, "_codex_revert_previous_prompt", False)),
            prompt_variant=_get_answer_prompt_variant(),
        )
        if slm_answer_prompt != canonical_teacher_prompt:
            raise RuntimeError(
                "Invariant violation: QA SLM answer prompt must exactly match the canonical teacher/full-context prompt."
            )
        shared_inputs = prepare_inputs(
            decoder.slm_tokenizer,
            shared_prompt,
            max_length=min(max_length, sketch_max_length),
            return_tensors="pt",
        )
        answer_inputs = prepare_inputs(decoder.slm_tokenizer, slm_answer_prompt, max_length=max_length, return_tensors="pt")
        sketch_inputs = prepare_inputs(
            decoder.slm_tokenizer,
            slm_sketch_prompt,
            max_length=sketch_max_length,
            return_tensors="pt",
        )
        shared_len = int(shared_inputs["input_ids"].shape[1])
        shared_ids_full = shared_inputs["input_ids"]
        shared_mask_full = shared_inputs["attention_mask"]
        answer_suffix_full = answer_inputs["input_ids"][:, shared_len:]
        sketch_suffix_full = sketch_inputs["input_ids"][:, shared_len:]
        answer_mask_full = torch.cat(
            [
                shared_inputs["attention_mask"],
                torch.ones_like(answer_suffix_full, dtype=shared_inputs["attention_mask"].dtype),
            ],
            dim=1,
        )
        sketch_mask_full = torch.cat(
            [
                shared_inputs["attention_mask"],
                torch.ones_like(sketch_suffix_full, dtype=shared_inputs["attention_mask"].dtype),
            ],
            dim=1,
        )

    slm_device = get_model_input_device(decoder.slm_model)
    lm_device = get_model_input_device(decoder.lm_model)
    phase_memory: Dict[str, Any] = {}

    shared_ids = shared_ids_full.to(slm_device)
    shared_mask = shared_mask_full.to(slm_device)
    answer_suffix_ids = answer_suffix_full.to(slm_device)
    sketch_suffix_ids = sketch_suffix_full.to(slm_device)
    answer_mask = answer_mask_full.to(slm_device)
    sketch_mask = sketch_mask_full.to(slm_device)

    # Optional KV-cache quantization for ONLY the SLM/context branch (default off).
    # The big long-context KV (shared prefill -> shared_past) is built into a HF
    # QuantizedCache, giving the context-reader a KV-memory advantage; the LM/query
    # branch is never quantized. None => byte-identical to the original code path.
    slm_kv_cache = decoder.build_slm_kv_quant_cache() if hasattr(decoder, "build_slm_kv_quant_cache") else None
    if slm_kv_cache is not None and sketch_max_new_tokens > 0:
        raise ValueError(
            "SLM KV-cache quantization (slm_kv_quant_bits>0) is only supported with the "
            "no-evidence-sketch method (sketch_max_new_tokens=0). Re-run with SKETCH_MAX_NEW=0."
        )

    _sync_models(decoder.slm_model)
    if measure_memory:
        reset_cuda_peak_memory()
    shared_prefill_start = time.perf_counter()
    with torch.no_grad():
        if need_attention and int(shared_ids.shape[1]) > 1:
            shared_prefix_ids = shared_ids[:, :-1]
            shared_prefix_mask = shared_mask[:, :-1]
            shared_last_ids = shared_ids[:, -1:]
            shared_prefix_kwargs = dict(
                input_ids=shared_prefix_ids,
                attention_mask=shared_prefix_mask,
                use_cache=True,
                output_attentions=False,
                logits_to_keep=1,
            )
            if slm_kv_cache is not None:
                shared_prefix_kwargs["past_key_values"] = slm_kv_cache
            shared_prefix_outputs = decoder.slm_model(**shared_prefix_kwargs)
            shared_outputs = decoder.slm_model(
                input_ids=shared_last_ids,
                attention_mask=shared_mask,
                past_key_values=shared_prefix_outputs.past_key_values,
                use_cache=True,
                output_attentions=True,
                logits_to_keep=1,
            )
        else:
            shared_plain_kwargs = dict(
                input_ids=shared_ids,
                attention_mask=shared_mask,
                use_cache=True,
                output_attentions=need_attention,
                logits_to_keep=1,
            )
            if slm_kv_cache is not None:
                shared_plain_kwargs["past_key_values"] = slm_kv_cache
            shared_outputs = decoder.slm_model(**shared_plain_kwargs)
        if slm_kv_cache is not None:
            print(
                f"[SLM-KV-quant] SLM/context branch cache = {type(shared_outputs.past_key_values).__name__} "
                f"(bits={decoder.slm_kv_quant_bits}, backend={decoder.slm_kv_backend}, "
                f"group_size={decoder.slm_kv_group_size}, residual_length={decoder.slm_kv_residual_length})",
                flush=True,
            )
    _sync_models(decoder.slm_model)
    shared_prefill_time = time.perf_counter() - shared_prefill_start
    if measure_memory:
        phase_memory["slm_shared_prefill"] = _attach_memory_extra(snapshot_cuda_memory(), memory_baseline)
    shared_past = shared_outputs.past_key_values

    if measure_memory:
        reset_cuda_peak_memory()
    with torch.no_grad():
        if slm_kv_cache is not None:
            # KV-quant path runs no-ES only (guarded above): the sketch is generated for
            # 0 tokens and discarded, so skip the sketch-suffix prefill rather than clone
            # the QuantizedCache (which has no safe deep-copy). shared_past is left intact.
            sketch_suffix_outputs = shared_outputs
        elif sketch_suffix_ids.shape[1] > 0:
            sketch_suffix_outputs = _advance_model_with_suffix(
                decoder.slm_model,
                sketch_suffix_ids,
                sketch_mask,
                past_key_values=_clone_past_key_values(shared_past),
                needs_attention=False,
            )
        else:
            sketch_suffix_outputs = shared_outputs
    if measure_memory:
        phase_memory["sketch_suffix_prefill"] = _attach_memory_extra(snapshot_cuda_memory(), memory_baseline)
        reset_cuda_peak_memory()
    sketch_text, sketch_decode_timing, sketch_hit_limit = _generate_from_existing_state(
        decoder.slm_model,
        decoder.slm_tokenizer,
        initial_outputs=sketch_suffix_outputs,
        initial_attention_mask=sketch_mask,
        max_new_tokens=sketch_max_new_tokens,
        stop_strings=sketch_stop_strings,
        needs_attention=False,
    )
    if measure_memory:
        phase_memory["sketch_decode"] = _attach_memory_extra(snapshot_cuda_memory(), memory_baseline)
    sketch_timing = GenerationTiming(
        prefill_s=shared_prefill_time,
        decode_s=sketch_decode_timing.decode_s,
        total_s=shared_prefill_time + sketch_decode_timing.decode_s,
    )
    sketch_text = _strip_generation_artifacts(sketch_text)
    if sketch_hit_limit:
        sketch_text = _trim_sketch_to_complete_lines(sketch_text)

    conditioned = SketchConditionedExample(example, dataset, sketch_text)
    lm_prompt = conditioned.prompt(include_docs=False, tokenizer=decoder.lm_tokenizer)
    lm_inputs = prepare_inputs(decoder.lm_tokenizer, lm_prompt, max_length=max_length, return_tensors="pt")
    lm_ids = lm_inputs["input_ids"].to(lm_device)
    lm_mask = lm_inputs["attention_mask"].to(lm_device)

    _sync_models(decoder.slm_model, decoder.lm_model)
    if measure_memory:
        reset_cuda_peak_memory()
    fusion_prefill_start = time.perf_counter()
    with torch.no_grad():
        if answer_suffix_ids.shape[1] > 0:
            # With KV-quant on, the sketch branch was skipped, so shared_past has no other
            # consumer and can be advanced in place (avoids cloning the QuantizedCache).
            answer_branch_past = shared_past if slm_kv_cache is not None else _clone_past_key_values(shared_past)
            slm_outputs = _advance_model_with_suffix(
                decoder.slm_model,
                answer_suffix_ids,
                answer_mask,
                past_key_values=answer_branch_past,
                needs_attention=need_attention,
            )
            slm_past = slm_outputs.past_key_values
        else:
            slm_outputs = shared_outputs
            slm_past = shared_past
        lm_outputs = decoder.lm_model(
            input_ids=lm_ids,
            attention_mask=lm_mask,
            use_cache=True,
        )
        lm_past = lm_outputs.past_key_values
    _sync_models(decoder.slm_model, decoder.lm_model)
    fusion_prefill_time = time.perf_counter() - fusion_prefill_start
    if measure_memory:
        phase_memory["fusion_prefill"] = _attach_memory_extra(snapshot_cuda_memory(), memory_baseline)

    slm_ids = answer_inputs["input_ids"].to(slm_device)
    slm_mask = answer_mask
    context_span = infer_context_token_span(slm_answer_prompt, decoder.slm_tokenizer, int(slm_ids.shape[1]))
    generated_ids: List[int] = []
    text_parts: List[str] = []
    eos_ids = get_eos_token_ids(decoder.lm_tokenizer, decoder.lm_model)
    terminated = False
    exported_samples: List[KLStepSample] = []
    forced_token_ids: List[int] = [int(x) for x in training_trace.get("token_ids", [])] if training_trace else []
    forced_steps: List[Dict[str, Any]] = list(training_trace.get("steps", [])) if training_trace else []

    can_parallel = (
        torch.cuda.is_available()
        and slm_device.type == "cuda"
        and lm_device.type == "cuda"
        and slm_device != lm_device
    )
    slm_stream = torch.cuda.Stream(device=slm_device) if can_parallel else None
    lm_stream = torch.cuda.Stream(device=lm_device) if can_parallel else None

    if measure_memory:
        reset_cuda_peak_memory()
    decode_start = time.perf_counter()
    num_decode_steps = len(forced_token_ids) if training_trace else decoder.max_new_tokens
    for step_idx in range(num_decode_steps):
        slm_logits = slm_outputs.logits[:, -1, :]
        lm_logits = lm_outputs.logits[:, -1, :]
        vocab_size = min(slm_logits.shape[-1], lm_logits.shape[-1])
        slm_logits = slm_logits[..., :vocab_size]
        lm_logits = lm_logits[..., :vocab_size]
        if slm_logits.device != lm_logits.device:
            slm_logits = slm_logits.to(lm_logits.device)

        if decoder.learned_gate is not None:
            lambda_w = decoder._learned_lambda(
                slm_logits,
                lm_logits,
                slm_attentions=slm_outputs.attentions if need_attention else None,
                context_span=context_span,
                kv_len=int(slm_ids.shape[1]),
            )
            fused_logits = lambda_w * slm_logits + (1 - lambda_w) * lm_logits
        else:
            fused_logits = decoder.lambda_weight * slm_logits + (1 - decoder.lambda_weight) * lm_logits

        if training_trace:
            teacher_topk = forced_steps[step_idx].get("top_k", [])
            sample = _build_export_sample_from_step(
                slm_logits=slm_logits[0],
                lm_logits=lm_logits[0],
                teacher_topk=teacher_topk,
                gate_feature_type=export_gate_feature_type,
                gate_top_k=export_gate_top_k,
                loss_top_k=export_loss_top_k,
                slm_attentions=slm_outputs.attentions if need_attention else None,
                context_span=context_span,
                kv_len=int(slm_ids.shape[1]),
            )
            if sample is not None:
                exported_samples.append(sample)
            next_token = torch.tensor([forced_token_ids[step_idx]], device=fused_logits.device, dtype=torch.long)
        else:
            next_token = torch.argmax(fused_logits, dim=-1)
        token_id = int(next_token[0].item())
        generated_ids.append(token_id)
        text_parts.append(decoder.lm_tokenizer.decode([token_id], skip_special_tokens=False))
        # full-decode the accumulated ids (NOT per-token join): SentencePiece/Llama lose leading spaces under
        # per-token decode+join ("FinalAnswer:"), which hides stop-strings + the "Final Answer:" marker from
        # extraction. Byte-BPE (Qwen) is unaffected → no-op there.
        current_text = decoder.lm_tokenizer.decode(generated_ids, skip_special_tokens=True)
        current_text, stopped = _apply_stop_strings(current_text, getattr(decoder, "stop_strings", []) or [])
        if stopped:
            terminated = True
            text_parts = [current_text]
            break
        # ★ 2026-07-29: this loop INLINES src/fusion.py's decode() for the shared-prefix ours path, but it had
        # dropped that loop's degenerate-output stops, so `--stop-on-repeat` was a NO-OP here and nothing capped a
        # runaway generation. Consequence: every fusion-FT run decoded to the full max_new (0/500 terminated) and
        # the ramble was scored — the "held-out FT is a wash" conclusion was an artifact of this. These two stops
        # are extraction-NEUTRAL (the canonical extractor already cuts at a repeated marker / chat-role leak);
        # they only stop generating text that is discarded anyway. The `answer_token_budget` cut is deliberately
        # NOT applied here: it would truncate the legitimately long answers of QASPER/ASQA/mtRAG.
        current_text, stopped_degenerate = _apply_degenerate_stops(current_text, decoder)
        if stopped_degenerate:
            terminated = True
            text_parts = [current_text]
            break

        next_token_tensor = next_token.unsqueeze(0)
        slm_ids = torch.cat([slm_ids, next_token_tensor.to(slm_ids.device)], dim=1)
        lm_ids = torch.cat([lm_ids, next_token_tensor.to(lm_ids.device)], dim=1)
        slm_mask = torch.cat(
            [slm_mask, torch.ones_like(next_token_tensor, device=slm_mask.device, dtype=slm_mask.dtype)],
            dim=1,
        )
        lm_mask = torch.cat(
            [lm_mask, torch.ones_like(next_token_tensor, device=lm_mask.device, dtype=lm_mask.dtype)],
            dim=1,
        )

        if token_id in eos_ids:
            terminated = True
            break

        if can_parallel:
            with torch.no_grad():
                with torch.cuda.stream(slm_stream):
                    slm_outputs = decoder.slm_model(
                        input_ids=next_token_tensor.to(slm_device),
                        attention_mask=slm_mask,
                        past_key_values=slm_past,
                        use_cache=True,
                        output_attentions=need_attention,
                    )
                with torch.cuda.stream(lm_stream):
                    lm_outputs = decoder.lm_model(
                        input_ids=next_token_tensor.to(lm_device),
                        attention_mask=lm_mask,
                        past_key_values=lm_past,
                        use_cache=True,
                    )
            torch.cuda.synchronize(slm_device)
            torch.cuda.synchronize(lm_device)
        else:
            with torch.no_grad():
                slm_outputs = decoder.slm_model(
                    input_ids=next_token_tensor.to(slm_device),
                    attention_mask=slm_mask,
                    past_key_values=slm_past,
                    use_cache=True,
                    output_attentions=need_attention,
                )
                lm_outputs = decoder.lm_model(
                    input_ids=next_token_tensor.to(lm_device),
                    attention_mask=lm_mask,
                    past_key_values=lm_past,
                    use_cache=True,
                )
        slm_past = slm_outputs.past_key_values
        lm_past = lm_outputs.past_key_values

    _sync_models(decoder.slm_model, decoder.lm_model)
    fusion_decode_time = time.perf_counter() - decode_start
    if measure_memory:
        phase_memory["fusion_decode"] = _attach_memory_extra(snapshot_cuda_memory(), memory_baseline)
    # ALWAYS full-decode the accumulated ids (SP-safe spaces for Llama; no-op for Qwen byte-BPE), then re-apply
    # stop-strings AND stop-after-answer so the answer line is clean and extraction can find "Final Answer:".
    fused_text = decoder.lm_tokenizer.decode(generated_ids, skip_special_tokens=True) if generated_ids else "".join(text_parts)
    fused_text, _ = _apply_stop_strings(fused_text, getattr(decoder, "stop_strings", []) or [])
    fused_text, _ = _apply_degenerate_stops(fused_text, decoder)   # the loop's break trims current_text, but the
    # stored text is re-decoded from generated_ids above, so the same trims must be re-applied here or the record
    # keeps the discarded ramble (that is why 47% of stored FT rows still carried a repeated 'Final Answer:').
    _sa = getattr(decoder, "stop_after_answer", None)
    if _sa and _sa in fused_text:
        _mi = fused_text.find(_sa)
        _nl = fused_text.find("\n", _mi + len(_sa))
        if _nl != -1:
            fused_text = fused_text[:_nl]
    fused_text = _strip_generation_artifacts(fused_text)

    return {
        "evidence_sketch": sketch_text,
        "lm_prompt_with_sketch": lm_prompt,
        "fused_text": fused_text,
        "terminated": terminated,
        "slm_debug": {
            "shared_prefix_tokens": int(shared_ids.shape[1]),
            "answer_suffix_tokens": int(answer_suffix_ids.shape[1]),
            "sketch_suffix_tokens": int(sketch_suffix_ids.shape[1]),
            "shared_prompt_sha256": _prompt_sha256(shared_prompt if dataset not in ASQA_STYLE_DATASETS else decoder.slm_tokenizer.decode(shared_ids_full[0], skip_special_tokens=False)),
            "slm_answer_prompt_sha256": _prompt_sha256(slm_answer_prompt),
            "slm_answer_prompt_tail": slm_answer_prompt[-400:],
            "slm_sketch_prompt_tail": slm_sketch_prompt[-400:],
        },
        "sketch_timing": sketch_timing,
        "fusion_timing": GenerationTiming(
            prefill_s=fusion_prefill_time,
            decode_s=fusion_decode_time,
            total_s=fusion_prefill_time + fusion_decode_time,
        ),
        "exported_samples": exported_samples,
        "phase_memory": phase_memory,
    }


def _score_qa_prediction(example, pred: str, dataset: str) -> Tuple[float, float]:
    if dataset in {"2wiki", "2wikimultihopqa"}:
        golds = getattr(example, "answers", None) or [example.answer]
        em, f1 = compute_2wiki_em_f1(pred, golds)
        return float(em), float(f1)
    if dataset == "musique":
        golds = getattr(example, "answers", None) or [example.answer]
        em, f1 = compute_musique_em_f1(pred, golds)
        return float(em), float(f1)
    if dataset == "babilong":
        # BABILong targets are short entities/labels; SQuAD-style max EM/F1 over
        # the (usually single) gold, with the robust gold-contained-in-pred EM rule.
        golds = getattr(example, "answers", None) or [example.answer]
        em, f1 = compute_best_em_f1(pred, golds)
        return float(em), float(f1)
    if dataset == "strategyqa":
        # StrategyQA is yes/no: canonicalize the prediction to yes/no and match the gold.
        pl = pred.strip().lower()
        if pl.startswith("yes") or "answer: yes" in pl or pl.startswith("true"):
            pred_yn = "yes"
        elif pl.startswith("no") or "answer: no" in pl or pl.startswith("false"):
            pred_yn = "no"
        else:
            pred_yn = pl
        correct = float(pred_yn == example.answer)
        return correct, correct
    if dataset in ("clutrr", "clutrr_multiq"):
        # CLUTRR targets are a single kinship word; same robust EM/F1 as BABILong.
        # (Authoritative CLUTRR-multiq relation-EM is applied post-hoc by scripts/clutrr_multiq_score.py;
        #  this inline score is for live monitoring only.)
        golds = getattr(example, "answers", None) or [example.answer]
        em, f1 = compute_best_em_f1(pred, golds)
        return float(em), float(f1)
    if dataset == "scifact":
        pred_label = normalize_scifact_label(pred)
        gold_label = normalize_scifact_label(example.answer)
        correct = float(pred_label == gold_label)
        return correct, correct
    if dataset == "qasper":
        golds = getattr(example, "answers", None) or [example.answer]
        # Robust QASPER scoring: canonicalize yes/no/unanswerable phrasings so the
        # inline score matches run_robust_em_f1_log_eval.py (no separate eval needed).
        pred = canonicalize_qasper_prediction(pred, golds)
        em, f1 = compute_best_em_f1(pred, golds)
        return float(em), float(f1)
    em, f1 = compute_em_f1(pred, example.answer)
    return float(em), float(f1)


def _extract_final_answer_for_dataset(text: str, dataset: str) -> str:
    pred = extract_final_answer(text)
    tag_match = _ANSWER_TAG_PAIR.search(pred)
    if tag_match is not None:
        return tag_match.group(1).strip()
    return _ANSWER_TAG.sub("", pred).strip()


def _maybe_shuffle_and_sample_examples(examples: List, *, sample: Optional[int], sample_seed: Optional[int]) -> List:
    if sample_seed is None:
        return examples
    shuffled = list(examples)
    random.Random(sample_seed).shuffle(shuffled)
    if sample is not None:
        shuffled = shuffled[: min(sample, len(shuffled))]
    print(
        f"[SampleSeed] Shuffled {len(examples)} loaded examples with seed={sample_seed}; "
        f"selected={len(shuffled)}."
    )
    return shuffled


_CITATION_RE = re.compile(r"\[\d+\]")


def _strip_citations(text: str) -> str:
    return _CITATION_RE.sub("", text)


def _normalize_output_for_alce_metrics(text: str) -> str:
    # ★ Use the FULL extractor, not extract_final_answer (which truncates to the first answer LINE —
    # correct for short-factoid EM, WRONG here). ASQA/LongBench-summ outputs are multi-sentence
    # long-form; truncating to the first line cut summaries roughly in half (gov_report teacher
    # 15.7→19.8 rougeLsum once fixed) and penalized whichever model phrased its summary across more
    # lines. rougeLsum/str_em must see the whole generated summary after the Final Answer marker.
    normalized = extract_final_answer_full(text)
    normalized = _strip_citations(normalized).strip()
    return normalized


def _normalize_answer(text: str) -> str:
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", re.sub(r"[^\w\s]", " ", text.lower())).split())


def _exact_presence(short_answers: Sequence[str], context: str) -> bool:
    norm_context = _normalize_answer(context)
    for answer in short_answers:
        if _normalize_answer(answer) in norm_context:
            return True
    return False


def _compute_rouge_lsum(text: str, example) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeLsum"], use_stemmer=True)
    hypothesis = "\n".join(re.findall(r"[^\n]+", text.lower()))
    if getattr(example, "annotations", None):
        refs = [ann.get("long_answer", "") for ann in example.annotations[:2] if isinstance(ann, dict)]
        refs = refs or [example.answer]
    else:
        refs = [example.answer]
    ref_scores = []
    for ref in refs:
        reference = "\n".join(re.findall(r"[^\n]+", ref.lower()))
        ref_scores.append(scorer.score(reference, hypothesis)["rougeLsum"].fmeasure)
    return 100.0 * max(ref_scores) if ref_scores else 0.0


def _score_asqa_prediction(text: str, example) -> Tuple[str, Dict[str, float]]:
    extracted = _normalize_output_for_alce_metrics(text)
    legacy_em, legacy_f1 = compute_em_f1(extracted, example.answer)
    qa_pairs = getattr(example, "qa_pairs", None) or []
    hits = [
        1.0 if _exact_presence(qa_pair.get("short_answers", []), extracted) else 0.0
        for qa_pair in qa_pairs
        if isinstance(qa_pair, dict)
    ]
    str_em = 100.0 * (sum(hits) / len(hits)) if hits else 0.0
    str_hit = 100.0 if hits and all(val == 1.0 for val in hits) else 0.0
    return extracted, {
        "em": float(legacy_em),
        "f1": legacy_f1,
        "str_em": str_em,
        "str_hit": str_hit,
        "rougeLsum": _compute_rouge_lsum(extracted, example),
    }


def _update_running(metrics: Dict[str, Dict[str, float]], mode: str, score_map: Dict[str, float]) -> Dict[str, float]:
    if mode not in metrics:
        metrics[mode] = {"count": 0.0}
    for key, value in score_map.items():
        metrics[mode][f"{key}_sum"] = metrics[mode].get(f"{key}_sum", 0.0) + float(value)
    metrics[mode]["count"] += 1.0
    count = metrics[mode]["count"]
    return {key: metrics[mode][f"{key}_sum"] / count for key in score_map}


def _update_timing(timings: Dict[str, Dict[str, float]], mode: str, timing: Dict[str, float]) -> None:
    if mode not in timings:
        timings[mode] = {}
    for key, value in timing.items():
        timings[mode][f"{key}_sum"] = timings[mode].get(f"{key}_sum", 0.0) + float(value)
    timings[mode]["count"] = timings[mode].get("count", 0.0) + 1.0


def _update_memory(memories: Dict[str, Dict[str, float]], mode: str, snapshot: Optional[Dict[str, Any]]) -> None:
    if not snapshot:
        return
    total = snapshot.get("total", {})
    if not total:
        return
    bucket = memories.setdefault(mode, {})
    for key in ("allocated_gib", "reserved_gib", "max_allocated_gib", "max_reserved_gib"):
        value = float(total.get(key, 0.0))
        bucket[f"{key}_sum"] = bucket.get(f"{key}_sum", 0.0) + value
        bucket[f"{key}_max"] = max(bucket.get(f"{key}_max", 0.0), value)
    bucket["count"] = bucket.get("count", 0.0) + 1.0


def _memory_summary(memories: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for mode, vals in memories.items():
        count = vals.get("count", 0.0)
        if count <= 0:
            continue
        mode_summary: Dict[str, float] = {}
        for key, value in vals.items():
            if key.endswith("_sum"):
                mode_summary[f"avg_{key[:-len('_sum')]}"] = value / count
            elif key.endswith("_max"):
                mode_summary[key] = value
        summary[mode] = mode_summary
    return summary


def _begin_memory_measurement(enabled: bool) -> None:
    if enabled:
        reset_cuda_peak_memory()


def _attach_memory_extra(snapshot: Dict[str, Any], baseline: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    extra = cuda_memory_extra(snapshot, baseline)
    if extra:
        snapshot["extra_from_after_load"] = extra
    return snapshot


def _finish_memory_measurement(
    enabled: bool,
    mode: str,
    memories: Dict[str, Dict[str, float]],
    *,
    baseline: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    if not enabled:
        return None
    snapshot = _attach_memory_extra(snapshot_cuda_memory(), baseline)
    _update_memory(memories, mode, snapshot)
    extra = snapshot.get("extra_from_after_load", {})
    extra_text = f" | {format_cuda_memory_extra(extra)}" if extra else ""
    print(f"[Memory][{mode}] {format_cuda_memory_summary(snapshot)}{extra_text}")
    return snapshot


def _format_phase_memory_trace(memory_trace: Dict[str, Any], baseline: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    formatted = {}
    for phase, snapshot in memory_trace.items():
        if isinstance(snapshot, dict):
            formatted[phase] = _attach_memory_extra(snapshot, baseline)
    return formatted


def _print_phase_memory_trace(mode: str, phase_memory: Dict[str, Any]) -> None:
    for phase, snapshot in phase_memory.items():
        extra = snapshot.get("extra_from_after_load", {}) if isinstance(snapshot, dict) else {}
        extra_text = f" | {format_cuda_memory_extra(extra)}" if extra else ""
        print(f"[Memory][{mode}.{phase}] {format_cuda_memory_summary(snapshot)}{extra_text}")


def _format_score_map(score_map: Dict[str, float], prefix: str = "") -> str:
    return " ".join(f"{prefix}{key}={value:.4f}" for key, value in score_map.items())


def _timing_avg(timings: Dict[str, Dict[str, float]], mode: str, key: str) -> Optional[float]:
    bucket = timings.get(mode)
    if not bucket:
        return None
    count = bucket.get("count", 0.0)
    sum_key = f"{key}_sum"
    if count <= 0 or sum_key not in bucket:
        return None
    return float(bucket[sum_key]) / float(count)


def _format_current_avg_timing(
    timings: Dict[str, Dict[str, float]],
    mode: str,
    *,
    current_prefill_s: Optional[float],
    current_decode_s: Optional[float],
    prefill_key: str,
    decode_key: str,
) -> str:
    if current_prefill_s is None or current_decode_s is None:
        return "prefill=n/a decode=n/a"
    avg_prefill = _timing_avg(timings, mode, prefill_key)
    avg_decode = _timing_avg(timings, mode, decode_key)
    if avg_prefill is None or avg_decode is None:
        return f"prefill={current_prefill_s:.2f}s decode={current_decode_s:.2f}s"
    return (
        f"prefill={current_prefill_s:.2f}s decode={current_decode_s:.2f}s "
        f"| avg_prefill={avg_prefill:.2f}s avg_decode={avg_decode:.2f}s"
    )


def _mode_gate_weight_name(mode: str, args: argparse.Namespace) -> str:
    if mode in {"slm_lm", "quantized_lm_lm"} and args.load_weight:
        return Path(args.load_weight).name
    return f"lambda={args.lambda_weight:.3f}" if mode in {"slm_lm", "quantized_lm_lm"} else "none"


def _simple_tokenize(text: str) -> List[str]:
    return text.lower().split()


def _bm25_scores(docs: Sequence[str], query: str) -> List[float]:
    if not docs:
        return []
    bm25 = BM25Okapi([_simple_tokenize(d) for d in docs])
    return bm25.get_scores(_simple_tokenize(query)).tolist()


def _order_by_scores(docs: Sequence[str], scores: Sequence[float], *, k: Optional[int] = None) -> Tuple[List[str], List[float]]:
    if not docs:
        return [], []
    ranked = list(reversed(sorted(range(len(scores)), key=lambda i: scores[i])))
    if k is not None:
        ranked = ranked[:k]
    return [docs[i] for i in ranked], [scores[i] for i in ranked]


def _load_alce_data(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        return data["data"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported ALCE data format in {path}")


def _split_candidates(split: str) -> List[str]:
    split = split.lower()
    if split in {"valid", "validation", "val", "dev"}:
        return [split, "validation", "valid", "val", "dev", "test"]
    return [split]


def _find_alce_data_file(base_dir: Path, dataset: str, split: str) -> Optional[Path]:
    dataset = dataset.lower()
    split_terms = _split_candidates(split)
    candidates = [p for p in base_dir.rglob("*.json") if p.is_file()] + [p for p in base_dir.rglob("*.jsonl") if p.is_file()]
    if not candidates:
        return None

    def score(path: Path) -> Tuple[int, int]:
        name = path.name.lower()
        score_val = 0
        if dataset in name:
            score_val += 10
        if any(term in name for term in split_terms):
            score_val += 5
        if any(key in name for key in ("retrieval", "retrieved", "ctx", "context", "passage", "passages")):
            score_val += 2
        if name.endswith(".jsonl"):
            score_val += 1
        return score_val, -len(name)

    ranked = sorted(candidates, key=score, reverse=True)
    best = ranked[0]
    if score(best)[0] == 0:
        return None
    return best


def _download_and_extract_alce(base_dir: Path) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    tar_path = base_dir / "ALCE-data.tar"
    if not tar_path.exists():
        print(f"Downloading ALCE data to {tar_path} (this is ~451 MB).")
        urllib.request.urlretrieve(ALCE_TAR_URL, tar_path)
    print(f"Extracting {tar_path} into {base_dir}.")
    with tarfile.open(tar_path, "r:*") as tf:
        tf.extractall(base_dir)


def _resolve_alce_data_path(args: argparse.Namespace) -> Path:
    if args.data is not None:
        candidate = Path(args.data)
        if candidate.is_dir():
            resolved = _find_alce_data_file(candidate, args.dataset, args.split)
            if resolved is not None:
                return resolved
            _download_and_extract_alce(candidate)
            resolved = _find_alce_data_file(candidate, args.dataset, args.split)
            if resolved is None:
                raise FileNotFoundError(f"No ALCE data file found in {candidate}.")
            return resolved
        return candidate
    resolved = _find_alce_data_file(DEFAULT_ALCE_DIR, args.dataset, args.split)
    if resolved is not None:
        return resolved
    _download_and_extract_alce(DEFAULT_ALCE_DIR)
    resolved = _find_alce_data_file(DEFAULT_ALCE_DIR, args.dataset, args.split)
    if resolved is None:
        raise FileNotFoundError(f"No ALCE data file found in {DEFAULT_ALCE_DIR}.")
    return resolved


def _extract_alce_passages(example: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ("ctxs", "contexts", "retrieved_passages", "retrievals", "docs", "passages"):
        if key in example and isinstance(example[key], list):
            return example[key]
    return []


def _format_passage_text(passage: Dict[str, Any]) -> str:
    title = passage.get("title") or passage.get("document_title") or ""
    text = passage.get("text") or passage.get("contents") or passage.get("passage") or ""
    if title and text:
        return f"{title}: {text}"
    return text or title


def _build_asqa_examples(rows: List[Dict[str, Any]], *, doc_number: int, max_corpus_examples: Optional[int]) -> List[ALCEExample]:
    corpus_texts: List[str] = []
    seen = set()
    for row in rows[: (max_corpus_examples or len(rows))]:
        for passage in _extract_alce_passages(row):
            text = _format_passage_text(passage)
            if text and text not in seen:
                seen.add(text)
                corpus_texts.append(text)
    bm25_corpus = BM25Okapi([_simple_tokenize(t) for t in corpus_texts]) if corpus_texts else None

    examples: List[ALCEExample] = []
    for idx, row in enumerate(rows):
        qid = row.get("sample_id") or row.get("id") or row.get("qid") or row.get("question_id") or str(idx)
        question = row.get("question") or row.get("query") or ""
        answer = row.get("answer") or row.get("answers") or ""
        passages = _extract_alce_passages(row)
        doc_meta = []
        for p_idx, passage in enumerate(passages):
            text = _format_passage_text(passage)
            if not text:
                continue
            doc_meta.append(
                {
                    "doc_id": passage.get("doc_id") or passage.get("id") or p_idx + 1,
                    "title": passage.get("title"),
                    "text": text,
                    "score": passage.get("score") or passage.get("retrieval_score") or passage.get("bm25_score"),
                }
            )
        selected = doc_meta[: min(doc_number, len(doc_meta))]
        selected_texts = [d["text"] for d in selected]
        if doc_number > len(selected) and bm25_corpus is not None:
            needed = doc_number - len(selected)
            scores = bm25_corpus.get_scores(_simple_tokenize(question))
            ranked_idx = list(reversed(sorted(range(len(scores)), key=lambda i: scores[i])))
            extras = []
            for ridx in ranked_idx:
                text = corpus_texts[ridx]
                if text in selected_texts:
                    continue
                extras.append({"doc_id": f"corpus-{ridx}", "title": None, "text": text, "score": float(scores[ridx])})
                if len(extras) >= needed:
                    break
            selected += extras
        ordered_texts, ordered_scores = _order_by_scores(
            [d["text"] for d in selected],
            _bm25_scores([d["text"] for d in selected], question),
        )
        score_map = {t: s for t, s in zip(ordered_texts, ordered_scores)}
        ordered_meta = sorted(selected, key=lambda d: score_map.get(d["text"], 0.0), reverse=True)
        examples.append(
            ALCEExample(
                qid,
                question,
                answer if isinstance(answer, str) else "",
                ordered_texts,
                ordered_meta,
                qa_pairs=row.get("qa_pairs"),
                annotations=row.get("annotations"),
                claims=row.get("claims"),
            )
        )
    return examples


def _load_learned_gate(args: argparse.Namespace, lm_model, slm_model):
    if not args.load_weight:
        return None, None, None
    ckpt = torch.load(args.load_weight, map_location="cpu")
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError(f"Invalid gate checkpoint format: {args.load_weight}")
    ckpt_cfg = ckpt.get("config", {})
    learned_gate_type = normalize_learned_gate_type(str(ckpt_cfg.get("gate_feature_type", "logit_topk")))
    if learned_gate_type not in _LEARNED_GATE_TYPES:
        raise ValueError(f"Unsupported learned gate type for this script: {learned_gate_type}")
    hidden_dim = int(ckpt_cfg.get("hidden_dim", 64))
    gate_mlp_layers = int(ckpt_cfg.get("gate_mlp_layers", 1))
    learned_gate_top_k = int(ckpt_cfg.get("gate_top_k", 10))
    ckpt_fusion_mode = str(ckpt_cfg.get("fusion_mode", "weighted_sum"))
    if ckpt_fusion_mode != args.fusion_mode:
        raise ValueError(
            f"Gate checkpoint fusion_mode={ckpt_fusion_mode!r} is incompatible with current fusion_mode={args.fusion_mode!r}."
        )
    if learned_gate_type != "logit_topk" and slm_model is not None:
        _ensure_eager_attention(slm_model, reason=f"learned_gate_type={learned_gate_type}")

    if learned_gate_type == "attn_linear_weighted":
        learned_gate = AttentionLinearCalibrator(
            num_layers=int(ckpt_cfg["num_layers"]),
            num_heads=int(ckpt_cfg["num_heads"]),
        )
    elif learned_gate_type == "attn_ratio_and_logit_topk":
        learned_gate = TokenAgnosticGateMLP(
            input_dim=2 * learned_gate_top_k + 1,
            hidden_dim=hidden_dim,
            mlp_layers=gate_mlp_layers,
        )
    elif learned_gate_type == "attn_ratio_mlp_weighted":
        learned_gate = AttentionRatioMLPGate(
            num_layers=int(ckpt_cfg["num_layers"]),
            num_heads=int(ckpt_cfg["num_heads"]),
            hidden_dim=hidden_dim,
            mlp_layers=gate_mlp_layers,
        )
    elif learned_gate_type == "attn_stats_mlp_unweighted":
        learned_gate = AttentionStatsGateMLP(
            input_dim=int(ckpt_cfg.get("attn_feature_dim", 10)),
            hidden_dim=hidden_dim,
            mlp_layers=gate_mlp_layers,
        )
    else:
        learned_gate = TokenAgnosticGateMLP(
            input_dim=2 * learned_gate_top_k,
            hidden_dim=hidden_dim,
            mlp_layers=gate_mlp_layers,
        )
    learned_gate.load_state_dict(ckpt["state_dict"])
    gate_device = get_model_output_device(lm_model) if lm_model is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    learned_gate = learned_gate.to(gate_device)
    learned_gate.eval()
    print(
        f"Loaded gate checkpoint: {args.load_weight} "
        f"(gate_type={learned_gate_type}, gate_top_k={learned_gate_top_k}, hidden_dim={hidden_dim}, mlp_layers={gate_mlp_layers})"
    )
    return learned_gate, learned_gate_top_k, learned_gate_type


def _build_decoder(
    args: argparse.Namespace,
    slm_model,
    slm_tok,
    lm_model,
    lm_tok,
    *,
    prompt_builder: Callable[[Any, bool, Any], str],
) -> FixedLambdaFusionDecoder:
    learned_gate, learned_gate_top_k, learned_gate_type = _load_learned_gate(args, lm_model, slm_model)
    return FixedLambdaFusionDecoder(
        slm_model,
        slm_tok,
        lm_model,
        lm_tok,
        lambda_weight=args.lambda_weight,
        fusion_mode=args.fusion_mode,
        entropy_scale=args.entropy_scale,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
        prompt_builder=prompt_builder,
        stop_strings=args.stop_strings,
        stop_on_repeat=(args.stop_on_repeat or None),
        stop_after_answer=(args.stop_after_answer or None),
        answer_token_budget=args.answer_token_budget,
        learned_gate=learned_gate,
        learned_gate_top_k=learned_gate_top_k or 10,
        learned_gate_type=learned_gate_type or "logit_topk",
        decode_temperature=args.decode_temperature,
        decode_top_p=args.decode_top_p,
        decode_seed=args.decode_seed,
        decode_sync_interval=getattr(args, "decode_sync_interval", 1),
        slm_kv_quant_bits=getattr(args, "slm_kv_quant_bits", 0),
        slm_kv_backend=getattr(args, "slm_kv_backend", "quanto"),
        slm_kv_group_size=getattr(args, "slm_kv_group_size", 64),
        slm_kv_residual_length=getattr(args, "slm_kv_residual_length", 128),
    )


def _print_debug_block(label: str, question: str, text: str) -> None:
    print(f"\n<<<{label} question>>>\n{question}")
    print(f"<<<{label} text>>>\n{text}")


def _print_debug_text(label: str, text: str) -> None:
    print(f"\n<<<{label}>>>\n{text}")


def _print_debug_answer(label: str, question: str, raw_text: str, extracted_text: str, gold_answer: str) -> None:
    print(f"\n<<<{label} question>>>\n{question}")
    print(f"<<<{label} raw>>>\n{raw_text}")
    print(f"<<<{label} extracted>>>\n{extracted_text}")
    print(f"<<<{label} gold>>>\n{gold_answer}")


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")
        f.flush()


def _serialize_kl_sample(sample: KLStepSample) -> Dict[str, Any]:
    return {
        "features": sample.features.tolist(),
        "teacher_logits": sample.teacher_logits.tolist(),
        "slm_logits": sample.slm_logits.tolist(),
        "lm_logits": sample.lm_logits.tolist(),
        "slm_base_logits": sample.slm_base_logits.tolist() if sample.slm_base_logits is not None else None,
    }


def _save_export_training_cache(
    path: Path,
    *,
    metadata: Dict[str, Any],
    samples: Sequence[KLStepSample],
    stats: SampleBuildStats,
    extra_stats: Dict[str, int],
    example_rows: Sequence[Dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "samples": list(samples),
        "stats": {
            "examples_seen": int(stats.examples_seen),
            "examples_used": int(stats.examples_used),
            "steps_used": int(stats.steps_used),
            "skipped_empty_teacher": int(stats.skipped_empty_teacher),
            "skipped_vocab_mismatch": int(stats.skipped_vocab_mismatch),
        },
        "extra_stats": {
            "skipped_missing_context": int(extra_stats.get("skipped_missing_context", 0)),
            "skipped_doc_id_mismatch": int(extra_stats.get("skipped_doc_id_mismatch", 0)),
        },
        "example_rows": list(example_rows),
    }
    torch.save(payload, path)


def _load_teacher_trace_lookup(path: Path) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue
            example_id = obj.get("example_id")
            if isinstance(example_id, str):
                lookup[example_id] = obj
    return lookup


def _maybe_refresh_teacher_trace_lookup(
    path: Optional[Path],
    lookup: Dict[str, Dict[str, Any]],
    *,
    expected_example_id: str,
) -> Dict[str, Dict[str, Any]]:
    if path is None:
        return lookup
    if expected_example_id in lookup:
        return lookup
    refreshed = _load_teacher_trace_lookup(path)
    if len(refreshed) > len(lookup):
        print(
            "[TeacherTrace] Refreshed teacher-trace lookup "
            f"{len(lookup)} -> {len(refreshed)} rows while waiting for example_id={expected_example_id}"
        )
        return refreshed
    return lookup


def _load_jsonl_example_ids(path: Path) -> set[str]:
    example_ids: set[str] = set()
    if not path.exists():
        return example_ids
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict) or "summary" in obj:
                continue
            example_id = obj.get("example_id")
            if isinstance(example_id, str):
                example_ids.add(example_id)
    return example_ids


def _load_export_training_cache(
    path: Path,
) -> Tuple[Optional[Dict[str, Any]], List[KLStepSample], SampleBuildStats, Dict[str, int], List[Dict[str, Any]]]:
    if not path.exists():
        return None, [], SampleBuildStats(), {"skipped_missing_context": 0, "skipped_doc_id_mismatch": 0}, []
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid export cache format: {path}")
    metadata = payload.get("metadata")
    samples = payload.get("samples", [])
    stats_payload = payload.get("stats", {})
    extra_stats = payload.get("extra_stats", {})
    example_rows = payload.get("example_rows", [])
    stats = SampleBuildStats(
        examples_seen=int(stats_payload.get("examples_seen", 0)),
        examples_used=int(stats_payload.get("examples_used", 0)),
        steps_used=int(stats_payload.get("steps_used", 0)),
        skipped_empty_teacher=int(stats_payload.get("skipped_empty_teacher", 0)),
        skipped_vocab_mismatch=int(stats_payload.get("skipped_vocab_mismatch", 0)),
    )
    return (
        metadata if isinstance(metadata, dict) else None,
        list(samples) if isinstance(samples, list) else [],
        stats,
        {
            "skipped_missing_context": int(extra_stats.get("skipped_missing_context", 0)),
            "skipped_doc_id_mismatch": int(extra_stats.get("skipped_doc_id_mismatch", 0)),
        },
        list(example_rows) if isinstance(example_rows, list) else [],
    )


def _collect_teacher_trace(
    model,
    tokenizer,
    prompt: str,
    *,
    max_length: int,
    max_new_tokens: int,
    top_k: int,
    stop_strings: Sequence[str],
) -> Dict[str, Any]:
    inputs = prepare_inputs(tokenizer, prompt, max_length=max_length, return_tensors="pt")
    device = get_model_input_device(model)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    past = outputs.past_key_values
    generated_ids: List[int] = []
    text_parts: List[str] = []
    steps: List[Dict[str, Any]] = []
    eos_ids = get_eos_token_ids(tokenizer, model)

    for _ in range(max_new_tokens):
        logits = outputs.logits[:, -1, :]
        k = max(1, min(int(top_k), int(logits.shape[-1])))
        top_vals, top_ids = torch.topk(logits, k=k, dim=-1)
        next_token = torch.argmax(logits, dim=-1)
        token_id = int(next_token[0].item())
        generated_ids.append(token_id)
        steps.append(
            {
                "token_id": token_id,
                "top_k": [
                    {"token_id": int(tid), "logit": float(val)}
                    for tid, val in zip(top_ids[0].tolist(), top_vals[0].tolist())
                ],
            }
        )
        text_parts.append(tokenizer.decode([token_id], skip_special_tokens=False))
        current_text = "".join(text_parts)
        current_text, stopped = _apply_stop_strings(current_text, stop_strings)
        if stopped:
            text_parts = [current_text]
            break
        if token_id in eos_ids:
            break
        next_ids = next_token.unsqueeze(0)
        input_ids = torch.cat([input_ids, next_ids.to(input_ids.device)], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(next_ids, device=attention_mask.device, dtype=attention_mask.dtype)],
            dim=1,
        )
        with torch.no_grad():
            outputs = model(
                input_ids=next_ids.to(device),
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
            )
        past = outputs.past_key_values

    text = "".join(text_parts) if text_parts else tokenizer.decode(generated_ids, skip_special_tokens=True)
    text, _ = _apply_stop_strings(text, stop_strings)
    text = _strip_generation_artifacts(text)
    return {"text": text, "token_ids": generated_ids, "steps": steps}


def _build_export_sample_from_step(
    *,
    slm_logits: torch.Tensor,
    lm_logits: torch.Tensor,
    teacher_topk: Sequence[Dict[str, Any]],
    gate_feature_type: str,
    gate_top_k: int,
    loss_top_k: int,
    slm_attentions: Optional[Sequence[torch.Tensor]],
    context_span: Optional[Tuple[int, int]],
    kv_len: int,
) -> Optional[KLStepSample]:
    vocab = min(int(slm_logits.shape[-1]), int(lm_logits.shape[-1]))
    if vocab <= 1:
        return None
    slm_logits = slm_logits[:vocab]
    lm_logits = lm_logits[:vocab]

    teacher_map: Dict[int, float] = {}
    teacher_ids: List[int] = []
    for row in teacher_topk:
        token_id = int(row["token_id"])
        if 0 <= token_id < vocab:
            teacher_map[token_id] = float(row["logit"])
            teacher_ids.append(token_id)
    if not teacher_map:
        return None

    k_loss = max(1, min(int(loss_top_k), vocab))
    slm_top_ids = torch.topk(slm_logits, k=k_loss, dim=-1).indices.tolist()
    lm_top_ids = torch.topk(lm_logits, k=k_loss, dim=-1).indices.tolist()

    union_ids: List[int] = []
    seen = set()
    for token_id in teacher_ids + slm_top_ids + lm_top_ids:
        if token_id not in seen:
            seen.add(token_id)
            union_ids.append(token_id)
    union_tensor = torch.tensor(union_ids, dtype=torch.long, device=slm_logits.device)
    slm_union = slm_logits.index_select(0, union_tensor).to(dtype=torch.float32)
    lm_union = lm_logits.index_select(0, union_tensor).to(dtype=torch.float32)
    teacher_union = torch.full((len(union_ids),), _NEG_INF, dtype=torch.float32, device=slm_logits.device)
    for idx, token_id in enumerate(union_ids):
        if token_id in teacher_map:
            teacher_union[idx] = teacher_map[token_id]

    if gate_feature_type in _HEADWISE_EXPORT_GATE_TYPES:
        if slm_attentions is None:
            raise ValueError(f"{gate_feature_type} requires SLM attentions for export.")
        ctx_heads, shr_heads = extract_attention_head_masses(
            slm_attentions,
            context_span=context_span,
            kv_len=int(kv_len),
        )
        feats = torch.cat([ctx_heads.reshape(-1), shr_heads.reshape(-1)], dim=-1).to(dtype=torch.float32)
    elif gate_feature_type == "attn_ratio_and_logit_topk":
        if slm_attentions is None:
            raise ValueError("attn_ratio_and_logit_topk requires SLM attentions for export.")
        ctx_heads, shr_heads = extract_attention_head_masses(
            slm_attentions,
            context_span=context_span,
            kv_len=int(kv_len),
        )
        ctx_ratio = attention_ctx_ratio(ctx_heads, shr_heads)
        feats = build_attn_ratio_and_logit_topk_features(
            slm_logits,
            lm_logits,
            gate_top_k=int(gate_top_k),
            ctx_attn_ratio=ctx_ratio,
        )
    elif gate_feature_type == "attn_stats_mlp_unweighted":
        if slm_attentions is None:
            raise ValueError("attn_stats_mlp_unweighted requires SLM attentions for export.")
        ctx_heads, shr_heads = extract_attention_head_masses(
            slm_attentions,
            context_span=context_span,
            kv_len=int(kv_len),
        )
        feats = build_attention_stat_features(ctx_heads, shr_heads)
    else:
        feats = build_gate_features(slm_logits, lm_logits, gate_top_k=int(gate_top_k))

    return KLStepSample(
        features=feats.detach().cpu(),
        teacher_logits=teacher_union.detach().cpu(),
        slm_logits=slm_union.detach().cpu(),
        lm_logits=lm_union.detach().cpu(),
        slm_base_logits=None,
    )


def main() -> None:
    args = parse_args()
    print("[Info] Parsed args " + json.dumps(vars(args), sort_keys=True, default=str))
    set_use_chat_template(args.use_chat_template)
    dataset_name = "2wiki" if args.dataset == "2wikimultihopqa" else args.dataset
    if dataset_name in {"counterfactual_hotpotqa", "cf_hotpotqa", "cf_qa"}:
        dataset_name = "counterfactual_qa"
    # BABILong: the dataset name carries the context length (babilong_<len>).
    # Collapse to the canonical "babilong" name (used for dispatch + scoring) and
    # keep the parsed length for the loader, so length can be swept by name.
    babilong_length: Optional[str] = None
    if dataset_name.startswith("babilong_"):
        babilong_length = dataset_name.split("babilong_", 1)[1]
        dataset_name = "babilong"
    # CLUTRR: the name carries the target context length (clutrr_<len>); collapse to "clutrr"
    # and keep the target_tokens for the loader.
    clutrr_target_tokens: Optional[int] = None
    if dataset_name in _CLUTRR_TARGET_TOKENS:
        clutrr_target_tokens = _CLUTRR_TARGET_TOKENS[dataset_name]
        dataset_name = "clutrr"
    # ProofWriter: same length-by-name collapse as CLUTRR.
    proofwriter_target_tokens: Optional[int] = None
    if dataset_name in _PROOFWRITER_TARGET_TOKENS:
        proofwriter_target_tokens = _PROOFWRITER_TARGET_TOKENS[dataset_name]
        dataset_name = "proofwriter"
    if dataset_name == "asqa" and any(mode in args.modes for mode in ("blockattention", "pcw")):
        raise ValueError("blockattention/pcw baselines are currently implemented for QA-style datasets, not ASQA.")
    if args.pass_number < 0:
        raise ValueError("--pass-number must be >= 0.")
    if args.export_output_for_training and "slm_lm" not in args.modes:
        raise ValueError("--export-output-for-training requires 'slm_lm' in --modes.")
    if args.teacher_trace_input and not Path(args.teacher_trace_input).exists():
        raise FileNotFoundError(f"--teacher-trace-input not found: {args.teacher_trace_input}")
    _prepare_devices(args)
    sketch_max_length = args.sketch_max_length or args.max_length
    use_generate = args.single_model_decoding == "generate"
    print(
        "[Info] Effective config "
        + json.dumps(
            {
                "dataset": args.dataset,
                "split": args.split,
                "doc_number": args.doc_number,
                "modes": args.modes,
                "sample": args.sample,
                "sample_seed": args.sample_seed,
                "max_new_tokens": args.max_new_tokens,
                "sketch_max_new_tokens": args.sketch_max_new_tokens,
                "sketch_max_length": sketch_max_length,
                "lambda_weight": args.lambda_weight,
                "fusion_mode": args.fusion_mode,
                "single_model_decoding": args.single_model_decoding,
                "quantized_lm_model": args.quantized_lm_model or args.lm_model,
                "quantized_lm_bits": args.quantized_lm_bits if "quantized_lm_lm" in args.modes else None,
                "revert_previous_prompt": bool(args.revert_previous_prompt),
                "asqa_prompt_version": args.asqa_prompt_version,
            },
            sort_keys=True,
            default=str,
        )
    )

    slm_model = slm_tok = None
    quantized_lm_model = quantized_lm_tok = None
    lm_model = lm_tok = None
    teacher_model = teacher_tok = None

    if "slm_lm" in args.modes:
        slm_model, slm_tok = load_causal_lm(
            args.slm_model,
            cache_dir=args.cache_dir,
            device_map=args.slm_device_map,
            max_memory=args._model_max_memory.get("slm_device_map"),
        )
        setattr(slm_tok, "_codex_revert_previous_prompt", bool(args.revert_previous_prompt))
        if getattr(args, "slm_lora", None):
            from peft import PeftModel
            slm_model = PeftModel.from_pretrained(slm_model, args.slm_lora)
            slm_model.eval()
            print(f"[Info] SLM-LoRA adapter loaded onto the fusion SLM (reader) branch: {args.slm_lora}", flush=True)
    if "quantized_lm_lm" in args.modes:
        quantized_lm_name = args.quantized_lm_model or args.lm_model
        quantized_lm_model, quantized_lm_tok = load_causal_lm(
            quantized_lm_name,
            cache_dir=args.cache_dir,
            device_map=args.quantized_lm_device_map,
            max_memory=args._model_max_memory.get("quantized_lm_device_map"),
            quantization_config=_build_quantized_lm_config(args),
        )
        setattr(quantized_lm_tok, "_codex_revert_previous_prompt", bool(args.revert_previous_prompt))
        print(
            f"[Info] Loaded quantized contextual LM branch model={quantized_lm_name} "
            f"bits={args.quantized_lm_bits}"
        )
    if any(mode in args.modes for mode in ("slm_lm", "quantized_lm_lm", "blockattention", "pcw", "pced")):
        lm_model, lm_tok = load_causal_lm(
            args.lm_model,
            cache_dir=args.cache_dir,
            device_map=args.lm_device_map,
            max_memory=args._model_max_memory.get("lm_device_map"),
        )
        setattr(lm_tok, "_codex_revert_previous_prompt", bool(args.revert_previous_prompt))
        if getattr(args, "lm_lora", None):
            from peft import PeftModel
            lm_model = PeftModel.from_pretrained(lm_model, args.lm_lora)
            lm_model.eval()
            print(f"[Info] LM-LoRA adapter loaded onto the fusion LM branch: {args.lm_lora}", flush=True)
        if any(mode in args.modes for mode in ("blockattention", "pcw")):
            _ensure_sdpa_attention(lm_model, reason="blockattention/pcw custom mask")
    teacher_trace_lookup: Dict[str, Dict[str, Any]] = {}
    teacher_trace_input_path: Optional[Path] = None
    if args.teacher_trace_input:
        teacher_trace_input_path = Path(args.teacher_trace_input)
        teacher_trace_lookup = _load_teacher_trace_lookup(teacher_trace_input_path)

    need_live_teacher_model = bool("teacher" in args.modes or args.export_teacher_trace or (args.export_output_for_training and not args.teacher_trace_input))
    if need_live_teacher_model:
        _teacher_quant_cfg = None
        if getattr(args, "teacher_quant_bits", 0):
            from transformers import BitsAndBytesConfig
            if args.teacher_quant_bits == 4:
                _teacher_quant_cfg = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
            elif args.teacher_quant_bits == 8:
                _teacher_quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
            else:
                raise ValueError(f"--teacher-quant-bits must be 0/4/8, got {args.teacher_quant_bits}")
            print(f"[Info] WEIGHT-QUANT teacher baseline: {args.teacher_quant_bits}-bit bitsandbytes (nf4/int8) — reduces WEIGHT peak mem (unlike KV-quant)")
        teacher_model, teacher_tok = load_causal_lm(
            args.lm_model,
            cache_dir=args.cache_dir,
            device_map=args.teacher_device_map,
            max_memory=args._model_max_memory.get("teacher_device_map"),
            quantization_config=_teacher_quant_cfg,
        )
        # 2026-08-10: --lm-lora was only wired into the FUSION LM branch, so `--modes teacher --lm-lora X`
        # silently ran the BASE model (caught when the "v5 reader alone" hotpot control came back 600/600
        # byte-identical to the naive floor). The single-model arm is how a distilled reader is scored
        # ALONE (the decomposition control), so the adapter must load here too.
        if getattr(args, "lm_lora", None):
            from peft import PeftModel
            teacher_model = PeftModel.from_pretrained(teacher_model, args.lm_lora)
            teacher_model.eval()
            print(f"[Info] LM-LoRA adapter loaded onto the SINGLE-MODEL (teacher/floor) branch: {args.lm_lora}", flush=True)
        setattr(teacher_tok, "_codex_revert_previous_prompt", bool(args.revert_previous_prompt))

    def decoder_prompt_builder(example, include_docs: bool, tokenizer) -> str:
        if dataset_name == "asqa":
            return example.prompt(include_docs=include_docs, tokenizer=tokenizer)
        if include_docs:
            return _build_qa_answer_prompt_preserving_query(
                example.base if hasattr(example, "base") else example,
                include_docs=True,
                tokenizer=tokenizer,
                max_length=args.max_length,
            )
        return _qa_answer_prompt(
            example,
            include_docs=False,
            evidence_sketch=getattr(example, "evidence_sketch", None),
            revert_previous_prompt=bool(getattr(tokenizer, "_codex_revert_previous_prompt", False)),
        )

    decoder = (
        _build_decoder(
            args,
            slm_model,
            slm_tok,
            lm_model,
            lm_tok,
            prompt_builder=decoder_prompt_builder,
        )
        if "slm_lm" in args.modes
        else None
    )
    quantized_decoder = (
        _build_decoder(
            args,
            quantized_lm_model,
            quantized_lm_tok,
            lm_model,
            lm_tok,
            prompt_builder=decoder_prompt_builder,
        )
        if "quantized_lm_lm" in args.modes
        else None
    )
    pced_decoder = None
    if "pced" in args.modes:
        pced_decoder = PCEDContrastiveDecoder(
            lm_model,
            lm_tok,
            beta0=args.pced_beta0,
            beta_mode=args.pced_beta_mode,
            beta_warmup=args.pced_beta_warmup,
            beta_reduce=args.pced_beta_reduce,
            gamma=args.pced_gamma,
            relevance_mode=args.pced_relevance_mode,
            max_new_tokens=args.max_new_tokens,
            max_length=args.max_length,
            prompt_builder=_build_pced_prompt_builder(args, dataset_name),
            stop_strings=args.stop_strings,
        )

    load_memory = None
    if args.log_memory:
        load_memory = snapshot_cuda_memory()
        print(f"[Memory][after_load] {format_cuda_memory_summary(load_memory)}")

    preprocess_start = time.perf_counter()
    data_path = None
    loader_sample = None if args.sample_seed is not None else args.sample
    if dataset_name == "hotpotqa":
        examples, preprocess_stats = load_hotpotqa_split(
            args.split,
            cache_dir=args.cache_dir,
            sample=loader_sample,
            doc_number=args.doc_number,
            retrieval_split=args.retrieval_split,
            max_corpus_examples=args.max_corpus_examples,
            return_stats=True,
            retriever=args.retriever,
        )
    elif dataset_name == "counterfactual_qa":
        examples, preprocess_stats = load_counterfactual_qa_split(
            args.split,
            cache_dir=args.cache_dir,
            sample=loader_sample,
            doc_number=args.doc_number,
            retrieval_split=args.retrieval_split,
            max_corpus_examples=args.max_corpus_examples,
            return_stats=True,
            retriever=args.retriever,
            data_path=args.data,
            corpus_path=args.counterfactual_corpus,
            count=args.counterfactual_count,
            seed=args.counterfactual_seed,
        )
        if args.doc_number == 0:
            print(
                "[Info] Counterfactual benchmark doc_number=0: no passages are provided "
                "(no synthetic gold passage, no distractors). This is the no-context / LM-only baseline."
            )
        else:
            print(
                "[Info] Counterfactual benchmark uses one fixed synthetic gold passage plus "
                f"{max(args.doc_number - 1, 0)} fixed-corpus distractor passage(s) per example."
            )
    elif dataset_name == "qasper":
        examples, preprocess_stats = load_qasper_split(
            args.split,
            cache_dir=args.cache_dir,
            sample=loader_sample,
            doc_number=args.doc_number,
            retrieval_split=args.retrieval_split,
            max_corpus_examples=args.max_corpus_examples,
            return_stats=True,
            retriever=args.retriever,
        )
        print(f"[Info] QASPER ignores doc_number={args.doc_number}: using full paper context for every question.")
    elif dataset_name == "2wiki":
        examples, preprocess_stats = load_2wiki_split(
            args.split,
            cache_dir=args.cache_dir,
            sample=loader_sample,
            doc_number=args.doc_number,
            retrieval_split=args.retrieval_split,
            max_corpus_examples=args.max_corpus_examples,
            return_stats=True,
            retriever=args.retriever,
        )
        print("[Info] 2WikiMultiHopQA evaluation uses official answer EM/F1 adapted from 2wikimultihop_evaluate_v1.1.py.")
    elif dataset_name == "musique":
        examples, preprocess_stats = load_musique_split(
            args.split,
            cache_dir=args.cache_dir,
            sample=loader_sample,
            doc_number=args.doc_number,
            retrieval_split=args.retrieval_split,
            max_corpus_examples=args.max_corpus_examples,
            return_stats=True,
            retriever=args.retriever,
        )
        print("[Info] MuSiQue evaluation uses official answer EM/F1 over answer aliases.")
    elif dataset_name == "scifact":
        examples, preprocess_stats = load_scifact_split(
            args.split,
            cache_dir=args.cache_dir,
            sample=loader_sample,
            doc_number=args.doc_number,
            retrieval_split=args.retrieval_split,
            max_corpus_examples=args.max_corpus_examples,
            return_stats=True,
            retriever=args.retriever,
        )
        print("[Info] SciFact uses BM25 open-corpus retrieval and official-style label accuracy/macro-F1 summaries.")
    elif dataset_name == "asqa":
        data_path = _resolve_alce_data_path(args)
        rows = _load_alce_data(data_path)
        if args.sample_seed is not None:
            random.Random(args.sample_seed).shuffle(rows)
            print(f"[SampleSeed] Shuffled {len(rows)} ASQA rows with seed={args.sample_seed}.")
        if args.sample:
            rows = rows[: args.sample]
        examples = _build_asqa_examples(rows, doc_number=args.doc_number, max_corpus_examples=args.max_corpus_examples)
        preprocess_stats = {"bm25_time_s": 0.0}
    elif dataset_name in _ALL_LONGBENCH_DATASETS:
        task = _ALL_LONGBENCH_DATASETS[dataset_name]
        examples = load_longbench_split(
            task,
            split=args.split,
            sample=loader_sample,
            cache_dir=args.cache_dir,
        )
        preprocess_stats = {"bm25_time_s": 0.0}
        print(
            f"[Info] LongBench {task}: single-document summarization, full context "
            f"(no BM25 retrieval; doc_number ignored), scored with rougeLsum."
        )
    elif dataset_name == "babilong":
        examples = load_babilong_split(
            length=babilong_length,
            task=args.babilong_task,
            split=args.split,
            sample=loader_sample,
            cache_dir=args.cache_dir,
        )
        preprocess_stats = {"bm25_time_s": 0.0}
        print(
            f"[Info] BABILong length={babilong_length} task={args.babilong_task or 'ALL(qa1..qa10 mix)'}: "
            f"long-context bAbI multi-fact reasoning, single context document "
            f"(no BM25 retrieval; doc_number ignored), scored with EM/F1 over the short target."
        )
    elif dataset_name == "strategyqa":
        examples = load_strategyqa_split(
            split=args.split,
            sample=loader_sample,
            cache_dir=args.cache_dir,
        )
        preprocess_stats = {"bm25_time_s": 0.0}
        print(
            "[Info] StrategyQA implicit multi-hop yes/no reasoning: gold facts as context "
            "(no BM25 retrieval; doc_number ignored), reason_then_answer + yes/no EM."
        )
    elif dataset_name == "clutrr":
        tgt = clutrr_target_tokens or 4000
        examples = load_clutrr_split(
            target_tokens=tgt,
            split=args.split,
            sample=loader_sample,
            cache_dir=args.cache_dir,
        )
        preprocess_stats = {"bm25_time_s": 0.0}
        print(
            f"[Info] CLUTRR multi-hop kinship reasoning (min_hops={os.environ.get('CLUTRR_MIN_HOPS', '4')}, "
            f"target_tokens={tgt}): relevant story padded with disjoint-entity distractor stories "
            f"(no BM25; reason_then_answer + EM/F1 over the single kinship-word target)."
        )
    elif dataset_name == "clutrr_multiq":
        # Multi-question KV-reuse CLUTRR (QASPER replacement): one story = shared context for several derived
        # kinship questions; clean single-word relation answer. Story count via env CLUTRR_MULTIQ_STORIES.
        examples = load_clutrr_multiq_split(sample=loader_sample, split=args.split, cache_dir=args.cache_dir)
        preprocess_stats = {"bm25_time_s": 0.0}
        print(f"[Info] CLUTRR multi-question (stories={os.environ.get('CLUTRR_MULTIQ_STORIES','100')}): one story "
              f"shared across derived kinship questions; single relation-word answer (clean EM).")
    elif dataset_name == "proofwriter":
        tgt = proofwriter_target_tokens or 4000
        examples = load_proofwriter_split(
            target_tokens=tgt,
            split=args.split,
            sample=loader_sample,
            cache_dir=args.cache_dir,
        )
        preprocess_stats = {"bm25_time_s": 0.0}
        print(
            f"[Info] ProofWriter deductive reasoning (min_depth={os.environ.get('PROOFWRITER_MIN_DEPTH', '2')}, "
            f"target_tokens={tgt}): relevant theory padded with disjoint-entity distractor theories "
            f"(no BM25; reason_then_answer + EM/F1 over the True/False target)."
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    if dataset_name != "asqa":
        examples = _maybe_shuffle_and_sample_examples(examples, sample=args.sample, sample_seed=args.sample_seed)
    preprocess_time_s = time.perf_counter() - preprocess_start
    print(
        f"Preprocessing time: {preprocess_time_s:.2f}s | Retrieval build: {preprocess_stats.get('bm25_time_s', 0.0):.2f}s"
    )
    total_loaded_examples = len(examples)
    if args.pass_number:
        skipped_by_pass_number = min(args.pass_number, total_loaded_examples)
        examples = examples[skipped_by_pass_number:]
        print(
            f"[PassNumber] Skipping first {skipped_by_pass_number} of {total_loaded_examples} loaded examples; "
            f"remaining={len(examples)}."
        )

    out_path = None
    completed_output_ids: set[str] = set()
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if args.resume and out_path.exists():
            completed_output_ids = _load_jsonl_example_ids(out_path)
    elif args.run_alce_eval and dataset_name == "asqa":
        tmp_dir = Path("/tmp")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        out_path = tmp_dir / f"evidence_sketch_{args.dataset}_{args.modes[0]}.json"
    if out_path is not None and not (args.resume and out_path.exists()):
        out_path.write_text("", encoding="utf-8")

    teacher_trace_export_path = None
    completed_teacher_trace_ids: set[str] = set()
    if args.export_teacher_trace:
        teacher_trace_export_path = Path(args.export_teacher_trace)
        teacher_trace_export_path.parent.mkdir(parents=True, exist_ok=True)
        if args.resume and teacher_trace_export_path.exists():
            completed_teacher_trace_ids = set(_load_teacher_trace_lookup(teacher_trace_export_path).keys())
        else:
            teacher_trace_export_path.write_text("", encoding="utf-8")

    export_cache_path = None
    export_manifest_path = None
    export_samples: List[KLStepSample] = []
    export_example_rows: List[Dict[str, Any]] = []
    export_stats = SampleBuildStats()
    export_extra_stats = {"skipped_missing_context": 0, "skipped_doc_id_mismatch": 0}
    export_metadata: Optional[Dict[str, Any]] = None
    completed_export_ids: set[str] = set()
    if args.export_output_for_training:
        export_cache_path = Path(args.export_output_for_training)
        export_manifest_path = export_cache_path.with_suffix(export_cache_path.suffix + ".manifest.jsonl")
        export_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        if slm_model is None or lm_model is None:
            raise RuntimeError("Training export requires SLM and LM models to be loaded.")
        export_metadata = {
            "source": "run_evidence_sketch_experiments.py",
            "dataset": dataset_name,
            "slm_model": args.slm_model,
            "lm_model": args.lm_model,
            "split": args.split,
            "doc_number": args.doc_number,
            "retrieval_split": args.retrieval_split,
            "retriever": args.retriever,
            "max_corpus_examples": args.max_corpus_examples,
            "sample": args.sample,
            "sample_seed": args.sample_seed,
            "pass_number": args.pass_number,
            "max_length": args.max_length,
            "sketch_max_length": sketch_max_length,
            "sketch_max_new_tokens": args.sketch_max_new_tokens,
            "max_new_tokens": args.max_new_tokens,
            "gate_feature_type": args.export_gate_feature_type,
            "fusion_mode": "weighted_sum",
            "gate_top_k": args.export_gate_top_k,
            "loss_top_k": args.export_loss_top_k,
            "revert_previous_prompt": bool(args.revert_previous_prompt),
            "teacher_trace_input": str(Path(args.teacher_trace_input).resolve()) if args.teacher_trace_input else None,
            "num_layers": int(getattr(slm_model.config, "num_hidden_layers", 0) or 0),
            "num_heads": int(getattr(slm_model.config, "num_attention_heads", 0) or 0),
            "attn_feature_dim": 10 if args.export_gate_feature_type == "attn_stats_mlp_unweighted" else None,
        }
        if args.resume and export_cache_path.exists():
            cached_metadata, export_samples, export_stats, export_extra_stats, export_example_rows = _load_export_training_cache(export_cache_path)
            if cached_metadata is not None and cached_metadata != export_metadata:
                raise ValueError(
                    f"Existing export cache metadata mismatch for {export_cache_path}. "
                    "Use a new path or remove the old cache before resuming."
                )
            completed_export_ids = {
                str(row.get("example_id"))
                for row in export_example_rows
                if isinstance(row, dict) and isinstance(row.get("example_id"), (str, int))
            }
            if export_manifest_path.exists():
                completed_export_ids |= _load_jsonl_example_ids(export_manifest_path)
        elif export_manifest_path is not None:
            export_manifest_path.write_text("", encoding="utf-8")

    completed_example_ids: set[str] = set()
    if args.output:
        completed_example_ids |= completed_output_ids
    if args.export_teacher_trace:
        completed_example_ids |= completed_teacher_trace_ids
    if args.export_output_for_training:
        completed_example_ids |= completed_export_ids

    n = len(examples)
    progress_total = total_loaded_examples
    progress_start = min(args.pass_number, total_loaded_examples) + 1
    records = []
    running_metrics: Dict[str, Dict[str, float]] = {}
    running_timings: Dict[str, Dict[str, float]] = {}
    running_memory: Dict[str, Dict[str, float]] = {}
    tokenizer_for_stats = slm_tok or lm_tok or teacher_tok

    for idx, ex in enumerate(examples, start=progress_start):
        if args.resume and str(ex.example_id) in completed_example_ids:
            print(f"[Resume] Skipping completed example_id={ex.example_id}")
            continue
        stats = _context_stats(ex.documents, tokenizer_for_stats)
        print(
            f"[{idx}/{progress_total}] docs={stats['doc_count']} "
            f"tokens={stats['context_tokens']} words={stats['context_words']}"
        )
        record: Dict[str, Any] = {
            "example_id": ex.example_id,
            "question": ex.question,
            "answer": ex.answer,
            "documents": ex.documents,
            "doc_count": stats["doc_count"],
            "context_tokens": stats["context_tokens"],
            "context_words": stats["context_words"],
            # ★ PROVENANCE ON EVERY ROW, EVERY MODE (2026-08-13). The accumulate harness has recorded a
            # fingerprint since the `--bench` incident, but this one recorded `prompt_audit` on the TEACHER
            # path only — so a musique/hotpot FUSION log carried nothing, and whether four arms of a sweep
            # shared their settings could not be checked from the logs at all, only by reading the launcher
            # and trusting it. That is exactly the gap the fingerprint rule exists to close.
            "_provenance": {
                "dataset": dataset_name, "split": args.split, "doc_number": args.doc_number,
                "sample": args.sample, "sample_seed": getattr(args, "sample_seed", None),
                "max_new_tokens": args.max_new_tokens, "max_length": args.max_length,
                "lm_model": args.lm_model, "slm_model": args.slm_model,
                "lm_lora": args.lm_lora, "slm_lora": args.slm_lora,
                "lambda_weight": args.lambda_weight, "fusion_mode": args.fusion_mode,
                "answer_prompt_variant": getattr(args, "answer_prompt_variant", None),
                "use_generate": bool(getattr(args, "use_generate", False)),
            },
        }
        if dataset_name == "asqa":
            record["doc_meta"] = getattr(ex, "doc_meta", None)
            record["qa_pairs"] = getattr(ex, "qa_pairs", None)
            record["annotations"] = getattr(ex, "annotations", None)
            record["claims"] = getattr(ex, "claims", None)

        teacher_prompt = None
        teacher_prompt_audit = None
        teacher_trace: Optional[Dict[str, Any]] = None
        teacher_prompt_tok = teacher_tok or lm_tok
        if "teacher" in args.modes or args.export_output_for_training or args.export_teacher_trace:
            teacher_prompt, teacher_prompt_audit = _build_teacher_prompt_with_audit(
                ex,
                dataset=dataset_name,
                tokenizer=teacher_prompt_tok,
                max_length=args.max_length,
                asqa_prompt_version=args.asqa_prompt_version,
            )
        if "teacher" in args.modes:
            _begin_memory_measurement(args.log_memory)
            teacher_memory_trace: Dict[str, Any] = {}
            if dataset_name in ASQA_STYLE_DATASETS:
                teacher_text, teacher_timing = _generate_single(
                    teacher_model,
                    teacher_tok,
                    teacher_prompt,
                    max_new_tokens=args.max_new_tokens,
                    max_length=args.max_length,
                    stop_strings=args.stop_strings,
                    use_generate=use_generate,
                    decode_temperature=args.decode_temperature,
                    decode_top_p=args.decode_top_p,
                    decode_seed=args.decode_seed,
                    stop_on_repeat=(args.stop_on_repeat or None),
                    memory_trace=teacher_memory_trace if args.log_memory else None,
                )
                memory_snapshot = _finish_memory_measurement(args.log_memory, "teacher", running_memory, baseline=load_memory)
                teacher_phase_memory = _format_phase_memory_trace(teacher_memory_trace, load_memory)
                if teacher_phase_memory:
                    _print_phase_memory_trace("teacher", teacher_phase_memory)
                teacher_text = _strip_generation_artifacts(teacher_text)
                extracted, score_map = _score_asqa_prediction(teacher_text, ex)
                avg = _update_running(running_metrics, "teacher", score_map)
                _update_timing(
                    running_timings,
                    "teacher",
                    {
                        "prefill_s": teacher_timing.prefill_s,
                        "decode_s": teacher_timing.decode_s,
                        "total_s": teacher_timing.total_s,
                    },
                )
                print(
                    f"[Eval][teacher] {_format_score_map(score_map)} {_format_score_map(avg, prefix='avg_')} "
                    f"| {_format_current_avg_timing(running_timings, 'teacher', current_prefill_s=teacher_timing.prefill_s, current_decode_s=teacher_timing.decode_s, prefill_key='prefill_s', decode_key='decode_s')}"
                )
                record["teacher"] = {
                    "text": teacher_text,
                    "extracted": extracted,
                    "metrics": score_map,
                    "prompt_audit": teacher_prompt_audit,
                    "timing": {
                        "prefill_s": teacher_timing.prefill_s,
                        "decode_s": teacher_timing.decode_s,
                        "total_s": teacher_timing.total_s,
                    },
                }
                if memory_snapshot is not None:
                    record["teacher"]["memory"] = memory_snapshot
                if teacher_phase_memory:
                    record["teacher"]["phase_memory"] = teacher_phase_memory
            else:
                teacher_text, teacher_timing = generate_with_single_model(
                    teacher_model,
                    teacher_tok,
                    teacher_prompt,
                    max_new_tokens=args.max_new_tokens,
                    min_new_tokens=int(os.environ.get("MIN_NEW_TOKENS", "0")),
                    max_length=args.max_length,
                    use_generate=use_generate,
                    return_timing=True,
                    memory_trace=teacher_memory_trace if args.log_memory else None,
                    decode_temperature=args.decode_temperature,
                    decode_top_p=args.decode_top_p,
                    decode_seed=args.decode_seed,
                    stop_on_repeat=(args.stop_on_repeat or None),
                )
                memory_snapshot = _finish_memory_measurement(args.log_memory, "teacher", running_memory, baseline=load_memory)
                teacher_phase_memory = _format_phase_memory_trace(teacher_memory_trace, load_memory)
                if teacher_phase_memory:
                    _print_phase_memory_trace("teacher", teacher_phase_memory)
                extracted = _extract_final_answer_for_dataset(teacher_text, dataset_name)
                em, f1 = _score_qa_prediction(ex, extracted, dataset_name)
                score_map = {"em": em, "f1": f1}
                avg = _update_running(running_metrics, "teacher", score_map)
                _update_timing(
                    running_timings,
                    "teacher",
                    {
                        "prefill_s": teacher_timing.prefill_s,
                        "decode_s": teacher_timing.decode_s,
                        "total_s": teacher_timing.total_s,
                    },
                )
                print(
                    f"[Eval][teacher] gate_weight=none em={em:.4f} f1={f1:.4f} {_format_score_map(avg, prefix='avg_')} "
                    f"| {_format_current_avg_timing(running_timings, 'teacher', current_prefill_s=teacher_timing.prefill_s, current_decode_s=teacher_timing.decode_s, prefill_key='prefill_s', decode_key='decode_s')}"
                )
                record["teacher"] = {
                    "text": teacher_text,
                    "extracted": extracted,
                    "em": em,
                    "f1": f1,
                    "prompt_audit": teacher_prompt_audit,
                    "timing": {
                        "prefill_s": teacher_timing.prefill_s,
                        "decode_s": teacher_timing.decode_s,
                        "total_s": teacher_timing.total_s,
                    },
                }
                if memory_snapshot is not None:
                    record["teacher"]["memory"] = memory_snapshot
                if teacher_phase_memory:
                    record["teacher"]["phase_memory"] = teacher_phase_memory
            if args.debug or args.teacher_prompt_audit:
                print(
                    "[TeacherPromptAudit] "
                    f"sha256={teacher_prompt_audit.get('prompt_sha256')} "
                    f"full_tokens={teacher_prompt_audit.get('full_prompt_tokens')} "
                    f"final_tokens={teacher_prompt_audit.get('final_prompt_tokens')} "
                    f"truncated={teacher_prompt_audit.get('truncated')}"
                )
            if args.debug or args.teacher_prompt_audit:
                _print_debug_text("Teacher Prompt Audit", json.dumps(teacher_prompt_audit, indent=2))
            if args.debug:
                _print_debug_answer("Teacher", ex.question, teacher_text, extracted, ex.answer)

        if "blockattention" in args.modes:
            _begin_memory_measurement(args.log_memory)
            block_memory_trace: Dict[str, Any] = {}
            block_out = _run_blockattention_baseline(
                lm_model,
                lm_tok,
                ex,
                dataset=dataset_name,
                max_new_tokens=args.max_new_tokens,
                max_length=args.max_length,
                debug=args.debug,
                use_generate=use_generate,
                mode_label="BlockAttention",
                use_pcw_positions=False,
                memory_trace=block_memory_trace if args.log_memory else None,
                decode_temperature=args.decode_temperature,
                decode_top_p=args.decode_top_p,
                decode_seed=args.decode_seed,
                stop_on_repeat=(args.stop_on_repeat or None),
            )
            memory_snapshot = _finish_memory_measurement(args.log_memory, "blockattention", running_memory, baseline=load_memory)
            block_phase_memory = _format_phase_memory_trace(block_memory_trace, load_memory)
            if block_phase_memory:
                _print_phase_memory_trace("blockattention", block_phase_memory)
            score_map = {"em": block_out["em"], "f1": block_out["f1"]}
            avg = _update_running(running_metrics, "blockattention", score_map)
            _update_timing(running_timings, "blockattention", block_out.get("timing", {}))
            timing = block_out.get("timing", {})
            print(
                f"[Eval][blockattention] gate_weight=none em={block_out['em']:.4f} f1={block_out['f1']:.4f} "
                f"{_format_score_map(avg, prefix='avg_')} "
                f"| {_format_current_avg_timing(running_timings, 'blockattention', current_prefill_s=timing.get('prefill_s'), current_decode_s=timing.get('decode_s'), prefill_key='prefill_s', decode_key='decode_s')}"
            )
            record["blockattention"] = block_out
            if memory_snapshot is not None:
                record["blockattention"]["memory"] = memory_snapshot
            if block_phase_memory:
                record["blockattention"]["phase_memory"] = block_phase_memory

        if "pcw" in args.modes:
            _begin_memory_measurement(args.log_memory)
            pcw_memory_trace: Dict[str, Any] = {}
            pcw_out = _run_blockattention_baseline(
                lm_model,
                lm_tok,
                ex,
                dataset=dataset_name,
                max_new_tokens=args.max_new_tokens,
                max_length=args.max_length,
                debug=args.debug,
                use_generate=use_generate,
                mode_label="PCW",
                use_pcw_positions=True,
                memory_trace=pcw_memory_trace if args.log_memory else None,
                decode_temperature=args.decode_temperature,
                decode_top_p=args.decode_top_p,
                decode_seed=args.decode_seed,
                stop_on_repeat=(args.stop_on_repeat or None),
            )
            memory_snapshot = _finish_memory_measurement(args.log_memory, "pcw", running_memory, baseline=load_memory)
            pcw_phase_memory = _format_phase_memory_trace(pcw_memory_trace, load_memory)
            if pcw_phase_memory:
                _print_phase_memory_trace("pcw", pcw_phase_memory)
            score_map = {"em": pcw_out["em"], "f1": pcw_out["f1"]}
            avg = _update_running(running_metrics, "pcw", score_map)
            _update_timing(running_timings, "pcw", pcw_out.get("timing", {}))
            timing = pcw_out.get("timing", {})
            print(
                f"[Eval][pcw] gate_weight=none em={pcw_out['em']:.4f} f1={pcw_out['f1']:.4f} "
                f"{_format_score_map(avg, prefix='avg_')} "
                f"| {_format_current_avg_timing(running_timings, 'pcw', current_prefill_s=timing.get('prefill_s'), current_decode_s=timing.get('decode_s'), prefill_key='prefill_s', decode_key='decode_s')}"
            )
            record["pcw"] = pcw_out
            if memory_snapshot is not None:
                record["pcw"]["memory"] = memory_snapshot
            if pcw_phase_memory:
                record["pcw"]["phase_memory"] = pcw_phase_memory

        if "pced" in args.modes and pced_decoder is not None:
            _begin_memory_measurement(args.log_memory)
            pced_out = _run_pced_baseline(
                pced_decoder,
                ex,
                dataset=dataset_name,
                debug=args.debug,
            )
            memory_snapshot = _finish_memory_measurement(args.log_memory, "pced", running_memory, baseline=load_memory)
            if dataset_name in ASQA_STYLE_DATASETS:
                score_map = pced_out["metrics"]
                avg = _update_running(running_metrics, "pced", score_map)
                _update_timing(running_timings, "pced", pced_out.get("timing", {}))
                timing = pced_out.get("timing", {})
                print(
                    f"[Eval][pced] gate_weight=none {_format_score_map(score_map)} {_format_score_map(avg, prefix='avg_')} "
                    f"| {_format_current_avg_timing(running_timings, 'pced', current_prefill_s=timing.get('prefill_s'), current_decode_s=timing.get('decode_s'), prefill_key='prefill_s', decode_key='decode_s')}"
                )
            else:
                score_map = {"em": pced_out["em"], "f1": pced_out["f1"]}
                avg = _update_running(running_metrics, "pced", score_map)
                _update_timing(running_timings, "pced", pced_out.get("timing", {}))
                timing = pced_out.get("timing", {})
                print(
                    f"[Eval][pced] gate_weight=none em={pced_out['em']:.4f} f1={pced_out['f1']:.4f} "
                    f"{_format_score_map(avg, prefix='avg_')} "
                    f"| {_format_current_avg_timing(running_timings, 'pced', current_prefill_s=timing.get('prefill_s'), current_decode_s=timing.get('decode_s'), prefill_key='prefill_s', decode_key='decode_s')}"
                )
            record["pced"] = pced_out
            if memory_snapshot is not None:
                record["pced"]["memory"] = memory_snapshot

        if teacher_trace_export_path is not None or args.export_output_for_training:
            if args.teacher_trace_input:
                teacher_trace_lookup = _maybe_refresh_teacher_trace_lookup(
                    teacher_trace_input_path,
                    teacher_trace_lookup,
                    expected_example_id=str(ex.example_id),
                )
                teacher_trace = teacher_trace_lookup.get(str(ex.example_id))
                if teacher_trace is None:
                    export_extra_stats["skipped_missing_context"] += 1
            else:
                if teacher_prompt is None:
                    teacher_prompt, teacher_prompt_audit = _build_teacher_prompt_with_audit(
                        ex,
                        dataset=dataset_name,
                        tokenizer=teacher_tok,
                        max_length=args.max_length,
                        asqa_prompt_version=args.asqa_prompt_version,
                    )
                teacher_trace = _collect_teacher_trace(
                    teacher_model,
                    teacher_tok,
                    teacher_prompt,
                    max_length=args.max_length,
                    max_new_tokens=args.max_new_tokens,
                    top_k=args.export_loss_top_k,
                    stop_strings=args.stop_strings,
                )
                teacher_trace_row = {
                    "example_id": str(ex.example_id),
                    "question": ex.question,
                    "teacher_prompt_sha256": _prompt_sha256(teacher_prompt),
                    "text": teacher_trace["text"],
                    "token_ids": teacher_trace["token_ids"],
                    "steps": teacher_trace["steps"],
                }
                if teacher_trace_export_path is not None:
                    _append_jsonl(teacher_trace_export_path, teacher_trace_row)

        if "slm_lm" in args.modes and decoder is not None:
            _begin_memory_measurement(args.log_memory)
            fused_run = _run_evidence_sketch_fusion(
                decoder,
                ex,
                dataset=dataset_name,
                max_length=args.max_length,
                sketch_max_length=sketch_max_length,
                sketch_max_new_tokens=args.sketch_max_new_tokens,
                sketch_stop_strings=args.sketch_stop_strings,
                asqa_prompt_version=args.asqa_prompt_version,
                measure_memory=args.log_memory,
                memory_baseline=load_memory,
            )
            memory_snapshot = _finish_memory_measurement(args.log_memory, "slm_lm", running_memory, baseline=load_memory)
            slm_lm_phase_memory = fused_run.get("phase_memory", {})
            if slm_lm_phase_memory:
                _print_phase_memory_trace("slm_lm", slm_lm_phase_memory)
            sketch_text = fused_run["evidence_sketch"]
            lm_prompt_with_sketch = fused_run["lm_prompt_with_sketch"]
            fused_text = fused_run["fused_text"]
            slm_debug = fused_run["slm_debug"]
            sketch_timing = fused_run["sketch_timing"]
            fusion_timing = fused_run["fusion_timing"]
            total_pipeline_s = sketch_timing.total_s + fusion_timing.total_s
            if dataset_name in ASQA_STYLE_DATASETS:
                extracted, score_map = _score_asqa_prediction(fused_text, ex)
                avg = _update_running(running_metrics, "slm_lm", score_map)
                _update_timing(
                    running_timings,
                    "slm_lm",
                    {
                        "sketch_prefill_s": sketch_timing.prefill_s if sketch_timing else 0.0,
                        "sketch_decode_s": sketch_timing.decode_s if sketch_timing else 0.0,
                        "sketch_total_s": sketch_timing.total_s if sketch_timing else 0.0,
                        "fusion_prefill_s": fusion_timing.prefill_s if fusion_timing else 0.0,
                        "fusion_decode_s": fusion_timing.decode_s if fusion_timing else 0.0,
                        "fusion_total_s": fusion_timing.total_s if fusion_timing else 0.0,
                        "pipeline_total_s": total_pipeline_s,
                    },
                )
                print(
                    f"[Eval][slm_lm] gate_weight={_mode_gate_weight_name('slm_lm', args)} "
                    f"{_format_score_map(score_map)} {_format_score_map(avg, prefix='avg_')} "
                    f"| {_format_current_avg_timing(running_timings, 'slm_lm', current_prefill_s=(fusion_timing.prefill_s if fusion_timing else None), current_decode_s=(fusion_timing.decode_s if fusion_timing else None), prefill_key='fusion_prefill_s', decode_key='fusion_decode_s')}"
                )
                record["slm_lm"] = {
                    "evidence_sketch": sketch_text,
                    "text": fused_text,
                    "extracted": extracted,
                    "metrics": score_map,
                        "timing": {
                            "sketch_prefill_s": sketch_timing.prefill_s if sketch_timing else None,
                            "sketch_decode_s": sketch_timing.decode_s if sketch_timing else None,
                            "sketch_total_s": sketch_timing.total_s if sketch_timing else None,
                            "fusion_prefill_s": fusion_timing.prefill_s if fusion_timing else None,
                            "fusion_decode_s": fusion_timing.decode_s if fusion_timing else None,
                            "fusion_total_s": fusion_timing.total_s if fusion_timing else None,
                            "pipeline_total_s": total_pipeline_s,
                        },
                        "slm_debug": slm_debug,
                        "terminated": fused_run["terminated"],
                    }
                if memory_snapshot is not None:
                    record["slm_lm"]["memory"] = memory_snapshot
                if slm_lm_phase_memory:
                    record["slm_lm"]["phase_memory"] = slm_lm_phase_memory
            else:
                extracted = _extract_final_answer_for_dataset(fused_text, dataset_name)
                em, f1 = _score_qa_prediction(ex, extracted, dataset_name)
                score_map = {"em": em, "f1": f1}
                avg = _update_running(running_metrics, "slm_lm", score_map)
                _update_timing(
                    running_timings,
                    "slm_lm",
                    {
                        "sketch_prefill_s": sketch_timing.prefill_s if sketch_timing else 0.0,
                        "sketch_decode_s": sketch_timing.decode_s if sketch_timing else 0.0,
                        "sketch_total_s": sketch_timing.total_s if sketch_timing else 0.0,
                        "fusion_prefill_s": fusion_timing.prefill_s if fusion_timing else 0.0,
                        "fusion_decode_s": fusion_timing.decode_s if fusion_timing else 0.0,
                        "fusion_total_s": fusion_timing.total_s if fusion_timing else 0.0,
                        "pipeline_total_s": total_pipeline_s,
                    },
                )
                print(
                    f"[Eval][slm_lm] gate_weight={_mode_gate_weight_name('slm_lm', args)} "
                    f"em={em:.4f} f1={f1:.4f} {_format_score_map(avg, prefix='avg_')} "
                    f"| {_format_current_avg_timing(running_timings, 'slm_lm', current_prefill_s=(fusion_timing.prefill_s if fusion_timing else None), current_decode_s=(fusion_timing.decode_s if fusion_timing else None), prefill_key='fusion_prefill_s', decode_key='fusion_decode_s')}"
                )
                record["slm_lm"] = {
                    "evidence_sketch": sketch_text,
                    "text": fused_text,
                    "extracted": extracted,
                    "em": em,
                    "f1": f1,
                        "timing": {
                            "sketch_prefill_s": sketch_timing.prefill_s if sketch_timing else None,
                            "sketch_decode_s": sketch_timing.decode_s if sketch_timing else None,
                            "sketch_total_s": sketch_timing.total_s if sketch_timing else None,
                            "fusion_prefill_s": fusion_timing.prefill_s if fusion_timing else None,
                            "fusion_decode_s": fusion_timing.decode_s if fusion_timing else None,
                            "fusion_total_s": fusion_timing.total_s if fusion_timing else None,
                            "pipeline_total_s": total_pipeline_s,
                        },
                        "slm_debug": slm_debug,
                        "terminated": fused_run["terminated"],
                    }
                if memory_snapshot is not None:
                    record["slm_lm"]["memory"] = memory_snapshot
                if slm_lm_phase_memory:
                    record["slm_lm"]["phase_memory"] = slm_lm_phase_memory
            if args.debug:
                _print_debug_text("Evidence Sketch", sketch_text)
                _print_debug_text("LM Prompt With Sketch", lm_prompt_with_sketch)
                _print_debug_text("SLM Debug", json.dumps(slm_debug, indent=2))
                _print_debug_answer("SLM+LM", ex.question, fused_text, extracted, ex.answer)

        if "quantized_lm_lm" in args.modes and quantized_decoder is not None:
            mode_name = "quantized_lm_lm"
            _begin_memory_measurement(args.log_memory)
            fused_run = _run_evidence_sketch_fusion(
                quantized_decoder,
                ex,
                dataset=dataset_name,
                max_length=args.max_length,
                sketch_max_length=sketch_max_length,
                sketch_max_new_tokens=args.sketch_max_new_tokens,
                sketch_stop_strings=args.sketch_stop_strings,
                asqa_prompt_version=args.asqa_prompt_version,
                measure_memory=args.log_memory,
                memory_baseline=load_memory,
            )
            memory_snapshot = _finish_memory_measurement(args.log_memory, mode_name, running_memory, baseline=load_memory)
            quant_phase_memory = fused_run.get("phase_memory", {})
            if quant_phase_memory:
                _print_phase_memory_trace(mode_name, quant_phase_memory)
            sketch_text = fused_run["evidence_sketch"]
            lm_prompt_with_sketch = fused_run["lm_prompt_with_sketch"]
            fused_text = fused_run["fused_text"]
            context_lm_debug = fused_run["slm_debug"]
            sketch_timing = fused_run["sketch_timing"]
            fusion_timing = fused_run["fusion_timing"]
            total_pipeline_s = sketch_timing.total_s + fusion_timing.total_s
            timing_payload = {
                "sketch_prefill_s": sketch_timing.prefill_s if sketch_timing else None,
                "sketch_decode_s": sketch_timing.decode_s if sketch_timing else None,
                "sketch_total_s": sketch_timing.total_s if sketch_timing else None,
                "fusion_prefill_s": fusion_timing.prefill_s if fusion_timing else None,
                "fusion_decode_s": fusion_timing.decode_s if fusion_timing else None,
                "fusion_total_s": fusion_timing.total_s if fusion_timing else None,
                "pipeline_total_s": total_pipeline_s,
            }
            _update_timing(
                running_timings,
                mode_name,
                {key: value or 0.0 for key, value in timing_payload.items()},
            )
            if dataset_name in ASQA_STYLE_DATASETS:
                extracted, score_map = _score_asqa_prediction(fused_text, ex)
                avg = _update_running(running_metrics, mode_name, score_map)
                print(
                    f"[Eval][{mode_name}] gate_weight={_mode_gate_weight_name(mode_name, args)} "
                    f"{_format_score_map(score_map)} {_format_score_map(avg, prefix='avg_')} "
                    f"| {_format_current_avg_timing(running_timings, mode_name, current_prefill_s=(fusion_timing.prefill_s if fusion_timing else None), current_decode_s=(fusion_timing.decode_s if fusion_timing else None), prefill_key='fusion_prefill_s', decode_key='fusion_decode_s')}"
                )
                record[mode_name] = {
                    "evidence_sketch": sketch_text,
                    "text": fused_text,
                    "extracted": extracted,
                    "metrics": score_map,
                    "timing": timing_payload,
                    "context_lm_debug": context_lm_debug,
                    "terminated": fused_run["terminated"],
                    "quantized_lm": {
                        "model": args.quantized_lm_model or args.lm_model,
                        "bits": args.quantized_lm_bits,
                    },
                }
            else:
                extracted = _extract_final_answer_for_dataset(fused_text, dataset_name)
                em, f1 = _score_qa_prediction(ex, extracted, dataset_name)
                score_map = {"em": em, "f1": f1}
                avg = _update_running(running_metrics, mode_name, score_map)
                print(
                    f"[Eval][{mode_name}] gate_weight={_mode_gate_weight_name(mode_name, args)} "
                    f"em={em:.4f} f1={f1:.4f} {_format_score_map(avg, prefix='avg_')} "
                    f"| {_format_current_avg_timing(running_timings, mode_name, current_prefill_s=(fusion_timing.prefill_s if fusion_timing else None), current_decode_s=(fusion_timing.decode_s if fusion_timing else None), prefill_key='fusion_prefill_s', decode_key='fusion_decode_s')}"
                )
                record[mode_name] = {
                    "evidence_sketch": sketch_text,
                    "text": fused_text,
                    "extracted": extracted,
                    "em": em,
                    "f1": f1,
                    "timing": timing_payload,
                    "context_lm_debug": context_lm_debug,
                    "terminated": fused_run["terminated"],
                    "quantized_lm": {
                        "model": args.quantized_lm_model or args.lm_model,
                        "bits": args.quantized_lm_bits,
                    },
                }
            if memory_snapshot is not None:
                record[mode_name]["memory"] = memory_snapshot
            if quant_phase_memory:
                record[mode_name]["phase_memory"] = quant_phase_memory
            if args.debug:
                _print_debug_text("Quantized LM Evidence Sketch", sketch_text)
                _print_debug_text("Quantized LM Prompt With Sketch", lm_prompt_with_sketch)
                _print_debug_text("Quantized LM Debug", json.dumps(context_lm_debug, indent=2))
                _print_debug_answer("Quantized LM+LM", ex.question, fused_text, extracted, ex.answer)

        if args.export_output_for_training:
            export_stats.examples_seen += 1
            if teacher_trace is None:
                if export_cache_path is not None and export_metadata is not None:
                    _save_export_training_cache(
                        export_cache_path,
                        metadata=export_metadata,
                        samples=export_samples,
                        stats=export_stats,
                        extra_stats=export_extra_stats,
                        example_rows=export_example_rows,
                    )
                continue
            export_run = _run_evidence_sketch_fusion(
                decoder,
                ex,
                dataset=dataset_name,
                max_length=args.max_length,
                sketch_max_length=sketch_max_length,
                sketch_max_new_tokens=args.sketch_max_new_tokens,
                sketch_stop_strings=args.sketch_stop_strings,
                asqa_prompt_version=args.asqa_prompt_version,
                training_trace=teacher_trace,
                export_gate_feature_type=args.export_gate_feature_type,
                export_gate_top_k=args.export_gate_top_k,
                export_loss_top_k=args.export_loss_top_k,
            )
            exported_samples_for_example = list(export_run.get("exported_samples", []))
            export_samples.extend(exported_samples_for_example)
            export_stats.steps_used += len(exported_samples_for_example)
            export_stats.skipped_empty_teacher += max(0, len(teacher_trace["steps"]) - len(exported_samples_for_example))
            if exported_samples_for_example:
                export_stats.examples_used += 1
            export_row = {
                "example_id": ex.example_id,
                "question": ex.question,
                "teacher_steps": len(teacher_trace["steps"]),
                "exported_steps": len(exported_samples_for_example),
                "teacher_text": teacher_trace["text"],
                "evidence_sketch": export_run.get("evidence_sketch"),
                "teacher_prompt_sha256": _prompt_sha256(teacher_prompt),
            }
            export_example_rows.append(export_row)
            if export_manifest_path is not None:
                _append_jsonl(export_manifest_path, export_row)
            if export_cache_path is not None and export_metadata is not None:
                _save_export_training_cache(
                    export_cache_path,
                    metadata=export_metadata,
                    samples=export_samples,
                    stats=export_stats,
                    extra_stats=export_extra_stats,
                    example_rows=export_example_rows,
                )

        records.append(record)
        if out_path is not None:
            _append_jsonl(out_path, record)

    summary: Dict[str, Any] = {
        "examples": n,
        "total_loaded_examples": total_loaded_examples,
        "pass_number": args.pass_number,
        "sample_seed": args.sample_seed,
    }
    if "quantized_lm_lm" in args.modes:
        summary["quantized_lm_lm"] = {
            "quantized_lm_model": args.quantized_lm_model or args.lm_model,
            "lm_model": args.lm_model,
            "bits": args.quantized_lm_bits,
            "bnb_4bit_quant_type": args.quantized_lm_4bit_quant_type if args.quantized_lm_bits == 4 else None,
            "bnb_4bit_compute_dtype": args.quantized_lm_4bit_compute_dtype if args.quantized_lm_bits == 4 else None,
            "bnb_4bit_use_double_quant": bool(args.quantized_lm_4bit_use_double_quant)
            if args.quantized_lm_bits == 4
            else None,
        }
    if running_metrics:
        summary["metrics"] = {
            mode: {
                key[: -len("_sum")]: value / vals["count"]
                for key, value in vals.items()
                if key.endswith("_sum")
            }
            for mode, vals in running_metrics.items()
            if vals.get("count", 0) > 0
        }
    if running_timings:
        summary["timing"] = {
            mode: {
                key[: -len("_sum")]: value / vals["count"]
                for key, value in vals.items()
                if key.endswith("_sum")
            }
            for mode, vals in running_timings.items()
            if vals.get("count", 0) > 0
        }
    if load_memory is not None:
        summary["memory_after_load"] = load_memory
    if running_memory:
        summary["memory"] = _memory_summary(running_memory)
    if dataset_name == "scifact":
        scifact_summary = {}
        for mode in args.modes:
            preds = [
                str(rec.get(mode, {}).get("extracted", ""))
                for rec in records
                if isinstance(rec.get(mode), dict)
            ]
            golds = [
                str(rec.get("answer", ""))
                for rec in records
                if isinstance(rec.get(mode), dict)
            ]
            scifact_summary[mode] = compute_scifact_label_metrics(preds, golds)
        summary["scifact_label_metrics"] = scifact_summary
    if args.export_output_for_training and export_cache_path is not None:
        if export_metadata is not None:
            _save_export_training_cache(
                export_cache_path,
                metadata=export_metadata,
                samples=export_samples,
                stats=export_stats,
                extra_stats=export_extra_stats,
                example_rows=export_example_rows,
            )
        summary["training_export"] = {
            "cache_path": str(export_cache_path),
            "manifest_path": str(export_manifest_path) if export_manifest_path is not None else None,
            "examples_seen": int(export_stats.examples_seen),
            "examples_used": int(export_stats.examples_used),
            "steps_used": int(export_stats.steps_used),
            "skipped_empty_teacher": int(export_stats.skipped_empty_teacher),
        }
    if teacher_trace_export_path is not None:
        summary["teacher_trace_export"] = {
            "path": str(teacher_trace_export_path),
            "source": "live_teacher_model",
        }

    if out_path is not None:
        _append_jsonl(out_path, {"summary": summary})
        if args.output:
            print("Summary written to", out_path)
    if args.export_output_for_training and export_cache_path is not None:
        print(f"Training export cache written to: {export_cache_path}")
        if export_manifest_path is not None:
            print(f"Training export manifest written to: {export_manifest_path}")
    if teacher_trace_export_path is not None:
        print(f"Teacher trace export written to: {teacher_trace_export_path}")
    print(json.dumps(summary, indent=2))

    if args.run_alce_eval and dataset_name == "asqa":
        if out_path is None:
            print("[ALCE eval] skipped: no output path available.")
            return
        eval_mode = args.alce_eval_mode or (args.modes[0] if args.modes else "teacher")
        eval_data_file = args.alce_data_file or str(data_path)
        eval_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_alce_eval.py"),
            "--preds",
            str(out_path),
            "--dataset",
            args.dataset,
            "--mode",
            eval_mode,
            "--alce-dir",
            args.alce_dir,
            "--data-file",
            eval_data_file,
        ]
        if args.skip_autoais:
            eval_cmd.append("--skip-autoais")
        if args.skip_mauve:
            eval_cmd.append("--skip-mauve")
        try:
            subprocess.run(eval_cmd, check=False)
        except Exception as exc:  # noqa: BLE001
            print(f"[ALCE eval] failed to launch: {exc}")


if __name__ == "__main__":
    main()
