#!/usr/bin/env python
"""results/timing/prompts.json — the exact prompt strings the four benchmarks run on, pulled from the code
so the appendix page never carries a typed-in prompt (user, 2026-09-19: "우리 prompt도 한 페이지 정도로
정리해두는게 좋을듯. 논문 appendix에 넣어야 할지도").

Sources: scripts/bench_config.py (which constant each benchmark pins), src/qa_prompts.py (the constants),
scripts/mtrag_accum.py (how a turn is assembled), scripts/reason_fix.py (the per-question reminder) and
scripts/chat_wrap.py (the per-family turn markers).
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ANSWER_PROMPT_VARIANT", "reason_v3")

import src.qa_prompts as QP                                            # noqa: E402
import scripts.bench_config as BC                                      # noqa: E402
import scripts.mtrag_accum as M                                        # noqa: E402
from scripts.mab_eval import q_turn as q_turn_terse, ref_answer        # noqa: E402
from src.qa_prompts import build_full_context_qa_prompt                # noqa: E402

BENCH = {"hotpotQA": "hotpotqa_st40_full", "MuSiQue": "musique_st40_full",
         "LooGLE": "loogle", "LoCoMo": "locomo"}
WRAPS = {  # scripts/chat_wrap.py, one definition shared by eval, the teacher generator and both trainers
    "Qwen2.5 (CHAT_WRAP=0)": ("", "", ""),
    "OLMo-3 (CHAT_WRAP=1)": ("<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n",
                             "<|im_end|>\n<|im_start|>user\n"),
    "Gemma-3 (CHAT_WRAP=gemma)": ("<bos><start_of_turn>user\n",
                                  "<end_of_turn>\n<start_of_turn>model\n",
                                  "<end_of_turn>\n<start_of_turn>user\n")}


def main():
    sha = lambda s: hashlib.sha256(s.encode()).hexdigest()[:16]
    out = {"benchmarks": {}, "chat_wrap": {k: dict(open=a, assistant=b, reopen=c)
                                           for k, (a, b, c) in WRAPS.items()}}
    for name, key in BENCH.items():
        b = BC.BENCHMARKS[key]
        txt = getattr(QP, b["instruction"])
        out["benchmarks"][name] = dict(bench_key=key, instruction_const=b["instruction"], instruction=txt,
                                       instruction_sha=sha(txt), metric=b["metric"], max_new=b["max_new"],
                                       ref=os.path.basename(str(b["ref"])), max_conv=b.get("max_conv"),
                                       order=b.get("order"), seed=b.get("seed"),
                                       doc_number=b.get("doc_number"), family=b.get("family"))
    out["question_turn"] = q_turn_terse("<the question>")
    out["history_block"] = q_turn_terse("<an earlier question>") + ref_answer("<its reference answer>")
    out["passage_block"] = M.fmt([{"text": "<passage 1>"}, {"text": "<passage 2>"}])
    out["assembly"] = ("The prompt is the instruction, then a blank line, then the passage block, then "
                       "the question turn. When the benchmark accumulates, each answered turn is "
                       "committed as a history block and the next turn's new passages and question "
                       "follow it in the same sequence.")
    p = build_full_context_qa_prompt("<the question>", [], prompt_variant="reason_v3")
    out["branch_kl_prompt"] = p
    out["branch_kl_sha"] = sha(p)
    out["_source"] = ("scripts/dump_prompts.py, from bench_config / qa_prompts / mtrag_accum / reason_fix "
                      "/ chat_wrap: the strings the harnesses build, not a transcription")
    os.makedirs("results/timing", exist_ok=True)
    json.dump(out, open("results/timing/prompts.json.tmp", "w"), indent=1)
    os.replace("results/timing/prompts.json.tmp", "results/timing/prompts.json")
    for n, d in out["benchmarks"].items():
        print(f"  {n:10s} {d['instruction_const']:24s} sha {d['instruction_sha']}  {d['ref']}")
    print("wrote results/timing/prompts.json")


if __name__ == "__main__":
    main()
