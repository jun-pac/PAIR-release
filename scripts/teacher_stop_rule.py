#!/usr/bin/env python
"""Apply the teacher's DEPLOYED stop rule to a generated target (2026-08-10).

`binding_teacher_gen.py` originally called `generate(max_new_tokens=200)` with no stop criteria, but every
eval harness runs the same teacher with `--stop-after-answer "Final Answer:"`. That mismatch matters: the
14B answers correctly and then LOOPS the whole "Step 1 … Final Answer: X" block until it hits max_new, so
the usability gate discarded **2,263/5,400 = 42%** of its targets — 100% of which had already emitted a
complete `Final Answer:`. (The 32B does it on 120/5,400 = 2%.) Cloning the teacher means cloning what it
emits UNDER ITS DEPLOYED DECODING, so the target is the text up to the end of its first Final-Answer line.

This is not curation: nothing is selected on correctness, and a row with no answer at all still fails the
usability gate.
"""
import re

_FA = re.compile(r"Final Answer:", re.IGNORECASE)


def apply_stop_after_answer(text: str):
    """(target, terminated). terminated=False iff no complete Final-Answer line was produced."""
    text = (text or "").strip()
    if not text:
        return "", False
    m = _FA.search(text)
    if not m:
        return text, False
    tail = text[m.end():]
    nl = tail.find("\n")
    line = tail if nl < 0 else tail[:nl]
    # The model may continue on the SAME LINE with a restarted scaffold OR a new chat turn. The original
    # pattern only covered "Step N (" / "Reasoning:", so 1,500 of the 2,000 replay targets in v5 (75%) kept
    # text like `Final Answer: 21Human: You are given context and a question...` — the teacher answered, did
    # not stop, and began a fresh turn. The DEPLOYED teacher stops at the marker and never emits that, so
    # cloning it trains the reader on something the teacher does not do in deployment. This is a defect in
    # the offline re-implementation of the stop rule, not teacher behaviour worth cloning.
    # ★ `Final Answer:` itself belongs in this alternation (2026-08-12). The teacher answers, does not
    # stop, and simply REPEATS the answer line: `Final Answer: search incident Final Answer: search
    # incident`. None of the patterns above match that, so 183 of 2,016 hard-retrieval targets (9.1%) and
    # 38 of 168 k=0 targets (22.6%) kept the duplicate. The deployed teacher stops at the FIRST marker and
    # never emits the second, so cloning it teaches a doubled answer the teacher does not produce.
    r = re.search(r"\s*(?:Step\s*\d+\s*[\(:]|\bReasoning\s*:|Human\s*:|Assistant\s*:|<\|im_(start|end)\|>"
                  r"|You are given context and a question|\bQuestion\s*:|Final\s*Answer\s*:)",
                  line, re.IGNORECASE)
    if r and r.start() > 0:
        line = line[: r.start()]
    return text[: m.end()] + line.rstrip(), True
