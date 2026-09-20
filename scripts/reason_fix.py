#!/usr/bin/env python
"""Shared reasoning-bug fix (2026-07-17). The prior reason_then_answer setup silently produced NO reasoning
(97% straight-to-'Final Answer:', literal '<one-line reasoning>' echoed) -> the SLM+LM fusion was starved and
collapsed to the SLM floor. This module makes reasoning STRUCTURALLY guaranteed and is imported by every harness
(mab_eval, mtrag_accum, eventqa_rawdump) so all methods share the identical, fair generation contract.

Fix = (a) prompt with no echoable <...> placeholder (qa_prompts.QA_REASON_V2*), (b) a teacher-forced 'Reasoning:'
cue in q_turn, (c) BLOCK the 'Final'/' Final' tokens for the first MIN_REASON generated steps so real reasoning
tokens must be produced first, (d) stop on chat/role leak markers, (e) strip datasets' 'answer only' overrides.
Enable per-harness via env REASON_FIX=1. Records provenance so validate_experiment.py can fairness-check."""
import hashlib

BLOCK_FINAL = [19357, 13023]      # 'Final', ' Final' (first token of 'Final Answer'; identical in Qwen2.5 14B & 3B)
BLOCK_LEAK = [33975, 11097, 71703, 21388]  # 'Human',' Human','Assistant',' Assistant' — never let the model escape to a chat turn
MIN_REASON = 24                    # force >= this many non-'Final Answer' tokens (the reasoning)
LEAK = ("Human:", "<|im_end|>", "<|im_start|>", "\nassistant", "\nuser", "\n\nQuestion:")
STRIP = ("In your response to me, only include the answer without anything else.",
         "In your response to me, only include the answer without anything else",)

def clean_q(q):
    q = str(q or "")
    for s in STRIP: q = q.replace(s, "")
    return q.strip()

# Instruction-at-END (2026-07-17, user's insight): the top instruction is buried behind the 66k/22.9k context and,
# by turn 2+, behind the accumulated Q/A history — long-context models attend to what's RECENT, so the far-away
# "reason first" instruction is ignored (turn 1 reasons ~60%, turn 2+ ~0%). Putting the reasoning instruction RIGHT
# BEFORE each question (most salient position) is the standard long-context CoT fix — a SETTING, not coercion.
REASON_REMINDER = ("Answer by reasoning first: in 1-2 sentences, use the context above to work out the answer "
                   "(say which turn/fact and, for a 'when' question, which session date). Then on a new line write "
                   "'Final Answer:' followed by the short answer.")

def q_turn_reason(q):
    return f"\n\nQuestion: {clean_q(q)}\n{REASON_REMINDER}\n"

def block_logits(logits, step, min_reason=MIN_REASON):
    """NO-OP now (natural reasoning; coercive 'Final'-token blocking was removed — it degraded answers).
    Kept for harness call-site compatibility; leaks are handled post-hoc by reason_stopped/trim_leak."""
    return logits

import re as _re
_FA = _re.compile(r"final\s*answer\s*:\s*", _re.I)
# answer ends at: newline / a new 'Step N' / '(Reasoning' / 'Question:' / a REPEATED 'Final Answer' / 'Note:'
# (models ramble AFTER the answer in model-specific ways: 14B -> 'Final Answer: X Step 1 (Reasoning)...',
#  3B -> 'Final Answer: X. Final Answer: X.' repeat. This single boundary handles both.)
_ANS_BOUND = _re.compile(r"(?:\n|step\s*\d|\(reasoning|reasoning\)|question\s*:|final\s*answer|note\s*:)", _re.I)

def reason_stopped(tok, gen):
    txt = tok.decode(gen)
    m = _FA.search(txt)
    if m:                                  # answer emitted: stop once it hits a boundary or gets long (stop the ramble)
        after = txt[m.end():]
        if _ANS_BOUND.search(after): return True
        if len(after.split()) >= 24: return True
    return any(mk in txt for mk in LEAK)

def reason_prefix(raw):
    """The model's reasoning text BEFORE 'Final Answer:' (for history that keeps reasoning but a reference answer)."""
    raw = str(raw or ""); m = _FA.search(raw)
    return raw[:m.start()].rstrip() if m else raw.rstrip()

def extract_answer(raw):
    """Robust answer span after the FIRST 'Final Answer:', truncated at the first ramble boundary. Handles both the
    14B 'X Step 1...' ramble and the 3B 'X. Final Answer: X.' repeat, so extraction is CONSISTENT across models."""
    raw = str(raw or ""); m = _FA.search(raw)
    if not m:
        return raw.strip().split("\n")[0].strip()      # no marker -> first line
    return _ANS_BOUND.split(raw[m.end():])[0].strip().strip(".").strip()

def trim_leak(txt):
    for m in LEAK: txt = txt.split(m)[0]
    return txt.strip()

def provenance(prompt_name, prompt_text, decoding, max_new, model, reason_fix, **extra):
    d = {"prompt_name": prompt_name, "prompt_sha": hashlib.sha256((prompt_text or "").encode()).hexdigest()[:16],
         "decoding": decoding, "max_new": int(max_new), "model": model, "reason_fix": bool(reason_fix)}
    if reason_fix: d["min_reason"] = MIN_REASON
    d.update(extra)
    return d
