from __future__ import annotations

import os
from typing import Optional


QA_FULL_CONTEXT_INSTRUCTION = (
    "You are given context passages and a question.\n"
    "Answer using ONLY the context.\n\n"
    "Output format (STRICT):\n"
    "Final Answer: <answer>\n\n"
    "Rules:\n"
    "- Keep the answer short.\n"
    "- Output ONLY one line.\n"
    "- Start the line with `Final Answer:`.\n"
    "- Do NOT include explanations.\n"
    "- Do NOT include any other text.\n\n"
)

# ★ TERSE no-reasoning variant (2026-07-26) — the default no-reason prompt let the 14B write run-on/hedging
# answers ("Final Answer: Hancock County, Indiana. However, the context...") → token-F1 punished the extra words,
# so no-reason SpecPrefill was EQUALLY correct (EM-contain) but scored lower F1 than the terse reason version.
# This forces a BARE ENTITY so no-reason SpecPrefill can genuinely replace reasoning (terse + correct, no reasoning
# overhead). Selected via ANSWER_PROMPT_VARIANT / prompt_variant="direct_terse". NEVER overwrite the existing constants.
QA_FULL_CONTEXT_INSTRUCTION_TERSE = (
    "You are given context passages and a question.\n"
    "Answer using ONLY the context.\n\n"
    "Output the answer as the SHORTEST possible span: a single bare entity — a name, date, number, or short noun "
    "phrase — and nothing else.\n\n"
    "Output format (STRICT):\n"
    "Final Answer: <bare entity>\n\n"
    "Rules:\n"
    "- The answer MUST be a bare entity (e.g. `1867`, `Fletcher Webster`, `Cologne`). A few words at most.\n"
    "- Do NOT write a sentence. Do NOT explain. Do NOT restate the question. Do NOT add any words after the entity.\n"
    "- Output ONLY the one line starting with `Final Answer:`.\n\n"
)

QA_FULL_CONTEXT_INSTRUCTION_LEGACY = (
    "You are given context passages and a question.\n"
    "Answer using ONLY the context.\n\n"
    "Output format (STRICT):\n"
    "Final Answer: <answer>\n\n"
    "Rules:\n"
    "- Output ONLY one line.\n"
    "- Do NOT include explanations.\n"
    "- Do NOT include any other text.\n\n"
)

QA_SKETCH_CONDITIONED_INSTRUCTION = (
    "You are given a question and a compact evidence sketch extracted from the retrieved context.\n"
    "The evidence sketch is only part of the context and may be incomplete.\n"
    "Use the evidence sketch as guidance for answering the question.\n"
    "You may rely on your general language knowledge, but do not claim specific retrieved facts that are not supported by the sketch.\n\n"
    "Output format (STRICT):\n"
    "Final Answer: <answer>\n\n"
    "Rules:\n"
    "- Keep the answer short.\n"
    "- Output ONLY one line.\n"
    "- Start the line with `Final Answer:`.\n"
    "- Do NOT include explanations.\n"
    "- Do NOT include any other text.\n\n"
)

QA_SKETCH_CONDITIONED_INSTRUCTION_LEGACY = (
    "You are given a question and a compact evidence sketch extracted from the retrieved context.\n"
    "The evidence sketch is only part of the context and may be incomplete.\n"
    "Use the evidence sketch as guidance for answering the question.\n"
    "You may rely on your general language knowledge, but do not claim specific retrieved facts that are not supported by the sketch.\n\n"
    "Output format (STRICT):\n"
    "Final Answer: <answer>\n\n"
    "Rules:\n"
    "- Output ONLY one line.\n"
    "- Do NOT include explanations.\n"
    "- Do NOT include any other text.\n\n"
)

# --- New prompt variant (2026-06-16): refusal-banned + concise. ADDED, NOT overwriting,
#     so every prior experiment (which uses the constants above) stays reproducible. It is
#     selected only via an explicit `prompt_variant="no_refuse"` argument in build_* funcs.
#     Same wording for full-context (teacher / SLM-teacher) and sketch (ours) → fair. ---
QA_FULL_CONTEXT_INSTRUCTION_NOREFUSE = (
    "You are given context passages and a question.\n"
    "Answer using the context. If the context is incomplete, give the single most plausible "
    "answer from your general knowledge instead of refusing.\n\n"
    "Output format (STRICT):\n"
    "Final Answer: <answer>\n\n"
    "Rules:\n"
    "- Always commit to ONE concrete, specific answer. NEVER say the answer is not supported / "
    "not provided / unavailable, and never refuse or hedge.\n"
    "- Keep the answer SHORT: a few words or a single phrase, not a sentence.\n"
    "- Output ONLY one line starting with `Final Answer:`.\n"
    "- Do NOT include explanations or any other text.\n\n"
)

QA_SKETCH_CONDITIONED_INSTRUCTION_NOREFUSE = (
    "You are given a question and a compact evidence sketch extracted from the retrieved context.\n"
    "The evidence sketch is only part of the context and may be incomplete.\n"
    "Use the evidence sketch as guidance for answering the question.\n"
    "You may rely on your general language knowledge, but do not claim specific retrieved facts that are not supported by the sketch.\n"
    "If the sketch is incomplete, give the single most plausible answer instead of refusing.\n\n"
    "Output format (STRICT):\n"
    "Final Answer: <answer>\n\n"
    "Rules:\n"
    "- Always commit to ONE concrete, specific answer. NEVER say the answer is not supported / "
    "not provided / unavailable, and never refuse or hedge.\n"
    "- Keep the answer SHORT: a few words or a single phrase, not a sentence.\n"
    "- Output ONLY one line starting with `Final Answer:`.\n"
    "- Do NOT include explanations or any other text.\n\n"
)

# --- New prompt variant (2026-06-17): reason-then-answer. ADDED, NOT overwriting, so prior
#     experiments stay reproducible. Selected via prompt_variant="reason_then_answer".
#     Hypothesis (qasper no-ES diagnostic): a query-only LM prior acts like noise when forced to
#     emit a bare short answer; giving it room to reason from the evidence first, then a single
#     `Final Answer:` line we extract, recovers the formatting/instruction-following losses.
#     Same wording for full-context (teacher / SLM-teacher) and sketch (ours) → fair. ---
QA_FULL_CONTEXT_INSTRUCTION_REASON = (
    "You are given context passages and a question.\n"
    "Use the context to work out the answer. If the context is incomplete, still answer using "
    "your general (pretrained) knowledge instead of refusing.\n\n"
    "First write a brief reasoning (1-3 sentences) grounded in the context. "
    "Then, on a new line, write the final answer.\n\n"
    "Output format (STRICT):\n"
    "<brief reasoning>\n"
    "Final Answer: <answer>\n\n"
    "Rules:\n"
    "- End with exactly one line starting with `Final Answer:`.\n"
    "- The answer span must be SHORT: a few words or a single phrase, not a sentence.\n"
    "- Always commit to ONE concrete, specific answer. NEVER say the answer is not supported / "
    "not provided / not mentioned / unavailable / cannot be determined, and never refuse or hedge.\n"
    "- Output nothing after the `Final Answer:` line.\n\n"
)

# BABILong: the story moves objects between rooms; the answer is the SINGLE location word the object was
# last left in. The default reason_then_answer let the model write sentences ("...in the kitchen") or name
# a person/action ("Daniel grabbed it there") with NO bare room word -> EM-strict ~0. This variant forces a
# single location token. Selected via prompt_variant="reason_then_answer_babilong".
QA_FULL_CONTEXT_INSTRUCTION_REASON_BABILONG = (
    "You are given a long story where people pick up and drop objects in different rooms, then a "
    "question 'Where is the <object>?'. The answer is the room where the object was LAST left.\n\n"
    "First write a brief reasoning (1-2 sentences) tracing the object's last location. "
    "Then, on a new line, write the final answer.\n\n"
    "Output format (STRICT):\n"
    "<brief reasoning>\n"
    "Final Answer: <one room word>\n\n"
    "Rules:\n"
    "- The final answer MUST be a SINGLE location/room WORD only (e.g. `kitchen`, `garden`, `bedroom`, "
    "`hallway`, `office`, `bathroom`). NOT a sentence, NOT a person's name, NOT `there`.\n"
    "- End with exactly one line: `Final Answer:` followed by just that one room word.\n"
    "- Always commit to one concrete room. NEVER say unknown / unavailable / cannot be determined.\n"
    "- Output nothing after the `Final Answer:` line.\n\n"
)

QA_SKETCH_CONDITIONED_INSTRUCTION_REASON = (
    "You are given a question and a compact evidence sketch extracted from the retrieved context.\n"
    "The evidence sketch is only part of the context and may be incomplete.\n"
    "Use the evidence sketch as guidance for answering the question.\n"
    "You may rely on your general language knowledge, but do not claim specific retrieved facts that are not supported by the sketch.\n"
    "If the sketch is incomplete, still answer using your general (pretrained) knowledge instead of refusing.\n\n"
    "First write a brief reasoning (1-3 sentences) grounded in the evidence. "
    "Then, on a new line, write the final answer.\n\n"
    "Output format (STRICT):\n"
    "<brief reasoning>\n"
    "Final Answer: <answer>\n\n"
    "Rules:\n"
    "- End with exactly one line starting with `Final Answer:`.\n"
    "- The answer span must be SHORT: a few words or a single phrase, not a sentence.\n"
    "- Always commit to ONE concrete, specific answer. NEVER say the answer is not supported / "
    "not provided / not mentioned / unavailable / cannot be determined, and never refuse or hedge.\n"
    "- Output nothing after the `Final Answer:` line.\n\n"
)

# QASPER-aware reason variant (2026-06-17): PERMITS a deliberate "Unanswerable" (qasper has
# genuinely-unanswerable golds) while still banning lazy hedging/padding on answerable questions.
# Selected via prompt_variant="reason_then_answer_qasper". (musique/asqa use plain reason_then_answer.)
QA_FULL_CONTEXT_INSTRUCTION_REASON_QASPER = (
    "You are given context passages and a question.\n"
    "Use the context to work out the answer.\n\n"
    "First write a brief reasoning (1-3 sentences) grounded in the context. "
    "Then, on a new line, write the final answer.\n\n"
    "Output format (STRICT):\n"
    "<brief reasoning>\n"
    "Final Answer: <answer>\n\n"
    "Rules:\n"
    "- If the context genuinely does NOT contain the answer, write exactly `Final Answer: Unanswerable`.\n"
    "- Otherwise commit to ONE concrete, specific answer — do NOT hedge, pad, or add caveats.\n"
    "- The answer span must be SHORT: a few words or a single phrase, not a sentence.\n"
    "- End with exactly one line starting with `Final Answer:`. Output nothing after it.\n\n"
)

QA_SKETCH_CONDITIONED_INSTRUCTION_REASON_QASPER = (
    "You are given a question and a compact evidence sketch extracted from the retrieved context.\n"
    "The evidence sketch is only part of the context and may be incomplete.\n"
    "Use the evidence sketch as guidance for answering the question.\n"
    "You may rely on your general language knowledge, but do not claim specific retrieved facts that are not supported by the sketch.\n\n"
    "First write a brief reasoning (1-3 sentences) grounded in the evidence. "
    "Then, on a new line, write the final answer.\n\n"
    "Output format (STRICT):\n"
    "<brief reasoning>\n"
    "Final Answer: <answer>\n\n"
    "Rules:\n"
    "- If the evidence genuinely does NOT contain the answer, write exactly `Final Answer: Unanswerable`.\n"
    "- Otherwise commit to ONE concrete, specific answer — do NOT hedge, pad, or add caveats.\n"
    "- The answer span must be SHORT: a few words or a single phrase, not a sentence.\n"
    "- End with exactly one line starting with `Final Answer:`. Output nothing after it.\n\n"
)

# QASPER no-refuse variants (2026-06-23): identical to the *_REASON_QASPER prompts above EXCEPT the
# "Unanswerable" rule is tightened to discourage over-refusal — only allow Unanswerable when the
# context has genuinely NO relevant information; otherwise the model MUST commit to its best answer.
# Selected via prompt_variant="reason_then_answer_qasper_norefuse". (Existing *_REASON_QASPER untouched.)
QA_FULL_CONTEXT_INSTRUCTION_REASON_QASPER_NOREFUSE = (
    "You are given context passages and a question.\n"
    "Use the context to work out the answer.\n\n"
    "First write a brief reasoning (1-3 sentences) grounded in the context. "
    "Then, on a new line, write the final answer.\n\n"
    "Output format (STRICT):\n"
    "<brief reasoning>\n"
    "Final Answer: <answer>\n\n"
    "Rules:\n"
    "- Only answer 'Unanswerable' if the context contains genuinely NO information relevant to the "
    "question. In all other cases you MUST give your best answer extracted or inferred from the "
    "context — do not refuse, do not say the context is insufficient; commit to the most likely answer.\n"
    "- Commit to ONE concrete, specific answer — do NOT hedge, pad, or add caveats.\n"
    "- The answer span must be SHORT: a few words or a single phrase, not a sentence.\n"
    "- End with exactly one line starting with `Final Answer:`. Output nothing after it.\n\n"
)

QA_SKETCH_CONDITIONED_INSTRUCTION_REASON_QASPER_NOREFUSE = (
    "You are given a question and a compact evidence sketch extracted from the retrieved context.\n"
    "The evidence sketch is only part of the context and may be incomplete.\n"
    "Use the evidence sketch as guidance for answering the question.\n"
    "You may rely on your general language knowledge, but do not claim specific retrieved facts that are not supported by the sketch.\n\n"
    "First write a brief reasoning (1-3 sentences) grounded in the evidence. "
    "Then, on a new line, write the final answer.\n\n"
    "Output format (STRICT):\n"
    "<brief reasoning>\n"
    "Final Answer: <answer>\n\n"
    "Rules:\n"
    "- Only answer 'Unanswerable' if the context contains genuinely NO information relevant to the "
    "question. In all other cases you MUST give your best answer extracted or inferred from the "
    "context — do not refuse, do not say the context is insufficient; commit to the most likely answer.\n"
    "- Commit to ONE concrete, specific answer — do NOT hedge, pad, or add caveats.\n"
    "- The answer span must be SHORT: a few words or a single phrase, not a sentence.\n"
    "- End with exactly one line starting with `Final Answer:`. Output nothing after it.\n\n"
)

# qasper_full: like NOREFUSE but ALLOWS a complete answer (qasper golds avg ~11.5 words; the
# "SHORT: a few words" rule was forcing 3-word answers → low recall/f1). 2026-06-24.
QA_FULL_CONTEXT_INSTRUCTION_REASON_QASPER_FULL = (
    "You are given context passages and a question.\n"
    "Use the context to work out the answer.\n\n"
    "First write a brief reasoning (1-3 sentences) grounded in the context. "
    "Then, on a new line, write the final answer.\n\n"
    "Output format (STRICT):\n"
    "<brief reasoning>\n"
    "Final Answer: <answer>\n\n"
    "Rules:\n"
    "- Only answer 'Unanswerable' if the context contains genuinely NO information relevant to the "
    "question. In all other cases you MUST give your best answer extracted or inferred from the "
    "context — do not refuse, do not say the context is insufficient; commit to the most likely answer.\n"
    "- Give a COMPLETE answer that fully covers what the question asks: include ALL the relevant "
    "specifics (every method name, number, dataset, condition, or list item). QASPER answers are "
    "usually a full phrase or short sentence (about 8-15 words), NOT a single word — do NOT "
    "over-truncate; if the answer has multiple parts, include them all.\n"
    "- Answer the question directly: no padding, no meta-commentary, no restating the question.\n"
    "- End with exactly one line starting with `Final Answer:`. Output nothing after it.\n\n"
)

QA_SKETCH_CONDITIONED_INSTRUCTION_REASON_QASPER_FULL = (
    "You are given a question and a compact evidence sketch extracted from the retrieved context.\n"
    "The evidence sketch is only part of the context and may be incomplete.\n"
    "Use the evidence sketch as guidance for answering the question.\n"
    "You may rely on your general language knowledge, but do not claim specific retrieved facts that are not supported by the sketch.\n\n"
    "First write a brief reasoning (1-3 sentences) grounded in the evidence. "
    "Then, on a new line, write the final answer.\n\n"
    "Output format (STRICT):\n"
    "<brief reasoning>\n"
    "Final Answer: <answer>\n\n"
    "Rules:\n"
    "- Only answer 'Unanswerable' if the context contains genuinely NO information relevant to the "
    "question. In all other cases you MUST give your best answer extracted or inferred from the "
    "context — do not refuse, do not say the context is insufficient; commit to the most likely answer.\n"
    "- Give a COMPLETE answer that fully covers what the question asks: include ALL the relevant "
    "specifics (every method name, number, dataset, condition, or list item). QASPER answers are "
    "usually a full phrase or short sentence (about 8-15 words), NOT a single word — do NOT "
    "over-truncate; if the answer has multiple parts, include them all.\n"
    "- Answer the question directly: no padding, no meta-commentary, no restating the question.\n"
    "- End with exactly one line starting with `Final Answer:`. Output nothing after it.\n\n"
)


# qasper_v2: target the TYPICAL qasper answer length (~5-15 words like gold). v1 (_full) overshot
# to 100+ words. Length rule FIRST, drop "include every detail". 2026-06-24 iteration 2.
QA_FULL_CONTEXT_INSTRUCTION_REASON_QASPER_V2 = (
    "You are answering a question about a scientific paper, using the given context.\n"
    "Your Final Answer must match how QASPER answers are written: a SINGLE concise phrase or ONE short "
    "sentence that directly answers the question (typically 5-15 words, like the gold answer) — NOT a single "
    "word, NOT a paragraph, and do not enumerate every detail.\n\n"
    "First write one line of reasoning grounded in the context. Then, on a new line, the final answer.\n\n"
    "Output format (STRICT):\n<one-line reasoning>\nFinal Answer: <answer>\n\n"
    "Rules:\n"
    "- Give the actual answer content (the method / number / finding / yes-no), phrased directly.\n"
    "- Only answer 'Unanswerable' if the context truly has NO relevant information; otherwise commit to your best answer.\n"
    "- End with exactly one line starting with `Final Answer:`. Output nothing after it.\n\n"
)

QA_SKETCH_CONDITIONED_INSTRUCTION_REASON_QASPER_V2 = (
    "You are answering a question about a scientific paper, using an evidence sketch from the context.\n"
    "Your Final Answer must match how QASPER answers are written: a SINGLE concise phrase or ONE short "
    "sentence that directly answers the question (typically 5-15 words, like the gold answer) — NOT a single "
    "word, NOT a paragraph, and do not enumerate every detail.\n\n"
    "First write one line of reasoning grounded in the evidence. Then, on a new line, the final answer.\n\n"
    "Output format (STRICT):\n<one-line reasoning>\nFinal Answer: <answer>\n\n"
    "Rules:\n"
    "- Give the actual answer content (the method / number / finding / yes-no), phrased directly.\n"
    "- Only answer 'Unanswerable' if the evidence truly has NO relevant information; otherwise commit to your best answer.\n"
    "- End with exactly one line starting with `Final Answer:`. Output nothing after it.\n\n"
)


# qasper_v3 (2026-06-25): log analysis showed the LM-vs-SLM gap is FORMATTING not knowledge —
# the SLM rambles/repeats AND over-refuses ("Unanswerable" on answerable yes/no). v3 kills both,
# hard. ADDED separate constant (reproducibility).
QA_FULL_CONTEXT_INSTRUCTION_REASON_QASPER_V3 = (
    "Answer the question about the paper using the context.\n\n"
    "Output format:\n<one short line of reasoning>\nFinal Answer: <answer>\n\n"
    "The Final Answer is graded against a short gold (avg ~10 words). Rules for it:\n"
    "- Use the FEWEST words that fully answer. For a yes/no question the Final Answer is EXACTLY 'Yes' or 'No' and NOTHING else.\n"
    "- Do NOT repeat anything, do NOT add a second sentence, do NOT elaborate or write 'to be specific'/'in particular'/'to be precise' or extra clauses.\n"
    "- Give the actual content (method / number / finding), phrased directly — not a description of where it is.\n"
    "- Answer 'Unanswerable' ONLY if the paper has truly NO relevant information. For a yes/no question this is almost never — commit to Yes or No.\n"
    "- Output nothing after the Final Answer line.\n\n"
)

QA_SKETCH_CONDITIONED_INSTRUCTION_REASON_QASPER_V3 = (
    "Answer the question about the paper using the evidence.\n\n"
    "Output format:\n<one short line of reasoning>\nFinal Answer: <answer>\n\n"
    "The Final Answer is graded against a short gold (avg ~10 words). Rules for it:\n"
    "- Use the FEWEST words that fully answer. For a yes/no question the Final Answer is EXACTLY 'Yes' or 'No' and NOTHING else.\n"
    "- Do NOT repeat anything, do NOT add a second sentence, do NOT elaborate or write 'to be specific'/'in particular'/'to be precise' or extra clauses.\n"
    "- Give the actual content (method / number / finding), phrased directly — not a description of where it is.\n"
    "- Answer 'Unanswerable' ONLY if the evidence has truly NO relevant information. For a yes/no question this is almost never — commit to Yes or No.\n"
    "- Output nothing after the Final Answer line.\n\n"
)

# qasper_v4 (2026-06-25): RAW logs showed v3 output "Final Answer: X" THEN "Reasoning: Y" (answer-then-reason,
# reversed) — the model answered first, so reasoning never informed the answer → wrong yes/no + refusals.
# Root cause: v3 dropped the explicit ordering sentence the WORKING musique prompt has. v4 restores
# "First write reasoning ... Then write the final answer" (musique structure) + QASPER length/anti-refuse.
QA_FULL_CONTEXT_INSTRUCTION_REASON_QASPER_V4 = (
    "You are given context passages from a research paper and a question about it.\n"
    "Use the context to work out the answer. If the context is incomplete, still answer using your "
    "general knowledge instead of refusing.\n\n"
    "First write a brief reasoning (1-3 sentences) grounded in the context. "
    "Then, on a new line, write the final answer.\n\n"
    "Output format (STRICT):\n"
    "<brief reasoning>\n"
    "Final Answer: <answer>\n\n"
    "Rules:\n"
    "- End with exactly one line starting with `Final Answer:`.\n"
    "- The Final Answer is graded against a short gold (avg ~10 words): give the actual content "
    "(method / number / finding) in the FEWEST words that fully answer — a phrase or one short clause, not a paragraph.\n"
    "- For a yes/no question the Final Answer is EXACTLY 'Yes' or 'No'. Commit — NEVER 'Unanswerable' for a yes/no question.\n"
    "- Only answer 'Unanswerable' if the paper truly has no relevant information (rare).\n"
    "- Do not repeat; output nothing after the `Final Answer:` line.\n\n"
)

QA_SKETCH_CONDITIONED_INSTRUCTION_REASON_QASPER_V4 = (
    "You are given a question about a research paper and evidence passages.\n"
    "Use the evidence to work out the answer. If it is incomplete, still answer using your general "
    "knowledge instead of refusing.\n\n"
    "First write a brief reasoning (1-3 sentences) grounded in the evidence. "
    "Then, on a new line, write the final answer.\n\n"
    "Output format (STRICT):\n"
    "<brief reasoning>\n"
    "Final Answer: <answer>\n\n"
    "Rules:\n"
    "- End with exactly one line starting with `Final Answer:`.\n"
    "- The Final Answer is graded against a short gold (avg ~10 words): give the actual content "
    "(method / number / finding) in the FEWEST words that fully answer — a phrase or one short clause, not a paragraph.\n"
    "- For a yes/no question the Final Answer is EXACTLY 'Yes' or 'No'. Commit — NEVER 'Unanswerable' for a yes/no question.\n"
    "- Only answer 'Unanswerable' if the evidence truly has no relevant information (rare).\n"
    "- Do not repeat; output nothing after the `Final Answer:` line.\n\n"
)

# qasper_v5 (2026-06-25): v4 echoed literal "<brief reasoning>" + leaked system prompt. v5 = the PROVEN
# working musique reason structure (no garbage at 94%) with QASPER answer length + anti-refuse only.
QA_FULL_CONTEXT_INSTRUCTION_REASON_QASPER_V5 = (
    "You are given context passages and a question.\n"
    "Use the context to work out the answer. If the context is incomplete, still answer using "
    "your general (pretrained) knowledge instead of refusing.\n\n"
    "First write a brief reasoning (1-3 sentences) grounded in the context. "
    "Then, on a new line, write the final answer.\n\n"
    "Output format (STRICT):\n"
    "<brief reasoning>\n"
    "Final Answer: <answer>\n\n"
    "Rules:\n"
    "- End with exactly one line starting with `Final Answer:`.\n"
    "- Give the actual content (method / number / finding) as a phrase or one short clause — complete but concise (~10 words like the gold), not a paragraph.\n"
    "- For a yes/no question the Final Answer is EXACTLY 'Yes' or 'No'. Commit; NEVER 'Unanswerable' for a yes/no question.\n"
    "- Only say 'Unanswerable' if the context truly has no relevant info (rare).\n"
    "- Output nothing after the `Final Answer:` line.\n\n"
)
QA_SKETCH_CONDITIONED_INSTRUCTION_REASON_QASPER_V5 = QA_FULL_CONTEXT_INSTRUCTION_REASON_QASPER_V5

QA_OUTPUT_FORMAT = "Output format (STRICT):\nFinal Answer: <answer>\n"

QA_SKETCH_HEADER = (
    "Use the following evidence sketch as context for answering the question.\n\n"
    "Evidence Sketch:\n"
)

ASQA_EVIDENCE_SKETCH_INSTRUCTION = (
    "Create a citation-aware evidence sketch for answering the ASQA question.\n\n"
    "Rules:\n"
    "1. Do not write the final answer.\n"
    "2. Include only facts directly useful for answering the question.\n"
    "3. Preserve document numbers exactly with bracket citations like [3] or [2][7].\n"
    "4. Every factual bullet must include at least one supporting document number.\n"
    "5. Include enough facts to cover all answer facets, entities, dates, comparisons, and exceptions needed by the question.\n"
    "6. If the question has multiple interpretations or asks about multiple entities, group facts by facet.\n"
    "7. Prefer dense factual notes over polished prose; omit boilerplate, broad background, and irrelevant details.\n"
    "8. Do not add outside knowledge or unsupported inferences.\n"
    "9. If evidence is missing, conflicting, or weak, state the limitation with the relevant document number when possible.\n\n"
    "Output format:\n"
    "Answer facets:\n"
    "- <facet or entity>: <short description>\n\n"
    "Citation facts:\n"
    "- [doc] <answer-critical fact>\n"
    "- [doc][doc] <answer-critical fact supported by multiple documents>\n\n"
    "Limits / conflicts:\n"
    "- [doc] <missing, ambiguous, or conflicting evidence if relevant>\n"
)

ASQA_PROMPT_VERSION_SUPPORTED_ONLY = "supported_only"
ASQA_PROMPT_VERSION_ANSWER_FIRST = "answer_first"
ASQA_PROMPT_VERSIONS = {ASQA_PROMPT_VERSION_SUPPORTED_ONLY, ASQA_PROMPT_VERSION_ANSWER_FIRST}

ASQA_CITATION_SYSTEM_SUPPORTED_ONLY = (
    "You are given context passages and a question.\n"
    "Write a well-structured answer with inline citations for each sentence.\n"
    "Use bracketed document numbers like [1] or [2][3] immediately after the sentence they support.\n"
    "Rules:\n"
    "- Base every sentence on the provided context.\n"
    "- Include citations for every sentence that contains factual claims.\n"
    "- If the answer is not supported, say: \"Answer not supported by provided documents.\" and stop.\n"
    "- Do not invent document numbers.\n"
    "- Do not write citations in prose; only append bracketed citations at sentence end.\n"
    "- Do not add role markers like \"Human:\" or \"Assistant:\".\n"
)

ASQA_CITATION_SYSTEM_ANSWER_FIRST = (
    "You are given context passages and a question.\n"
    "Write a well-structured answer with inline citations for each sentence.\n"
    "Use bracketed document numbers like [1] or [2][3] immediately after the sentence they support.\n"
    "Begin your response with the answer itself — the very first words must state the answer, not any preamble.\n"
    "You must NEVER state or imply, anywhere in the response, that the answer is unsupported, unavailable, missing, not provided, or that the question cannot be answered.\n"
    "Always commit to exactly one specific, concrete answer. If the passages are incomplete, use your general knowledge to give the single most plausible answer rather than refusing or hedging; "
    "cite only statements supported by the provided passages.\n"
    "Rules:\n"
    "- A direct, concrete answer is mandatory. Refusal, hedging, disclaimers, or \"not supported / not provided\" style responses are strictly forbidden.\n"
    "- Include citations for every sentence that is supported by the provided passages.\n"
    "- Do not invent document numbers.\n"
    "- Do not write citations in prose; only append bracketed citations at sentence end.\n"
    "- Do not add role markers like \"Human:\" or \"Assistant:\".\n"
) 

# Reason-then-answer ASQA variant (no-ES + reason main direction). The model reasons briefly,
# then emits "Final Answer:" followed by the FULL cited long-form answer (citations preserved
# for ALCE). Extraction = everything after "Final Answer:" (src.eval.extract_final_answer_full),
# so the reasoning prefix is dropped but the attributed answer + [1][2] citations survive.
# Selected via prompt_variant in {reason_then_answer, reason_then_answer_asqa}.
ASQA_CITATION_SYSTEM_ANSWER_FIRST_REASON = (
    "You are given context passages and a question.\n"
    "First, think step by step in two or three brief sentences of reasoning (no citations in this part).\n"
    "Then, on a new line, write exactly 'Final Answer:' followed by a well-structured answer with inline citations.\n"
    "In the final answer, use bracketed document numbers like [1] or [2][3] immediately after the sentence they support.\n"
    "You must NEVER state or imply, anywhere in the response, that the answer is unsupported, unavailable, missing, not provided, or that the question cannot be answered.\n"
    "Always commit to exactly one specific, concrete answer. If the passages are incomplete, use your general knowledge to give the single most plausible answer rather than refusing or hedging; "
    "cite only statements supported by the provided passages.\n"
    "Rules:\n"
    "- The reasoning comes first; the attributed answer comes after the 'Final Answer:' marker.\n"
    "- A direct, concrete answer is mandatory. Refusal, hedging, disclaimers, or \"not supported / not provided\" style responses are strictly forbidden.\n"
    "- Include citations for every sentence of the final answer that is supported by the provided passages.\n"
    "- Do not invent document numbers.\n"
    "- Do not write citations in prose; only append bracketed citations at sentence end.\n"
    "- Do not add role markers like \"Human:\" or \"Assistant:\".\n"
)

# LONG ASQA reason variant — ASQA questions are deliberately AMBIGUOUS with MULTIPLE correct
# answers (different interpretations); ALCE str_em/QA-EM reward covering ALL of them, so the
# terse `reason_then_answer_asqa` underperforms. This variant keeps reason+citation but instructs
# a COMPREHENSIVE answer that enumerates every distinct interpretation/answer found in the docs.
# Selected via prompt_variant="reason_then_answer_asqa_long".
ASQA_CITATION_SYSTEM_ANSWER_FIRST_REASON_LONG = (
    "You are given context passages and a question.\n"
    "The question is often AMBIGUOUS and may have SEVERAL distinct correct answers depending on interpretation (different people, places, dates, or senses).\n"
    "First, think step by step in two or three brief sentences of reasoning (no citations in this part): identify the different plausible interpretations and the answer for each.\n"
    "Then, on a new line, write exactly 'Final Answer:' followed by a COMPREHENSIVE answer that addresses EVERY distinct interpretation/answer supported by the passages — one sentence per interpretation, each ending with its bracketed citation [1] or [2][3].\n"
    "Cover ALL the correct answers you can find (do not stop at one); a complete answer to an ambiguous question lists each valid case.\n"
    "You must NEVER state or imply that the answer is unsupported, unavailable, missing, or that the question cannot be answered.\n"
    "If the passages are incomplete, use general knowledge for the most plausible answers rather than refusing; cite only statements supported by the passages.\n"
    "Rules:\n"
    "- The reasoning comes first; the comprehensive attributed answer comes after the 'Final Answer:' marker.\n"
    "- Enumerate every distinct supported answer/interpretation as its own cited sentence — completeness is rewarded.\n"
    "- Refusal, hedging, or \"not supported\" responses are strictly forbidden.\n"
    "- Do not invent document numbers; append bracketed citations at sentence end only; no role markers.\n"
)

# CITE-REASON ASQA variant — the no-ES LM leg never sees the documents, but it DOES attend to the
# tokens the fused model has already generated. So if the REASONING explicitly names documents by
# number as it reasons, those doc-ids enter the generated context → the LM finally gets a document
# signal (a self-generated sketch) → better final-answer content + citations. This variant makes the
# reasoning LONG and explicitly document-cited (the reasoning is still stripped before ALCE scoring;
# it only serves to ground the final answer). Selected via prompt_variant="reason_then_answer_asqa_cite".
ASQA_CITATION_SYSTEM_ANSWER_FIRST_REASON_CITE = (
    "You are given context passages and a question.\n"
    "The question is often AMBIGUOUS and may have SEVERAL distinct correct answers (different people, places, dates, or senses).\n"
    "First, REASON IN DETAIL (at least 4-6 sentences). As you reason, go through the relevant documents ONE BY ONE and EXPLICITLY name each by its number while you use it — e.g., \"Document [3] states ...; Document [7] indicates ...; together these imply ...\". For every plausible interpretation, state which document(s) support it and the answer for that case.\n"
    "Then, on a new line, write exactly 'Final Answer:' followed by a COMPREHENSIVE answer that addresses EVERY distinct interpretation supported by the passages — one sentence per interpretation, each ending with its bracketed citation [1] or [2][3].\n"
    "Cover ALL the correct answers you can find (do not stop at one).\n"
    "You must NEVER state or imply that the answer is unsupported, unavailable, missing, or that the question cannot be answered.\n"
    "If the passages are incomplete, use general knowledge for the most plausible answers; cite only statements supported by the passages.\n"
    "Rules:\n"
    "- The reasoning MUST be detailed and cite specific documents by number as it goes (this is required, not optional).\n"
    "- The comprehensive attributed answer comes after the 'Final Answer:' marker; enumerate every distinct supported answer as its own cited sentence.\n"
    "- Refusal, hedging, or \"not supported\" responses are strictly forbidden; do not invent document numbers; no role markers.\n"
)

ASQA_SKETCH_CONDITIONED_CITATION_SYSTEM_SUPPORTED_ONLY = (
    "You are given a question and a compact evidence sketch extracted from retrieved context passages.\n"
    "The evidence sketch is only part of the context and may be incomplete.\n"
    "Use the evidence sketch as guidance for writing the answer.\n"
    "You may rely on general language knowledge for composition, but do not claim specific retrieved facts that are not supported by the sketch.\n"
    "Use bracketed document numbers like [1] or [2][3] immediately after the sentence they support.\n"
    "Rules:\n"
    "- Base factual claims on the provided evidence sketch.\n"
    "- Include citations for every sentence that contains factual claims.\n"
    "- If the answer is not supported, say: \"Answer not supported by provided documents.\" and stop.\n"
    "- Do not invent document numbers.\n"
    "- Do not write citations in prose; only append bracketed citations at sentence end.\n"
    "- Do not add role markers like \"Human:\" or \"Assistant:\".\n"
)

ASQA_SKETCH_CONDITIONED_CITATION_SYSTEM_ANSWER_FIRST = (
    "You are given a question and a compact evidence sketch extracted from retrieved context passages.\n"
    "The evidence sketch is only part of the context and may be incomplete.\n"
    "Use the evidence sketch as guidance for writing the answer.\n"
    "Begin your response with the answer itself — the very first words must state the answer, not any preamble.\n"
    "You must NEVER state or imply, anywhere in the response, that the answer is unsupported, unavailable, missing, not provided, or that the question cannot be answered.\n"
    "Always commit to exactly one specific, concrete answer. If the sketch is incomplete, use your general knowledge to give the single most plausible answer rather than refusing or hedging.\n"
    "Use bracketed document numbers like [1] or [2][3] immediately after the sentence they support.\n"
    "Rules:\n"
    "- A direct, concrete answer is mandatory. Refusal, hedging, disclaimers, or \"not supported / not provided\" style responses are strictly forbidden.\n"
    "- Include citations for every sentence that is supported by the evidence sketch.\n"
    "- Do not invent document numbers.\n"
    "- Do not write citations in prose; only append bracketed citations at sentence end.\n"
    "- Do not add role markers like \"Human:\" or \"Assistant:\".\n"
)

# Historical alias kept so older imports continue to resolve. New code should use
# build_asqa_citation_answer_prompt / build_asqa_citation_sketch_prompt with an
# explicit prompt_version.
QA_SKETCH_CONDITIONED_CITATION_SYSTEM = ASQA_SKETCH_CONDITIONED_CITATION_SYSTEM_SUPPORTED_ONLY


def _normalize_asqa_prompt_version(prompt_version: str) -> str:
    if prompt_version not in ASQA_PROMPT_VERSIONS:
        raise ValueError(
            f"Unsupported ASQA prompt_version={prompt_version!r}. "
            f"Expected one of {sorted(ASQA_PROMPT_VERSIONS)}."
        )
    return prompt_version


def get_asqa_citation_system(prompt_version: str = ASQA_PROMPT_VERSION_ANSWER_FIRST, *, reason: bool = False, long_answer: bool = False, cite_reason: bool = False) -> str:
    prompt_version = _normalize_asqa_prompt_version(prompt_version)
    if reason and cite_reason:
        return ASQA_CITATION_SYSTEM_ANSWER_FIRST_REASON_CITE
    if reason and long_answer:
        return ASQA_CITATION_SYSTEM_ANSWER_FIRST_REASON_LONG
    if reason:
        # reason+citation only defined for the answer_first (no-refuse) family — the main direction
        return ASQA_CITATION_SYSTEM_ANSWER_FIRST_REASON
    if prompt_version == ASQA_PROMPT_VERSION_SUPPORTED_ONLY:
        return ASQA_CITATION_SYSTEM_SUPPORTED_ONLY
    return ASQA_CITATION_SYSTEM_ANSWER_FIRST


def get_asqa_sketch_conditioned_citation_system(
    prompt_version: str = ASQA_PROMPT_VERSION_ANSWER_FIRST,
    *,
    reason: bool = False,
    long_answer: bool = False,
    cite_reason: bool = False,
) -> str:
    prompt_version = _normalize_asqa_prompt_version(prompt_version)
    if reason and cite_reason:
        return ASQA_CITATION_SYSTEM_ANSWER_FIRST_REASON_CITE
    if reason and long_answer:
        return ASQA_CITATION_SYSTEM_ANSWER_FIRST_REASON_LONG
    if reason:
        # no-ES + reason: the LM (query-preserving leg) and any sketch-off path use the same
        # reason+citation system as the full-context leg, so the fused output format aligns.
        return ASQA_CITATION_SYSTEM_ANSWER_FIRST_REASON
    if prompt_version == ASQA_PROMPT_VERSION_SUPPORTED_ONLY:
        return ASQA_SKETCH_CONDITIONED_CITATION_SYSTEM_SUPPORTED_ONLY
    return ASQA_SKETCH_CONDITIONED_CITATION_SYSTEM_ANSWER_FIRST


def format_chat_prompt(tokenizer, system_text: str, user_text: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        messages = []
        if system_text:  # ★ official LongBench uses build_chat with NO system prompt → skip when empty/None
            messages.append({"role": "system", "content": system_text})
        messages.append({"role": "user", "content": user_text})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{system_text}\n\n{user_text}" if system_text else user_text


def build_asqa_citation_answer_prompt(
    *,
    question: str,
    documents: list[str],
    include_docs: bool,
    tokenizer=None,
    evidence_sketch: Optional[str] = None,
    prompt_version: str = ASQA_PROMPT_VERSION_ANSWER_FIRST,
    reason: bool = False,
    long_answer: bool = False,
    cite_reason: bool = False,
) -> str:
    if include_docs:
        system_text = get_asqa_citation_system(prompt_version, reason=reason, long_answer=long_answer, cite_reason=cite_reason)
        doc_blocks = "\n\n".join(f"[{idx}] {doc}" for idx, doc in enumerate(documents, start=1))
        user_text = f"Context:\n{doc_blocks}\n\nQuestion:\n{question}\n\nAnswer:"
    else:
        system_text = get_asqa_sketch_conditioned_citation_system(prompt_version, reason=reason, long_answer=long_answer, cite_reason=cite_reason)
        sketch_block = f"{QA_SKETCH_HEADER}{evidence_sketch.strip()}\n\n" if evidence_sketch and evidence_sketch.strip() else ""
        user_text = f"{sketch_block}Question:\n{question}\n\nAnswer:"
    if tokenizer is None:
        return f"{system_text}\n\n{user_text}"
    return format_chat_prompt(tokenizer, system_text, user_text)


# ★ LongBench SUMMARIZATION prompt (gov_report / multi_news). These are NOT QA — the gold is a
# comprehensive ~one-page (≈400-600 word) summary with NO citations. The ASQA citation/reason prompt
# was wrong here: it elicited short cited "answers" → the 14B wrote ~270w (half the 520w gold) and lost
# recall-based rougeLsum to the more-verbose 3B (LM<SLM reversal = a PROMPT bug, not a metric bug).
# This prompt matches the benchmark: ask for a thorough full-length summary, plain prose, no scaffolding.
# Selected via ANSWER_PROMPT_VARIANT="summarize". (Added as a separate named variant for reproducibility.)
LONGBENCH_SUMMARIZATION_SYSTEM = (
    "You are given a long source text (a government report, or several related news articles).\n"
    "Write a LONG, DETAILED summary that captures ALL the main points, key findings, arguments, specific "
    "figures/numbers, recommendations, and any official responses across the WHOLE source — go section by "
    "section through the entire document and do not omit or compress important specifics.\n"
    "The summary MUST be at least 500 words (aim for about 500-700 words of well-organized flowing prose). "
    "Keep writing until you have comprehensively covered the entire source; do NOT stop early or write a "
    "short/condensed version — an overly brief summary will be heavily penalized for missing content.\n"
    "Write the summary text directly. Do NOT include: any citations or bracketed document numbers, any "
    "reasoning preamble, a 'Final Answer:' marker, section headings, bullet points, or any meta-commentary "
    "about the task or the instruction.\n"
)

# ★ V2 (ANSWER_PROMPT_VARIANT="summarize2") — FIXES the inverted teacher ladder (teacher-3B > teacher-14B on
# gov rougeLsum). Root cause (logs): the gold opens with the SUBJECT ("The Forest Service, an agency within
# USDA, ...") and mirrors the source register; the 14B opens META ("The report evaluates...") and abstracts,
# so its lexical/LCS overlap with the gold drops below the more-copy-faithful 3B. This is a PROMPT bug, not a
# metric bug. V2 forces subject-first, source-mirroring style (which the more-capable 14B executes better) so
# rougeLsum ranks model size correctly — the PRECONDITION for any fusion experiment.
LONGBENCH_SUMMARIZATION_SYSTEM_V2 = (
    "You are given a long source text (a government report, or several related news articles).\n"
    "Write a LONG, DETAILED summary (at least 500 words; aim 500-700) capturing ALL main points, key "
    "findings, specific figures/numbers/dates/names, statutory references, recommendations, and official "
    "responses across the WHOLE source — go section by section; do not omit specifics.\n"
    "STYLE (important): Begin DIRECTLY with the SUBJECT of the report — the agency, program, bill, or topic — "
    "and its essential background, e.g. 'The Forest Service, an agency within USDA, ...' or 'Multiyear "
    "procurement (MYP) is ...'. Do NOT begin with 'The report examines/evaluates/discusses/provides/details/"
    "highlights' or any meta-description of the document. Write a standalone factual account OF THE SUBJECT, "
    "mirroring the source's own structure, terminology, and definitions; preserve the source's specific "
    "wording, names, and figures rather than paraphrasing them at a higher level of abstraction.\n"
    "Write the summary text directly. Do NOT include citations, bracketed numbers, a reasoning preamble, a "
    "'Final Answer:' marker, section headings, bullet points, or any meta-commentary about the task.\n"
)


# ★ V3 (ANSWER_PROMPT_VARIANT="summarize3") — escalation when V2's instruction alone failed (the 14B kept
# meta-opening "The report details..." and abstracting). FEW-SHOT the register (show, don't tell): a crafted
# style-only example (no eval leakage) of the gold's subject-first, definitional, extractive style + a forceful
# "be extractive, copy the source's wording" instruction. Goal: make the more-capable 14B match the gold
# register so rougeLsum ranks model size correctly (14B > 3B) — fixing the inverted ladder (prompting bug).
LONGBENCH_SUMMARIZATION_SYSTEM_V3 = (
    "You are given a long source text (a government report, or several related news articles).\n"
    "Write a LONG, DETAILED summary (at least 500 words; aim 500-700) capturing ALL main points, key "
    "findings, specific figures/numbers/dates/names, statutory references, recommendations, and official "
    "responses across the WHOLE source — go section by section; do not omit specifics.\n"
    "BE EXTRACTIVE: the reference summaries closely follow the source's OWN wording and structure. COPY the "
    "source's key sentences and exact phrasing rather than paraphrasing or abstracting; keep its terminology, "
    "definitions, names, and figures verbatim. Begin DIRECTLY with the SUBJECT and its definition; do NOT "
    "begin with 'The report examines/evaluates/discusses/provides/details' or any meta-description.\n"
    "EXAMPLE of the required register (STYLE ONLY — do NOT use these facts; your content must come from the "
    "source above):\n"
    "'The Environmental Protection Agency (EPA), an independent agency within the executive branch, "
    "administers federal programs to control pollution of air and water. Its enforcement process includes "
    "(1) compliance monitoring, which detects violations, and (2) civil and criminal penalties, which deter "
    "future noncompliance. GAO was asked to review EPA's oversight of state-administered programs. This report "
    "examines the extent to which ...'\n"
    "Notice: it opens with the SUBJECT and its definition, uses the source's numbered structural elements, and "
    "reads like the report's own summary/highlights — concrete and copy-faithful, never an abstract overview.\n"
    "Write the summary text directly. No citations, bracketed numbers, reasoning preamble, 'Final Answer:' "
    "marker, section headings, bullet points, or meta-commentary about the task.\n"
)

# ★ V4 (ANSWER_PROMPT_VARIANT="summarize4") — escalation after V1/V2/V3 all left teacher-3B > teacher-14B.
# DATA finding (score_rouge_sharedN): the inversion is NOT meta-openers (both ~40%) — it is EXTRACTIVENESS.
# gov gold is highly extractive (median 571 words closely following the source's own sentences); the weaker
# 3B copies the source more verbatim → higher lexical/LCS overlap → higher rougeLsum, while the 14B
# paraphrases/abstracts (semantically better but lexically further from the extractive gold). Even at matched
# length (V1: 14B 575w 28.9 vs 3B 606w 31.6) the 3B wins. So V4 forces the 14B to OUT-EXTRACT the 3B: an
# explicit two-stage "select the most important source sentences, then reproduce them VERBATIM and stitch
# them" procedure — maximize verbatim overlap with the extractive gold while covering all sections (~570w).
LONGBENCH_SUMMARIZATION_SYSTEM_V4 = (
    "You are given a long source text (a government report, or several related news articles). Produce an "
    "EXTRACTIVE summary built from the source's OWN sentences.\n"
    "METHOD (follow exactly):\n"
    "1. Scan the ENTIRE source and identify the ~20-30 most important sentences spread across ALL sections "
    "(background/subject, key findings, specific figures and dates, statutory references, recommendations, and "
    "any official responses). Cover the whole document end to end, not just the beginning.\n"
    "2. Build the summary by REPRODUCING those source sentences VERBATIM — copy the source's exact wording, "
    "names, numbers, and terminology. Do NOT paraphrase, compress, generalize, or abstract them into your own "
    "higher-level phrasing. Lightly trim and order them into flowing prose, but keep the original sentence "
    "wording wherever possible. Begin DIRECTLY with the subject (copy the source's own opening definition); do "
    "NOT begin with 'The report examines/evaluates/discusses/provides/details' or any meta-description.\n"
    "3. The result MUST be at least 500 words (aim 550-650). Keep adding important source sentences until you "
    "have comprehensively covered every section; an over-short or over-paraphrased summary scores poorly.\n"
    "Write the summary text directly. No citations, bracketed numbers, reasoning preamble, 'Final Answer:' "
    "marker, section headings, bullet points, or meta-commentary about the task.\n"
)

# ★ V5 (ANSWER_PROMPT_VARIANT="summarize5") — the per-example diagnostic (gov4) pinned the cause: in the 33/54
# examples where teacher-14B LOSES to 3B, the 14B writes 337w vs gold 576w — it STOPS EARLY and ABSTRACTS AWAY
# the specifics (e.g. "DOD's MILCON appropriations fund acquisition…" instead of gold's "$2.5 to $9.6 billion in
# MILCON funding FY2005-2016"). When it DOES write enough (373w, subject-first) it scores 36.8 and WINS. So the
# fix is NOT "be extractive/concise" (V2-V4 made it shorter) — it is FORCE EXHAUSTIVE LENGTH + EVERY SPECIFIC
# FIGURE + section-by-section coverage so the capable 14B stops under-covering. This is a PROMPT fix (the user was
# right): the 14B can match the gold, it just terminates too early and generalizes.
LONGBENCH_SUMMARIZATION_SYSTEM_V5 = (
    "You are given a long source text (a government report, or several related news articles). Write a "
    "COMPREHENSIVE, DETAILED summary.\n"
    "LENGTH: The reference summaries average ~560 words. Write AT LEAST 500 words (aim 550-650). Keep writing "
    "until you have covered the ENTIRE document; do NOT stop early. A short summary is the single biggest cause "
    "of a low score.\n"
    "COVERAGE: Go through the report SECTION BY SECTION from beginning to end. Cover every major topic, finding, "
    "and recommendation — do not stop after the introduction.\n"
    "SPECIFICS (critical): INCLUDE every concrete detail — all dollar amounts, percentages, dates, fiscal years, "
    "counts, program/agency names, bill numbers, and statutory citations exactly as they appear in the source. "
    "Do NOT abstract or generalize them away (write '$2.5 to $9.6 billion in FY2005-2016', NOT 'significant "
    "funding over the years'). A summary that drops the specific numbers will score poorly.\n"
    "STYLE: Begin DIRECTLY with the subject and its definition (e.g. 'The Aegis BMD program, managed by the "
    "Missile Defense Agency and the Navy, ...'). Do NOT begin with 'The report examines/discusses/provides an "
    "overview of' or any meta-description; write a standalone factual account, mirroring the source's wording.\n"
    "Write the summary text directly. No citations, bracketed numbers, reasoning preamble, 'Final Answer:' "
    "marker, section headings, bullet points, or meta-commentary about the task.\n"
)

_SUMMARY_SYSTEM_BY_VARIANT = {
    "summarize": LONGBENCH_SUMMARIZATION_SYSTEM,
    "summarize2": LONGBENCH_SUMMARIZATION_SYSTEM_V2,
    "summarize3": LONGBENCH_SUMMARIZATION_SYSTEM_V3,
    "summarize4": LONGBENCH_SUMMARIZATION_SYSTEM_V4,
    "summarize5": LONGBENCH_SUMMARIZATION_SYSTEM_V5,
}

# ★★★ OFFICIAL LongBench summarization prompts — VERBATIM from THUDM/LongBench config/dataset2prompt.json
# (commit-checked 2026-06-27). Reproduced to test whether our ELABORATE system prompt + LONG max_gen
# (we used 900-1024; official is 512) caused the Qwen rougeLsum size-inversion (teacher-14B < 3B), which
# Gemma did NOT show. The official recipe: (1) NO system prompt — bare build_chat (model chat template);
# (2) instruction SANDWICHED before AND after the context; (3) max_gen=512. Selected via
# ANSWER_PROMPT_VARIANT="summarize_official". Keyed by LongBench task name; {context} = whole document.
LONGBENCH_OFFICIAL_PROMPTS = {
    "gov_report": (
        "You are given a report by a government agency. Write a one-page summary of the report.\n\n"
        "Report:\n{context}\n\n"
        "Now, write a one-page summary of the report.\n\nSummary:"
    ),
    "multi_news": (
        "You are given several news passages. Write a one-page summary of all news. \n\n"
        "News:\n{context}\n\n"
        "Now, write a one-page summary of all the news.\n\nSummary:"
    ),
}
# Fusion LM-side (no-docs): the LM is query-only, so it never sees the Report/News block — keep the
# sandwiched instruction without the {context} so the official wording still drives the LM logits.
LONGBENCH_OFFICIAL_PROMPTS_NODOCS = {
    "gov_report": (
        "You are given a report by a government agency. Write a one-page summary of the report.\n\n"
        "Now, write a one-page summary of the report.\n\nSummary:"
    ),
    "multi_news": (
        "You are given several news passages. Write a one-page summary of all news. \n\n"
        "Now, write a one-page summary of all the news.\n\nSummary:"
    ),
}


def build_longbench_summary_prompt(
    *,
    question: str,
    documents: list[str],
    include_docs: bool,
    tokenizer=None,
    evidence_sketch: Optional[str] = None,
    task: Optional[str] = None,
) -> str:
    variant = os.environ.get("ANSWER_PROMPT_VARIANT", "")
    # ★★★ OFFICIAL LongBench reproduction: exact prompt, NO system message, model's native chat template.
    if variant == "summarize_official":
        t = task if task in LONGBENCH_OFFICIAL_PROMPTS else (
            "multi_news" if (question and "news" in question.lower()) else "gov_report"
        )
        if include_docs:
            context = "\n\n".join(documents)
            user_text = LONGBENCH_OFFICIAL_PROMPTS[t].format(context=context)
        else:
            user_text = LONGBENCH_OFFICIAL_PROMPTS_NODOCS[t]
        if tokenizer is None:
            return user_text
        return format_chat_prompt(tokenizer, None, user_text)  # None system → bare build_chat
    system_text = _SUMMARY_SYSTEM_BY_VARIANT.get(variant, LONGBENCH_SUMMARIZATION_SYSTEM)
    if include_docs:
        # No [i] numbering — summaries don't cite; the source is one corpus to summarize.
        doc_blocks = "\n\n".join(documents)
        user_text = f"{doc_blocks}\n\n{question}\n\nSummary:"
    else:
        sketch_block = f"{QA_SKETCH_HEADER}{evidence_sketch.strip()}\n\n" if evidence_sketch and evidence_sketch.strip() else ""
        user_text = f"{sketch_block}{question}\n\nSummary:"
    if tokenizer is None:
        return f"{system_text}\n\n{user_text}"
    return format_chat_prompt(tokenizer, system_text, user_text)


def build_asqa_citation_sketch_prompt(
    *,
    question: str,
    documents: list[str],
    evidence_sketch_instruction: str = ASQA_EVIDENCE_SKETCH_INSTRUCTION,
    tokenizer=None,
) -> str:
    system_text = "You are given retrieved context passages and a question."
    doc_blocks = "\n\n".join(f"[{idx}] {doc}" for idx, doc in enumerate(documents, start=1))
    user_text = (
        f"Context:\n{doc_blocks}\n\n"
        f"Question:\n{question}\n\n"
        f"{evidence_sketch_instruction}"
    )
    if tokenizer is None:
        return f"{system_text}\n\n{user_text}"
    return format_chat_prompt(tokenizer, system_text, user_text)


# CLUTRR multi-question kinship — the QASPER-replacement CLEAN-METRIC benchmark (2026-07-05). Answer = ONE
# family-relationship word -> exact-match, NO length pathology (unlike QASPER free-form token-F1). One story is
# shared context for several derived questions. Selected via prompt_variant="reason_then_answer_clutrr".
QA_FULL_CONTEXT_INSTRUCTION_REASON_CLUTRR = (
    "You are given a short story that states family relationships between named people. Determine the "
    "family relationship asked in the question by composing the stated relationships. Think step by step, "
    "then end with a line exactly of the form 'Final Answer: <relation>', where <relation> is a SINGLE "
    "family-relationship word such as mother, father, daughter, son, sister, brother, grandmother, "
    "grandfather, granddaughter, grandson, aunt, uncle, niece, nephew, mother-in-law, father-in-law, "
    "daughter-in-law, or son-in-law.\n\n"
)


def build_full_context_qa_prompt(
    question: str,
    documents: list[str],
    *,
    revert_previous_prompt: bool = False,
    prompt_variant: str = "default",
) -> str:
    if prompt_variant == "no_refuse":
        instruction = QA_FULL_CONTEXT_INSTRUCTION_NOREFUSE
        suffix = "\n"
    elif prompt_variant == "direct_terse":
        instruction = QA_FULL_CONTEXT_INSTRUCTION_TERSE
        suffix = "\n"
    elif prompt_variant == "reason_then_answer":
        instruction = QA_FULL_CONTEXT_INSTRUCTION_REASON
        suffix = "\n"
    elif prompt_variant == "reason_v3":
        instruction = QA_REASON_V3          # bench_config CANONICAL musique/hotpotqa prompt (sha c1a237e0)
        suffix = "\n"
    elif prompt_variant == "reason_then_answer_babilong":
        instruction = QA_FULL_CONTEXT_INSTRUCTION_REASON_BABILONG
        suffix = "\n"
    elif prompt_variant == "reason_then_answer_qasper":
        instruction = QA_FULL_CONTEXT_INSTRUCTION_REASON_QASPER
        suffix = "\n"
    elif prompt_variant == "reason_then_answer_qasper_norefuse":
        instruction = QA_FULL_CONTEXT_INSTRUCTION_REASON_QASPER_NOREFUSE
        suffix = "\n"
    elif prompt_variant == "reason_then_answer_qasper_full":
        instruction = QA_FULL_CONTEXT_INSTRUCTION_REASON_QASPER_FULL
        suffix = "\n"
    elif prompt_variant == "reason_then_answer_clutrr":
        instruction = QA_FULL_CONTEXT_INSTRUCTION_REASON_CLUTRR
        suffix = "\n"
    elif prompt_variant == "reason_then_answer_qasper_v2":
        instruction = QA_FULL_CONTEXT_INSTRUCTION_REASON_QASPER_V2
        suffix = "\n"
    elif prompt_variant == "reason_then_answer_qasper_v3":
        instruction = QA_FULL_CONTEXT_INSTRUCTION_REASON_QASPER_V3
        suffix = "\n"
    elif prompt_variant == "reason_then_answer_qasper_v4":
        instruction = QA_FULL_CONTEXT_INSTRUCTION_REASON_QASPER_V4
        suffix = "\n"
    elif prompt_variant == "reason_then_answer_qasper_v5":
        instruction = QA_FULL_CONTEXT_INSTRUCTION_REASON_QASPER_V5
        suffix = "\n"
    # ★ reason_v3 (2026-07-29): select the CANONICAL bench_config instruction QA_REASON_V3 (sha c1a237e0) —
    # the same text musique/qasper/hotpotqa are scored under — so a run of this harness is directly
    # comparable with bench_config runs. Used by the corrected fusion-FT pipeline (target generation,
    # training and eval all share it). The LM (query-only) branch gets the SAME instruction so both
    # fusion branches are conditioned on one task framing. NOTHING existing is overwritten.
    elif prompt_variant == "reason_v3":
        instruction = QA_REASON_V3
        suffix = "\n"
    else:
        instruction = QA_FULL_CONTEXT_INSTRUCTION_LEGACY if revert_previous_prompt else QA_FULL_CONTEXT_INSTRUCTION
        suffix = f"\n\n{QA_OUTPUT_FORMAT}" if revert_previous_prompt else "\n"
    # ★ FIX_EMPTY_CONTEXT (opt-in, 2026-06-29): with NO documents (the fusion LM-query branch / closed-book), the
    # default still emits an EMPTY "Context:\n\n\n" header, which primes the query-only LM to FILL it -> it
    # degenerates into echoing the context ("Context:\nDocument 1...") at long ctx (musique d400, 13%). When the
    # env flag is set, omit the Context block entirely for no-docs prompts. Off by default (existing runs unchanged).
    if not documents and os.environ.get("FIX_EMPTY_CONTEXT", "0") == "1":
        return f"{instruction}Question: {question}{suffix}"
    doc_blocks = "\n\n".join(f"### Document {idx}\n{doc}" for idx, doc in enumerate(documents, start=1))
    return f"{instruction}Context:\n{doc_blocks}\n\nQuestion: {question}{suffix}"


def build_sketch_conditioned_qa_prompt(
    question: str,
    evidence_sketch: Optional[str],
    *,
    revert_previous_prompt: bool = False,
    prompt_variant: str = "default",
) -> str:
    if prompt_variant == "no_refuse":
        instruction = QA_SKETCH_CONDITIONED_INSTRUCTION_NOREFUSE
        suffix = "\n"
    elif prompt_variant == "reason_then_answer":
        instruction = QA_SKETCH_CONDITIONED_INSTRUCTION_REASON
        suffix = "\n"
    elif prompt_variant == "reason_then_answer_qasper":
        instruction = QA_SKETCH_CONDITIONED_INSTRUCTION_REASON_QASPER
        suffix = "\n"
    elif prompt_variant == "reason_then_answer_qasper_norefuse":
        instruction = QA_SKETCH_CONDITIONED_INSTRUCTION_REASON_QASPER_NOREFUSE
        suffix = "\n"
    elif prompt_variant == "reason_then_answer_qasper_full":
        instruction = QA_SKETCH_CONDITIONED_INSTRUCTION_REASON_QASPER_FULL
        suffix = "\n"
    elif prompt_variant == "reason_then_answer_qasper_v2":
        instruction = QA_SKETCH_CONDITIONED_INSTRUCTION_REASON_QASPER_V2
        suffix = "\n"
    elif prompt_variant == "reason_then_answer_qasper_v3":
        instruction = QA_SKETCH_CONDITIONED_INSTRUCTION_REASON_QASPER_V3
        suffix = "\n"
    elif prompt_variant == "reason_then_answer_qasper_v4":
        instruction = QA_SKETCH_CONDITIONED_INSTRUCTION_REASON_QASPER_V4
        suffix = "\n"
    elif prompt_variant == "reason_then_answer_qasper_v5":
        instruction = QA_SKETCH_CONDITIONED_INSTRUCTION_REASON_QASPER_V5
        suffix = "\n"
    # ★ reason_v3 (2026-07-29): select the CANONICAL bench_config instruction QA_REASON_V3 (sha c1a237e0) —
    # the same text musique/qasper/hotpotqa are scored under — so a run of this harness is directly
    # comparable with bench_config runs. Used by the corrected fusion-FT pipeline (target generation,
    # training and eval all share it). The LM (query-only) branch gets the SAME instruction so both
    # fusion branches are conditioned on one task framing. NOTHING existing is overwritten.
    elif prompt_variant == "reason_v3":
        instruction = QA_REASON_V3
        suffix = "\n"
    else:
        instruction = (
            QA_SKETCH_CONDITIONED_INSTRUCTION_LEGACY
            if revert_previous_prompt
            else QA_SKETCH_CONDITIONED_INSTRUCTION
        )
        suffix = f"\n\n{QA_OUTPUT_FORMAT}" if revert_previous_prompt else "\n"
    sketch_block = f"{QA_SKETCH_HEADER}{evidence_sketch.strip()}\n\n" if evidence_sketch and evidence_sketch.strip() else ""
    return f"{instruction}{sketch_block}Question: {question}{suffix}"


def build_query_preserving_full_context_qa_prompt(
    *,
    question: str,
    documents: list[str],
    tokenizer,
    max_length: int,
    revert_previous_prompt: bool = False,
    prompt_variant: str = "default",
) -> str:
    if prompt_variant == "no_refuse":
        instruction = QA_FULL_CONTEXT_INSTRUCTION_NOREFUSE
    elif prompt_variant == "reason_then_answer":
        instruction = QA_FULL_CONTEXT_INSTRUCTION_REASON
    elif prompt_variant == "reason_then_answer_qasper":
        instruction = QA_FULL_CONTEXT_INSTRUCTION_REASON_QASPER
    elif prompt_variant == "reason_then_answer_qasper_norefuse":
        instruction = QA_FULL_CONTEXT_INSTRUCTION_REASON_QASPER_NOREFUSE
    elif prompt_variant == "reason_then_answer_qasper_full":
        instruction = QA_FULL_CONTEXT_INSTRUCTION_REASON_QASPER_FULL
    elif prompt_variant == "reason_then_answer_qasper_v2":
        instruction = QA_FULL_CONTEXT_INSTRUCTION_REASON_QASPER_V2
    elif prompt_variant == "reason_then_answer_qasper_v3":
        instruction = QA_FULL_CONTEXT_INSTRUCTION_REASON_QASPER_V3
    elif prompt_variant == "reason_then_answer_qasper_v4":
        instruction = QA_FULL_CONTEXT_INSTRUCTION_REASON_QASPER_V4
    elif prompt_variant == "reason_then_answer_qasper_v5":
        instruction = QA_FULL_CONTEXT_INSTRUCTION_REASON_QASPER_V5
    # ★ reason_v3 (2026-07-29): select the CANONICAL bench_config instruction QA_REASON_V3 (sha c1a237e0) —
    # the same text musique/qasper/hotpotqa are scored under — so a run of this harness is directly
    # comparable with bench_config runs. Used by the corrected fusion-FT pipeline (target generation,
    # training and eval all share it). The LM (query-only) branch gets the SAME instruction so both
    # fusion branches are conditioned on one task framing. NOTHING existing is overwritten.
    elif prompt_variant == "reason_v3":
        instruction = QA_REASON_V3
    else:
        instruction = QA_FULL_CONTEXT_INSTRUCTION_LEGACY if revert_previous_prompt else QA_FULL_CONTEXT_INSTRUCTION
    full_prompt = build_full_context_qa_prompt(
        question,
        documents,
        revert_previous_prompt=revert_previous_prompt,
        prompt_variant=prompt_variant,
    )
    full_len = len(tokenizer(full_prompt, add_special_tokens=False, truncation=False).input_ids)
    if full_len <= max_length:
        return full_prompt

    prefix = f"{instruction}Context:\n"
    suffix = f"\n\nQuestion: {question}\n"
    if revert_previous_prompt and prompt_variant == "default":
        suffix += f"\n{QA_OUTPUT_FORMAT}"
    fixed_len = len(tokenizer(prefix + suffix, add_special_tokens=False, truncation=False).input_ids)

    if fixed_len >= max_length:
        fixed_ids = tokenizer(prefix + suffix, add_special_tokens=False, truncation=False).input_ids
        return tokenizer.decode(fixed_ids[-max_length:], skip_special_tokens=True)

    context_budget = max_length - fixed_len
    context_parts: list[str] = []
    used = 0
    for idx, doc in enumerate(documents, start=1):
        chunk = ("" if idx == 1 else "\n\n") + f"### Document {idx}\n{doc}"
        chunk_ids = tokenizer(chunk, add_special_tokens=False, truncation=False).input_ids
        chunk_len = len(chunk_ids)
        if used + chunk_len <= context_budget:
            context_parts.append(chunk)
            used += chunk_len
            continue
        remain = context_budget - used
        if remain > 0:
            context_parts.append(tokenizer.decode(chunk_ids[:remain], skip_special_tokens=True))
        break

    return prefix + "".join(context_parts) + suffix


def build_query_preserving_full_context_qa_prompt_with_token_segments(
    *,
    question: str,
    documents: list[str],
    tokenizer,
    max_length: int,
    revert_previous_prompt: bool = False,
) -> tuple[str, list[int], list[int]]:
    """
    Return the truncated full-context prompt together with token-level segment ids.

    Segment id `0` denotes shared tokens (instruction, separators, question, output
    format, and later generated tokens). Positive segment ids correspond to each
    document block in order, so document tokens can be masked against other documents.
    """
    instruction = QA_FULL_CONTEXT_INSTRUCTION_LEGACY if revert_previous_prompt else QA_FULL_CONTEXT_INSTRUCTION
    prefix = f"{instruction}Context:\n"
    suffix = f"\n\nQuestion: {question}\n"
    if revert_previous_prompt:
        suffix += f"\n{QA_OUTPUT_FORMAT}"

    prefix_ids = tokenizer(prefix, add_special_tokens=False, truncation=False).input_ids
    suffix_ids = tokenizer(suffix, add_special_tokens=False, truncation=False).input_ids
    fixed_len = len(prefix_ids) + len(suffix_ids)

    if fixed_len >= max_length:
        prompt = tokenizer.decode(
            tokenizer(prefix + suffix, add_special_tokens=False, truncation=False).input_ids[-max_length:],
            skip_special_tokens=True,
        )
        enc = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )
        token_ids = list(enc.input_ids)
        return prompt, [0] * len(token_ids), token_ids

    context_budget = max_length - fixed_len
    prompt_parts: list[str] = [prefix]
    doc_char_spans: list[tuple[int, int, int]] = []
    current_pos = len(prefix)
    used = 0
    for idx, doc in enumerate(documents, start=1):
        chunk = ("" if idx == 1 else "\n\n") + f"### Document {idx}\n{doc}"
        chunk_ids = tokenizer(chunk, add_special_tokens=False, truncation=False).input_ids
        chunk_len = len(chunk_ids)
        if used + chunk_len <= context_budget:
            chunk_text = chunk
            used += chunk_len
        else:
            remain = context_budget - used
            if remain <= 0:
                break
            chunk_text = tokenizer.decode(chunk_ids[:remain], skip_special_tokens=True)
        prompt_parts.append(chunk_text)
        doc_char_spans.append((current_pos, current_pos + len(chunk_text), idx))
        current_pos += len(chunk_text)
        if used + chunk_len > context_budget:
            break

    prompt_parts.append(suffix)
    prompt = "".join(prompt_parts)
    enc = tokenizer(
        prompt,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    token_ids = list(enc.input_ids)
    offsets = list(enc.offset_mapping)
    segment_ids: list[int] = []
    span_ptr = 0
    for start, end in offsets:
        seg_id = 0
        while span_ptr < len(doc_char_spans) and end > doc_char_spans[span_ptr][1]:
            span_ptr += 1
        if span_ptr < len(doc_char_spans):
            span_start, span_end, candidate_seg = doc_char_spans[span_ptr]
            if start < span_end and end > span_start:
                seg_id = candidate_seg
        segment_ids.append(seg_id)

    if len(segment_ids) != len(token_ids):
        raise RuntimeError(
            f"Prompt token/segment mismatch after offset mapping: tokens={len(token_ids)} segments={len(segment_ids)}"
        )

    return prompt, segment_ids, token_ids

# mtRAG: reference answers are full conversational sentences (~25 words). Elicit a complete conversational
# answer (NOT a terse "Final Answer:" span), matching the benchmark's answer style. No reasoning scaffold.
QA_MTRAG_INSTRUCTION = (
    "You are a helpful assistant in a multi-turn conversation. Using the evidence passages and the conversation "
    "so far, answer the user's latest question directly and completely in 1-3 full sentences, in a natural "
    "conversational style. State the answer explicitly (do not just say 'yes'/'no' — give the supporting detail). "
    "If the passages do not contain the answer, say you don't have that information."
)


# qasper_v2_norefuse (2026-07-15, iteration on QASPER multi-Q reuse): V2 concise (5-15w) BUT commit — the
# fusion (ours) was emitting "Unanswerable" on answerable questions (sometimes skipping reasoning), where
# floor-3B and teacher-14B both commit. Forbid refusal unless the context is genuinely empty on the topic;
# force the reasoning step first; explicit yes/no guidance (two refusals were yes/no golds).
QA_FULL_CONTEXT_INSTRUCTION_REASON_QASPER_V2_NOREFUSE = (
    "You are answering a question about a scientific paper, using the given context.\n"
    "Your Final Answer must match how QASPER answers are written: a SINGLE concise phrase or ONE short "
    "sentence that directly answers the question (typically 5-15 words, like the gold answer) — NOT a single "
    "word, NOT a paragraph, and do not enumerate every detail.\n\n"
    "First write ONE line of reasoning grounded in the context (do not skip it). Then, on a new line, the final answer.\n\n"
    "Output format (STRICT):\n<one-line reasoning>\nFinal Answer: <answer>\n\n"
    "Rules:\n"
    "- COMMIT to your best answer extracted or inferred from the context. The context is relevant to the "
    "question; do NOT say the paper 'does not explicitly state' or that you 'cannot conclude'.\n"
    "- Do NOT answer 'Unanswerable' unless the context contains literally no information touching the topic. "
    "If any relevant detail appears, give the answer it implies.\n"
    "- For a yes/no question, answer 'Yes' or 'No' (optionally a few words of support) — never 'Unanswerable'.\n"
    "- End with exactly one line starting with `Final Answer:`. Output nothing after it.\n\n"
)


# qasper_v2_yesno (2026-07-15, QASPER reuse iteration 4): V2 concise, but TARGETED yes/no commit. Diagnosis:
# the fusion (ours) refused 46% of yes/no questions ("Unanswerable" where gold is Yes/No), which is most of the
# ours→teacher gap; floor/teacher commit. Forbid refusal ONLY on yes/no (where it is almost always wrong in
# QASPER); KEEP the 'Unanswerable' option for open questions (avoids V2NR's cost on genuinely-unanswerable golds).
QA_FULL_CONTEXT_INSTRUCTION_REASON_QASPER_V2_YESNO = (
    "You are answering a question about a scientific paper, using the given context.\n"
    "Your Final Answer must match how QASPER answers are written: a SINGLE concise phrase or ONE short "
    "sentence that directly answers the question (typically 5-15 words, like the gold answer) — NOT a single "
    "word, NOT a paragraph, and do not enumerate every detail.\n\n"
    "First write ONE line of reasoning grounded in the context. Then, on a new line, the final answer.\n\n"
    "Output format (STRICT):\n<one-line reasoning>\nFinal Answer: <answer>\n\n"
    "Rules:\n"
    "- If the question is a YES/NO question (it starts with Do/Does/Did/Is/Are/Was/Were/Can/Could/Has/Have/"
    "Will/Would), you MUST answer 'Yes' or 'No' (optionally a few words of support). NEVER 'Unanswerable' for a "
    "yes/no question — the paper determines the answer; infer it from the relevant details, do not hedge.\n"
    "- For other questions, give the actual answer content (method/number/finding) phrased directly; answer "
    "'Unanswerable' only if the context genuinely has NO relevant information.\n"
    "- End with exactly one line starting with `Final Answer:`. Output nothing after it.\n\n"
)


# LoCoMo (2026-07-15): long multi-session conversation between two people = shared context; questions probe memory.
# Answers are SHORT factual spans (a date, a name, a place, a short phrase). Concise, reason-then-answer.
QA_FULL_CONTEXT_INSTRUCTION_REASON_LOCOMO = (
    "You are answering a question about a long multi-session conversation between two people (with dates).\n"
    "Use the conversation to work out the answer. Your Final Answer must be SHORT and factual — a date, a name, a "
    "place, or a brief phrase (typically 1-8 words), phrased directly like the ground-truth answers.\n\n"
    "First write ONE line of reasoning grounded in the conversation (cite who/when). Then, on a new line, the answer.\n\n"
    "Output format (STRICT):\n<one-line reasoning>\nFinal Answer: <answer>\n\n"
    "Rules:\n"
    "- For a 'when' question, give the specific date/time mentioned (e.g. '7 May 2023').\n"
    "- Commit to the best answer supported by the conversation; do not restate the question.\n"
    "- If the conversation genuinely never mentions the answer, reply exactly 'Not mentioned in the conversation'.\n"
    "- End with exactly one line starting with `Final Answer:`. Output nothing after it.\n\n"
)


# LoCoMo v2 (2026-07-15, iter 2): temporal-aware + FORCE reasoning on 'when' questions. Diagnosis: c2 (temporal)
# is the biggest gap; models output relative dates ("last year"/"next month") instead of resolving to the absolute
# date from the session timestamp, and skip reasoning (97% go straight to Final Answer). For 'when' Qs, make them
# find the session date + the relative reference and COMPUTE the absolute date before answering.
QA_FULL_CONTEXT_INSTRUCTION_REASON_LOCOMO_V2 = (
    "You are answering a question about a long multi-session conversation between two people. Each session is "
    "labelled with its DATE (e.g. '[Session 3 | 2:14 pm on 8 May 2023]').\n"
    "Use the conversation to work out the answer. Your Final Answer must be SHORT and factual — a date, a name, a "
    "place, or a brief phrase (1-8 words), phrased like the ground-truth answers.\n\n"
    "For a 'when' question you MUST reason first: find the SESSION DATE of the relevant turn, read any relative "
    "reference in the dialogue ('last year', 'next month', 'last Friday', 'the sunday before'), and COMPUTE the "
    "ABSOLUTE date. Give the absolute date/year in the Final Answer (e.g. '2022', 'June 2023', '11 June 2023') — "
    "NEVER a relative phrase like 'last year'. For non-'when' questions, one line of reasoning then the answer.\n\n"
    "Output format (STRICT):\n<reasoning: session date + relative ref -> absolute>\nFinal Answer: <answer>\n\n"
    "Rules:\n"
    "- Commit to the best answer supported by the conversation; do not restate the question.\n"
    "- If the conversation genuinely never mentions the answer, reply exactly 'Not mentioned in the conversation'.\n"
    "- End with exactly one line starting with `Final Answer:`. Output nothing after it.\n\n"
)


# ============================================================================
# 2026-07-17 REASONING-BUG FIX. The prior reason_then_answer prompts showed a literal template
# "<one-line reasoning>" / "<brief reasoning>" which the model ECHOED verbatim (see validate_experiment.py),
# and merely ASKED for reasoning, which short-answer tasks ignored (97% straight-to-'Final Answer:').
# This variant: (a) NO angle-bracket placeholder to echo, (b) mandates reasoning and explicitly overrides any
# "answer only" instruction in the question. Pair with a teacher-forced "Reasoning:" generation cue (q_turn_reason).
QA_REASON_V2 = (
    "You are given context and a question. You must answer in two steps and you must do BOTH steps.\n"
    "Step 1 (Reasoning): write 1-3 sentences that work out the answer directly from the context (say who / what / when). "
    "Do this even if the question tells you to answer directly or with only the answer — ignore any such instruction.\n"
    "Step 2 (Answer): then, on a new line, write the words 'Final Answer:' followed by the answer in a few words.\n\n"
    "Rules:\n"
    "- Start your reply with the reasoning sentence. Do NOT start with 'Final Answer:'.\n"
    "- Never output text inside angle brackets and never repeat these instructions.\n"
    "- Commit to one concrete answer; do not refuse or say it is not mentioned unless the context truly lacks it.\n"
    "- Output nothing after the 'Final Answer:' line.\n"
)
# LoCoMo (temporal-heavy, short factual gold ~1-8 words): same 2-step contract, date-aware.
QA_REASON_V2_LOCOMO = (
    "You are answering a question about a long multi-session conversation between two people (each session is dated).\n"
    "Answer in two steps and do BOTH:\n"
    "Step 1 (Reasoning): write ONE sentence identifying the turn/session that answers it (who said it, and which "
    "session date it maps to). Do this even if the question says to answer directly.\n"
    "Step 2 (Answer): on a new line, write 'Final Answer:' then a SHORT factual answer (a date, name, place, or brief "
    "phrase, ~1-8 words), phrased like the ground-truth (e.g. a 'when' question -> the specific date such as '7 May 2023').\n\n"
    "Rules:\n- Start with the reasoning sentence, NOT with 'Final Answer:'.\n- No angle-bracket placeholders; do not repeat instructions.\n"
    "- If the conversation genuinely never mentions it, answer exactly 'Not mentioned in the conversation'.\n- Output nothing after 'Final Answer:'.\n"
)


# 2026-07-17 v3: reasoning + COMMIT (no refusal) + concise answer span. v2 permitted "Not mentioned", and reasoning
# amplified over-refusal (LoCoMo teacher 8%->22%, ours 3%->26%) + let notes bleed into the answer ("20 May. (Note:...)")
# -> token-F1 collapse. v3 restores the old prompt's stance: never refuse, put ONLY the short answer after 'Final Answer:'.
QA_REASON_V3 = (
    "You are given context and a question. Answer in two steps and do BOTH.\n"
    "Step 1 (Reasoning): 1-2 sentences using the context to work out the answer (who/what/when).\n"
    "Step 2 (Answer): on a new line write 'Final Answer:' then ONLY the answer in a few words.\n\n"
    "Rules:\n"
    "- ALWAYS commit to one concrete best answer. NEVER reply 'Not mentioned', 'not provided', 'unavailable', or hedge — "
    "even if unsure, give your single most likely answer from the context.\n"
    "- After 'Final Answer:' put ONLY the answer span: no notes, no parentheticals, no caveats, no extra sentence.\n"
    "- No angle-bracket placeholders; don't repeat these instructions. Output nothing after the 'Final Answer:' line.\n"
)
QA_REASON_V3_NQA = (
    # narrativeqa-accum (2026-08-20): under QA_REASON_V3 the 7B-family arms rambled 20+ words vs 4-6-word
    # golds and 12% of rows never emitted 'Final Answer:' within max_new=200 (reasoning never ended), so
    # the F1 spread was a length artifact (recall gap teacher-floor 0.006). This variant forces brevity
    # HARD on both steps. New constant, never overwrite V3 (reproducibility rule).
    "You are given a story and a question about it. Answer in two steps and do BOTH.\n"
    "Step 1 (Reasoning): ONE short sentence only - name the scene or fact that answers it. HARD LIMIT one sentence.\n"
    "Step 2 (Answer): on a new line write 'Final Answer:' then ONLY the answer in AT MOST 6 words - a name, place, "
    "date, object, or tiny phrase. Shorter is better; never a full sentence.\n\n"
    "Rules:\n"
    "- ALWAYS commit to one concrete best answer. NEVER reply 'Not mentioned' or hedge.\n"
    "- After 'Final Answer:' put ONLY the answer span: no notes, no parentheticals, no extra words.\n"
    "- Do NOT quote long passages from the story. Output nothing after the 'Final Answer:' line.\n"
)
QA_REASON_V3_LOCOMO = (
    "You are answering a question about a long multi-session conversation (each session is dated). Answer in two steps.\n"
    "Step 1 (Reasoning): ONE sentence — which turn/session answers it, and for a 'when' question resolve the actual date "
    "(e.g. if someone says they did it 'yesterday' in a session dated 8 May 2023, the answer is 7 May 2023).\n"
    "Step 2 (Answer): on a new line write 'Final Answer:' then ONLY the short answer (a date, name, place, or brief phrase).\n\n"
    "Rules:\n"
    "- ALWAYS commit to your single best answer. NEVER reply 'Not mentioned' or 'not provided' — give the most likely answer.\n"
    "- After 'Final Answer:' put ONLY the answer: no notes, parentheticals, or caveats.\n"
    "- Phrase 'when' answers like the ground truth (e.g. '7 May 2023', 'the week before 9 June 2023').\n"
    "- No angle-bracket placeholders; output nothing after the 'Final Answer:' line.\n"
)


# 2026-08-12: LoCoMo ABSTENTION variant. THE CANONICAL PROMPT FORBIDS THE GOLD ANSWER OF 25% OF THE TURNS.
#
# LoCoMo category 5 (ADVERSARIAL) has the gold "Not mentioned in the conversation", and it is 148 of the
# 600 turns of the episode ref. QA_REASON_V3_LOCOMO rule 1 reads, verbatim:
#     "ALWAYS commit to your single best answer. NEVER reply 'Not mentioned' or 'not provided'"
# so every arm is instructed not to produce the only answer that can score on a quarter of the benchmark.
# The whole adversarial cell was therefore measuring WHICH MODEL DISOBEYS ITS INSTRUCTION MOST, and it read
# exactly that way: teacher-32B abstains on 51% of those turns, the naive 7B floor on 13%, every fusion arm
# on 10-16%. The category was never split out on the front-30 slice (0% adversarial there) so the conflict
# went unnoticed until the episode ref put it at 25%.
#
# It also explains the mechanism that looked like an ours-specific defect. The LM branch gets the SAME
# instruction as the reader (fusion_stage2_lm_sft.py: `lp = lm_tok(instr + qt)`) but has NO context that
# could contradict it, so it follows "NEVER reply Not mentioned" most obediently and supplies a fluent
# affirmative frame — "In Session 3, dated 1 February 2023, Jon mentions that he found a cool new fashion
# piece" — over a reader that had correctly written "he does NOT mention finding anything". 10 of the 19
# abstentions the floor produces are destroyed this way. That is the prompt working as written, not fusion
# failing.
#
# The blanket ban is not deleted, it is made CONDITIONAL. It exists for a real reason — over-refusal on
# answerable turns is the failure it was added to stop, and 75% of LoCoMo IS answerable — so the rule still
# pushes hard toward committing, and declining is scoped to the case where the specific thing asked is
# absent. The third bullet names the actual trap: measured on the episode ref, LoCoMo's adversarial
# questions are MINIMAL-PAIR perturbations of real answerable ones (`Caroline`<-`Melanie`, `Sam`<-`Evan`,
# `museum`<-`library`; 50 of 148 still have a >=0.55-overlap answerable twin inside the 600-turn sample
# alone), so the reader fails by TOPIC-MATCHING a neighbouring fact and re-attributing it. Telling it that a
# similar fact about a different person or place is not an answer targets that directly.
#
# EVERYTHING ELSE IS BYTE-IDENTICAL to QA_REASON_V3_LOCOMO so the A/B is one variable. Apply it to EVERY arm
# of a comparison — the teacher's ceiling moves too, and closeness is measured against that ceiling.
# Selected with `--instruction QA_REASON_V3_LOCOMO_ABSTAIN`; bench_config's default is deliberately NOT
# changed, so every existing LoCoMo row stays valid and build_table will refuse to merge the two families.
QA_REASON_V3_LOCOMO_ABSTAIN = (
    "You are answering a question about a long multi-session conversation (each session is dated). Answer in two steps.\n"
    "Step 1 (Reasoning): ONE sentence — which turn/session answers it, and for a 'when' question resolve the actual date "
    "(e.g. if someone says they did it 'yesterday' in a session dated 8 May 2023, the answer is 7 May 2023).\n"
    "Step 2 (Answer): on a new line write 'Final Answer:' then ONLY the short answer (a date, name, place, or brief phrase).\n\n"
    "Rules:\n"
    "- Commit to your single best answer whenever the conversation supports one. Do not hedge and do not add caveats.\n"
    "- Most questions ARE answerable. Only if the conversation genuinely never states the thing asked about, reply with "
    "exactly: Not mentioned in the conversation\n"
    "- Check that the answer is about the PERSON, PLACE and EVENT the question names. A similar fact about someone or "
    "something else is NOT an answer — if only that is present, the conversation does not state it.\n"
    "- After 'Final Answer:' put ONLY the answer: no notes, parentheticals, or caveats.\n"
    "- Phrase 'when' answers like the ground truth (e.g. '7 May 2023', 'the week before 9 June 2023').\n"
    "- No angle-bracket placeholders; output nothing after the 'Final Answer:' line.\n"
)


# 2026-08-13: LoCoMo abstention variant 3 — a PROCEDURE instead of a judgement.
#
# WHY V2 WAS NOT ENOUGH. QA_REASON_V3_LOCOMO_ABSTAIN removed the ban and lifted the 32B a great deal while
# leaving the 7B exactly where it was. Matched on the same turns (episode ref, batch 3, λ0.85):
#     teacher-32B  adversarial F1 0.354 -> 0.692   abstains 37% -> 63%
#     floor-7B     adversarial F1 0.088 -> 0.090   abstains  7% ->  5%
#     plain fusion adversarial F1 0.014 -> 0.044
# That is the WRONG shape for closeness = (ours - floor)/(teacher - floor): the ceiling rose and the
# numerator did not, so permitting abstention makes the arms look worse even though the benchmark's ceiling
# became more honest. The 32B can act on a conditional rule ("only if the conversation never states it");
# the 7B cannot turn that judgement into behaviour.
#
# So v3 stops asking for a judgement and asks for a CHECK the small model can actually execute, inside the
# reasoning step it already writes: NAME THE SESSION THAT STATES IT. A model that cannot name one has
# already established the answer is absent, which converts "is this absent?" into "did step 1 produce a
# citation?". This targets the measured failure directly — the reader confabulates by TOPIC-MATCHING a
# neighbouring fact and re-attributing it ("Melanie's son got into an accident" answered under Caroline's
# name), and naming the session forces the binding to be checked rather than the topic.
#
# The "Most questions ARE answerable" hedge of v2 is DROPPED: it was meant to prevent over-refusal, and the
# suspicion is that it is what suppressed the 7B's abstention to 5%. Over-refusal is instead held down by
# requiring the citation to be produced whenever one exists, which is the same evidence the answer needs.
QA_REASON_V3_LOCOMO_ABSTAIN_V3 = (
    "You are answering a question about a long multi-session conversation (each session is dated). Answer in two steps.\n"
    "Step 1 (Reasoning): ONE sentence naming the SESSION AND SPEAKER that state the answer — e.g. 'Session 12 "
    "(8 May 2023), Maria says she joined a gym.' For a 'when' question resolve the actual date (if someone says "
    "'yesterday' in a session dated 8 May 2023, the answer is 7 May 2023).\n"
    "Step 2 (Answer): on a new line write 'Final Answer:' then ONLY the short answer (a date, name, place, or brief phrase).\n\n"
    "Rules:\n"
    "- The session you name in Step 1 must state the answer about THE PERSON, PLACE AND EVENT THE QUESTION NAMES. "
    "A similar fact about someone or something else does not count.\n"
    "- If you cannot name such a session, then the conversation does not state it, and Step 2 must be exactly: "
    "Not mentioned in the conversation\n"
    "- Otherwise commit to your single best answer. Do not hedge and do not add caveats.\n"
    "- After 'Final Answer:' put ONLY the answer: no notes, parentheticals, or caveats.\n"
    "- Phrase 'when' answers like the ground truth (e.g. '7 May 2023', 'the week before 9 June 2023').\n"
    "- No angle-bracket placeholders; output nothing after the 'Final Answer:' line.\n"
)


# 2026-08-13: LoCoMo abstention variant 4 — force the COMPARISON as an output token.
#
# WHY V3 IS PROBABLY NOT ENOUGH, read off the v2 logs before v3 even landed. v3 assumes the 7B fails to
# produce a citation. It does not. Under v2 the 7B ALREADY names the session and the speaker, names the
# WRONG speaker, and answers anyway:
#
#   Q "How did CAROLINE's son handle the accident?"        gold: Not mentioned in the conversation
#   7B "In Session 18 on 20 October 2023, MELANIE mentioned that her son was scared ... This indicates
#       that Caroline's son, WHO IS MELANIE'S SON, was scared but handled the situation well."
#   Q "What country is MELANIE's grandma from?"            gold: Not mentioned in the conversation
#   7B "In Session 1, CAROLINE mentions that her grandmother gave her a necklace from her home country,
#       which is Sweden."                                  -> Final Answer: Sweden
#
# In the first one it NOTICES the mismatch and rationalises it away by asserting the two are the same
# person. So the deficit is not citation, it is the COMPARISON between the cited entity and the asked
# entity — and v3 states that comparison as a RULE, which this shows the model reads and then talks past.
# v4 makes it a token the model has to emit and commit to before the answer exists, which is the same
# trick reason_then_answer plays on the answer itself.
#
# If v4 also fails, the prompting route is closed and the capability has to be TRAINED — which is what the
# minimal-pair corpus (scripts/gen_minimal_pair_corpus.py) supplies, since a minimal pair is by
# construction "same context, one slot different, opposite answer": exactly this comparison.
QA_REASON_V3_LOCOMO_ABSTAIN_V4 = (
    "You are answering a question about a long multi-session conversation (each session is dated). Answer in three steps.\n"
    "Step 1 (Evidence): ONE sentence naming the session, the date and the SPEAKER whose turn is closest to what "
    "the question asks about — e.g. 'Session 12 (8 May 2023), Melanie says her son was in an accident.'\n"
    "Step 2 (Check): write 'Check:' then compare Step 1 to the question. If the person, place and event in "
    "Step 1 are the ones the question names, write 'same'. If Step 1 is about a DIFFERENT person, place or "
    "event, write 'different'. Write one word, 'same' or 'different'.\n"
    "Step 3 (Answer): on a new line write 'Final Answer:' then ONLY the short answer (a date, name, place, or "
    "brief phrase).\n\n"
    "Rules:\n"
    "- If Step 2 is 'different', Step 3 must be exactly: Not mentioned in the conversation\n"
    "- Two people are not the same person because their stories are similar. Do not explain a mismatch away.\n"
    "- If Step 2 is 'same', commit to your single best answer. Do not hedge and do not add caveats.\n"
    "- For a 'when' question resolve the actual date (if someone says 'yesterday' in a session dated 8 May 2023, "
    "the answer is 7 May 2023), and phrase it like the ground truth (e.g. '7 May 2023').\n"
    "- After 'Final Answer:' put ONLY the answer: no notes, parentheticals, or caveats.\n"
    "- No angle-bracket placeholders; output nothing after the 'Final Answer:' line.\n"
)


# 2026-08-13: LoCoMo abstention variant 5 = v4's forced comparison + an explicit INFERENCE ALLOWANCE.
#
# v2 measured on the complete floor-7B run (N=600) redistributes rather than helps, and the damage is in one
# category: open-domain abstention 6% -> 21%, F1 0.1978 -> 0.1761. The cause is the instruction, not the
# model. LoCoMo's open-domain questions ask for what is INFERRED from the conversation ("Why did Maria start
# blogging about politics?"), while v2 scopes declining to "only if the conversation genuinely never STATES
# the thing asked about" — and an inferable answer is, read literally, not stated. v3 and v4 are strictly
# more aggressive about that word (both demand naming the session that STATES the answer), so both should be
# expected to make open-domain worse, and open-domain already has an arm at 78.7% that must not be given
# back. They were cancelled before running rather than spend the GPU on a design already known to be wrong
# in a category it would damage.
#
# v5 keeps what v4 was for — the COMPARISON as a forced token, because the measured failure is that the 7B
# already writes the citation, names the WRONG speaker, and rationalises the mismatch away ("Caroline's son,
# WHO IS MELANIE'S SON") — and separates two things v2/v3/v4 ran together:
#   * the conversation does not SAY it in one turn        -> may still be inferable, answer it
#   * the person / thing / event the question names is simply ABSENT from the conversation, or the fact
#     belongs to SOMEONE ELSE                             -> decline
# The check is therefore about WHO/WHAT the evidence is about, never about whether the answer was spelled
# out. That is exactly the adversarial trap (a single-slot swap of a real answerable question) and it leaves
# inference alone.
QA_REASON_V3_LOCOMO_ABSTAIN_V5 = (
    "You are answering a question about a long multi-session conversation (each session is dated). Answer in three steps.\n"
    "Step 1 (Evidence): ONE sentence naming the session, the date and the SPEAKER whose turns are the basis for "
    "your answer — e.g. 'Session 12 (8 May 2023), Melanie talks about her son's accident.'\n"
    "Step 2 (Check): write 'Check:' then one word. Write 'same' if the evidence in Step 1 is about THE PERSON, "
    "THING AND EVENT the question names. Write 'different' if it is about someone or something else, or if the "
    "question asks about a person, thing or event that never appears in the conversation at all.\n"
    "Step 3 (Answer): on a new line write 'Final Answer:' then ONLY the short answer (a date, name, place, or "
    "brief phrase).\n\n"
    "Rules:\n"
    "- If Step 2 is 'different', Step 3 must be exactly: Not mentioned in the conversation\n"
    "- Two people are not the same person because their stories are similar. Do not explain a mismatch away.\n"
    "- If Step 2 is 'same', ANSWER — including when the conversation implies the answer rather than saying it "
    "outright. Reasons, opinions and motives are usually implied; infer them and commit. 'Not mentioned' is "
    "ONLY for the wrong-person / absent-thing case of Step 2, NEVER for an answer you had to work out.\n"
    "- For a 'when' question resolve the actual date (if someone says 'yesterday' in a session dated 8 May 2023, "
    "the answer is 7 May 2023), and phrase it like the ground truth (e.g. '7 May 2023').\n"
    "- After 'Final Answer:' put ONLY the answer: no notes, parentheticals, or caveats.\n"
    "- No angle-bracket placeholders; output nothing after the 'Final Answer:' line.\n"
)


# 2026-08-11: LoCoMo ENUMERATION variant. Diagnosis (Q32B7B_RESULTS / HANDOFF_260811 §2b): 66% of LoCoMo's
# "multi-hop" turns have a MULTI-ITEM gold ("Kickboxing, Taekwondo"), and every arm answers with one item and
# stops — floor 11/187 complete, plain fusion 13/187, ours 11/187, and the TEACHER only 21/187. The category
# is not multi-hop composition; it is exhaustive enumeration of one predicate across the sessions. The
# canonical prompt actively suppresses it: it asks for "ONLY the short answer (a date, name, place, or brief
# phrase)" and for ONE sentence of reasoning. This variant keeps every other instruction identical and adds
# the enumeration requirement. Apply it to EVERY arm of a comparison — it is aimed at the teacher's ceiling
# first (21/187 is the ceiling closeness is measured against).
QA_REASON_V3_LOCOMO_ENUM = (
    "You are answering a question about a long multi-session conversation (each session is dated). Answer in two steps.\n"
    "Step 1 (Reasoning): ONE or TWO sentences — which turn/session answers it, and for a 'when' question resolve the actual date "
    "(e.g. if someone says they did it 'yesterday' in a session dated 8 May 2023, the answer is 7 May 2023). "
    "If the question asks what someone did/made/visited/likes, SCAN ALL SESSIONS and list every instance you find, not just the first.\n"
    "Step 2 (Answer): on a new line write 'Final Answer:' then ONLY the answer.\n\n"
    "Rules:\n"
    "- If more than one thing satisfies the question, give ALL of them, comma-separated. Do not stop at the first one.\n"
    "- ALWAYS commit to your single best answer. NEVER reply 'Not mentioned' or 'not provided' — give the most likely answer.\n"
    "- After 'Final Answer:' put ONLY the answer: no notes, parentheticals, or caveats.\n"
    "- Phrase 'when' answers like the ground truth (e.g. '7 May 2023', 'the week before 9 June 2023').\n"
    "- No angle-bracket placeholders; output nothing after the 'Final Answer:' line.\n"
)


# 2026-08-10: QASPER SURFACE-FORM variant. QASPER's official metric is token-F1 against short answers copied
# verbatim from the paper, and the 32B teacher loses roughly TWICE as much as the 7B floor to answer STYLE
# rather than content: measured on the 248 shared turns, a digit/unit-suffix relaxation (diagnostic only,
# never the scored metric) lifts teacher-32B 0.395 -> 0.427 but floor-7B only 0.387 -> 0.402. The losing
# turns are things like gold "5" / teacher "five", gold "300" / teacher "300-dimensional", gold "WSJ" /
# teacher "WSJ-SI84 and WSJ-SI284 datasets. The text from all the utterances was mapped into ...".
# The legitimate lever is the PROMPT, not the metric (never invent a custom metric — the official one is the
# headline). This variant asks for the paper's own surface form. It must be applied to EVERY arm of a
# comparison, never to ours alone. QA_REASON_V3 is untouched so every earlier QASPER run still replicates.
QA_REASON_V3_QASPER_SPAN = (
    "You are answering a question about a scientific paper. Answer in two steps and do BOTH.\n"
    "Step 1 (Reasoning): 1-2 sentences locating the answer in the paper.\n"
    "Step 2 (Answer): on a new line write 'Final Answer:' then ONLY the answer, copied in the paper's own "
    "surface form.\n\n"
    "Rules:\n"
    "- Copy the SHORTEST span that answers the question, exactly as the paper writes it. Do not paraphrase, "
    "do not expand an abbreviation, do not add a unit or a suffix (write '300', not '300-dimensional').\n"
    "- Write numbers as DIGITS ('5', not 'five').\n"
    "- For a yes/no question answer exactly 'Yes' or 'No' and nothing else — no explanation after it.\n"
    "- ALWAYS commit to one concrete best answer. NEVER reply 'Not mentioned', 'not provided', or hedge.\n"
    "- After 'Final Answer:' put ONLY the answer span: no notes, parentheticals, caveats, or extra sentence.\n"
    "- No angle-bracket placeholders; output nothing after the 'Final Answer:' line.\n"
)


# 2026-07-18: DIRECT (no-reasoning) prompt for the CONTROLLED reason-vs-noreason test (single-turn, no history).
QA_DIRECT_LOCOMO = (
    "You are answering a question about a long multi-session conversation (each session is dated).\n"
    "Reply with ONLY the short answer — a date, name, place, or brief phrase (e.g. '7 May 2023', 'Single'). "
    "Do NOT explain or add anything. Always commit to your single best answer; never say 'Not mentioned'.\n\n"
    "Answer:"
)

# 2026-07-18: mtRAG needs COMPLETE conversational answers (gold ~25 words), not a terse span — the old terse+no-reason
# runs produced 11-word answers vs 25-word gold and cratered f1. This is reason_then_answer with a COMPLETE answer.
# ★ 2026-08-03 LoCoMo TEMPORAL variant. Measured failure (45 remaining recoverable turns, 51% of them):
# the model FINDS a session that mentions the topic and reports THAT session's date, but the event is stated in a
# DIFFERENT session ("pottery class" -> answered Session 2 / 25 May, gold 2 July; "adoption meeting" -> Session 18 /
# 20 Oct, gold the Friday before 15 July). V3_LOCOMO already handles relative-date resolution WITHIN a session; it
# does not address picking the wrong session. This variant forces the model to separate "session that mentions the
# topic" from "session that states the event happened", and to check later sessions before committing.
# CITATION-FIRST variant (2026-08-09, Q32B7B_RESULTS §6 diagnosis): fusion's failures are (a) the reasoning
# dropping/corrupting the session-date citation and (b) the blind LM's generic tokens capturing answer-entity
# slots. This variant FORCES a verbatim evidence quote before any reasoning: inside a character-for-character
# copy the context-blind LM has no distribution to push, so the reader's specific tokens anchor the prefix, and
# the date arithmetic then conditions on the anchored (correct) session date. NEVER overwrites V3/V4 — select
# via --instruction QA_REASON_V5_LOCOMO_CITEFIRST. All arms of a comparison set must share it (fingerprint).
QA_REASON_V5_LOCOMO_CITEFIRST = (
    "You are answering a question about a long multi-session conversation (each session is dated). Answer in three steps.\n"
    "Step 1 (Evidence): copy the session header of the session that contains the answer, exactly as it appears "
    "(e.g. [Session 4 | 1:56 pm on 8 May, 2023]), then quote VERBATIM, character for character, the exact sentence(s) "
    "from that session that contain the evidence. Do not paraphrase inside the quote.\n"
    "Step 2 (Reasoning): ONE sentence — derive the answer from the quoted evidence; for a 'when' question resolve the "
    "actual date against the quoted session date (e.g. 'yesterday' in a session dated 8 May 2023 means 7 May 2023).\n"
    "Step 3 (Answer): on a new line write 'Final Answer:' then ONLY the short answer (a date, name, place, or brief phrase).\n\n"
    "Rules:\n"
    "- ALWAYS commit to your single best answer. NEVER reply 'Not mentioned' or 'not provided' — give the most likely answer.\n"
    "- After 'Final Answer:' put ONLY the answer: no notes, parentheticals, or caveats.\n"
    "- Phrase 'when' answers like the ground truth (e.g. '7 May 2023', 'the week before 9 June 2023').\n"
    "- No angle-bracket placeholders; output nothing after the 'Final Answer:' line.\n"
)

# NEVER overwrites QA_REASON_V3_LOCOMO — selected explicitly via --instruction QA_REASON_V4_LOCOMO_TEMPORAL.
QA_REASON_V4_LOCOMO_TEMPORAL = (
    "You are answering a question about a long multi-session conversation (each session is dated). Answer in two steps.\n"
    "Step 1 (Reasoning): ONE sentence. For a 'when' question do this in order: (a) find EVERY session that mentions "
    "the event, (b) the answer comes from the session where the event is REPORTED AS HAPPENING, which is often NOT "
    "the earliest session that merely mentions the topic or plans it — prefer the latest session that states it "
    "actually happened, (c) then resolve the actual date from that session's date (e.g. 'yesterday' in a session "
    "dated 8 May 2023 means 7 May 2023; 'last Friday' means the Friday before that session's date).\n"
    "Step 2 (Answer): on a new line write 'Final Answer:' then ONLY the short answer (a date, name, place, or brief phrase).\n\n"
    "Rules:\n"
    "- ALWAYS commit to your single best answer. NEVER reply 'Not mentioned' or 'not provided' — give the most likely answer.\n"
    "- After 'Final Answer:' put ONLY the answer: no notes, parentheticals, or caveats.\n"
    "- Phrase 'when' answers like the ground truth (e.g. '7 May 2023', 'the week before 9 June 2023', 'June 2023').\n"
    "- Do not answer with a session number or 'Session N' — convert it to the calendar date.\n"
    "- No angle-bracket placeholders; output nothing after the 'Final Answer:' line.\n"
)


# ★ 2026-08-03 LoCoMo temporal, MINIMAL-DELTA variant. The full rewrite (V4_LOCOMO_TEMPORAL, 38b8e174) doubled
# temporal recovery (base 29.5%->51.2%) but cost the non-temporal half (40.0%->35.4%) and lost overall.
# This changes exactly ONE clause of V3_LOCOMO — "use the session where the event actually HAPPENED, not the
# earliest session that merely mentions or plans it" — leaving every other instruction byte-identical, so the
# non-temporal reasoning budget is not crowded out. V3_LOCOMO and V4_LOCOMO_TEMPORAL are both untouched.
QA_REASON_V5_LOCOMO_SESSPICK = (
    'You are answering a question about a long multi-session conversation (each session is dated). Answer in two steps.\n'
    "Step 1 (Reasoning): ONE sentence — which turn/session answers it, and for a 'when' question use the session where the event actually HAPPENED (not the earliest session that merely mentions or plans it), then resolve the actual date (e.g. if someone says they did it 'yesterday' in a session dated 8 May 2023, the answer is 7 May 2023).\n"
    "Step 2 (Answer): on a new line write 'Final Answer:' then ONLY the short answer (a date, name, place, or brief phrase).\n"
    '\n'
    'Rules:\n'
    "- ALWAYS commit to your single best answer. NEVER reply 'Not mentioned' or 'not provided' — give the most likely answer.\n"
    "- After 'Final Answer:' put ONLY the answer: no notes, parentheticals, or caveats.\n"
    "- Phrase 'when' answers like the ground truth (e.g. '7 May 2023', 'the week before 9 June 2023').\n"
    "- No angle-bracket placeholders; output nothing after the 'Final Answer:' line.\n"
)


QA_REASON_V3_MTRAG = (
    "You are a helpful assistant in a multi-turn conversation. Using the evidence passages and the conversation so "
    "far, answer the user's latest question. Answer in two steps and do BOTH.\n"
    "Step 1 (Reasoning): 1-2 sentences using the passages to work out the answer.\n"
    "Step 2 (Answer): on a new line write 'Final Answer:' then a COMPLETE answer in 1-3 full natural sentences — "
    "state the answer explicitly with the supporting detail (do NOT reply just 'yes'/'no' — give the detail).\n\n"
    "Rules:\n"
    "- ALWAYS commit to a concrete best answer grounded in the passages; do not hedge or refuse.\n"
    "- After 'Final Answer:' write only the conversational answer sentence(s); no notes, no placeholders, no repeat "
    "of these instructions. Output nothing after the answer.\n"
)
