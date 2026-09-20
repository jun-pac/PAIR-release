"""Turn-wrapping for chat-only model families — ONE definition shared by eval (mtrag_accum), the
teacher generator (binding_teacher_gen) and both SFT trainers, so the three stages render the same
bytes (2026-09-05).

CHAT_WRAP=0      raw prompt (Qwen2.5: no BOS, follows the instruction as plain text)
CHAT_WRAP=1      ChatML  <|im_start|>user … <|im_end|>\n<|im_start|>assistant\n   (OLMo-3; 2026-08-30)
CHAT_WRAP=gemma  Gemma-3 <bos><start_of_turn>user … <end_of_turn>\n<start_of_turn>model\n

Why gemma is its own mode (user, 2026-09-05: one-off, Gemma gets its own format). Gemma-3-it needs
<bos> and its turn markers; fed the raw prompt with add_special_tokens=False it never sees either, and
it then treats the instruction as text to continue: it echoes the reminder, answers with no reasoning
(teacher 21% straight to 'Final Answer:', 4B floor 47%, the λ0.7 pool 68%, the adapters cloned from
those outputs 100%). Under the July harness, which tokenised with BOS, the same pair reasoned on
60–81% of outputs. The tokenizer parses the literal markers as single special tokens
(<bos>=2, <start_of_turn>=105, <end_of_turn>=106), checked 2026-09-05.
"""
import os

MODE = os.environ.get("CHAT_WRAP", "0")
ON = MODE in ("1", "gemma")
if MODE == "gemma":
    OPEN, ASSIST, REOPEN = ("<bos><start_of_turn>user\n",
                            "<end_of_turn>\n<start_of_turn>model\n",
                            "<end_of_turn>\n<start_of_turn>user\n")
    END_TOKEN = "<end_of_turn>"
elif MODE == "1":
    OPEN, ASSIST, REOPEN = ("<|im_start|>user\n",
                            "<|im_end|>\n<|im_start|>assistant\n",
                            "<|im_end|>\n<|im_start|>user\n")
    END_TOKEN = "<|im_end|>"
else:
    OPEN = ASSIST = REOPEN = ""
    END_TOKEN = None


def wrap(body):
    """the full single-turn rendering: OPEN + body + ASSIST (identity when off)"""
    return (OPEN + body + ASSIST) if ON else body


def end_token_id(tok):
    """the turn-end id to add to the stop set, or None"""
    if not END_TOKEN:
        return None
    i = tok.convert_tokens_to_ids(END_TOKEN)
    return i if (i is not None and i >= 0) else None
