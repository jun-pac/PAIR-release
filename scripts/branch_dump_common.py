"""Shared basis for every reader of the teacher-forced branch dump (2026-09-05).

The dump (scripts/dump_branch_logits.py, job 3012898) ran each teacher trajectory to a fixed 200 tokens
and stopped only on the tokenizer's eos id (<|im_end|>), but the teacher ends its answer with
<|endoftext|> (id 151643). Everything after that token is the model continuing past its own end —
prompt recitation, a new "Human:" turn — and 7,232 of the 12,000 dumped positions are that.
The segment triple (reason_end, answer_start, answer_end) was cut at the 'Final Answer:' marker
only, so 1,799 of those junk positions were labelled reasoning or answer. Every analysis reads the
segment through clean_seg(), which clips all three bounds at the first <|endoftext|> (the EOT position
itself is kept: predicting the end of the answer is a real decision).
"""
import numpy as np

EOT = 151643


def first_eot(gen):
    g = np.asarray(gen)
    w = np.where(g == EOT)[0]
    return int(w[0]) + 1 if len(w) else len(g)


def clean_seg(gen, seg):
    """(reason_end, answer_start, answer_end) clipped at the teacher's first <|endoftext|>."""
    cut = first_eot(gen)
    r, a0, a1 = (int(v) for v in seg)
    return min(r, cut), min(a0, cut), min(a1, cut)
