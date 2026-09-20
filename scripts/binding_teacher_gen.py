#!/usr/bin/env python
"""TEACHER-32B target generation over the synthetic binding corpus (2026-08-10) — the FAIRNESS FIX.

The v1–v4 reader SFTs trained on the generator's GOLD targets. That violates the project's FT doctrine
(FT clones the TEACHER, never gold) and pollutes closeness-to-teacher: the student received supervision
the teacher never provides, so "recovering the teacher" stops being the measured thing. All v1–v4
reader-FT results are VOID as method claims (kept as diagnostics only).

This script produces the doctrine-compliant targets: teacher-32B reads each synthetic context and
generates under the row's own instruction; whatever it outputs IS the target (no gold gate, no
curation — usability gates only, same as ft_mix_teacher_gen). Resumable: skips example_ids already in
OUT. Shardable via --shard k --nshards n.
"""
import argparse, json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import load_causal_lm
import src.qa_prompts as QP
from scripts.mab_eval import q_turn
from scripts.teacher_stop_rule import apply_stop_after_answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="synthetic rows {context,question,instruction,...}")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
    ap.add_argument("--batch", type=int, default=8,
                    help="rows per generate() call. MUST be >= 2 (project rule: flash-attention's graph "
                         "does not compile at batch 1, so batch 1 is a guaranteed throughput loss).")
    a = ap.parse_args()
    # ★ batch >= 2 is a hard project rule, enforced here rather than in prose. This script looped one row
    #   at a time until 2026-08-16 and took 1h47m for 1400 short rows; that is the cost of the rule living
    #   only in CLAUDE.md. FORCE_BATCH1 requires a written reason, same contract as the eval harnesses.
    if a.batch < 2:
        why = os.environ.get("FORCE_BATCH1", "")
        if not why:
            print(f"[ABORT] --batch {a.batch} < 2. Set FORCE_BATCH1='<reason>' to override.", file=sys.stderr)
            return 3
        print(f"[FORCE_BATCH1] {why}", flush=True)
    rows = [json.loads(l) for l in open(a.corpus)]
    rows = [r for i, r in enumerate(rows) if i % a.nshards == a.shard]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["example_id"] for l in open(a.out)}
        print(f"[resume] {len(done)} already generated", flush=True)
    model, tok = load_causal_lm(a.model, cache_dir=a.cache_dir, device_map="cuda:0"); model.eval()
    dev = next(model.parameters()).device
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"          # decoder-only generation requires left padding

    todo = [r for r in rows if r["example_id"] not in done]
    # CHAT_WRAP=1 renders the SAME ChatML turn wrapping that scripts/mtrag_accum.py applies at eval
    # time (its _CW_OPEN / _CW_ASSIST). Required for OLMo-3, which emits EOS at step 0 on this
    # harness's raw prompt, and it must be identical in ALL THREE stages -- this generation, the
    # reader SFT, and stage 2 -- or the reader is distilled against a prompt it never sees again.
    # Default off, so every Qwen and gemma prompt stays byte-identical to what has already been run.
    from scripts import chat_wrap as _CWM          # 2026-09-05: one definition for all three stages
    _cw = _CWM.ON
    _OPEN, _ASSIST = _CWM.OPEN, _CWM.ASSIST
    prompts = {}
    for r in todo:
        instr = getattr(QP, r.get("instruction") or "QA_REASON_V3_LOCOMO")
        body = instr + "\n\n" + r["context"] + "\n" + q_turn(r["question"])
        prompts[r["example_id"]] = (_OPEN + body + _ASSIST) if _cw else body
    if _cw:
        print(f"[Info] CHAT_WRAP={_CWM.MODE} — turn wrapping {_OPEN!r} … {_ASSIST!r}, matching mtrag_accum's eval rendering",
              flush=True)
    # length-bucket so a batch is padded to its own longest member rather than the corpus's
    todo.sort(key=lambda r: len(prompts[r["example_id"]]))

    f = open(a.out, "a"); t0 = time.perf_counter(); n = 0
    for s in range(0, len(todo), a.batch):
        chunk = todo[s:s + a.batch]
        enc = tok([prompts[r["example_id"]] for r in chunk], return_tensors="pt",
                  add_special_tokens=False, padding=True).to(dev)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        gen = out[:, enc.input_ids.shape[1]:]        # left padding ⇒ one shared input length
        for r, g in zip(chunk, gen):
            txt = tok.decode(g, skip_special_tokens=True).strip()
            # the teacher is DEPLOYED with --stop-after-answer "Final Answer:"; generate() here has no stop
            # criteria, so apply the same rule to what it wrote (see scripts/teacher_stop_rule.py)
            stopped, terminated = apply_stop_after_answer(txt)
            rec = dict(r); rec["teacher_text_raw"] = txt; rec["teacher_text"] = stopped
            rec["hit_max_new"] = int(not terminated)
            f.write(json.dumps(rec) + "\n"); n += 1
        f.flush()
        print(f"[{n}/{len(todo)}] {(time.perf_counter()-t0)/max(n,1):.2f}s/ex (batch {len(chunk)})", flush=True)
    f.close(); print(f"[Done] {n} teacher targets -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
