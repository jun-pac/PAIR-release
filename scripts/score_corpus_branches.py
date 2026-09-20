#!/usr/bin/env python
"""Score every corpus row by what EACH BRANCH already predicts — to find where L-fusion can learn to DEFER.

WHY. Four corpus versions (v8 → v9 DERIVE → CHAIN → CHAIN2) changed WHAT the LM branch says and none of
them changed WHEN it intervenes. The evidence is that the damage rate never moved: on CLUTRR, the fraction
of already-correct turns the arm breaks is 31 / 29 / 28 / 31 % across those four. On hotpotQA the wins and
losses are structurally indistinguishable — two candidate separating features (the answer dropping every
gold named entity; the answer being the right type but the wrong instance) were proposed from verbatim
reads and BOTH died when counted over the whole set (9/19 vs 8/20, and 82% vs 81%). The intervention is
unselective.

There is a mechanical reason it cannot become selective under the current training. The loss is
cross-entropy on the FUSED logits toward the teacher's text with the reader FROZEN. Where the reader alone
already predicts the teacher's tokens, the fused distribution is already right and the loss is already
small, so the LM branch receives almost no gradient — it is never taught to hold still. Where the reader is
wrong the loss is large, so the only lesson the branch ever gets is to PUSH. No amount of new row CONTENT
changes that, because it is a property of which rows carry gradient.

The rows that DO carry a deference gradient are the ones where the reader is right AND the LM branch's own
query-only prediction disagrees: there the fused distribution is being dragged off a correct answer at
(1−λ) weight, the loss is NOT small, and the gradient points at the branch backing off. This script finds
them, by teacher-forcing both branches over each row's target and recording where each one already agrees
with it.

Both branches see exactly what they see in `fusion_stage2_lm_sft.py` and in `mtrag_accum.py`:
    reader : INSTRUCTION + "\n\n" + context + history + question
    LM     : INSTRUCTION                   + history + question      (no context)
so the numbers describe the branches as they are actually deployed.

Per row it writes: agreement over ALL target tokens and over the ANSWER tokens only (those following
"Final Answer:"), for each branch. The answer positions are what matters — a branch can track the teacher's
reasoning prose and still fight it on the committed answer.

Forward-only and batched (length-bucketed): no optimizer, no backward.
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import load_causal_lm
import src.qa_prompts as _QP
from scripts.mab_eval import q_turn


def _fwd(model, ids, keep):
    """forward returning only the last `keep` positions' logits.

    transformers exposes this as `logits_to_keep` (older builds: `num_logits_to_keep`); if neither is
    accepted we fall back to the full tensor, which is correct but is what OOM'd on the longest rows."""
    for kw in ("logits_to_keep", "num_logits_to_keep"):
        try:
            return model(ids, use_cache=False, **{kw: keep}).logits
        except TypeError:
            continue
    return model(ids, use_cache=False).logits


def _agree(logits, tids, start):
    """fraction of target positions whose argmax equals the target token; (all, answer-only)."""
    pred = logits.argmax(-1)
    hit = (pred == tids).float()
    all_acc = hit.mean().item()
    ans_acc = hit[start:].mean().item() if start < hit.numel() else float("nan")
    return all_acc, ans_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--slm-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--slm-lora", required=True, help="the frozen reader whose deployment this describes")
    ap.add_argument("--lm-model", default="Qwen/Qwen2.5-32B-Instruct")
    ap.add_argument("--lm-lora", default=None, help="optional: score an already-L-fusion'd branch instead")
    ap.add_argument("--max-length", type=int, default=28000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
    a = ap.parse_args()
    if a.batch < 2:
        print("[ABORT] --batch must be >= 2 (project rule: flash-attention does not compile at batch 1)",
              file=sys.stderr)
        return 3

    rows = [json.loads(l) for l in open(a.corpus)]
    rows = [r for r in rows if str(r.get("target") or "").strip()]
    if a.limit:
        rows = rows[:a.limit]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["example_id"] for l in open(a.out)}
        print(f"[resume] {len(done)} rows already scored", flush=True)
    rows = [r for r in rows if r["example_id"] not in done]
    if not rows:
        print("[Done] nothing to score"); return 0

    from peft import PeftModel
    slm, stok = load_causal_lm(a.slm_model, cache_dir=a.cache_dir, device_map=a.device)
    slm = PeftModel.from_pretrained(slm, a.slm_lora); slm.eval()
    lm, ltok = load_causal_lm(a.lm_model, cache_dir=a.cache_dir, device_map=a.device)
    if a.lm_lora:
        lm = PeftModel.from_pretrained(lm, a.lm_lora)
    lm.eval()
    sdev, ldev = next(slm.parameters()).device, next(lm.parameters()).device
    for tk in (stok, ltok):
        if tk.pad_token_id is None:
            tk.pad_token = tk.eos_token

    # build both prompts per row exactly as the trainer/harness does
    items = []
    for r in rows:
        instr = getattr(_QP, r.get("instruction") or "QA_REASON_V3_LOCOMO")
        qt = r.get("history", "") + q_turn(r["question"])
        tgt = r["target"] + "\n"
        tids = stok(tgt, return_tensors="pt", add_special_tokens=False).input_ids
        if tids.shape[1] < 2:
            continue
        # first ANSWER token = the one after the last "Final Answer:" in the target
        cut = tgt.rfind("Final Answer:")
        start = 0 if cut < 0 else stok(tgt[:cut + len("Final Answer:")],
                                       return_tensors="pt", add_special_tokens=False).input_ids.shape[1]
        items.append(dict(r=r, sp=instr + "\n\n" + r["context"] + qt, lp=instr + qt,
                          tids=tids, start=min(start, tids.shape[1] - 1)))
    items.sort(key=lambda x: len(x["sp"]))          # length-bucket so a batch pads to its own longest

    fh = open(a.out, "a"); t0 = time.perf_counter(); n = 0
    for s in range(0, len(items), a.batch):
        chunk = items[s:s + a.batch]
        out = []
        for it in chunk:                            # per-row forward; the batching win is the sort+loop
            T = it["tids"].shape[1]
            sp = stok(it["sp"], return_tensors="pt", add_special_tokens=False).input_ids
            if sp.shape[1] + T > a.max_length:
                sp = sp[:, -(a.max_length - T):]
            lp = ltok(it["lp"], return_tensors="pt", add_special_tokens=False).input_ids
            # ★ keep only the logits we score. A full [1, seq, vocab] tensor is 28000 x 152k x 2B = 8.5 GiB
            #   at max_length, which OOM'd this job at row 8008 of 8682 — exactly on the longest REPLAY
            #   contexts, the rows the ASSERT question needs. We only ever read the T positions ending one
            #   before the last, so ask the model for the last T+1.
            with torch.no_grad():
                so = _fwd(slm, torch.cat([sp, it["tids"]], 1).to(sdev), T + 1)
                lo = _fwd(lm, torch.cat([lp, it["tids"]], 1).to(ldev), T + 1)
            sl = so[0, -(T + 1):-1, :]
            ll = lo[0, -(T + 1):-1, :]
            tt = it["tids"][0]
            r_all, r_ans = _agree(sl, tt.to(sl.device), it["start"])
            l_all, l_ans = _agree(ll, tt.to(ll.device), it["start"])

            # ★ PRIOR — the number the A/B split actually needs, and the one the teacher-forced scores above
            #   CANNOT give. Under teacher forcing both branches receive the target PREFIX, and the teacher's
            #   reasoning has normally already named the answer ("...therefore the plotter is the lightest.
            #   Final Answer: plotter"), so a branch scores well by COPYING FROM ITS OWN INPUT. Measured that
            #   way the context-blind LM matched the reader on `chain` rows (0.630 vs 0.641) — on rows whose
            #   answer is a specific entity it cannot see. That is the copy channel, not knowledge.
            #   Here the reasoning is REMOVED: each branch gets only its prompt plus "\nFinal Answer:" and
            #   must produce the answer tokens from what it actually knows.
            ans_ids = it["tids"][:, it["start"]:]
            r_pri = l_pri = float("nan")
            if ans_ids.shape[1] > 0:
                sfa = stok(it["sp"] + "\nFinal Answer:", return_tensors="pt",
                           add_special_tokens=False).input_ids
                if sfa.shape[1] + ans_ids.shape[1] > a.max_length:
                    sfa = sfa[:, -(a.max_length - ans_ids.shape[1]):]
                lfa = ltok(it["lp"] + "\nFinal Answer:", return_tensors="pt",
                           add_special_tokens=False).input_ids
                A = ans_ids.shape[1]
                with torch.no_grad():
                    sp2 = _fwd(slm, torch.cat([sfa, ans_ids], 1).to(sdev), A + 1)
                    lp2 = _fwd(lm, torch.cat([lfa, ans_ids], 1).to(ldev), A + 1)
                r_pri = _agree(sp2[0, -(A + 1):-1, :], ans_ids[0].to(sp2.device), 0)[0]
                l_pri = _agree(lp2[0, -(A + 1):-1, :], ans_ids[0].to(lp2.device), 0)[0]
            out.append(dict(example_id=it["r"]["example_id"],
                            kind=(it["r"].get("meta") or {}).get("kind", "v8-retrieve/unans"),
                            replay=("-replay-" in str(it["r"]["example_id"])),
                            n_target_tokens=int(T), n_answer_tokens=int(T - it["start"]),
                            reader_all=r_all, reader_ans=r_ans, lm_all=l_all, lm_ans=l_ans,
                            reader_prior=r_pri, lm_prior=l_pri))
        for o in out:
            fh.write(json.dumps(o) + "\n")
        fh.flush(); n += len(out)
        if (s // a.batch) % 20 == 0:
            print(f"[{n}/{len(items)}] {(time.perf_counter()-t0)/max(n,1):.2f}s/row", flush=True)
    fh.close()
    print(f"[Done] {n} rows -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
