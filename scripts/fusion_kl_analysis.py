#!/usr/bin/env python
"""The fusion analysis, rebuilt on divergence instead of top-1 coverage (2026-09-01). CPU ONLY.

WHY THIS REPLACES §1 AND THE THEORY OF reports/fusion_microscope.html
--------------------------------------------------------------------
The old §1 asked "is the teacher's next token the top-1 of at least one branch?" and answered 92%.
Two things are wrong with that number. It is a hard 0/1 on the argmax, so a branch that puts 0.49 on
the teacher's token and 0.50 on a synonym counts as a total failure, identical to a branch that puts
nothing there at all. And read as a ceiling it is not even consistent with our own results — hotpot
token-F1 is above it. Top-5 only moves the cutoff; embedding cosine leaves the next-token mechanism
entirely. The right object is the one the method actually manipulates: the whole predictive
distribution, measured against the teacher's own.

The old theory section derived linear fusion from an exponential-tilt objective — maximise the inner
product with the SLM's "evidence score" subject to a KL budget around the LM. The constraint was
never motivated; it is a construction that happens to yield the rule we already use.

Replaced by an IDENTITY, not a derivation. With z_F = λ z_S + (1−λ) z_L,
      p_F(v) ∝ p_S(v)^λ · p_L(v)^(1−λ)                                        (log-opinion pool)
      p_F    = argmin_q [ λ·KL(q‖p_S) + (1−λ)·KL(q‖p_L) ]                     (variational form)
Both are exact and assumption-free — they are what the operation IS, not a story about why it should
work, and both are two-line PROOFS (stated on the report page). This script does NOT "verify" them
numerically. An earlier version did, and reported max |Δ log p| = 1.1e-5 between mixing logits and
mixing log-probabilities as if that were evidence: the two expressions differ by exactly a per-row
constant, which softmax discards, so the residual measured float32 rounding (eps x the log-prob
magnitude, ~7.5e-6 here) and nothing about the claim. It was also computed on ONE of the 60
examples while the page implied the whole dump. Removed rather than rescoped, because a bigger
sample of a vacuous check is still vacuous.

WHAT IS MEASURED
----------------
Teacher-forced, so the teacher, both branches and every fusion see the identical prefix at every
position (`dump_branch_logits.py`; 60 hotpotQA examples × 200 positions = 12,000 positions).

  D_S(t) = KL(p_T ‖ p_S)     reader, full context
  D_L(t) = KL(p_T ‖ p_L)     query-only LM, never sees the documents
  D_F(t) = KL(p_T ‖ p_F)     the fused distribution
  D_oracle(t) = min(D_S, D_L)          an unimplementable per-position best-branch reference

SCOPE — and this matters more than anything else here. The dump runs each generation to a fixed 200
tokens. The teacher answers well before that and then, having nothing to do, RECITES THE PROMPT
TEMPLATE back ("Step 1 (Reasoning): 1-2 sentences using the context ... Rules: ALWAYS commit to one
concrete best answer ..."). Those positions are 5,839 of the 12,000 — 49% — and they are trivial for
both branches, because both have the instruction in their own prompt. Averaging over them dilutes
every number: the reader's mean KL reads 0.385 over all positions and 0.538 over the task span, the
oracle 0.143 against 0.226. So the headline scope is TASK = reason + answer, and the all-positions
figures are kept only so the dilution can be seen. 3 of the 60 examples never emit the "Final
Answer:" marker within 200 tokens (their 600 positions are all labelled reason, and they contribute
no answer-span positions); that is reported rather than patched.

Three stages, each a (reader, LM) pair, so what fine-tuning changes is isolated:
  plain  = S   + L      neither branch fine-tuned
  ssolo  = Sft + L      reader fine-tuned only
  lfus   = Sft + Lft    both — this is the deployed arm

λ. The adapters in this dump are the λ0.85 family and were trained at that λ, so λ=0.85 is the
train-λ = eval-λ point and is the headline. λ=0.7 is reported beside it because it is the canonical
λ of the hotpot ACCURACY arm; a λ0.7 row for a λ0.85-trained adapter is a mismatch and is labelled.

Output: results/timing/fusion_kl.json + printed tables.   ~5 min, no GPU.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from scripts.branch_dump_common import clean_seg

DIR = os.environ.get("BRANCH_DIR", "/work/hdd/myproject/anon/analysis/branch_logits")
LAMS = (0.85, 0.70)
STAGES = {"plain": ("S", "L"), "ssolo": ("Sft", "L"), "lfus": ("Sft", "Lft")}


def logsoftmax(x):
    x = x.astype(np.float32)
    m = x.max(-1, keepdims=True)
    e = np.exp(x - m)
    return x - m - np.log(e.sum(-1, keepdims=True))


def segments(seg, n):
    """seg = (reason_end, answer_start, answer_end) as written by dump_branch_logits."""
    lab = np.array(["other"] * n, dtype=object)
    r_end, a0, a1 = int(seg[0]), int(seg[1]), int(seg[2])
    lab[:r_end] = "reason"
    lab[a0:a1] = "answer"
    return lab


def main():
    files = sorted(glob.glob(f"{DIR}/hotpotqa_ex*.npz"))
    if not files:
        raise SystemExit(f"no dump under {DIR}")
    cols = {k: [] for k in ("ex", "t", "seg", "D_S", "D_L", "D_Sft", "D_Lft")}
    for lam in LAMS:
        for st in STAGES:
            cols[f"D_F_{st}_{int(lam * 100)}"] = []
    for i, f in enumerate(files):
        A = np.load(f)
        z = {k: A[k].astype(np.float32) for k in ("T", "S", "L", "Sft", "Lft")}
        lg = {k: logsoftmax(v) for k, v in z.items()}
        pT = np.exp(lg["T"])

        def kl(lq):
            return (pT * (lg["T"] - lq)).sum(-1)

        n = len(A["gen"])
        cols["ex"].append(np.full(n, i))
        cols["t"].append(np.arange(n))
        cols["seg"].append(segments(clean_seg(A["gen"], A["seg"]), n))
        for k, name in (("S", "D_S"), ("L", "D_L"), ("Sft", "D_Sft"), ("Lft", "D_Lft")):
            cols[name].append(kl(lg[k]))
        for lam in LAMS:
            for st, (rk, lk) in STAGES.items():
                lF = logsoftmax(lam * z[rk] + (1 - lam) * z[lk])
                cols[f"D_F_{st}_{int(lam * 100)}"].append(kl(lF))
        if (i + 1) % 10 == 0:
            print(f"  ... {i + 1}/{len(files)} examples")
    D = {k: (np.concatenate(v) if k != "seg" else np.concatenate(v)) for k, v in cols.items()}
    N = len(D["D_S"])
    print(f"\n{N:,} teacher-forced positions, {len(files)} hotpotQA examples")
    out = {"n": int(N), "n_examples": len(files),
           "lams": list(LAMS), "stages": {k: list(v) for k, v in STAGES.items()}}

    TASK = (D["seg"] == "reason") | (D["seg"] == "answer")
    out["n_task"] = int(TASK.sum())
    out["n_other"] = int((~TASK).sum())
    out["n_examples_without_answer_span"] = int(
        sum(1 for e in range(len(files)) if not (D["seg"][D["ex"] == e] == "answer").any()))

    def stats(a, m=None, scope=None):
        """scope=None -> the TASK span (reason+answer), the headline. scope='all' -> every
        position, kept only to show how much the prompt-echo tail dilutes things."""
        sel = TASK if scope is None else np.ones(len(a), bool)
        if m is not None:
            sel = sel & m
        a = a[sel]
        return dict(mean=float(a.mean()), median=float(np.median(a)),
                    q25=float(np.quantile(a, .25)), q75=float(np.quantile(a, .75)),
                    frac_under_01=float((a < 0.1).mean()), n=int(len(a)))

    # ── the complementarity picture, per stage ──────────────────────────────────────────────────
    for st, (rk, lk) in STAGES.items():
        dR, dQ = D[f"D_{rk}"], D[f"D_{lk}"]
        orc = np.minimum(dR, dQ)
        blk = {"reader": stats(dR), "query_lm": stats(dQ), "oracle": stats(orc)}
        prefR = dQ > dR
        blk["reader_preferred_frac"] = float(prefR[TASK].mean())
        for lam in LAMS:
            dF = D[f"D_F_{st}_{int(lam * 100)}"]
            blk[f"fused_{int(lam * 100)}"] = stats(dF)
            blk[f"fused_{int(lam * 100)}_beats_both"] = float(
                ((dF < dR) & (dF < dQ))[TASK].mean())
            # how much of the branch→oracle headroom the fixed-λ mixture actually takes
            # NOT comparable across stages (the oracle itself moves), so it is not published
            blk[f"fused_{int(lam * 100)}_by_regime"] = {
                "reader_preferred": {"reader": stats(dR, prefR), "query_lm": stats(dQ, prefR),
                                     "fused": stats(dF, prefR)},
                "lm_preferred": {"reader": stats(dR, ~prefR), "query_lm": stats(dQ, ~prefR),
                                 "fused": stats(dF, ~prefR)}}
        blk["by_segment"] = {
            s: {"reader": stats(dR, D["seg"] == s, "all"),
                "query_lm": stats(dQ, D["seg"] == s, "all"),
                "oracle": stats(orc, D["seg"] == s, "all"),
                "fused_85": stats(D[f"D_F_{st}_85"], D["seg"] == s, "all")}
            for s in ("reason", "answer", "other")}
        # the same four, over EVERY position, so the dilution is visible rather than asserted
        blk["all_positions"] = {"reader": stats(dR, None, "all"),
                                "query_lm": stats(dQ, None, "all"),
                                "oracle": stats(orc, None, "all"),
                                "fused_85": stats(D[f"D_F_{st}_85"], None, "all")}
        out[st] = blk

    # ── what L-fusion changed, reader held FIXED at Sft ─────────────────────────────────────────
    out["l_fusion"] = {
        "query_lm_alone": {"before": stats(D["D_L"]), "after": stats(D["D_Lft"])},
        "fused_85": {"before": stats(D["D_F_ssolo_85"]), "after": stats(D["D_F_lfus_85"])},
        "delta_q_only_mean": float(D["D_L"].mean() - D["D_Lft"].mean()),
        "delta_fusion_mean": float(D["D_F_ssolo_85"].mean() - D["D_F_lfus_85"].mean()),
        "delta_q_only_median": float(np.median(D["D_L"]) - np.median(D["D_Lft"])),
        "delta_fusion_median": float(np.median(D["D_F_ssolo_85"]) - np.median(D["D_F_lfus_85"])),
    }

    os.makedirs("results/timing", exist_ok=True)
    json.dump(out, open("results/timing/fusion_kl.json", "w"), indent=2)
    print("wrote results/timing/fusion_kl.json\n")

    def line(nm, s):
        return (f"  {nm:26s} mean {s['mean']:7.4f}  median {s['median']:8.5f}  "
                f"IQR [{s['q25']:.5f}, {s['q75']:.4f}]  under 0.1 nats {100*s['frac_under_01']:5.1f}%")

    for st in STAGES:
        b = out[st]
        print(f"== stage {st}  ({' + '.join(STAGES[st])})  KL(teacher ‖ ·), nats")
        print(line("reader", b["reader"]))
        print(line("query-only LM", b["query_lm"]))
        print(line("oracle min of the two", b["oracle"]))
        for lam in LAMS:
            k = int(lam * 100)
            print(line(f"fused λ={lam}", b[f"fused_{k}"])
                  + f"   beats both branches {100*b[f'fused_{k}_beats_both']:.1f}%")
        print(f"  reader-preferred positions: {100*b['reader_preferred_frac']:.1f}%\n")

    L = out["l_fusion"]
    print("== what L-fusion changed (reader held fixed at Sft; only the LM branch differs)")
    print(f"  query-LM ALONE     {L['query_lm_alone']['before']['mean']:.4f} -> "
          f"{L['query_lm_alone']['after']['mean']:.4f}   (Δ {L['delta_q_only_mean']:+.4f} mean, "
          f"{L['delta_q_only_median']:+.5f} median)")
    print(f"  FUSED λ0.85        {L['fused_85']['before']['mean']:.4f} -> "
          f"{L['fused_85']['after']['mean']:.4f}   (Δ {L['delta_fusion_mean']:+.4f} mean, "
          f"{L['delta_fusion_median']:+.5f} median)")


if __name__ == "__main__":
    main()
