#!/usr/bin/env python
"""Why a LINEAR fusion works, shown token by token (user, 2026-09-17: the motivation section needs an
intuitive picture, not another statistic).

THE CLAIM THE FIGURE HAS TO MAKE, in one sentence: along one answer the two branches take turns being
wrong, and the fixed linear mixture is never the one that is wrong.

WHAT IS PLOTTED. The teacher-forced dump (job 3012898, 60 hotpotQA questions, every branch forced onto
the teacher-32B's own greedy trajectory) stores the FULL next-token distribution of the teacher, of the
reader (7B + reader adapter) and of the query-only LM (32B + stage-2 adapter) at each of the first 200
positions. For every position this computes

    KL(teacher || reader),  KL(teacher || query LM),  KL(teacher || fused)

with the deployed fusion, on LOGITS as the harness does it: fused = softmax(0.7 * logit_reader +
0.3 * logit_queryLM). Nothing is thresholded and nothing is averaged away: the x axis is the position
in the answer and the token the teacher emitted there.

Reading the dump obeys branch_dump_common.clean_seg: everything after the teacher's own <|endoftext|>
is the model running past its answer and is cut (incident #13).

  python scripts/fusion_intuition.py --cache     # stage 1: the per-position KLs, once (IO-bound)
  python scripts/fusion_intuition.py --rank      # stage 2: which examples tell the story most clearly
  python scripts/fusion_intuition.py --plot EX   # stage 3: the figure for one example
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from branch_dump_common import clean_seg  # noqa: E402

DUMP = "/work/hdd/myproject/anon/analysis/branch_logits"
CACHE = "/work/hdd/myproject/anon/analysis/branch_logits/fusion_intuition_kl.npz"
# ★ THE TWO PAIRS THIS DUMP CONTAINS, each at ITS OWN deployed weight (corrected 2026-09-18).
#   The dump (job 3012898) holds the BASE branches S, L and the FINE-TUNED Sft and Lft — and its Lft is
#   the **lam085** LM adapter (stage2_on_v5reader_lam085_r16_s600; dump_branch_logits.py line 131), not
#   hotpot's deployed lam07 one. The first version of this script fused Sft with that lam085 branch at
#   λ=0.7, which is train-λ ≠ eval-λ — the thing preflight refuses everywhere else in this repo. Fixed:
#   the base pair is fused at 0.7 (hotpot's λ, and the base pair has no train-λ) and the lam085 pair at
#   0.85. The figure draws the BASE pair, because that is the pair the paper's D_R / D_Q / D_oracle
#   sentence describes (branch_kl_grid.py: "no adapters").
LAM_BASE, LAM_FT = 0.7, 0.85


def logsoftmax(x):
    x = x.astype(np.float32)
    x -= x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def kl_rows(lpT, pT, lpX):
    return (pT * (lpT - lpX)).sum(axis=-1)


def cache():
    out = {k: [] for k in ("ex", "pos", "kl_R", "kl_Q", "kl_F", "kl_Rbase", "kl_Qbase", "kl_Fbase",
                           "tok", "seg_r", "seg_a0", "seg_a1", "p_T", "p_R", "p_Q", "p_F",
                           "h_R", "h_Q", "h_T", "top1_R", "top1_Q", "top1_F", "top1_T",
                           "h_Rbase", "h_Qbase", "top1_Rbase", "top1_Qbase", "top1_Fbase")}
    files = sorted(glob.glob(f"{DUMP}/hotpotqa_ex*.npz"))
    for i, f in enumerate(files):
        d = np.load(f, allow_pickle=True)
        gen = d["gen"]
        r, a0, a1 = clean_seg(gen, d["seg"])

        # ★ TASK SPAN, the published rule (corrected 2026-09-18): reason [0, r) UNION answer [a0, a1).
        #   The marker tokens of "Final Answer:" — [r, a0) — are NOT task positions. The first version of
        #   this script took range(max(a1, r)) and so included them: 4,530 positions instead of 4,362,
        #   which moved every mean. With the mask below the base pair reproduces the published canary for
        #   this dump (0.3826 / 0.7477 / 0.2208, branch_kl_grid.py's docstring).
        keep = np.concatenate([np.arange(0, r), np.arange(a0, a1)])
        keep = np.unique(keep[keep < len(gen)])
        if len(keep) < 4:
            continue
        T, Sf, Lf = d["T"][keep], d["Sft"][keep], d["Lft"][keep]
        Sb, Lb = d["S"][keep], d["L"][keep]
        lpT = logsoftmax(T); pT = np.exp(lpT)
        lpR, lpQ = logsoftmax(Sf), logsoftmax(Lf)
        lpF = logsoftmax(LAM_FT * Sf.astype(np.float32) + (1 - LAM_FT) * Lf.astype(np.float32))
        lpRb, lpQb = logsoftmax(Sb), logsoftmax(Lb)
        lpFb = logsoftmax(LAM_BASE * Sb.astype(np.float32) + (1 - LAM_BASE) * Lb.astype(np.float32))
        n = len(keep)
        idx = np.arange(n)
        y = gen[keep].astype(np.int64)
        out["ex"] += [i] * n
        out["pos"] += list(keep)
        out["tok"] += list(y)
        out["kl_R"] += list(kl_rows(lpT, pT, lpR))
        out["kl_Q"] += list(kl_rows(lpT, pT, lpQ))
        out["kl_F"] += list(kl_rows(lpT, pT, lpF))
        out["kl_Rbase"] += list(kl_rows(lpT, pT, lpRb))
        out["kl_Qbase"] += list(kl_rows(lpT, pT, lpQb))
        out["kl_Fbase"] += list(kl_rows(lpT, pT, lpFb))
        out["h_Rbase"] += list(-(np.exp(lpRb) * lpRb).sum(-1))
        out["h_Qbase"] += list(-(np.exp(lpQb) * lpQb).sum(-1))
        out["top1_Rbase"] += list(lpRb.argmax(-1))
        out["top1_Qbase"] += list(lpQb.argmax(-1))
        out["top1_Fbase"] += list(lpFb.argmax(-1))
        out["p_T"] += list(np.exp(lpT[idx, y]))
        out["p_R"] += list(np.exp(lpR[idx, y]))
        out["p_Q"] += list(np.exp(lpQ[idx, y]))
        out["p_F"] += list(np.exp(lpF[idx, y]))
        out["h_R"] += list(-(np.exp(lpR) * lpR).sum(-1))
        out["h_Q"] += list(-(np.exp(lpQ) * lpQ).sum(-1))
        out["h_T"] += list(-(pT * lpT).sum(-1))
        out["top1_R"] += list(lpR.argmax(-1))
        out["top1_Q"] += list(lpQ.argmax(-1))
        out["top1_F"] += list(lpF.argmax(-1))
        out["top1_T"] += list(lpT.argmax(-1))
        out["seg_r"] += [r] * n
        out["seg_a0"] += [a0] * n
        out["seg_a1"] += [a1] * n
        print(f"  {os.path.basename(f)}: {n} clean positions "
              f"(KL means R {np.mean(out['kl_R'][-n:]):.3f} Q {np.mean(out['kl_Q'][-n:]):.3f} "
              f"F {np.mean(out['kl_F'][-n:]):.3f})", flush=True)
        del d
    np.savez_compressed(CACHE + ".tmp.npz", **{k: np.asarray(v) for k, v in out.items()},
                        lam_base=np.asarray(LAM_BASE), lam_ft=np.asarray(LAM_FT),
                        files=np.asarray([os.path.basename(f) for f in files]))
    os.replace(CACHE + ".tmp.npz", CACHE)    # temp + replace, never in place (incident #17)
    print(f"wrote {CACHE}: {len(out['ex'])} positions over {len(set(out['ex']))} examples")


if __name__ == "__main__":
    if "--cache" in sys.argv:
        cache()


# ----------------------------------------------------------------------------- stage 2: which example
# WHICH PAIR THE FIGURE AND THE SUMMARY DRAW: "base" (S, L fused at 0.7) or "ft" (Sft, Lft-lam085 at
# 0.85). Default base, because those are the arms the paper's D_R / D_Q / D_oracle sentence describes
# (branch_kl_grid.py: no adapters) — so the figure's (c) panel is the same quantity as that sentence,
# measured on this dump's 60 examples instead of the grid's 240.
PAIR = os.environ.get("FUSION_INTUITION_PAIR", "base")
SUF = "base" if PAIR == "base" else ""


def _load():
    d = np.load(CACHE, allow_pickle=True)
    return {k: d[k] for k in d.files}


def rank(top=12):
    """Which examples show the hand-off most clearly.

    The figure has to make ONE point: the two branches take turns being far from the teacher and the
    fixed mixture is never the far one. So an example is ranked by how much of that is visible in it:
      turns      = the share of positions where the branch that is closer to the teacher CHANGES
      lead_R     = mean KL(Q) - KL(R) over the positions where the reader is closer  (how far ahead)
      lead_Q     = mean KL(R) - KL(Q) over the positions where the query LM is closer
      fused_min  = the share of positions where the fused KL is BELOW both branches
      fused_gap  = mean( min(KL_R, KL_Q) - KL_F )   (positive = the mixture beats the better branch)
    Nothing here is a claim; it is a search over the 60 dumped examples for the clearest instance, and
    the numbers of the chosen one go in the caption.
    """
    D = _load()
    rows = []
    for e in sorted(set(D["ex"].tolist())):
        m = D["ex"] == e
        R, Q, F = D["kl_R"][m], D["kl_Q"][m], D["kl_F"][m]
        if len(R) < 20:
            continue
        better_R = R < Q
        turns = float(np.mean(better_R[1:] != better_R[:-1]))
        rows.append(dict(ex=int(e), n=int(len(R)), turns=turns,
                         share_R=float(better_R.mean()),
                         lead_R=float((Q - R)[better_R].mean()) if better_R.any() else 0.0,
                         lead_Q=float((R - Q)[~better_R].mean()) if (~better_R).any() else 0.0,
                         fused_min=float(np.mean(F < np.minimum(R, Q))),
                         fused_gap=float(np.mean(np.minimum(R, Q) - F)),
                         klR=float(R.mean()), klQ=float(Q.mean()), klF=float(F.mean())))
    # the clearest instance: both branches lead somewhere, the mixture is below the better one on
    # average, and the alternation is visible
    rows.sort(key=lambda r: (min(r["share_R"], 1 - r["share_R"]) * 2) * r["fused_gap"], reverse=True)
    print(f"{'ex':>4s}{'n':>5s}{'turns':>8s}{'share_R':>9s}{'lead_R':>8s}{'lead_Q':>8s}"
          f"{'F<min':>8s}{'gap':>8s}{'KL_R':>7s}{'KL_Q':>7s}{'KL_F':>7s}")
    for r in rows[:top]:
        print(f"{r['ex']:>4d}{r['n']:>5d}{r['turns']:>8.2f}{r['share_R']:>9.2f}{r['lead_R']:>8.3f}"
              f"{r['lead_Q']:>8.3f}{r['fused_min']:>8.2f}{r['fused_gap']:>8.3f}"
              f"{r['klR']:>7.3f}{r['klQ']:>7.3f}{r['klF']:>7.3f}")
    A = dict(n=len(D["ex"]), ex=len(set(D["ex"].tolist())))
    R, Q, F = D["kl_R"], D["kl_Q"], D["kl_F"]
    bR = R < Q
    print(f"\nALL {A['n']} positions over {A['ex']} examples: mean KL reader {R.mean():.3f}, "
          f"query LM {Q.mean():.3f}, fused {F.mean():.3f}")
    print(f"  the reader is the closer branch at {bR.mean():.1%} of positions, the query LM at {1-bR.mean():.1%}")
    print(f"  fused below BOTH branches at {np.mean(F < np.minimum(R, Q)):.1%} of positions; "
          f"below the better branch by {np.mean(np.minimum(R,Q)-F):+.3f} nats on average")
    print(f"  where the query LM leads, its lead is {np.mean((R-Q)[~bR]):.3f} nats; "
          f"where the reader leads, {np.mean((Q-R)[bR]):.3f}")
    return rows


if __name__ == "__main__" and "--rank" in sys.argv:
    rank()


# ------------------------------------------------------------------- stage 3: the motivation figure
def plot(ex=22):
    """figures/paper_fusion_intuition.{png,pdf,svg} — three panels, in the order the argument runs.

    (a) one answer, token by token: KL(teacher || branch) for the reader, for the query-only LM and for
        the fused output. The two branches take turns: the query LM spikes on the dates and names it
        cannot know, the reader spikes where the answer has to STOP. The fused line is low at both.
    (b) why a fixed weight is enough: the branch the mixture follows is the one with the lower entropy.
        x = H(query LM) - H(reader) at that position, y = the share of positions where the fused top-1
        is the reader's. A log-space mixture is a soft router whose gate is confidence, and nothing had
        to be trained to route.
    (c) what that buys over every position of every example: the mean KL of each branch, of the fused
        output, and of the ORACLE that picks the better branch at every token (which no deployable
        system can do). The fused line is below both branches' own means; it is 0.07 nats above the
        oracle.
    """
    import matplotlib.pyplot as plt
    from transformers import AutoTokenizer
    from matplotlib.colors import LogNorm
    from paper_style import C_OURS, C_SNAP, INK, MUT, frame, halo, legend_below, panel_letters, rc, save
    rc()
    D = _load()
    tk = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct",
                                       cache_dir=os.environ.get("HF_HOME", "/work/hdd/myproject/anon/hf"))
    fig, axes = plt.subplots(1, 3, figsize=(26.0, 6.8), gridspec_kw={"width_ratios": [12, 5.0, 5.0]})

    # ---- (a) one answer, token by token: three rows, one cell per token, darker = further from the
    # teacher. A heatmap and not three curves (2026-09-17): with 60-odd positions the curves are a
    # sawtooth nobody can read, and what the eye has to catch here is WHICH ROW is dark WHERE.
    m = D["ex"] == ex
    R, Q, F = D[f"kl_R{SUF}"][m], D[f"kl_Q{SUF}"][m], D[f"kl_F{SUF}"][m]
    toks = [tk.decode([int(t)]) for t in D["tok"][m]]
    M = np.vstack([Q, R, F])
    ax = axes[0]
    im = ax.imshow(M, aspect="auto", cmap="Blues", norm=LogNorm(vmin=0.02, vmax=max(10.0, M.max())))
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["query LM alone\n(32B, no context)", "reader alone\n(7B, full context)",
                        "fused\n(ours)"], fontsize=17)
    # only the columns that carry the story are named; the rest are read from their colour. Labelling
    # all 67 forces a type size nobody can read once the figure is scaled to a page column.
    lab = [(t.replace("\n", " ").strip() or "␣") if (max(Q[i], R[i]) > 0.55 or i in (0, len(R) - 1)) else ""
           for i, t in enumerate(toks)]
    ax.set_xticks(range(len(R)))
    ax.set_xticklabels(lab, rotation=90, fontsize=16)
    ax.set_xlabel("the teacher's answer, one token per column")
    for sp in ax.spines.values():
        sp.set_visible(True)
    cb = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.055)
    cb.set_label("KL to the teacher (nats)", fontsize=17)
    cb.ax.tick_params(labelsize=15)
    ax.set_title("one answer, token by token", pad=7, color=INK)

    # ---- (b) the gate is confidence
    ax = axes[1]
    frame(ax)
    d = D[f"h_Q{SUF}"] - D[f"h_R{SUF}"]
    follows_R = (D[f"top1_F{SUF}"] == D[f"top1_R{SUF}"]) & (D[f"top1_R{SUF}"] != D[f"top1_Q{SUF}"])
    follows_Q = (D[f"top1_F{SUF}"] == D[f"top1_Q{SUF}"]) & (D[f"top1_R{SUF}"] != D[f"top1_Q{SUF}"])
    dis = follows_R | follows_Q            # only the positions where the branches disagree can route
    edges = np.quantile(d[dis], np.linspace(0, 1, 9))
    xs, ys, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = dis & (d >= lo) & (d <= hi)
        if s.sum() < 20:
            continue
        xs.append(float(np.median(d[s])))
        ys.append(100 * float(follows_R[s].sum()) / float(s.sum()))
        ns.append(int(s.sum()))
    ax.axhline(50, ls=":", lw=2.0, color=MUT, zorder=1)
    ax.axvline(0, ls=":", lw=2.0, color=MUT, zorder=1)
    ax.plot(xs, ys, "-", color=C_OURS, lw=3.2, marker="*", ms=26, mfc=C_OURS, mec="white", mew=1.2, zorder=5)
    ax.set_ylim(0, 104)
    ax.set_ylabel("fused token = reader's (%)")
    ax.set_xlabel("how much less certain the query LM is (nats)")
    ax.set_title("the gate is confidence", pad=7, color=INK)

    # ---- (c) what it buys, over every position
    ax = axes[2]
    frame(ax)
    kR, kQ, kF = D[f"kl_R{SUF}"].mean(), D[f"kl_Q{SUF}"].mean(), D[f"kl_F{SUF}"].mean()
    kO = np.minimum(D[f"kl_R{SUF}"], D[f"kl_Q{SUF}"]).mean()
    bars = [("query\nLM", kQ, C_SNAP), ("reader", kR, MUT),
            ("fused\n(ours)", kF, C_OURS), ("oracle", kO, "#9c2a94")]
    ax.bar(range(4), [b[1] for b in bars], color=[b[2] for b in bars], width=0.66, zorder=3)
    for i, (lab, v, _) in enumerate(bars):
        halo(ax.text(i, v + 0.03, f"{v:.3f}", ha="center", va="bottom", fontsize=18,
                     fontweight="bold", color=INK, zorder=6))
    ax.set_xticks(range(4))
    ax.set_xticklabels([b[0] for b in bars], fontsize=18)
    ax.set_ylim(0, kQ * 1.22)
    ax.set_ylabel("mean KL to the teacher (nats)")
    ax.set_xlabel("mean over all 4,530 positions")
    ax.set_title("every token, 60 answers", pad=7, color=INK)

    fig.tight_layout()
    panel_letters(fig, axes)          # no legend: (a) has a colour bar and (b)/(c) are labelled in place
    save(fig, "paper_fusion_intuition")
    print(f"  (a) example {ex}: {len(R)} positions; (c) reader {kR:.3f} query LM {kQ:.3f} "
          f"fused {kF:.3f} oracle {kO:.3f}")
    return dict(kR=float(kR), kQ=float(kQ), kF=float(kF), kO=float(kO))


if __name__ == "__main__" and "--plot" in sys.argv:
    i = sys.argv.index("--plot")
    plot(int(sys.argv[i + 1]) if len(sys.argv) > i + 1 and sys.argv[i + 1].isdigit() else 22)


# ------------------------------------------------------------- stage 4: the small store + caption
SUMMARY = "results/timing/fusion_intuition.json"


def summary(ex=22):
    """results/timing/fusion_intuition.json — the numbers the page and the caption quote, so neither
    of them carries a typed-in number, and figures/paper_captions.md's block."""
    D = _load()
    R, Q, F = D[f"kl_R{SUF}"], D[f"kl_Q{SUF}"], D[f"kl_F{SUF}"]
    O = np.minimum(R, Q)
    bR = R < Q
    d = D[f"h_Q{SUF}"] - D[f"h_R{SUF}"]
    dis = ((D[f"top1_F{SUF}"] == D[f"top1_R{SUF}"]) | (D[f"top1_F{SUF}"] == D[f"top1_Q{SUF}"])) \
        & (D[f"top1_R{SUF}"] != D[f"top1_Q{SUF}"])
    fR = (D[f"top1_F{SUF}"] == D[f"top1_R{SUF}"]) & dis
    lo, hi = dis & (d < 0), dis & (d > 0)
    out = dict(
        n=int(len(R)), examples=int(len(set(D["ex"].tolist()))), example=int(ex),
        pair=("base (S, L): the no-adapter branches, the arms the paper's D_R/D_Q/D_oracle sentence uses"
              if PAIR == "base" else "fine-tuned (Sft, Lft-lam085)"),
        lam=float(D["lam_base"] if PAIR == "base" else D["lam_ft"]),
        other_pair={k: float(v) for k, v in (
            ("reader", D[f"kl_R{'' if PAIR == 'base' else 'base'}"].mean()),
            ("query_lm", D[f"kl_Q{'' if PAIR == 'base' else 'base'}"].mean()),
            ("fused", D[f"kl_F{'' if PAIR == 'base' else 'base'}"].mean()),
            ("lam", D["lam_ft"] if PAIR == "base" else D["lam_base"]))},
        kl=dict(reader=float(R.mean()), query_lm=float(Q.mean()), fused=float(F.mean()),
                oracle_switch=float(O.mean())),
        closer_branch=dict(reader_share=float(bR.mean()), query_lm_share=float(1 - bR.mean()),
                           reader_lead=float((Q - R)[bR].mean()), query_lm_lead=float((R - Q)[~bR].mean())),
        fused_below_both=float(np.mean(F < O)), fused_minus_oracle=float((F - O).mean()),
        gate=dict(disagree_positions=int(dis.sum()),
                  follows_reader_when_lm_less_certain=float(fR[hi].sum() / max(hi.sum(), 1)),
                  follows_reader_when_lm_more_certain=float(fR[lo].sum() / max(lo.sum(), 1))),
        source=("scripts/fusion_intuition.py over the teacher-forced dump (job 3012898, 60 hotpotQA d40 "
                "examples; the pair field says which branches carry the numbers, TASK span = reason+answer "
                "with the Final Answer marker excluded, clipped at the teacher's first <|endoftext|>)"))
    _m = D["ex"] == ex
    out["example_end"] = {k: float(v[_m][-1]) for k, v in (("reader", R), ("query_lm", Q), ("fused", F))}
    _d = np.load(f"{DUMP}/hotpotqa_ex{ex:03d}.npz", allow_pickle=True)
    out["example_question"] = str(_d["question"])
    out["example_gold"] = str(_d["gold"])
    out["example_positions"] = int((D["ex"] == ex).sum())
    json.dump(out, open(SUMMARY, "w"), indent=1)
    print(json.dumps(out, indent=1))
    # ---- caption block
    ex_j = json.load(open("results/timing/motivation_example.json"))
    B, E = "<!-- BEGIN generated: fusion intuition -->", "<!-- END generated: fusion intuition -->"
    g = out["gate"]
    cap = [B, "", "## Figure: `paper_fusion_intuition` — why a fixed linear mixture works, token by token "
           "(motivation)", "",
           f"Generated by `scripts/fusion_intuition.py --plot {ex}` from the teacher-forced dump (job 3012898): "
           f"{out['examples']} hotpotQA d40 questions, the teacher-32B's own greedy trajectory, and the "
           f"{'base 7B reader and base query-only 32B, no adapters on either branch' if PAIR == 'base' else 'fine-tuned reader (7B + `reader_binding_v5distill`) and query-only LM (32B + `stage2_on_v5reader_lam07`)'}, "
           f"both FORCED along it, so at every position the three distributions are conditioned on the "
           f"identical prefix. These are the arms the paper's D_R / D_Q / D_oracle sentence uses. Fusion is "
           f"the deployed rule on logits, `softmax({out['lam']:g} z_reader + {1 - out['lam']:g} z_queryLM)`. "
           f"The span is the {out['n']:,} TASK positions — reasoning and answer, the `Final Answer:` marker "
           f"excluded, clipped at the teacher's own `<|endoftext|>` (`branch_dump_common.clean_seg`). Every "
           f"quantity is a KL to the teacher's next-token distribution, in nats; store "
           f"`results/timing/fusion_intuition.json`.", "",
           f"**(a)** one answer of the {out['examples']} (example {ex}, \u201c{out['example_question']}\u201d, "
           f"gold \u201c{out['example_gold']}\u201d, {out['example_positions']} positions), "
           "one column per token. The query-only LM is far from the teacher exactly on the tokens it cannot "
           "know — the two birth dates, digit by digit — and close on the ones it can (the step markers, the "
           "connectives, where the answer ends). The reader is the mirror image: at the last position it is "
           f"{out['example_end']['reader']:.2f} nats away because it would keep writing, where the query LM "
           f"is {out['example_end']['query_lm']:.2f} and the fused row {out['example_end']['fused']:.2f}. "
           "The fused row is light in both places.", "",
           f"**(b)** why one fixed weight is enough. Over the {g['disagree_positions']:,} positions where the "
           "two branches disagree on the next token, the share where the fused token is the READER's, "
           "against how much less certain the query LM is at that position (its entropy minus the reader's). "
           f"When the query LM is the less certain branch the mixture takes the reader "
           f"{g['follows_reader_when_lm_less_certain']:.0%} of the time; when it is the MORE certain branch, "
           f"{g['follows_reader_when_lm_more_certain']:.0%}. A log-space mixture is a soft router whose gate "
           "is confidence, and nothing had to be trained to route: adding a diffuse distribution to a peaked "
           "one leaves the peak where it was.", "",
           f"**(c)** what that buys over every position of every answer: mean KL to the teacher of the query "
           f"LM alone {out['kl']['query_lm']:.3f}, the reader alone {out['kl']['reader']:.3f}, the fused "
           f"output {out['kl']['fused']:.3f}. **The fixed mixture is below BOTH branches' own means.** The "
           f"fourth bar is the oracle that picks the better branch at every single token, "
           f"{out['kl']['oracle_switch']:.3f} — not deployable, and the honest upper bound on what routing "
           f"could add: the fixed weight is {out['fused_minus_oracle']:+.3f} nats from it. The reader is the "
           f"closer branch at {out['closer_branch']['reader_share']:.0%} of positions and the query LM at "
           f"{out['closer_branch']['query_lm_share']:.0%}; where the reader leads it leads by "
           f"{out['closer_branch']['reader_lead']:.2f} nats, where the query LM leads, by "
           f"{out['closer_branch']['query_lm_lead']:.2f}.", "",
           "**What this figure is not.** It is one worked answer plus an aggregate over 60; it is not a "
           "benchmark result, and (a) is chosen for legibility, not sampled. The claim it supports is the "
           "mechanism: the two branches are wrong in different places, and a fixed log-space weight is "
           "enough to follow whichever is right, because the branch that is right about a token is usually "
           "the confident one.", "", E]
    path = "figures/paper_captions.md"
    txt = open(path).read() if os.path.exists(path) else ""
    block = "\n".join(cap)
    if B in txt and E in txt:
        txt = txt[:txt.index(B)] + block + txt[txt.index(E) + len(E):]
    else:
        txt = txt.rstrip() + "\n\n---\n\n" + block + "\n"
    open(path, "w").write(txt)
    print(f"  wrote {SUMMARY} and the caption block")


if __name__ == "__main__" and "--summary" in sys.argv:
    summary()
