#!/usr/bin/env python
"""F1 by question position on the two accumulate benchmarks (user, 2026-09-14, appendix: does the score
of PAIR and of the baselines differ between a conversation's FIRST question and the ones that follow?).

WHAT IS MEASURED. The canonical accuracy cells (the same logs the main tables publish), every row re-scored
with the current metric (src.eval.compute_best_em_f1 on the stored prediction, as rescore_headline_tables
does), on the shared (conversation, turn) keys of the seven arms — 355 rows on LooGLE (70 conversations,
1-14 questions each) and 300 on LoCoMo (10 conversations x 30 questions). Each arm's all-row mean is
printed beside its store value as the mapping check; they must agree.

TWO CONFOUNDS, BOTH CONTROLLED — position is not a random assignment on either bench:
 1. LooGLE conversations have different numbers of questions, and the longer ones are HARDER: the teacher's
    first-question F1 is 0.319 on the 40 conversations that reach 5 questions against 0.468 on the 30 that do
    not. So a per-position mean over all 70 mixes two populations. Headline basis = the conversations that
    have a question in every column (BALANCED: 32 of the 70 at one column per question, 1..6); the
    all-conversation view is printed under it.
 2. LoCoMo's question CATEGORY composition varies with position: category 2 (temporal, the easiest for every
    arm: teacher 0.658 against 0.492 on category 1) is 60% of the ten first questions and ~43% later, and the
    first question is one per conversation (N=10). So the bench's first-question point is a composition
    artefact, and the position table is also printed per category (1 and 2, the two with usable N, on coarse
    bins). Neither control is optional: without them the LoCoMo first-question point reads 0.535 for the
teacher against 0.452 over questions 2-5, which is category mix, not position.

WHAT IT FINDS (numbers in the caption block). The query-DEPENDENT presses select their kept tokens with the
FIRST question and then freeze the cache: run_single_batched calls compress_append(ctxs, qc) / 
specprefill_append(ctxs, qc) with qc = turn 1's question. So snapKV and SpecPrefill are measured on the one
question their compression was built for and lose accuracy on every later one. ExpectedAttention is
query-agnostic, KV quantization keeps every token, and PAIR's reader holds the whole context — none of them
depends on which question came first.

Writes results/timing/turn_position.json, the caption block "turn position" in figures/paper_captions.md, and
with --plot the figure figures/paper_turn_position.{png,pdf,svg}.
"""
import json
import os
import statistics as st
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from src.eval import compute_best_em_f1  # noqa: E402

# figure label -> (LooGLE log tag, LoCoMo log tag, store key on each bench)
ARMS = [
    ("Full-context 32B", "lg_teacher", "c30_teacher32b", "teacher", "teacher"),
    ("PAIR 32B + 7B (ours)", "lg_SsoloLfus_l07", "c30b3_7b_SsoloLfus_LMNA", "ours", "ours"),
    ("Full-context 7B", "lg_floor7", "c30_floor7b", "floor7", "floor7"),
    ("snapKV (32B)", "lg_snapkv_r78125", "c30b3_snapkv_frozen_r078125", "snap219", "snap78125"),
    ("SpecPrefill (32B)", "gf_lg_acc_spec21875", "c30b3_specprefill_k21875", "spec219", "spec21875"),
    ("ExpectedAttention (32B)", "lga_expected781_ACC", "c30b3_h2o_r078125", "expected219", "expected78125"),
    ("KV quantization (32B)", "lgc_quant_int4", "lcao_int4", "quant_int4", "quant_int4"),
]
QUERY_DEPENDENT = ("snapKV (32B)", "SpecPrefill (32B)")     # kept set chosen with turn 1's question
# bins: ONE COLUMN PER QUESTION where the basis allows it (user, 2026-09-14: "x축 점도 더 찍고").
#   LooGLE - questions 1..6 on the 32 conversations that have six, so every column is the SAME 32
#   conversations (n=32 each). Per-turn row counts over all 70 conversations fall 70/65/58/46/40/32 and the
#   long conversations are the hard ones, so an all-conversation per-turn curve would drift with the
#   population, not with the position; that view is still written to the store.
#   LoCoMo - ten conversations x 30 questions; five-question columns give six columns of n=50 with every
#   conversation in every column (ten three-question columns fit the data too, but their labels collide at
#   print size, and six columns already resolve the trend).
# cat_bins: the category-controlled views stay COARSE - splitting 122 and 131 rows ten ways leaves ~12 per
# cell, which is not a number to quote.
BENCH = {
    "loogle": dict(name="LooGLE", col=0, store="results/timing/loogle_accuracy.json",
                   bins=[(t, t) for t in range(1, 7)], cats=None, cat_bins=None),
    "locomo": dict(name="LoCoMo", col=1, store="results/timing/lcf_accuracy.json",
                   bins=[(a, a + 4) for a in range(1, 30, 5)], cats=(1, 2),
                   cat_bins=[(1, 5), (6, 10), (11, 20), (21, 30)],
                   ref="/work/hdd/myproject/anon/locomo/locomo_reuse_ref_full.jsonl"),
}
OUT = "results/timing/turn_position.json"


def load(tag):
    d = {}
    for l in open(f"results/fusionft/{tag}.jsonl"):
        if l.strip():
            r = json.loads(l)
            if "acc_f1" in r:
                d[(r["conv"], int(r["turn"]))] = r
    return d


def bin_label(a, b):
    return str(a) if a == b else f"{a}–{b}"


def means(sc, ks, bins):
    return [round(st.fmean([sc[k] for k in ks if a <= k[1] <= b]), 4) for a, b in bins]


def analyse():
    res = {}
    for b, cfg in BENCH.items():
        col, bins = cfg["col"], cfg["bins"]
        store = json.load(open(cfg["store"]))
        logs = {lab: load(tags[col]) for lab, *tags in ((a[0], a[1], a[2]) for a in ARMS)}
        shared = sorted(set.intersection(*(set(v) for v in logs.values())))
        sc = {lab: {k: compute_best_em_f1(logs[lab][k]["acc_pred"], [logs[lab][k]["gold"]])[1] for k in shared}
              for lab in logs}
        convs = sorted({k[0] for k in shared})
        balanced = [c for c in convs if all(any(k[0] == c and a <= k[1] <= bb for k in shared) for a, bb in bins)]
        e = dict(name=cfg["name"], shared_n=len(shared), conversations=len(convs),
                 balanced_conversations=len(balanced), max_turn=max(k[1] for k in shared),
                 bin_labels=[bin_label(a, bb) for a, bb in bins], views={}, arms={}, first_vs_later={})
        print(f"\n== {cfg['name']}: shared-N={len(shared)}, {len(convs)} conversations, turns 1..{e['max_turn']}; "
              f"{len(balanced)} conversations have a question in every bin")
        views = [("balanced", [k for k in shared if k[0] in balanced]), ("all", shared)]
        if cfg["cats"]:
            cat = {(r["conversation_id"], int(r["turn"])): r["category"]
                   for r in (json.loads(l) for l in open(cfg["ref"]) if l.strip())}
            _cb = cfg.get("cat_bins") or bins
            e["category_mix"] = [{"bin": bin_label(a, bb),
                                  "counts": {str(c): sum(1 for k in shared if a <= k[1] <= bb and cat.get(k) == c)
                                             for c in sorted({v for v in cat.values()})}}
                                 for a, bb in _cb]
            e["category_mix_first_question"] = {str(c): sum(1 for k in shared if k[1] == 1 and cat.get(k) == c)
                                               for c in sorted({v for v in cat.values()})}
            views += [(f"category {c}", [k for k in shared if cat.get(k) == c]) for c in cfg["cats"]]
        cb = cfg.get("cat_bins") or bins
        e["cat_bin_labels"] = [bin_label(a, bb) for a, bb in cb]
        for vname, ks in views:
            bs = cb if vname.startswith("category") else bins
            e["views"][vname] = dict(n=len(ks), conversations=len({k[0] for k in ks}),
                                     bins=[list(x) for x in bs],
                                     bin_labels=[bin_label(a, bb) for a, bb in bs],
                                     bin_n=[sum(1 for k in ks if a <= k[1] <= bb) for a, bb in bs],
                                     arms={lab: means(sc[lab], ks, bs) for lab, *_ in ARMS})
            t, o, f = (means(sc[x], ks, bs) for x in ("Full-context 32B", "PAIR 32B + 7B (ours)", "Full-context 7B"))
            e["views"][vname]["closeness_pct"] = [None if ti == fi else round(100 * (oi - fi) / (ti - fi), 1)
                                                  for ti, oi, fi in zip(t, o, f)]
            print(f"  -- view {vname}: n={len(ks)}, {len({k[0] for k in ks})} conversations")
            print("      " + "".join(f"{lb:>9s}" for lb in e["views"][vname]["bin_labels"])
                  + f"{'all':>9s}{'store':>8s}")
            print("   N  " + "".join(f"{n:>9d}" for n in e["views"][vname]["bin_n"]) + f"{len(ks):>9d}")
            for lab, lgt, lct, ksl, kslc in ARMS:
                v = e["views"][vname]["arms"][lab]
                sk = ksl if col == 0 else kslc
                print(f"   {lab:24s}" + "".join(f"{x:>9.3f}" for x in v)
                      + f"{st.fmean(sc[lab][k] for k in ks):>9.3f}"
                      + (f"{store[sk]:>8.3f}" if vname == "all" else ""))
        # first question vs later, per arm: absolute means (raw, both shown) + paired per conversation
        pool = [c for c in convs if (c, 1) in sc["Full-context 32B"] and any(k[0] == c and k[1] >= 2 for k in shared)]
        k1 = [(c, 1) for c in pool]
        kL = [k for k in shared if k[0] in pool and k[1] >= 2]
        e["first_vs_later"] = dict(conversations=len(pool), n_first=len(k1), n_later=len(kL), arms={})
        print(f"  -- first question vs later questions, the {len(pool)} conversations with at least two "
              f"(first n={len(k1)}, later n={len(kL)}):")
        for lab, *_ in ARMS:
            first, later = st.fmean(sc[lab][k] for k in k1), st.fmean(sc[lab][k] for k in kL)
            d = [st.fmean([sc[lab][k] for k in shared if k[0] == c and k[1] >= 2]) - sc[lab][(c, 1)] for c in pool]
            e["first_vs_later"]["arms"][lab] = dict(
                first=round(first, 4), later=round(later, 4),
                paired_mean=round(st.fmean(d), 4), up=sum(x > 0 for x in d), down=sum(x < 0 for x in d))
            print(f"   {lab:24s} first {first:.3f}  later {later:.3f}  paired {st.fmean(d):+.3f} "
                  f"({sum(x > 0 for x in d)}:{sum(x < 0 for x in d)} of {len(d)})")
        for lab, *_ in ARMS:
            e["arms"][lab] = dict(query_dependent=lab in QUERY_DEPENDENT,
                                  tag=(ARMS[[a[0] for a in ARMS].index(lab)][1 + col]))
        res[b] = e
    os.makedirs("results/timing", exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")
    return res


# ------------------------------------------------------------------ figure + caption
def plot(res):
    import matplotlib.pyplot as plt
    from paper_style import (C_H2O, C_OURS, C_QUANT, C_SNAP, C_SPEC, INK, MUT, frame, legend_below,
                             panel_letters, rc, save)
    rc()
    STYLE = {"Full-context 32B": (INK, "D", "--"), "PAIR 32B + 7B (ours)": (C_OURS, "*", "-"),
             "Full-context 7B": (MUT, "X", ":"), "snapKV (32B)": (C_SNAP, "o", "-"),
             "SpecPrefill (32B)": (C_SPEC, "^", "-"), "ExpectedAttention (32B)": (C_H2O, "s", "-"),
             "KV quantization (32B)": (C_QUANT, "v", "-")}
    VIEW = {"loogle": "balanced", "locomo": "all"}       # LooGLE: the 40 conversations present in every bin
    fig, axes = plt.subplots(1, 3, figsize=(25.0, 6.6), gridspec_kw={"width_ratios": [6, 6, 5]})
    for ax, b in zip(axes, ("loogle", "locomo")):
        e = res[b]
        v = e["views"][VIEW[b]]
        frame(ax)
        x = list(range(len(v["bin_labels"])))
        top = 0.0
        for lab, *_ in ARMS:
            c, mk, ls = STYLE[lab]
            y = v["arms"][lab]
            top = max(top, max(y))
            ax.plot(x, y, ls, color=c, lw=3.2 if lab in ("PAIR 32B + 7B (ours)", "Full-context 32B") else 2.2,
                    marker=mk, ms=26 if mk == "*" else 11, mfc=c, mec="white", mew=0.8, label=lab,
                    zorder=6 if lab == "PAIR 32B + 7B (ours)" else 4)
        ax.set_xticks(x)
        ax.set_xticklabels(v["bin_labels"])
        ax.set_xlim(-0.35, len(x) - 0.65)
        ax.set_ylim(0, top * 1.14)
        ax.set_xlabel("question position in the conversation")
        ax.set_title(e["name"], pad=7, color=INK)
    axes[0].set_ylabel("F1")
    # third panel: the first question against the later ones, both absolute, one row per arm (LooGLE, N=65 pairs)
    e = res["loogle"]
    fv = e["first_vs_later"]
    ax = axes[2]
    frame(ax)
    ax.grid(axis="y", visible=False)
    labs = [lab for lab, *_ in ARMS]
    ys = list(range(len(labs)))[::-1]
    for y, lab in zip(ys, labs):
        c, mk, _ = STYLE[lab]
        a = fv["arms"][lab]
        ax.plot([a["first"], a["later"]], [y, y], "-", color=c, lw=2.8, alpha=0.55, zorder=3)
        ax.plot([a["first"]], [y], mk, color=c, ms=19 if mk == "*" else 11, mfc="white", mec=c, mew=2.0, zorder=5)
        ax.plot([a["later"]], [y], mk, color=c, ms=26 if mk == "*" else 11, mfc=c, mec="white", mew=0.8, zorder=5)
    ax.set_yticks(ys)
    ax.set_yticklabels(labs)
    ax.set_ylim(-0.7, len(labs) - 0.3)
    ax.set_xlim(0, max(max(a["first"], a["later"]) for a in fv["arms"].values()) * 1.16)
    ax.set_xlabel("F1")
    ax.set_title("LooGLE: first (hollow) vs later (filled)", pad=7, color=INK)
    h, l = axes[1].get_legend_handles_labels()
    legend_below(fig, h, l, ncol=4, letters=axes)
    save(fig, "paper_turn_position")
    # ------------- caption block
    B, E = "<!-- BEGIN generated: turn position -->", "<!-- END generated: turn position -->"
    out = [B, "", "## figures/paper_turn_position — F1 by question position on the two accumulate "
           "benchmarks (appendix)", "",
           "Generated by `scripts/turn_position_analysis.py --plot` from the canonical accuracy cells (the logs the "
           "main tables publish), every row re-scored with the current metric on the shared (conversation, turn) "
           "keys; store `results/timing/turn_position.json`. Full-context 32B = teacher, Full-context 7B = floor; "
           "presses at kept 21.9%, KV quantization at int4."]
    for b in ("loogle", "locomo"):
        e = res[b]
        vname = VIEW[b]
        v = e["views"][vname]
        basis = (f"the {e['balanced_conversations']} conversations that have a question in every column"
                 if vname == "balanced" else f"all {e['conversations']} conversations")
        out += ["", f"**{e['name']}** — shared-N={e['shared_n']} over {e['conversations']} conversations "
                f"(questions 1..{e['max_turn']}); the panel and this table are on {basis}, n={v['n']}.", "",
                "| arm | " + " | ".join(f"Q{lb}" for lb in v["bin_labels"]) + " | all rows | first question | later questions | later − first, paired per conversation |",
                "|---|" + "---:|" * (len(v["bin_labels"]) + 4),
                "| N | " + " | ".join(str(n) for n in v["bin_n"]) + f" | {v['n']} | {e['first_vs_later']['n_first']} | {e['first_vs_later']['n_later']} | {e['first_vs_later']['conversations']} conversations |"]
        for lab, *_ in ARMS:
            a = e["first_vs_later"]["arms"][lab]
            allm = st.fmean(v["arms"][lab])
            out.append(f"| {lab} | " + " | ".join(f"{x:.3f}" for x in v["arms"][lab])
                       + f" | {allm:.3f} | {a['first']:.3f} | {a['later']:.3f} | {a['paired_mean']:+.3f} ({a['up']}:{a['down']}) |")
        if b == "locomo":
            cm = e["category_mix"]
            out += ["", "LoCoMo question CATEGORY composition, the confound this bench carries (category 2 is the "
                    "easiest for every arm — teacher 0.658 against 0.492 on category 1 — and it is 60% of "
                    "the ten first questions against ~43% later, so the bench's first-question point is a "
                    "composition artefact and is not drawn as its own column):", "",
                    "| position | " + " | ".join(f"category {c}" for c in cm[0]["counts"]) + " |",
                    "|---|" + "---:|" * len(cm[0]["counts"]),
                    "| question 1 only (N=10) | " + " | ".join(str(n) for n in e["category_mix_first_question"].values()) + " |"]
            for row in cm:
                out.append(f"| questions {row['bin']} | " + " | ".join(str(n) for n in row["counts"].values()) + " |")
            out += ["", "Within one category the position effect survives (view `category 1` / `category 2` in the "
                    "store): on category 1, snapKV falls 0.468 → 0.291 and SpecPrefill 0.415 → 0.328 from "
                    "questions 1–5 to 21–30 while PAIR rises 0.324 → 0.497 and the teacher 0.442 → "
                    "0.562; on category 2, snapKV 0.484 → 0.442, SpecPrefill 0.513 → 0.403, PAIR 0.506 "
                    "→ 0.610." , "", "On LoCoMo the two query-dependent presses are the only arms that fall in BOTH categories, but arms with no query dependence move here as well — the 7B floor loses ground in both (0.443 → 0.430 and 0.464 → 0.401) and ExpectedAttention goes down on one and up on the other (0.443 → 0.298, 0.185 → 0.306). Ten conversations cannot separate a selection effect from whatever else changes along a 30-question conversation, so the claim rests on LooGLE's 65 paired conversations; LoCoMo is the agreeing direction, not a second measurement of the same size."]
    out += ["", "**Measurement sentence.** x (left and middle) is the question's position in its conversation, "
            "binned; y is F1 on the current metric, the mean over the rows in that bin. The right panel is "
            "LooGLE's first question (hollow mark) against the mean of that conversation's later questions "
            "(filled mark), both absolute, over the 65 conversations that have at least two questions. "
            "'later − first, paired' = per conversation, the mean of its questions from position 2 on minus its "
            "first question, averaged over those conversations (up:down = conversations where it is "
            "positive:negative).", "",
            "**Why the two groups separate.** snapKV and SpecPrefill are QUERY-DEPENDENT: the harness selects the "
            "kept tokens with the FIRST question (`run_single_batched` calls `compress_append(ctxs, qc)` / "
            "`specprefill_append(ctxs, qc)` with `qc` = turn 1's question) and the compressed cache is then frozen "
            "for the whole conversation. They are therefore measured on the one question their compression was "
            "built for, and lose accuracy on every later one — LooGLE paired: snapKV −0.115 (25:35 of 65), "
            "SpecPrefill −0.051 (29:33). Everything that does not select on the query is flat or slightly up on "
            "the same rows: teacher +0.042, PAIR +0.043, floor-7B +0.020, ExpectedAttention +0.013 (query-agnostic "
            "by construction), KV quantization +0.067. Read in the logs: on the five worst late-question rows the "
            "turn-1 question is about a different part of the document (conversation lg-0a904ef3, question 1 \"in "
            "what two parts of Asia are the countries with these stories located?\" against question 6 \"how many "
            "stories, according to Jan-Öjvind Swahn?\" — teacher 1.00, snapKV 0.00). PAIR's reader holds "
            "the whole context and its query LM carries no history (`lm_no_accum`), so neither branch depends on "
            "which question came first.", "",
            "**What this means for a single-question benchmark.** A benchmark that asks one question per context "
            "measures the query-dependent presses at their best point. On LooGLE the first question alone "
            "puts snapKV at 0.455 against PAIR's 0.420; over the later questions it is 0.341 against 0.453.", "", E]
    path = "figures/paper_captions.md"
    txt = open(path).read() if os.path.exists(path) else ""
    block = "\n".join(out)
    if B in txt and E in txt:
        txt = txt[:txt.index(B)] + block + txt[txt.index(E) + len(E):]
    else:
        txt = txt.rstrip() + "\n\n---\n\n" + block + "\n"
    open(path, "w").write(txt)
    print(f"  wrote the turn-position caption into {path}")


if __name__ == "__main__":
    r = analyse()
    if "--plot" in sys.argv:
        plot(r)
