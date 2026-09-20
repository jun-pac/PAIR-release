#!/usr/bin/env python
"""Score the CONTEXT-LENGTH axis under one uniform answer boundary (2026-09-03).

WHY THIS FILE EXISTS, AND WHY IT IS SCOPED TO THIS EXPERIMENT ONLY.
The harness is used as-is everywhere else. What changes with context is the MODEL: the arm that
actually reads the long context stops emitting EOS promptly after its answer and appends another
sentence, and since `mab_eval.done_final_answer` requires a newline after "Final Answer:" -- which
this prompt never produces, at ANY context length -- nothing in the harness cuts at the answer
boundary and the continuation lands in the extracted answer. Measured, characters emitted after the
marker (median / share over 60 chars):

    teacher-32B   d40 18 / 2.5%    d80 18 / 5.8%    d120 23 / 19.5%   d160 28 / 35.0%
    ours 32B+7B   d40 14 / 1.7%    d80 14 / 2.0%    d120 14 /  2.0%   d160 14 /  1.8%
    floor-7B      d40 14 / 2.0%    d80 14 / 2.7%    d120 14 /  3.2%   d160 14 /  3.6%

Only the context-reading arm drifts, and it drifts with context, so leaving it uncorrected biases
the axis in OUR favour -- the direction that may never stand. The phrase is not in the documents
(searched all 96,000 at d160: zero hits); the model generates it.

WHAT THIS DOES. It applies the project's OWN deployed stop rule -- scripts/teacher_stop_rule.py,
already used when generating training targets -- to every arm of the axis identically, then
re-extracts and re-scores with the ONE canonical extractor and metric. The rule's restart-marker
alternation already lists "You are given context and a question"; the marker seen here is
"You are given a new question and a set of background knowledge", the same species of restart, so
the entry is generalised to "You are given" FOR THIS SCORER ONLY. A hotpot answer is a short span
(median 14 characters after the marker), so that phrase cannot occur inside a real answer.

WHAT IT DOES NOT DO. It does not touch scripts/teacher_stop_rule.py, the canonical extractor, or
any published number outside this axis. User instruction, 2026-09-03: uniform within the context
experiment, including the teacher, and "context 실험에서만" -- only there.

THE CHECK THAT MAKES IT HONEST: at d40 the trim must be a NO-OP, because d40 is the published cell.
If d40 moves, this is a metric change and not a boundary fix, and the script says so.
"""
import json
import os
import re
import sys

sys.path.insert(0, ".")
from scripts.reason_fix import extract_answer          # noqa: E402  the ONE canonical extractor
from src.eval import compute_best_em_f1                # noqa: E402  the ONE canonical metric

_FA = re.compile(r"Final Answer:", re.IGNORECASE)
# scripts/teacher_stop_rule.py's alternation, extended to the restart openers ACTUALLY OBSERVED in
# this axis. Every one of these was read off the longest surviving predictions, not guessed:
#   "You are given a new question and a set of background knowledge…"   (the first one found)
#   "You are to act as a logical inference engine…"
#   "You are to act as a detective solving a case…"
#   "You are an AI assistant. You will be given a task…"
#   "Answer by reasoning first: …"
# They are all one thing: the model finishes its answer and opens a NEW instruction turn in the
# second person. So the entry is `You (are|will|may|must)` plus that one imperative, rather than a
# growing list of individual sentences. A hotpot answer is a short span -- median 2 words after the
# trim -- so none of these can occur inside a real answer.
# ★ NO LEADING \b ON THE "You are" ALTERNATIVE. The leak is glued to the answer with no separator
# ("Army of the Holy Roman EmpireYou are given a new question..."), so requiring a word boundary
# before "You" misses exactly the case this exists for: with \b it trimmed 41 of 600 instead of
# 140, and the teacher scored LOWER than the narrower literal it replaced. The absence of a
# boundary IS the signature.
# THE DISCIPLINE THIS LIST NEEDS: a curated marker list can drift into a bespoke metric. Three
# things stop it. (1) d40, the published cell, must stay an EXACT no-op -- the script fails loudly
# if it moves. (2) The same list is applied to every arm, never only to the one that misbehaves.
# (3) The residual is reported, so what is still uncut stays visible instead of being assumed away.
_RESTART = re.compile(r"\s*(?:Step\s*\d+\s*[\(:]|\bReasoning\s*:|Human\s*:|Assistant\s*:"
                      r"|<\|im_(?:start|end)\|>|You\s+(?:are|will|may|must)\b"
                      r"|\bAnswer\s+by\s+reasoning\b|\bQuestion\s*:|Final\s*Answer\s*:)",
                      re.IGNORECASE)


def stop_after_answer(text):
    text = (text or "").strip()
    m = _FA.search(text)
    if not m:
        return text
    tail = text[m.end():]
    nl = tail.find("\n")
    line = tail if nl < 0 else tail[:nl]
    r = _RESTART.search(line)
    if r and r.start() > 0:
        line = line[: r.start()]
    return text[: m.end()] + line.rstrip()


# The three compression baselines are carried at kept 21.875% ONLY — the level at which a press
# holds the same share of the teacher's KV that ours' reader does, so the axis compares arms that
# have paid the same memory price. Running the whole kept grid at every context length would be 4x
# the GPU for a question this axis does not ask.
AXIS = [(40, 6.1, dict(teacher="hofa_teacher", ours="hofa_ours", floor="hofa_floor7",
                       snapkv="hofa_snap781", expected="hofa_expected781", spec="hofa_spec21875")),
        (80, 11.8, dict(teacher="ctx_ho80_t", ours="ctx_ho80_o", floor="ctx_ho80_f",
                        snapkv="ctxb_ho80_snap219", expected="ctxb_ho80_expected219",
                        spec="ctxb_ho80_spec219")),
        (120, 18.0, dict(teacher="ctx_ho120_t", ours="ctx_ho120_o", floor="ctx_ho120_f",
                         snapkv="ctxb_ho120_snap219", expected="ctxb_ho120_expected219",
                         spec="ctxb_ho120_spec219")),
        (160, 23.3, dict(teacher="ctx_ho160_t", ours="ctx_ho160_o", floor="ctx_ho160_f",
                         snapkv="ctxb_ho160_snap219", expected="ctxb_ho160_expected219",
                         spec="ctxb_ho160_spec219"))]
ARMS = ("teacher", "ours", "floor", "snapkv", "expected", "spec")

# ── the axis PAST the native window (2026-09-11, user: 200 / 240 / 280 / 320 documents with YaRN) ──
# Same 600 questions, prompt, metric, boundary and scorer; both models run with ROPE_YARN_FACTOR=4
# (stamped axis_rope_yarn_factor), which the store carries as `yarn` on each of these depths so the
# figure can mark where the model changes. Teacher / ours / floor only — the presses were not run
# past d160. The context label is measured the way the four native labels were: the median
# `acc_ctx_tok` of the teacher's 600 rows (d40 6.1k, d80 11.8k, d160 23.3k reproduce exactly);
# until that arm lands the label is PROVISIONAL: the Qwen-tokenized document block's median
# (results/timing/context_tokens_by_depth.json) plus the ~1.0k of instruction and question.
def _ktok(teacher_tag, provisional):
    p = f"results/fusionft/{teacher_tag}.jsonl"
    if os.path.exists(p):
        import statistics as _st
        v = [json.loads(l)["acc_ctx_tok"] for l in open(p) if l.strip() and '"acc_ctx_tok"' in l]
        if len(v) >= 600:
            return round(_st.median(v) / 1000, 1)
    return provisional


YARN_DEPTHS = [(200, 29.0), (240, 34.8), (280, 40.7), (320, 46.5)]
AXIS += [(d, _ktok(f"ctx_ho{d}_t", k),
          dict(teacher=f"ctx_ho{d}_t", ours=f"ctx_ho{d}_o", floor=f"ctx_ho{d}_f"))
         for d, k in YARN_DEPTHS]


def score(tag):
    p = f"results/fusionft/{tag}.jsonl"
    if not os.path.exists(p):
        return None
    rows = [r for r in (json.loads(l) for l in open(p) if l.strip()) if "acc_f1" in r]
    if not rows:
        return None
    asis = sum(r["acc_f1"] for r in rows) / len(rows)
    trimmed, moved, resid = 0.0, 0, 0
    # reasoning share and empty answers (2026-09-11): past the native window the 7B floor starts
    # answering without its reasoning step and sometimes with nothing (probe 3129254: 6/24 reasoned,
    # 4 empty at 58-67k tokens; d200's longest 120 rows: 61/120, 18 empty). The harness neither forces
    # nor suppresses reasoning (reason_fix.block_logits is a no-op), so this is the model's behaviour
    # and every table that prints the score prints these two counts beside it.
    straight = sum(1 for r in rows if (r.get("raw") or "").lstrip().startswith("Final Answer"))
    empty = sum(1 for r in rows if not (r.get("acc_pred") or "").strip())
    for r in rows:
        t = stop_after_answer(r["raw"])
        pred = extract_answer(t)
        if pred != r["acc_pred"]:
            moved += 1
        if len(pred.split()) > 15:            # the residual: still long after the trim
            resid += 1
        trimmed += compute_best_em_f1(pred, [r["gold"]])[1]
    return dict(n=len(rows), asis=round(asis, 4), trimmed=round(trimmed / len(rows), 4),
                moved=moved, residual_over15w=resid,
                reasoned_pct=round(100 * (len(rows) - straight) / len(rows), 1), empty=empty,
                yarn=rows[0].get("_provenance", {}).get("axis_rope_yarn_factor"))


# ── the cross-family context axis (2026-09-03) ─────────────────────────────────────────────────
# Two points per family, d40 (already measured, ledger §8) and d160, on the SAME uniform answer
# boundary as the Qwen axis — that is the whole point of a boundary rule: it is applied to every arm
# of every family identically, or it is a per-arm adjustment and worth nothing.
# d80 and d120 added 2026-09-04 (jobs 3084593-3084605, user: the middle of the axis was empty).
XFAM = {
    # Gemma-3 raw-prompt cells (gmho*) withdrawn 2026-09-05: the pair never reasoned on that input
    # (no <bos>, no turn markers). The re-run with Gemma's own format is family gemmacw (gmcwho*).
    "Gemma-3 12B+4B (own chat format)": [
        (40, 6.1, dict(teacher="gmcwho_teacher12", ours="gmcwho_ft_ours", floor="gmcwho_floor4")),
        (80, 11.8, dict(teacher="gmcwho80_teacher12", ours="gmcwho80_ft_ours", floor="gmcwho80_floor4")),
        (120, 18.0, dict(teacher="gmcwho120_teacher12", ours="gmcwho120_ft_ours",
                         floor="gmcwho120_floor4")),
        (160, 23.3, dict(teacher="gmcwho160_teacher12", ours="gmcwho160_ft_ours",
                         floor="gmcwho160_floor4"))],
    # Qwen2.5 32B+3B (2026-09-05, user): the teacher is the Qwen axis's own at each depth; the
    # 3B floor and the fine-tuned 32B+3B fusion arm are q3ho<D>_* (jobs of 2026-09-05).
    "Qwen2.5 32B+3B": [
        (40, 6.1, dict(teacher="hofa_teacher", ours="q3ho40_ft_ours", floor="q3ho40_floor3")),
        (80, 11.8, dict(teacher="ctx_ho80_t", ours="q3ho80_ft_ours", floor="q3ho80_floor3")),
        (120, 18.0, dict(teacher="ctx_ho120_t", ours="q3ho120_ft_ours", floor="q3ho120_floor3")),
        (160, 23.3, dict(teacher="ctx_ho160_t", ours="q3ho160_ft_ours", floor="q3ho160_floor3"))],
    # OLMo-3 fusion arm = the adapters RE-TRAINED under the ChatML rendering the evals use (family
    # olmocw, 2026-09-06, jobs 3099231-36); the 09-03 adapters were trained on the raw rendering
    # (omho*_ft_ours, kept on disk, off the tables). Teacher and floor arms are unchanged.
    "OLMo-3 32B+7B": [
        (40, 6.1, dict(teacher="omho_teacher32", ours="omcwho40_ft_ours", floor="omho_floor7")),
        (80, 11.8, dict(teacher="omho80_teacher32", ours="omcwho80_ft_ours", floor="omho80_floor7")),
        (120, 18.0, dict(teacher="omho120_teacher32", ours="omcwho120_ft_ours",
                         floor="omho120_floor7")),
        (160, 23.3, dict(teacher="omho160_teacher32", ours="omcwho160_ft_ours",
                         floor="omho160_floor7"))],
}


def score_xfam():
    """Write results/timing/xfam_context.json, or say which arms are still missing."""
    out, missing = {}, []
    for fam, axis in XFAM.items():
        for d, ktok, arms in axis:
            for arm in ("teacher", "ours", "floor"):
                r = score(arms[arm])
                # A file is COMPLETE at the bench's 600 questions. A job still writing leaves a
                # partial file on disk, and scoring it produced a teacher below its own d160 value
                # and a "-700%" closeness on 2026-09-04 — longest-first bucketing puts the hardest
                # questions first, so a partial score is not an estimate of anything.
                if r is None or r["n"] < 600:
                    missing.append(f"{fam} d{d} {arm} ({arms[arm]}"
                                   + (f", PARTIAL {r['n']}/600" if r else "") + ")")
                    continue
                out.setdefault(fam, {}).setdefault(f"d{d}", {})[arm] = r
                out[fam][f"d{d}"]["ctx_ktok"] = ktok
    for fam in out:
        for k, e in out[fam].items():
            if all(a in e for a in ("teacher", "ours", "floor")):
                T, O, F = (e[a]["trimmed"] for a in ("teacher", "ours", "floor"))
                e["closeness_pct"] = round((O - F) / (T - F) * 100, 1) if T != F else None
    print("\ncross-family context axis — the same uniform boundary, applied to every family:")
    for fam in XFAM:
        for d, ktok, _ in XFAM[fam]:
            e = out.get(fam, {}).get(f"d{d}")
            if not e or "closeness_pct" not in e:
                continue
            print("  %-16s d%-4d %4.1fk tok   teacher %.4f  ours %.4f  floor %.4f   closeness %5.1f%%"
                  % (fam, d, ktok, e["teacher"]["trimmed"], e["ours"]["trimmed"],
                     e["floor"]["trimmed"], e["closeness_pct"]))
    if missing:
        print("  still missing: " + ", ".join(missing))
    if out:
        out["_source"] = ("scripts/score_context_axis.py score_xfam() — the Qwen axis's uniform "
                          "answer boundary and canonical extractor, applied to Gemma-3 and OLMo-3 "
                          "at four context lengths (d80/d120 added 2026-09-04). Accuracy only.")
        json.dump(out, open("results/timing/xfam_context.json", "w"), indent=2)
        print("  wrote results/timing/xfam_context.json")


def main():
    out, bad = {}, False
    print("hotpotQA context axis — the SAME 600 questions and golds at four context lengths, one "
          "uniform answer boundary applied to every arm\n")
    print("%-14s %5s %9s %9s %9s   %s" % ("context", "arm", "as-scored", "trimmed", "delta",
                                          "rows whose answer changed"))
    for d, ktok, arms in AXIS:
        for arm in ARMS:
            s = score(arms[arm]) if arm in arms else None
            if s is None:
                print("  d%-4d %-9s  not present" % (d, arm))
                continue
            # a file is COMPLETE at the bench's 600 questions; a running job's partial file is not a
            # cell (longest-first bucketing puts the hardest questions first, so a partial score is not
            # an estimate of anything) — printed, never stored (same rule as score_xfam, 2026-09-11)
            if s["n"] < 600:
                print("  d%-4d %-9s  PARTIAL %d/600 — printed, not stored (inline %.4f, reasoned %.1f%%, empty %d)"
                      % (d, arm, s["n"], s["asis"], s["reasoned_pct"], s["empty"]))
                continue
            out.setdefault(f"d{d}", {})[arm] = s
            out[f"d{d}"]["ctx_ktok"] = ktok
            if d >= 200:                       # the YaRN half of the axis, stamped from provenance
                out[f"d{d}"]["yarn"] = s.get("yarn")
                out[f"d{d}"]["ctx_ktok_provisional"] = not os.path.exists(f"results/fusionft/ctx_ho{d}_t.jsonl")
            print("  d%-4d %-9s %9.4f %9.4f %+9.4f   %4d/%d   reasoned %5.1f%%  empty %d" %
                  (d, arm, s["asis"], s["trimmed"], s["trimmed"] - s["asis"], s["moved"], s["n"],
                   s["reasoned_pct"], s["empty"]))
            if s["residual_over15w"]:
                print("       residual: %d/%d predictions are still over 15 words after the trim"
                      % (s["residual_over15w"], s["n"]))
            # THE d40 GATE, restated 2026-09-03 when the baselines joined the axis. The trim is a
            # no-op at d40 for teacher / ours / floor, but it moves SpecPrefill by +0.0105 (8 rows
            # of 600) -- UPWARD, i.e. it makes a competitor better. So the gate cannot be "nothing
            # moves"; it is "nothing moves in the direction that flatters us":
            #   * ours may not go UP,
            #   * no competitor (teacher, snapKV, ExpectedAttention, SpecPrefill) may go DOWN.
            # Movement the other way is conservative, and is printed rather than hidden.
            if d == 40:
                dlt = s["trimmed"] - s["asis"]
                if (arm == "ours" and dlt > 5e-4) or (arm != "ours" and dlt < -5e-4):
                    bad = True
                elif abs(dlt) > 5e-4:
                    print("       d40 %s moves %+0.4f under the uniform boundary — AGAINST us "
                          "(this row's published cell is the untrimmed %.4f)" % (arm, dlt, s["asis"]))
    print("\nd40 is the PUBLISHED cell: the trim must never move it in the direction that "
          "flatters ours.  ->  %s" %
          ("❌ IT MOVED OUR WAY — a metric change, not a boundary fix; do not use these numbers"
           if bad else "✅ no-op for teacher/ours/floor; SpecPrefill moves +0.0105 AGAINST us"))
    if not bad:
        for d, ktok, arms in AXIS:
            k = f"d{d}"
            if k in out and all(a in out[k] for a in ("teacher", "ours", "floor")):
                T, O, F = (out[k][a]["trimmed"] for a in ("teacher", "ours", "floor"))
                out[k]["closeness_pct"] = round((O - F) / (T - F) * 100, 1)
        out["_source"] = ("scripts/score_context_axis.py — the project's deployed stop rule applied "
                          "uniformly to every arm, canonical extractor and metric, hotpotQA d40 s42 "
                          "FULL-600 questions at up to eight document counts (d200-d320 with "
                          "ROPE_YARN_FACTOR=4 on both models, 2026-09-11). Scoped to this axis only.")
        json.dump(out, open("results/timing/context_axis.json", "w"), indent=2)
        print("\nwrote results/timing/context_axis.json")
        print("\nclosest-to-teacher, on the uniform boundary:")
        for d, ktok, _ in AXIS:
            k = f"d{d}"
            if "closeness_pct" in out.get(k, {}):
                b = " ".join("%s %.4f" % (a[:4], out[k][a]["trimmed"])
                             for a in ("snapkv", "expected", "spec") if a in out[k])
                print("  d%-4d %4.1fk tok   teacher %.4f  ours %.4f  floor %.4f   closeness %5.1f%%"
                      "   | at kept 21.9%%: %s" %
                      (d, ktok, out[k]["teacher"]["trimmed"], out[k]["ours"]["trimmed"],
                       out[k]["floor"]["trimmed"], out[k]["closeness_pct"], b))
    score_xfam()
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
