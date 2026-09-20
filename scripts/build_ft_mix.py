#!/usr/bin/env python
"""Build a BENCHMARK-AGNOSTIC fusion-FT training corpus (the fairness fix).

WHY (user, 2026-07-24): fine-tuning on musique-train and evaluating on musique-val gives ours an
in-domain advantage the baselines never got. Instead we train ONE adapter on a MIX that contains
**none of the evaluation benchmarks**, and evaluate that same adapter everywhere. Then there is no
benchmark-specific FT, and the claim becomes "adapting the FUSION ROLE" rather than "learning a task".

SECOND motivation — answer-length calibration. The musique-only adapter lost the ability to modulate
answer length (measured on QASPER: recall +0.030 but precision −0.080; it described where gold was a
span). A mix spanning TERSE multi-hop QA and LONG-FORM summarization supplies both regimes, which is
the natural way to teach conditional length.

Sources are normalized to ONE record shape so a single prompt builder covers all of them:
    {source, example_id, question, documents[], golds[]}
For summarization tasks the LongBench task instruction becomes the `question` (the LM branch then sees
only that instruction — the hardest, most honest version of the fusion setting).

Output: a JSONL corpus. Teacher targets are added later by scripts/ft_mix_teacher_gen.py, and
scripts/fusion_distill_train.py --mix-corpus consumes the result.

Eval benchmarks (musique, qasper) are REFUSED as sources — the whole point is that they stay held out.
"""
import os, sys, json, argparse, random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ★ EVERY benchmark that any results table evaluates on must be held out — otherwise "trained on a held-out mix"
# is false for that table. hotpotqa was a TRAINING source until 2026-07-29 while §8.9 evaluates on it; that made
# the fairness claim wrong for the HotpotQA rows. Keep this in sync with scripts/bench_config.py's BENCHES.
HELD_OUT = {"musique", "qasper", "hotpotqa", "locomo", "eventqa", "mtrag", "clutrr", "asqa", "babilong"}


def load_source(name, n, cache_dir, doc_number, seed):
    """Return a list of normalized records for one source."""
    out = []
    if name in ("hotpotqa", "2wiki"):
        from src.data import load_hotpotqa_split, load_2wiki_split
        fn = load_hotpotqa_split if name == "hotpotqa" else load_2wiki_split
        for ex in fn("train", cache_dir=cache_dir, sample=n, doc_number=doc_number):
            golds = list(getattr(ex, "answers", None) or [ex.answer])
            out.append({"source": name, "example_id": str(ex.example_id), "question": ex.question,
                        "documents": list(ex.documents), "golds": golds})
    elif name in ("strategyqa", "scifact"):
        from src.data import load_strategyqa_split, load_scifact_split
        fn = load_strategyqa_split if name == "strategyqa" else load_scifact_split
        for ex in fn("train", cache_dir=cache_dir, sample=n, doc_number=doc_number):
            golds = list(getattr(ex, "answers", None) or [ex.answer])
            out.append({"source": name, "example_id": str(ex.example_id), "question": ex.question,
                        "documents": list(ex.documents), "golds": golds})
    elif name in ("gov_report", "multi_news", "qmsum", "passage_count", "passage_retrieval_en"):
        from src.longbench_data import load_longbench_split, LONGBENCH_SUMM_TASKS, LONGBENCH_COUNT_TASKS
        instr = {**LONGBENCH_SUMM_TASKS, **LONGBENCH_COUNT_TASKS}[name]
        for ex in load_longbench_split(name, cache_dir=cache_dir, sample=n):
            q = getattr(ex, "question", None) or instr          # qmsum has a real per-example query
            docs = list(getattr(ex, "documents", None) or [getattr(ex, "context_text", "")])
            golds = list(getattr(ex, "answers", None) or [getattr(ex, "answer", "")])
            out.append({"source": name, "example_id": str(getattr(ex, "example_id", f"{name}-{len(out)}")),
                        "question": q, "documents": docs, "golds": [g for g in golds if g]})
    else:
        raise SystemExit(f"unknown source '{name}'")
    random.Random(seed).shuffle(out)
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="hotpotqa:1200,2wiki:1200,gov_report:600,qmsum:600",
                    help="comma list of name:count (TRAIN splits / non-eval sets only)")
    ap.add_argument("--doc-number", type=int, default=40, help="retrieval docs for the QA sources")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", default=os.environ.get("HF_HOME", "/work/hdd/myproject/anon/hf"))
    ap.add_argument("--id-suffix", default="",
                    help="appended to every example_id. REQUIRED when the same source is built at several "
                         "doc-numbers, because ft_mix_teacher_gen.py dedups by example_id and would otherwise "
                         "silently skip the longer-context copy of an example it already generated.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    specs = []
    for part in a.sources.split(","):
        nm, _, cnt = part.partition(":")
        nm = nm.strip()
        if nm in HELD_OUT:
            raise SystemExit(f"❌ '{nm}' is an EVAL benchmark — it must stay held out of the training mix")
        specs.append((nm, int(cnt or 500)))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    total = 0
    with open(a.out, "w") as f:
        for nm, cnt in specs:
            recs = load_source(nm, cnt, a.cache_dir, a.doc_number, a.seed)
            for r in recs:
                if a.id_suffix:
                    r["example_id"] = f"{r['example_id']}{a.id_suffix}"
                f.write(json.dumps(r) + "\n")
            total += len(recs)
            gl = sum(len(str(g).split()) for r in recs for g in r["golds"][:1]) / max(1, len(recs))
            print(f"  {nm:12s} n={len(recs):5d}  mean gold words={gl:6.1f}", flush=True)
    print(f"[Done] {total} records -> {a.out}", flush=True)
    print(f"       held-out (never trained on): {sorted(HELD_OUT)}", flush=True)


if __name__ == "__main__":
    main()
