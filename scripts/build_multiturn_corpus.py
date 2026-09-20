#!/usr/bin/env python
"""EVAL-MATCHED (multi-turn) training corpus — 2026-08-11.

THE MISMATCH THIS FIXES. Every fusion adapter so far was trained on a FLAT, single-turn sequence:

    reader:  INSTRUCTION ++ context ++ question ++ target
    LM:      INSTRUCTION ++ question ++ target

but `mtrag_accum` decodes something else entirely. It prefills the context ONCE, and from turn 2 on both
branches carry an accumulated history — `q_turn(q_i) + <reasoning> + "Final Answer: a_i"` — committed to
their KV, so the reader at turn k reads

    INSTRUCTION ++ context ++ [q1 a1 q2 a2 ... q_{k-1} a_{k-1}] ++ q_k

and the LM reads that same history with NO context. Training never showed either branch a history. The two
branches are trained to be each other's complement at a specific pair of inputs; if that pair is not the
one they meet at decode time, the complementarity is fitted to the wrong thing — which is exactly the
regime where a JOINT (fused-loss) objective can score worse than a standalone one even while its training
loss is lower. So this is a candidate cause, not a verdict, and it is now testable: same corpus content,
same objective, only the input format changes.

WHAT IT BUILDS. Rows sharing a context are chained into a conversation: for a context with questions
q1..qm and their TEACHER targets a1..am (gold is never used — the history answer is the teacher's own
generation, same doctrine as the target), it emits one row per turn k with `history` = the k-1 preceding
q/a blocks and `target` = a_k. Turn 1 rows are the original flat rows, so the corpus still covers the
single-turn case the benchmarks with one question per context (CLUTRR, musique) actually use.

Usage:
  build_multiturn_corpus.py --in results/fusionft/binding_corpus_v5.jsonl \
                            --out results/fusionft/binding_corpus_v5_MULTITURN.jsonl --max-turns 5
"""
import argparse, hashlib, json, random, sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.mab_eval import q_turn
import scripts.reason_fix as RF


def hist_block(question, target, reason_hist="gen"):
    """byte-identical to scripts/mtrag_accum.py::_hist_reason.

    ★ The runs use REASON_HIST='gen' (the env default, and what every LoCoMo provenance records), i.e. the
    history holds the model's OWN raw generation — no reference answer anywhere. The gold-free stand-in in
    training is the teacher's target, verbatim and rstripped. The 'ref' branch is kept only because the code
    has it; matching it when the harness runs 'gen' produces a corpus that matches nothing (16% of blocks
    differ, because reason_prefix trims the ramble those targets carry).
    """
    if reason_hist == "ref":
        rp = RF.reason_prefix(target)
        aref = target.split("Final Answer:")[-1].strip()
        return q_turn(question) + (rp + "\n" if rp else "") + f"Final Answer: {aref}\n"
    return q_turn(question) + str(target).rstrip() + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-turns", type=int, default=5, help="cap on conversation length")
    ap.add_argument("--reason-hist", choices=("gen", "ref"), default="gen",
                    help="must equal the harness's REASON_HIST (env default 'gen')")
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    groups = {}
    for line in open(a.inp):
        r = json.loads(line)
        if not (r.get("target") or "").strip():
            continue
        groups.setdefault(hashlib.md5(r["context"].encode()).hexdigest(), []).append(r)

    n_out = 0
    hist_turns = 0
    n_conv = 0
    with open(a.out, "w") as f:
        for ctx_key, rows in groups.items():
            rng.shuffle(rows)
            for start in range(0, len(rows), a.max_turns):
                conv = rows[start:start + a.max_turns]
                conv_key = f"{ctx_key[:12]}-{start//a.max_turns}"
                n_conv += 1
                history = ""
                for k, r in enumerate(conv):
                    out = dict(r)
                    out["history"] = history                    # "" for turn 1 == the old flat format
                    out["turn_index"] = k
                    out["conv_key"] = conv_key                  # so the growing-prefix invariant is checkable
                    out["example_id"] = f"{r.get('example_id','row')}#t{k}"
                    f.write(json.dumps(out) + "\n")
                    n_out += 1
                    hist_turns += 1 if history else 0
                    history += hist_block(r["question"], r["target"], a.reason_hist)

    print(f"[multiturn] {n_out} rows -> {a.out}  ({hist_turns} carry a history, "
          f"{n_out - hist_turns} are turn-1/flat; {len(groups)} contexts, {n_conv} conversations)")


if __name__ == "__main__":
    main()
