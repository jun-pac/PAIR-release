#!/usr/bin/env python
"""Convert the CLUTRR multi-question benchmark (results/clutrr_multiq/clutrr_multiq.json) -> mtRAG-style
conversation JSONL for the stateful ACCUMULATE harness (scripts/mtrag_accum.py). This is the QASPER-style
KV-reuse packaging so CLUTRR gets the IDENTICAL fair sandwich + baselines as LoCoMo/QASPER (§7.0-V).

  * one STORY = one 'conversation'; each derived kinship question = a turn (order = the benchmark's question order,
    Q1 = the labeled CLUTRR target/full-k-hop, Q2+ = other derived facts).
  * the WHOLE padded story is introduced at turn 1 (contexts); turns 2+ have empty contexts -> mtrag_accum's pkey
    dedup makes Q1 prefill/compress the context once, Q2+ REUSE it (append-only KV).
  * the reference relation word is teacher-forced into history; gold = ONE relation word -> relation-vocabulary
    exact match (metric `clutrr_relation_em`, no length pathology).

This only REGROUPS the existing built benchmark (same story text via item['context'], same question phrasing via
src.clutrr_data.build_clutrr_question, same gold q['rel']). It does NOT regenerate or modify any CLUTRR data or the
non-accumulation results (results/clutrr_multiq*, results/clutrr_multiqd0_*).
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.clutrr_data import build_clutrr_question


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="results/clutrr_multiq/clutrr_multiq.json")
    ap.add_argument("--out", default="/work/hdd/myproject/anon/clutrr/clutrr_mq_reuse_ref.jsonl")
    # 2026-09-06 (user: the accumulate packaging is not the CLUTRR setting to show — "accum이 아닌 세팅이
    # 오히려 좋을거같음"): --single-question REGROUPS the accumulate ref so that EVERY question is its own
    # one-turn conversation with the whole story as its context. Same 331 stories, 1862 questions, story
    # text, question phrasing, golds and example_ids as the accumulate ref (it is read FROM that file, so the
    # bytes cannot drift); what changes is only that no question sees an earlier question's teacher-forced
    # reference answer, and nothing is reused across questions. The first 254 rows are the first 70
    # stories — the identical question set to the CLUTRR-accum N=254 cells (bench_config `clutrr_sq`).
    ap.add_argument("--single-question", action="store_true",
                    help="write one 1-turn conversation per question (read from --from-ref, not --src)")
    ap.add_argument("--from-ref", default="/work/hdd/myproject/anon/clutrr/clutrr_mq_reuse_ref.jsonl")
    a = ap.parse_args()
    if a.single_question:
        rows = [json.loads(l) for l in open(a.from_ref)]
        story = {r["conversation_id"]: r["contexts"] for r in rows if r["contexts"]}
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w") as fout:
            for r in rows:
                rec = dict(r)
                rec["conversation_id"] = f'{r["conversation_id"]}-q{r["q_index"]}'
                rec["turn"] = "1"
                rec["contexts"] = story[r["conversation_id"]]
                fout.write(json.dumps(rec) + "\n")
        print(f"[Done] single-question: {len(story)} stories, {len(rows)} one-turn conversations -> {a.out}")
        return
    data = json.load(open(a.src))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fout = open(a.out, "w"); nstories = nturns = 0
    for item in data:
        sid = item["story_id"]; ctx = item["context"]; qs = item["questions"]
        cid = f"clutrr-{sid}"
        contexts = [{"document_id": cid, "text": ctx}]
        for qi, q in enumerate(qs):
            question = build_clutrr_question(q["b"], q["a"])
            gold = q["rel"]
            rec = {
                "conversation_id": cid, "turn": qi + 1,
                "input": [{"speaker": "user", "text": question}],
                "contexts": contexts if qi == 0 else [],   # story only at turn 1 -> Q1 introduces, Q2+ reuse
                "targets": [{"text": gold}],
                "Answerability": ["ANSWERABLE"],
                "golds": [gold], "example_id": f"{sid}_{qi}", "q_index": qi,
            }
            fout.write(json.dumps(rec) + "\n"); nturns += 1
        nstories += 1
    fout.close()
    print(f"[Done] {nstories} stories, {nturns} questions -> {a.out}")


if __name__ == "__main__":
    main()
