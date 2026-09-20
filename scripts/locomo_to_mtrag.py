#!/usr/bin/env python
"""Convert LoCoMo (locomo10.json) -> mtRAG-style conversation JSONL for the stateful KV-reuse harness.
Each LoCoMo conversation = one "conversation": the FULL multi-session dialogue is the shared context introduced at
turn 1 (→ dedup makes Q1 compress/prefill it, Q2+ reuse); each QA = a turn; the answer is teacher-forced into history.
LoCoMo context is long (~18k tokens) → the reuse penalty (Q1-compression evicting Q2+ evidence) should be pronounced.
Category kept for per-category scoring (5 = adversarial/unanswerable → answer = 'Not mentioned in the conversation')."""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from locomo_serialize import serialize_conversation

ADV = "Not mentioned in the conversation"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="/work/hdd/myproject/anon/locomo/locomo10.json")
    ap.add_argument("--out", default="/work/hdd/myproject/anon/locomo/locomo_reuse_ref.jsonl")
    ap.add_argument("--max-q", type=int, default=30, help="cap questions/conversation (iteration; 0 = all)")
    ap.add_argument("--max-doc-chars", type=int, default=128000, help="~20k token cap on the conversation context")
    a = ap.parse_args()
    data = json.load(open(a.inp))
    fout = open(a.out, "w"); nconv = nq = 0
    for ci, c in enumerate(data):
        conv = c.get("conversation", c)
        ctx, _, _ = serialize_conversation(conv)
        if len(ctx) > a.max_doc_chars:
            ctx = ctx[:a.max_doc_chars]
        cid = c.get("sample_id") or f"locomo{ci}"
        contexts = [{"document_id": cid, "text": ctx}]
        qas = c.get("qa", [])
        if a.max_q: qas = qas[:a.max_q]
        for qi, q in enumerate(qas):
            cat = q.get("category")
            ans = q.get("answer", q.get("adversarial_answer", ""))
            if cat == 5 and not q.get("answer"):
                ans = ADV
            ans = str(ans)
            if not q.get("question") or not ans:
                continue
            rec = {
                "conversation_id": cid, "turn": qi + 1,
                "input": [{"speaker": "user", "text": q["question"]}],
                "contexts": contexts if qi == 0 else [],
                "targets": [{"text": ans}],
                "Answerability": ["ANSWERABLE"],
                "golds": [ans], "category": cat, "evidence": q.get("evidence"),
            }
            fout.write(json.dumps(rec) + "\n"); nq += 1
        nconv += 1
    fout.close()
    print(f"[Done] {nconv} conversations, {nq} questions -> {a.out}")

if __name__ == "__main__":
    main()
