#!/usr/bin/env python
"""The full-context 32B's distribution at ITS OWN decisive token (user, 2026-09-20: show the teacher at the
Baz it wrote, rather than forcing it onto PAIR's prefix).

The decisive-token probe forced every arm along the two published texts of PAIR and of the full-context 7B,
so the teacher's numbers there are conditioned on PAIR's prefix. Its own answer does not carry the
Step 1 scaffold, and there is no reason to align it: this run forces the teacher along ITS OWN text and
reads the distribution at the token right after "Therefore,", which is exactly what it produced while
decoding freely. One forward of the 32B, no adapters.

Writes at_own_fork into results/timing/decisive_token.json and leaves every existing field alone.
ANALYSIS PROBE: one forced forward, no score and no timing.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ANSWER_PROMPT_VARIANT", "reason_v3")

CONV = "ho-5a794119554299029c4b5f3c"
STORE = "results/timing/decisive_token.json"
K = 20


def main():
    import torch
    import scripts.mtrag_accum as MA
    import scripts.bench_config as BC
    from scripts.dump_divergence_logits import build_blocks, enc, load_log
    from transformers import AutoTokenizer
    from src.models import load_causal_lm

    cache = os.environ.get("HF_HOME", "/work/hdd/myproject/anon/hf")
    cfg = BC.BENCHMARKS["hotpotqa_st40_full"]
    by = {}
    for line in open(cfg["ref"]):
        if line.strip():
            r = json.loads(line)
            by.setdefault(r.get("conversation_id") or r.get("conv"), []).append(r)
    T = load_log("results/fusionft/hofa_teacher.jsonl")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct", cache_dir=cache)
    instr, ctx, q = build_blocks(MA, by[CONV])
    gt = enc(tok, T[CONV]["raw"])
    # its own decisive position: the first token after "Therefore," in its own text
    dec = tok.decode(gt)
    cut = dec.index("Therefore,") + len("Therefore,")
    i_own = len(enc(tok, dec[:cut]))
    print(f"[plan] the teacher's own answer is {len(gt)} tokens; its decisive position is {i_own}, "
          f"token {tok.decode([gt[i_own]])!r}")
    print(f"  prefix ends: ...{tok.decode(gt[max(0, i_own - 12):i_own])!r}")
    assert tok.decode([gt[i_own]]).strip().startswith("Baz"), "that position is not the name token"

    prompt = enc(tok, instr) + enc(tok, ctx) + enc(tok, q)
    print("[32B] loading, no adapter", flush=True)
    lm, _ = load_causal_lm("Qwen/Qwen2.5-32B-Instruct", device_map="cuda:0", cache_dir=cache)
    lm.eval()
    with torch.no_grad():
        ids = torch.tensor([prompt + gt], device="cuda:0")
        out = lm(ids, use_cache=False, logits_to_keep=len(gt) + 1)
    z = out.logits[0, : len(gt)].float().cpu()          # row j predicts gt[j]
    lp = torch.log_softmax(z[i_own], -1)
    top = torch.topk(lp, K)
    ent = float(-(lp.exp() * lp).sum())
    rows = [[tok.decode([int(i)]), float(v)] for v, i in zip(top.values, top.indices)]
    print(f"[result] entropy {ent:.4f} nats")
    for t_, v in rows[:6]:
        print(f"   {t_!r:14s} {v:+.4f}  p={pow(2.718281828, v):.4f}")

    D = json.load(open(STORE))
    D["at_own_fork"] = D.get("at_own_fork", {})
    D["at_own_fork"]["teacher"] = dict(
        i_own=i_own, entropy=ent, top=rows, n_tokens=len(gt),
        prefix_tail=tok.decode(gt[max(0, i_own - 12):i_own]),
        token=tok.decode([gt[i_own]]),
        _source=("scripts/teacher_own_fork.py: the full-context 32B forced along ITS OWN published answer "
                 "(hofa_teacher), read at the token after 'Therefore,', which is the token it generated "
                 "there while decoding freely. No adapter."))
    json.dump(D, open(STORE + ".tmp", "w"), indent=1)
    os.replace(STORE + ".tmp", STORE)
    print(f"wrote at_own_fork into {STORE}")


if __name__ == "__main__":
    main()
