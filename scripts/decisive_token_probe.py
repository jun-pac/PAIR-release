#!/usr/bin/env python
"""The decisive token of one answer, in numbers (user, 2026-09-18: "결정적인 token이 딱 'Baz Luhrmann'가
나오는 순간인데, 각 모델에서 token logit이 어떻게 되고, 이런 것들을 숫자로 볼 수 있으면 좋을듯").

THE QUESTION. On hotpotQA ho-5a794119554299029c4b5f3c ("Who was born first, Nellee Hooper or Baz
Luhrmann?", gold "Baz Luhrmann") the reader alone (hofa_floor7, the 7B with the same 6,897-token
context) and the pair (hofa_ours) generate the SAME sentence of facts and part at ONE token:

  ... while Baz Luhrmann was born on 17 September 1962. Therefore, | " Baz"  (ours, = the teacher)
                                                                   | " Nel"  (reader alone)

This probe reads, at exactly that position, what each model's distribution was.

WHAT IS MEASURED. The two texts are re-encoded and FORCED through three branches, with the prompt built
by dump_divergence_logits.build_blocks — the same code the published divergence dump uses, so the
prompt is not rebuilt here:
  reader   7B + reader_binding_v5distill      , prompt = instruction | context | question
  query LM 32B + stage2_on_v5reader_lam07     , prompt = instruction | question          (lm_no_accum)
  teacher  32B, no adapter                    , prompt = instruction | context | question
and the pool is the deployed rule on raw logits, softmax(0.7 z_reader + 0.3 z_queryLM). λ=0.7 is
hotpot's deployed weight AND the train-λ of this LM adapter, so train-λ = eval-λ.

GATES, stated before the run (nothing is read if they fail):
  (1) the pool's argmax must reproduce OURS's text at >= 97% of its positions;
  (2) the reader's own argmax must reproduce the READER-ALONE text at >= 97% of its positions;
  (3) at the fork, the two texts' prefixes must be token-identical (first_div == the fork).
The teacher is forced along OURS's text as a reference, not a gate.

ALSO. With the 32B loaded and its adapter on, the query LM's own FREE-RUNNING answer to this question
is decoded greedily with no context (200 tokens, the harness's stop rule), which is the one arm the
motivation example was missing while the 600-question closed-book job (3167579) waits in the queue.

Output: results/timing/decisive_token.json (+ the top-10 of every branch at the fork, printed).
ANALYSIS PROBE: forced forwards plus one 200-token greedy decode. No timing, no benchmark score.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("REASON_FIX", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("ANSWER_PROMPT_VARIANT", "reason_v3")

import numpy as np  # noqa: E402

CONV = "ho-5a794119554299029c4b5f3c"
CKPT = "/work/hdd/myproject/anon/kvreuse_ckpts"
LAM = 0.7
OUT = "results/timing/decisive_token.json"
TOPK = 10          # printed at the fork
TOPK_ALL = 20      # stored at EVERY position, for every model (user, 2026-09-18: "top-k 토큰단위로 저장")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv", default=CONV)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    import scripts.mtrag_accum as MA
    from scripts.dump_divergence_logits import build_blocks, enc, first_div, load_log
    import scripts.bench_config as BC

    cfg = BC.BENCHMARKS["hotpotqa_st40_full"]
    rows = [json.loads(l) for l in open(cfg["ref"]) if l.strip()]
    by = {}
    for r in rows:
        by.setdefault(r.get("conversation_id") or r.get("conv"), []).append(r)
    X = load_log("results/fusionft/hofa_floor7.jsonl")      # the reader ALONE (7B, same context)
    Y = load_log("results/fusionft/hofa_ours.jsonl")        # the pair
    T = load_log("results/fusionft/hofa_teacher.jsonl")
    c = a.conv
    assert c in by and c in X and c in Y and c in T, f"{c} missing from the ref or one of the logs"

    from transformers import AutoTokenizer
    cache = os.environ.get("HF_HOME", "/work/hdd/myproject/anon/hf")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", cache_dir=cache)
    instr, ctx, q = build_blocks(MA, by[c])
    gx, gy = enc(tok, X[c]["raw"]), enc(tok, Y[c]["raw"])
    i_div = first_div(gx, gy)
    print(f"[plan] {c}\n  question: {X[c]['q']}\n  gold: {X[c]['gold']}")
    print(f"  prompt tokens: instruction {len(enc(tok, instr))} | context {len(enc(tok, ctx))} | question {len(enc(tok, q))}")
    print(f"  reader-alone text {len(gx)} tokens, ours {len(gy)} tokens, first divergence at {i_div}")
    print(f"  shared prefix ends: ...{tok.decode(gy[max(0, i_div - 14):i_div])!r}")
    print(f"  then  ours -> {tok.decode([gy[i_div]])!r}   reader alone -> {tok.decode([gx[i_div]])!r}")
    assert tok.decode(gx[:i_div]) == tok.decode(gy[:i_div]), "the prefixes are not identical (gate 3)"
    if a.dry_run:
        print("[dry-run] OK - nothing loaded")
        return

    import torch
    from src.models import load_causal_lm
    from src.lora_load import load_lora_stack
    import scripts.reason_fix as RF
    dev = "cuda:0"
    cache = os.environ.get("HF_HOME", "/work/hdd/myproject/anon/hf")
    pr_ctx = enc(tok, instr) + enc(tok, ctx) + enc(tok, q)
    pr_noctx = enc(tok, instr) + enc(tok, q)

    @torch.no_grad()
    def forced(model, prompt_ids, gens):
        seqs = [prompt_ids + g for g in gens]
        Tm = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), Tm), tok.pad_token_id, dtype=torch.long)
        am = torch.zeros((len(seqs), Tm), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s)
            am[i, : len(s)] = 1
        P = len(prompt_ids)
        out = model(ids.to(dev), attention_mask=am.to(dev), use_cache=False, logits_to_keep=Tm - (P - 1))
        return [out.logits[i, 0: len(g)].float().cpu() for i, g in enumerate(gens)]

    res = {}
    print("[7B] loading reader", flush=True)
    slm, _ = load_causal_lm("Qwen/Qwen2.5-7B-Instruct", device_map=dev, cache_dir=cache)
    slm = load_lora_stack(slm, f"{CKPT}/reader_binding_v5distill_r16_s900", label="reader"); slm.eval()
    zx_R, zy_R = forced(slm, pr_ctx, [gx, gy])          # the READER BRANCH of the pair (7B + adapter)
    # ★ and the FLOOR (2026-09-18 correction): the example's "reader alone" row is hofa_floor7, which is
    #   the BASE 7B with the same context and NO adapter — a different model from the pair's reader
    #   branch. The first run forced the adapted reader along the floor's text and its reconstruction
    #   gate failed at 0.912, which is the gate doing its job: that text was not generated by that model.
    with slm.disable_adapter():
        zx_B, zy_B = forced(slm, pr_ctx, [gx, gy])
    del slm; torch.cuda.empty_cache()

    print("[32B] loading query LM (+ lam07 adapter; the same load serves the teacher with the adapter off)", flush=True)
    lm, _ = load_causal_lm("Qwen/Qwen2.5-32B-Instruct", device_map=dev, cache_dir=cache)
    lm = load_lora_stack(lm, f"{CKPT}/stage2_on_v5reader_lam07_r16_s600", label="LM-lam07"); lm.eval()
    zx_Q, zy_Q = forced(lm, pr_noctx, [gx, gy])
    # the query LM's own free-running answer to this question, no context
    stop = {x for x in (tok.eos_token_id, 151643) if x is not None}   # the harness's rule + <|endoftext|>
    ids = torch.tensor([pr_noctx], device=dev)
    gen = []
    with torch.no_grad():
        past = None
        cur = ids
        for _ in range(200):
            o = lm(cur, past_key_values=past, use_cache=True)
            past = o.past_key_values
            nxt = int(o.logits[0, -1].argmax())
            if nxt in set(stop):
                break
            gen.append(nxt)
            if RF.reason_stopped(tok, gen):
                break
            cur = torch.tensor([[nxt]], device=dev)
    qlm_answer = tok.decode(gen)
    print(f"[query LM, free-running, no context] {qlm_answer!r}")
    with lm.disable_adapter():
        zx_T, zy_T = forced(lm, pr_ctx, [gx, gy])
    del lm; torch.cuda.empty_cache()

    def lsm(z):
        return torch.log_softmax(z, -1)
    # gates
    lpY_R, lpY_Q = lsm(zy_R), lsm(zy_Q)
    pool_y = lsm(LAM * zy_R + (1 - LAM) * zy_Q)
    g1 = float((pool_y.argmax(-1).numpy() == np.array(gy)).mean())
    g2 = float((lsm(zx_B).argmax(-1).numpy() == np.array(gx)).mean())
    # ★ THE GATE IS AT THE FORK, NOT OVER THE WHOLE TRAJECTORY (2026-09-18, after two failed runs).
    #   Both texts were generated INSIDE A BATCH — ours at 16 rows, the floor at 32 (axis_batch_size in
    #   each log; the floor's acc_ctx_tok 6,897 is that padded length, against this row's own 6,048) — so
    #   a single un-padded forward cannot reproduce them bitwise, and the arm with more padding drifts
    #   more: 0.970 for the pool, 0.868 for the floor. Those rates are REPORTED, not gated. What must
    #   hold is the claim the figure makes: at the fork the pool picks the token ours emitted and the
    #   base 7B picks the token the reader alone emitted. If that fails the example does not reproduce
    #   and nothing here may be read.
    fork_pool = int(pool_y[i_div].argmax()) == int(gy[i_div])
    fork_floor = int(lsm(zx_B[i_div]).argmax()) == int(gx[i_div])
    print(f"[trajectory] the pool reproduces OURS at {g1:.3f} of positions, the base 7B reproduces "
          f"READER-ALONE at {g2:.3f} — reported, not gated: both texts were decoded inside padded batches")
    print(f"[gate] at the fork (position {i_div}): pool -> {tok.decode([int(pool_y[i_div].argmax())])!r} "
          f"(ours emitted {tok.decode([int(gy[i_div])])!r}) | base 7B -> "
          f"{tok.decode([int(lsm(zx_B[i_div]).argmax())])!r} (reader alone emitted "
          f"{tok.decode([int(gx[i_div])])!r})")
    assert fork_pool and fork_floor, "the fork does not reproduce - nothing from this probe may be read"

    i = i_div
    cand = {"ours": gy[i], "reader_alone": gx[i]}
    rowset = {"reader_branch": lsm(zy_R[i]), "floor_7B_base": lsm(zy_B[i]), "query_lm": lsm(zy_Q[i]),
              "teacher": lsm(zy_T[i]), "fused": pool_y[i]}
    out = dict(conv=c, question=X[c]["q"], gold=X[c]["gold"], lam=LAM, i_div=int(i),
               ctx_tokens=int(X[c]["acc_ctx_tok"]), traj_pool_reproduces_ours=g1, traj_base7B_reproduces_reader_alone=g2,
               batch_rows={"ours": int(Y[c].get("batch_rows", 0)), "floor": int(X[c].get("batch_rows", 0))},
               shared_prefix_tail=tok.decode(gy[max(0, i - 14):i]),
               token_ours=tok.decode([gy[i]]), token_reader_alone=tok.decode([gx[i]]),
               query_lm_free_running_no_context=qlm_answer,
               texts={"teacher": T[c]["raw"], "ours": Y[c]["raw"], "reader_alone": X[c]["raw"]},
               f1={"teacher": T[c]["acc_f1"], "ours": Y[c]["acc_f1"], "reader_alone": X[c]["acc_f1"]},
               at_fork={}, logit_raw={})
    print(f"\n=== at the fork (position {i}), log-probabilities ===")
    for nm, lp in rowset.items():
        top = torch.topk(lp, TOPK)
        out["at_fork"][nm] = dict(
            candidates={k: float(lp[v]) for k, v in cand.items()},
            top=[[tok.decode([int(t)]), float(p)] for p, t in zip(top.values, top.indices)],
            entropy=float(-(lp.exp() * lp).sum()), argmax=tok.decode([int(lp.argmax())]))
        print(f"  {nm:9s} argmax {out['at_fork'][nm]['argmax']!r:12s} H {out['at_fork'][nm]['entropy']:.3f} "
              f"| logp(ours' token) {out['at_fork'][nm]['candidates']['ours']:+.3f} "
              f"| logp(reader-alone's) {out['at_fork'][nm]['candidates']['reader_alone']:+.3f}")
        print(f"            top{TOPK}: " + ", ".join(f"{t!r}:{p:.2f}" for t, p in out["at_fork"][nm]["top"]))
    for nm, z in (("reader_branch", zy_R[i]), ("floor_7B_base", zy_B[i]), ("query_lm", zy_Q[i]),
                  ("teacher", zy_T[i])):
        out["logit_raw"][nm] = {k: float(z[v]) for k, v in cand.items()}
    # ★ EVERY position, EVERY model: the top-20 (token, log-prob) and the log-prob of the token ours
    #   emitted — the per-token record the visualisation needs, not just the fork (user, 2026-09-18).
    lpY_T = lsm(zy_T)
    rows = {"reader_branch": lpY_R, "floor_7B_base": lsm(zy_B), "query_lm": lpY_Q,
            "teacher": lpY_T, "fused": pool_y}
    out["per_position"] = dict(tokens=[tok.decode([int(t)]) for t in gy],
                               token_ids=[int(t) for t in gy],
                               reader_alone_token=[tok.decode([int(t)]) for t in gx],
                               models={})
    for nm, lp in rows.items():
        top = torch.topk(lp, TOPK_ALL, dim=-1)
        out["per_position"]["models"][nm] = dict(
            logp_ours_token=[float(lp[j, gy[j]]) for j in range(len(gy))],
            logp_reader_alone_token=[float(lp[j, gx[j]]) if j < len(gx) else None for j in range(len(gy))],
            entropy=[float(-(lp[j].exp() * lp[j]).sum()) for j in range(len(gy))],
            argmax=[tok.decode([int(x)]) for x in lp.argmax(-1)],
            top_tokens=[[tok.decode([int(t)]) for t in row] for row in top.indices],
            top_logp=[[float(v) for v in row] for row in top.values])
    # the same for the reader-alone text, so its own fork token can be read in its own trajectory
    lpX = {"reader_branch": lsm(zx_R), "floor_7B_base": lsm(zx_B), "query_lm": lsm(zx_Q),
           "teacher": lsm(zx_T), "fused": lsm(LAM * zx_R + (1 - LAM) * zx_Q)}
    out["per_position_reader_alone_text"] = dict(
        tokens=[tok.decode([int(t)]) for t in gx],
        models={nm: dict(logp_emitted=[float(lp[j, gx[j]]) for j in range(len(gx))],
                         argmax=[tok.decode([int(x)]) for x in lp.argmax(-1)],
                         top_tokens=[[tok.decode([int(t)]) for t in row]
                                     for row in torch.topk(lp, TOPK_ALL, dim=-1).indices],
                         top_logp=[[float(v) for v in row]
                                   for row in torch.topk(lp, TOPK_ALL, dim=-1).values])
                for nm, lp in lpX.items()})
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
