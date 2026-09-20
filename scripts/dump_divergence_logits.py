#!/usr/bin/env python
"""Branch logits along FREE-RUNNING generations, at the tokens where two arms part ways (2026-09-05).

WHY (user, 2026-09-05: "학습전에는 이 토큰때 이런식으로 답변했었다면 이후에는 어떻게 됐는지").
The existing branch dump (scripts/dump_branch_logits.py) is teacher-forced along the TEACHER's text, on
the sketch harness's prompt and the lam085 adapter. Along that text the LM adapter moves per-token
accuracy by 0.2 points (4,362 clean positions), yet in free generation on the deployed prompt it is
worth 5.5 F1 points on hotpot (abl_ho_ssolo 0.6633 -> hofa_ours 0.7178). The gain lives in the
trajectory, so it has to be read where the trajectories diverge, with the DEPLOYED adapters
(reader_binding_v5distill / stage2_on_v5reader_lam07, lam 0.7) and the DEPLOYED prompt
(mtrag_accum --bench hotpotqa_st40_full: QA_REASON_V3 + reason_fix question turn).

WHAT. For each selected conversation, the two arms' raw generations X (reader adapter only, pool
Sft+L) and Y (both adapters, pool Sft+Lft) are re-encoded and forced through FOUR branches — the 7B
reader with its adapter off (S) and on (Sft), the 32B query-only LM with its adapter off (L) and on
(Lft) — using ONE copy of each model (PEFT disable_adapter()). The prompt is rebuilt from the same
pieces the batched decoder forwards, block by block: INSTRUCTION()+"\n\n" | fmt(passages) | q_turn(q)
| generated ids, each tokenised separately with add_special_tokens=False (src/batched_stateful.py
prefill/turn); the LM branch gets instruction | q_turn | generated (lm_no_accum, no context).

Per conversation the .npz holds, for both texts: the generated ids, each branch's top-K ids and
log-probs at every position, each branch's log-prob of the actual next token, the four pools'
(S+L, Sft+L, S+Lft, Sft+Lft) argmax at every position at the deployed lambda, and at the first
divergence position i_div (identical prefix in X and Y) the FULL log-prob rows of the four branches.

SELF-CONSISTENCY GATE (stated before the run): the pool that generated each text must reproduce it.
argmax(lam*z_Sft + (1-lam)*z_L) == next token of X, and argmax(lam*z_Sft + (1-lam)*z_Lft) == next
token of Y, at >= 97% of positions (re-encoding and trim_leak can perturb a few boundary tokens).
Below that the prompt reconstruction is wrong and NOTHING from this dump may be read.

ANALYSIS PROBE. Forced forwards only (no decode loop); B=2 right-padded sequences per forward (the
two texts of one conversation share the prompt). No timing, no score is produced here.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("REASON_FIX", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402

CKPT = "/work/hdd/myproject/anon/kvreuse_ckpts"
TOPK = 20


def load_log(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return {r["conv"]: r for r in rows if "acc_f1" in r}


def select_convs(X, Y, n_ctrl, seed=0):
    """all conversations whose answer or score differs between the arms, plus n_ctrl same-answer controls"""
    both = [c for c in X if c in Y]
    diff = [c for c in both if X[c]["acc_pred"] != Y[c]["acc_pred"] or X[c]["acc_f1"] != Y[c]["acc_f1"]]
    same = [c for c in both if c not in set(diff) and X[c]["raw"] != Y[c]["raw"]]
    rng = np.random.default_rng(seed)
    ctrl = list(rng.choice(same, min(n_ctrl, len(same)), replace=False)) if n_ctrl else []
    return diff, ctrl


def build_blocks(MA, tasks):
    turns = MA.prep(tasks)
    first = next((t for t in turns if t["newk"]), None)
    ctx = MA.fmt(first["newk"]) if first else ""
    instr = MA.INSTRUCTION() + "\n\n"
    q = MA.q_turn(turns[0]["q"])
    return instr, ctx, q


def enc(tok, s):
    return tok(s, add_special_tokens=False).input_ids


def first_div(a, b):
    n = min(len(a), len(b))
    return next((j for j in range(n) if a[j] != b[j]), n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="hotpotqa_st40_full")
    ap.add_argument("--x", default="results/fusionft/abl_ho_ssolo.jsonl", help="arm X log (reader adapter only)")
    ap.add_argument("--y", default="results/fusionft/hofa_ours.jsonl", help="arm Y log (both adapters)")
    ap.add_argument("--n-ctrl", type=int, default=40)
    ap.add_argument("--lam", type=float, default=0.7)
    ap.add_argument("--slm-lora", default=f"{CKPT}/reader_binding_v5distill_r16_s900")
    ap.add_argument("--lm-lora", default=f"{CKPT}/stage2_on_v5reader_lam07_r16_s600")
    ap.add_argument("--out", default="/work/hdd/myproject/anon/analysis/divergence_logits")
    ap.add_argument("--dry-run", action="store_true", help="rebuild prompts + round-trip the texts, no models")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    import scripts.mtrag_accum as MA
    from scripts import bench_config as BC
    from src import qa_prompts
    cfg = BC.get(a.bench)
    MA.INSTR_OVERRIDE = getattr(qa_prompts, cfg["instruction"])
    MA.FULLANS = False
    rows = [json.loads(l) for l in open(cfg["ref"]) if l.strip()]
    by = {}
    for r in rows:
        by.setdefault(r["conversation_id"], []).append(r)
    by = dict(list(by.items())[: cfg["max_conv"]])

    X, Y = load_log(a.x), load_log(a.y)
    diff, ctrl = select_convs(X, Y, a.n_ctrl)
    convs = diff + ctrl
    if a.limit:
        convs = convs[: a.limit]
    print(f"[select] {len(diff)} conversations differ in answer/score, {len(ctrl)} same-answer controls -> {len(convs)}")
    for c in convs:
        assert c in by, f"conv {c} not in ref {cfg['ref']}"

    from transformers import AutoTokenizer
    cache = os.environ.get("HF_HOME", "/work/hdd/myproject/anon/hf")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", cache_dir=cache)
    ltok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct", cache_dir=cache)
    assert tok.get_vocab() == ltok.get_vocab(), "reader and LM tokenizers differ"

    plan = []
    rt_bad = 0
    for c in convs:
        instr, ctx, q = build_blocks(MA, by[c])
        gx, gy = enc(tok, X[c]["raw"]), enc(tok, Y[c]["raw"])
        rt_bad += int(tok.decode(gx) != X[c]["raw"]) + int(tok.decode(gy) != Y[c]["raw"])
        plan.append(dict(conv=c, instr=enc(tok, instr), ctx=enc(tok, ctx), q=enc(tok, q), gx=gx, gy=gy,
                         i_div=first_div(gx, gy), fx=X[c]["acc_f1"], fy=Y[c]["acc_f1"],
                         px=X[c]["acc_pred"], py=Y[c]["acc_pred"], gold=X[c]["gold"], question=X[c]["q"]))
    L = [len(p["instr"]) + len(p["ctx"]) + len(p["q"]) + max(len(p["gx"]), len(p["gy"])) for p in plan]
    print(f"[plan] reader sequence length: median {int(np.median(L))}, max {max(L)}; instruction {len(plan[0]['instr'])} tokens;"
          f" generated: median {int(np.median([len(p['gx']) for p in plan]))} tokens; i_div median "
          f"{int(np.median([p['i_div'] for p in plan]))}; text round-trip failures {rt_bad}/{2 * len(plan)}")
    print(f"[plan] first instruction tokens: {tok.decode(plan[0]['instr'][:12])!r} ... q block: {tok.decode(plan[0]['q'])[:90]!r}")
    if a.dry_run:
        print("[dry-run] OK — no model loaded")
        return

    import torch
    from src.models import load_causal_lm
    from src.lora_load import load_lora_stack
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda:0"

    @torch.no_grad()
    def forced(model, prompt_ids, gens):
        """B=2 right-padded forced forward -> list of [G_i, V] log-prob rows (float32, on CPU)"""
        seqs = [prompt_ids + g for g in gens]
        T = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), T), tok.pad_token_id, dtype=torch.long)
        am = torch.zeros((len(seqs), T), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s)
            am[i, : len(s)] = 1
        P = len(prompt_ids)
        K = T - (P - 1)                       # keep only the tail: from the last prompt token onwards
        out = model(ids.to(dev), attention_mask=am.to(dev), use_cache=False, logits_to_keep=K)
        res = []
        for i, g in enumerate(gens):
            lg = out.logits[i, 0: len(g)].float()     # row 0 of the kept tail == position P-1
            res.append(torch.log_softmax(lg, -1).cpu())
        return res

    def pass_branch(model, prompt_key, name, store):
        t0 = time.time()
        for k, p in enumerate(plan):
            pr = p["instr"] + (p["ctx"] if prompt_key == "ctx" else []) + p["q"]
            lx, ly = forced(model, pr, [p["gx"], p["gy"]])
            d = store.setdefault(p["conv"], {})
            for tag, lp, g in (("x", lx, p["gx"]), ("y", ly, p["gy"])):
                top = torch.topk(lp, TOPK, dim=-1)
                d[f"{name}_top_ids_{tag}"] = top.indices.numpy().astype(np.int32)
                d[f"{name}_top_lp_{tag}"] = top.values.numpy().astype(np.float16)
                d[f"{name}_lp_next_{tag}"] = lp[torch.arange(len(g)), torch.tensor(g)].numpy().astype(np.float32)
                d[f"{name}_argmax_{tag}"] = lp.argmax(-1).numpy().astype(np.int32)
                # the pool argmax needs both branches at once; keep the full rows of this pass on disk
                # only at i_div, and keep a compact fp16 copy of the whole row set for the pool in RAM
                d[f"_{name}_rows_{tag}"] = lp.to(torch.float16)
            d[f"{name}_row_div"] = lx[p["i_div"]].numpy().astype(np.float16) if p["i_div"] < len(p["gx"]) else np.zeros(0, np.float16)
            if k % 20 == 0:
                print(f"[{name}] {k + 1}/{len(plan)} {time.time() - t0:.0f}s", flush=True)

    store = {}
    slm, _ = load_causal_lm("Qwen/Qwen2.5-7B-Instruct", device_map=dev, cache_dir=cache)
    slm.eval()
    slm = load_lora_stack(slm, a.slm_lora, label="reader")
    slm.eval()
    with slm.disable_adapter():
        pass_branch(slm, "ctx", "S", store)
    pass_branch(slm, "ctx", "Sft", store)
    del slm
    torch.cuda.empty_cache()
    lm, _ = load_causal_lm("Qwen/Qwen2.5-32B-Instruct", device_map=dev, cache_dir=cache)
    lm.eval()
    lm = load_lora_stack(lm, a.lm_lora, label="LM")
    lm.eval()
    with lm.disable_adapter():
        pass_branch(lm, "noctx", "L", store)
    pass_branch(lm, "noctx", "Lft", store)
    del lm
    torch.cuda.empty_cache()

    # pools at the deployed lambda, per text; self-consistency of the generating pool
    lam = a.lam
    ok = {"x": [0, 0], "y": [0, 0]}
    for p in plan:
        d = store[p["conv"]]
        for tag, g in (("x", p["gx"]), ("y", p["gy"])):
            S, Sft, L, Lft = (d.pop(f"_{n}_rows_{tag}").float() for n in ("S", "Sft", "L", "Lft"))
            pools = {"SL": lam * S + (1 - lam) * L, "SftL": lam * Sft + (1 - lam) * L,
                     "SLft": lam * S + (1 - lam) * Lft, "SftLft": lam * Sft + (1 - lam) * Lft}
            for k, P in pools.items():
                d[f"pool_{k}_argmax_{tag}"] = P.argmax(-1).numpy().astype(np.int32)
                d[f"pool_{k}_lp_next_{tag}"] = torch.log_softmax(P, -1)[torch.arange(len(g)), torch.tensor(g)].numpy().astype(np.float32)
            gen_pool = "SftL" if tag == "x" else "SftLft"
            hit = int((d[f"pool_{gen_pool}_argmax_{tag}"] == np.array(g)).sum())
            ok[tag][0] += hit
            ok[tag][1] += len(g)
        d.update(gx=np.array(p["gx"], np.int32), gy=np.array(p["gy"], np.int32), i_div=p["i_div"],
                 fx=p["fx"], fy=p["fy"], px=p["px"], py=p["py"], gold=p["gold"], question=p["question"],
                 lam=lam, in_ctrl=p["conv"] in set(ctrl))
        np.savez_compressed(f"{a.out}/{p['conv']}.npz", **d)
    for tag, (h, n) in ok.items():
        print(f"[self-consistency] text {tag}: generating pool reproduces the next token at {h}/{n} = {h / n:.4f}"
              f" ({'PASS' if h / n >= 0.97 else 'FAIL — prompt reconstruction wrong, do not read this dump'})")
    json.dump({"convs": [p["conv"] for p in plan], "ctrl": list(ctrl), "lam": lam, "x": a.x, "y": a.y,
               "self_consistency": {t: {"hit": h, "n": n} for t, (h, n) in ok.items()},
               "slm_lora": a.slm_lora, "lm_lora": a.lm_lora, "bench": a.bench},
              open(f"{a.out}/manifest.json", "w"), indent=1)
    print(f"[Done] {len(plan)} conversations -> {a.out}")


if __name__ == "__main__":
    main()
