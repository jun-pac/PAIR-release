#!/usr/bin/env python
"""D_fused on the PAPER's basis: the 240-example branch-KL cell, with the pool added (user, 2026-09-18:
"240개 예시에 forced pass를 다시 돌려야 합니다 이건 바로 돌리자").

WHY THIS EXISTS. results/timing/branch_kl_grid_n240/lm32B.json carries the sentence's three numbers for
the 7B reader and the 32B query LM — D_R = 0.337, D_Q = 0.689, D_oracle = 0.193 over 17,216 TASK
positions — but NOT the pool: scripts/branch_kl_grid.py computes the KLs on the fly and stores only the
per-position D_S and D_L scalars, never the distributions, so D_fused cannot be recovered from the store.
This adds it on exactly that basis.

WHAT IS AND IS NOT RE-RUN. The teacher's trajectory and its full logit rows are READ FROM THE CACHE
(/work/hdd/myproject/anon/analysis/branch_kl_grid_n240/lm32B_Tpass.npz, job 3107257's pass), so nothing is
regenerated and the floating-point question does not arise: every branch is forced along the SAME stored
token sequence the published numbers used. Re-run: the 32B no-context forward (L) and the 7B
full-context forward (S), because their distributions were never stored.

  D_S = KL(p_T‖p_S)   D_L = KL(p_T‖p_L)   D_F = KL(p_T‖softmax(λ z_S + (1−λ) z_L))   oracle = min(D_S, D_L)

λ = 0.7, hotpot's deployed weight.

WHICH ARM (corrected 2026-09-18, user: "애초에 ours자체가 FT를 포함하는 방법인데"). --ft runs the DEPLOYED
pair — reader 7B + reader_binding_v5distill, query LM 32B + stage2_on_v5reader_lam07, the adapters and the λ
of the published hotpot accuracy arm — and that is what "ours" means. Without --ft the branches carry no
adapter, which is the arm the published D_R / D_Q / D_oracle sentence quotes. Teacher forcing makes no
distinction between them: the basis is the teacher's stored trajectory, the prompts, the span and the 240
examples, all identical; the adapters are the arm, not the basis. Both rows therefore compare directly.

GATES, before any number is read:
  (1) the prompt rebuilt here must be byte-identical to the cached prompt_ctx of every example;
  (2) the recomputed D_L must reproduce the cached D_L (mean |Δ| < 1e-3) — the same forced inputs on the
      same card, which is also the check that the trajectory and the prompt are the published ones;
  (3) the TASK-span means of D_S and D_L must reproduce 0.337 and 0.689 to three decimals.

Output: results/timing/branch_kl_grid_n240/lm32B_s7B_fused.json and .../lm32B_s7B_fused.npz
(per-position D_S, D_L, D_F, seg, ex, t). Nothing published is overwritten.
ANALYSIS PROBE: B=1 forced forwards only. No timing, no benchmark score.
"""
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("ANSWER_PROMPT_VARIANT", "reason_v3")

import numpy as np  # noqa: E402

from scripts.branch_kl_grid import (NAME, forced_logits, kl_rows, stats,  # noqa: E402
                                    _load_examples, _build_teacher_prompt_with_audit)

DUMP = "/work/hdd/myproject/anon/analysis/branch_kl_grid_n240"
PUBLISHED = {"reader": 0.337, "query_lm": 0.689, "oracle": 0.193, "n_task": 17216}
LAM = 0.7


def logsoftmax(x):
    x = x.astype(np.float32)
    x -= x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm", default="32B")
    ap.add_argument("--reader", default="7B")
    ap.add_argument("--sample", type=int, default=240)
    ap.add_argument("--lam", type=float, default=LAM)
    ap.add_argument("--ft", action="store_true",
                    help="the DEPLOYED pair: both branches with their hotpot adapters (this is `ours`)")
    ap.add_argument("--slm-lora", default="reader_binding_v5distill_r16_s900")
    ap.add_argument("--lm-lora", default="stage2_on_v5reader_lam07_r16_s600")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    t0 = time.perf_counter()
    cache = os.environ.get("HF_HOME", "/work/hdd/myproject/anon/hf")

    Z = np.load(f"{DUMP}/lm{a.lm}_Tpass.npz", allow_pickle=True)
    idx = [int(i) for i in Z["idx"]]
    print(f"[cache] {len(idx)} examples in lm{a.lm}_Tpass.npz")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(NAME[a.lm], cache_dir=cache)
    # the grid's own namespace, field for field (scripts/branch_kl_grid.py), so the sample is the same
    ns = argparse.Namespace(dataset="hotpotqa", doc_number=40, sample=a.sample, sample_seed=42,
                            retriever="BM25", split="validation", pass_number=0, cache_dir=cache,
                            max_length=30000, retrieval_split="train", max_corpus_examples=None,
                            babilong_split="qa2", babilong_length="64k")
    dataset_name, examples, _ = _load_examples(ns)
    print(f"[examples] loader returned {len(examples)} {dataset_name} examples")
    plan = []
    for i in idx:
        ex = examples[i]
        pc, _ = _build_teacher_prompt_with_audit(ex, dataset=dataset_name, tokenizer=tok, max_length=30000)
        assert pc == str(Z[f"pc_{i}"]), f"gate 1 failed: rebuilt prompt differs from the cache at ex{i}"
        exq = copy.copy(ex); exq.documents = []
        pn, _ = _build_teacher_prompt_with_audit(exq, dataset=dataset_name, tokenizer=tok, max_length=30000)
        plan.append(dict(i=i, gen=[int(t) for t in Z[f"gen_{i}"]], pc=pc, pn=pn,
                         seg=Z[f"seg_{i}"].astype(object), DL=Z[f"DL_{i}"]))
    npos = sum(len(p["gen"]) for p in plan)
    print(f"[gate 1] every rebuilt prompt matches the cache; {npos} positions, "
          f"median generation {int(np.median([len(p['gen']) for p in plan]))} tokens")
    if a.dry_run:
        print("[dry-run] OK — nothing loaded")
        return

    import torch
    from src.models import load_causal_lm
    from src.lora_load import load_lora_stack
    CKPT = "/work/hdd/myproject/anon/kvreuse_ckpts"
    dev = "cuda:0"
    tmp = f"{DUMP}/lm{a.lm}_Lpass_tmp{'_ft' if a.ft else ''}.npz"

    print(f"[{a.lm}] loading for the no-context pass", flush=True)
    lm, ltok = load_causal_lm(NAME[a.lm], device_map=dev, cache_dir=cache)
    if a.ft:
        lm = load_lora_stack(lm, f"{CKPT}/{a.lm_lora}", label="LM")
    lm.eval()
    Lrows, dl_gap = {}, []
    for k, p in enumerate(plan):
        L = forced_logits(lm, ltok, p["pn"], p["gen"], dev).numpy()
        Lrows[f"L_{p['i']}"] = L.astype(np.float16)
        dl_gap.append(np.abs(kl_rows(Z[f"T_{p['i']}"], L) - p["DL"]).mean())
        if k % 40 == 0:
            print(f"  [{a.lm}] ex{p['i']} ({k + 1}/{len(plan)}) mean|ΔD_L| {dl_gap[-1]:.2e}", flush=True)
    np.savez(tmp + ".part.npz", **Lrows); os.replace(tmp + ".part.npz", tmp)   # temp + replace
    gap = float(np.mean(dl_gap))
    if a.ft:
        # with the LM adapter on, D_L MUST differ from the cached no-adapter D_L — that difference is the
        # check that the adapter loaded and is being used (accident #8: verify the behaviour, not the flag)
        print(f"[gate 2, --ft] D_L against the cached NO-ADAPTER D_L: mean |Δ| = {gap:.3e} "
              f"(must be > 1e-3, i.e. the adapter changed the branch)")
        assert gap > 1e-3, "gate 2 failed — the LM adapter did not change the branch, so it is not loaded"
    else:
        print(f"[gate 2] recomputed D_L against the cached D_L: mean |Δ| = {gap:.3e} (needs < 1e-3)")
        assert gap < 1e-3, "gate 2 failed — the forced inputs are not the published ones"
    del lm
    import gc; gc.collect(); torch.cuda.empty_cache()

    print(f"[{a.reader}] loading for the full-context pass", flush=True)
    slm, stok = load_causal_lm(NAME[a.reader], device_map=dev, cache_dir=cache)
    if a.ft:
        slm = load_lora_stack(slm, f"{CKPT}/{a.slm_lora}", label="reader")
    slm.eval()
    cols = {k: [] for k in ("ex", "t", "seg", "D_S", "D_L", "D_F")}
    for k, p in enumerate(plan):
        T = Z[f"T_{p['i']}"]
        S = forced_logits(slm, stok, p["pc"], p["gen"], dev).numpy()
        L = Lrows[f"L_{p['i']}"]
        lpT = logsoftmax(T); pT = np.exp(lpT)
        F = logsoftmax(a.lam * S.astype(np.float32) + (1 - a.lam) * L.astype(np.float32))
        n = len(p["gen"])
        cols["ex"].append(np.full(n, p["i"])); cols["t"].append(np.arange(n)); cols["seg"].append(p["seg"])
        # ★ D_L IS RECOMPUTED FROM THE ROWS THIS RUN PRODUCED, never taken from the cache (2026-09-18):
        #   with --ft the L branch carries an adapter, so the cached no-adapter D_L belongs to a DIFFERENT
        #   arm. The first --ft run appended the cached value and reported query_lm 0.6886 — the base LM's
        #   number under the FT label — and its oracle mixed the two arms. D_F was unaffected (it is built
        #   from these same rows) and the store was repaired on CPU from the saved rows.
        cols["D_S"].append(kl_rows(T, S)); cols["D_L"].append(kl_rows(T, L))
        cols["D_F"].append((pT * (lpT - F)).sum(-1))
        if k % 40 == 0:
            print(f"  [{a.reader}] ex{p['i']} ({k + 1}/{len(plan)})", flush=True)
    del slm; gc.collect(); torch.cuda.empty_cache()

    D = {k: np.concatenate(v) for k, v in cols.items()}
    TASK = (D["seg"] == "reason") | (D["seg"] == "answer")
    orc = np.minimum(D["D_S"], D["D_L"])
    out = dict(lm=a.lm, reader_size=a.reader, lam=a.lam, n_examples=len(plan),
               arm=("deployed (both adapters): reader + " + a.slm_lora + ", query LM + " + a.lm_lora)
                   if a.ft else "no adapters (the arm the published D_R/D_Q/D_oracle sentence quotes)",
               n_task=int(TASK.sum()), n_all=int(len(orc)),
               reader=stats(D["D_S"][TASK]), query_lm=stats(D["D_L"][TASK]),
               fused=stats(D["D_F"][TASK]), oracle=stats(orc[TASK]),
               fused_below_both_frac=float((D["D_F"] < orc)[TASK].mean()),
               reader_preferred_frac=float((D["D_S"] <= D["D_L"])[TASK].mean()),
               by_segment={sg: {"reader": stats(D["D_S"][D["seg"] == sg]),
                                "query_lm": stats(D["D_L"][D["seg"] == sg]),
                                "fused": stats(D["D_F"][D["seg"] == sg]),
                                "oracle": stats(orc[D["seg"] == sg])} for sg in ("reason", "answer")},
               gate_DL_mean_abs_delta=gap, published_cell=PUBLISHED,
               _source=("scripts/branch_kl_fused.py — the teacher trajectory and its logits read from "
                        f"lm{a.lm}_Tpass.npz (not regenerated); the 32B no-context and {a.reader} "
                        "full-context forwards re-run; no adapters; TASK span = reason+answer"))
    SUF = "_ft" if a.ft else ""
    for k in ("reader", "query_lm", "oracle"):
        d = abs(out[k]["mean"] - PUBLISHED[k])
        if a.ft:
            print(f"[--ft] {k}: {out[k]['mean']:.4f} (the no-adapter arm reads {PUBLISHED[k]}; "
                  f"a different arm on the same basis, not a reproduction)")
        else:
            print(f"[gate 3] {k}: {out[k]['mean']:.4f} against the published {PUBLISHED[k]} (|Δ| {d:.4f})")
    print(f"[result] TASK n={out['n_task']} (published {PUBLISHED['n_task']})  "
          f"reader {out['reader']['mean']:.4f}  query_lm {out['query_lm']['mean']:.4f}  "
          f"FUSED {out['fused']['mean']:.4f}  oracle {out['oracle']['mean']:.4f}  "
          f"| fused below both at {out['fused_below_both_frac']:.1%} of positions")
    os.makedirs("results/timing/branch_kl_grid_n240", exist_ok=True)
    json.dump(out, open(f"results/timing/branch_kl_grid_n240/lm{a.lm}_s{a.reader}_fused{SUF}.json", "w"), indent=1)
    np.savez_compressed(f"{DUMP}/lm{a.lm}_s{a.reader}_fused{SUF}.npz", ex=D["ex"], t=D["t"],
                        seg=D["seg"].astype("U8"), D_S=D["D_S"].astype(np.float32),
                        D_L=D["D_L"].astype(np.float32), D_F=D["D_F"].astype(np.float32))
    print(f"[done] {time.perf_counter() - t0:.0f}s -> results/timing/branch_kl_grid_n240/lm{a.lm}_s{a.reader}_fused{SUF}.json")


if __name__ == "__main__":
    main()
