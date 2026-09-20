#!/usr/bin/env python
"""Branch-KL grid over model sizes (user, 2026-09-07: "1.5B, 3B, 7B, 14B, 32B, 72B … 모든 combination을
잡아서 이걸 분석해서 heatmap" — "어 이건 다 해도 되겠다").

WHAT IT MEASURES, exactly as scripts/dump_branch_logits.py + scripts/fusion_kl_analysis.py did for the
published 32B+7B numbers (reader 0.3826 / query-LM 0.7477 / oracle 0.2208 nats, TASK span, 4,362
positions), but for one LM size against every smaller reader size, with no adapters and no full-vocab
dump: the KLs are computed on the fly and only per-position scalars are stored.

  For the LM  l  (Qwen2.5-<l>-Instruct):
    T = l with the full context, greedy, its generation-time logits (the same forced logits);
    L = l with NO context (documents=[]), forced along T's tokens;
  for each reader  s < l :
    S = s with the full context, forced along T's tokens;
  D_S = KL(p_T || p_S), D_L = KL(p_T || p_L), oracle = min(D_S, D_L), per position, log-softmax in fp32.

TRAJECTORY. Greedy from the LM's context prompt, stopped at the tokenizer eos (<|im_end|>, not kept) or
at <|endoftext|> id 151643 (KEPT as the last position, as clean_seg keeps it). That is the prefix the
published analysis scored after clean_seg, so for l=32B, s=7B the numbers must reproduce (the canary).

SCOPE. TASK = reason [0, 'Final Answer:') + answer ['Final Answer:'+len, next newline), the segment
rule of the dump, on the clipped trajectory. Same 60 hotpotQA d40 seed-42 examples, same prompt
builders, ANSWER_PROMPT_VARIANT=reason_v3, B=1 forced forwards. ANALYSIS PROBE: no timing, no score.

Output: results/timing/branch_kl_grid/lm<l>.json (summary per reader) and
/work/hdd/myproject/anon/analysis/branch_kl_grid/lm<l>_s<s>.npz (per-position D_S, D_L, seg, ex, t).
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

import numpy as np
import torch

from scripts.run_h2o_teacher_experiments import _load_examples
from scripts.run_evidence_sketch_experiments import _build_teacher_prompt_with_audit
from scripts.branch_dump_common import EOT
from src.models import load_causal_lm

MAX_NEW = 200
OUT_NPZ = "/work/hdd/myproject/anon/analysis/branch_kl_grid"
OUT_JSON = "results/timing/branch_kl_grid"
NAME = {"0.5B": "Qwen/Qwen2.5-0.5B-Instruct",   # added 2026-09-11 (user: "0.5B도 한번 돌려서 heatmap에 추가")
        "1.5B": "Qwen/Qwen2.5-1.5B-Instruct", "3B": "Qwen/Qwen2.5-3B-Instruct", "7B": "Qwen/Qwen2.5-7B-Instruct",
        "14B": "Qwen/Qwen2.5-14B-Instruct", "32B": "Qwen/Qwen2.5-32B-Instruct", "72B": "Qwen/Qwen2.5-72B-Instruct"}
PUBLISHED_32B_7B = {"reader": 0.3826, "query_lm": 0.7477, "oracle": 0.2208, "n_task": 4362}


@torch.no_grad()
def greedy_traj(model, tok, prompt, dev):
    ids = tok(prompt, return_tensors="pt", truncation=False).input_ids.to(dev)
    out = model(ids, use_cache=True)
    past = out.past_key_values
    logits = [out.logits[0, -1].to(torch.float16).cpu()]
    gen = []
    eos = tok.eos_token_id
    for _ in range(MAX_NEW):
        nxt = int(logits[-1].float().argmax())
        if nxt == eos:                      # <|im_end|>: the dump stopped here without keeping it
            break
        gen.append(nxt)
        if nxt == EOT:                      # <|endoftext|>: the teacher's real end; kept, then stop
            break
        out = model(torch.tensor([[nxt]], device=dev), past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits.append(out.logits[0, -1].to(torch.float16).cpu())
    if not gen:
        return [], None
    return gen, torch.stack(logits[: len(gen)])


@torch.no_grad()
def forced_logits(model, tok, prompt, gen_ids, dev):
    p = tok(prompt, return_tensors="pt", truncation=False).input_ids.to(dev)
    g = torch.tensor([gen_ids], device=dev)
    ids = torch.cat([p, g], dim=1)
    out = model(ids, use_cache=False)
    lp = p.shape[1]
    return out.logits[0, lp - 1: lp - 1 + len(gen_ids)].to(torch.float16).cpu()


def seg_of(tok, gen):
    """(reason_end, answer_start, answer_end) by the dump's rule on the (already clipped) trajectory."""
    text = tok.decode(gen, skip_special_tokens=False)
    fa = text.find("Final Answer:")
    offs, acc = [], ""
    for t_id in gen:
        acc += tok.decode([t_id], skip_special_tokens=False)
        offs.append(len(acc))
    def tok_at(char):
        for j, o in enumerate(offs):
            if o > char:
                return j
        return len(gen)
    r_end = tok_at(fa) if fa >= 0 else len(gen)
    a0 = tok_at(fa + len("Final Answer:")) if fa >= 0 else len(gen)
    nl = text.find("\n", fa) if fa >= 0 else -1
    a1 = tok_at(nl) if nl >= 0 else len(gen)
    lab = np.array(["other"] * len(gen), dtype=object)
    lab[:r_end] = "reason"
    lab[a0:a1] = "answer"
    return lab


def logsoftmax(z):
    z = z.astype(np.float32)
    m = z.max(-1, keepdims=True)
    e = np.exp(z - m)
    return z - m - np.log(e.sum(-1, keepdims=True))


def kl_rows(T, Q):
    """KL(p_T || p_Q) per row. Qwen2.5 sizes pad the SAME tokenizer to different logit widths
    (0.5B/1.5B/3B: 151936; 7B and up: 152064; real vocab 151,665) — the extra columns are never-used
    padding ids, so both are cut to the common width before the softmax (the published 32B+7B pair
    had equal widths and is unaffected)."""
    V = min(T.shape[-1], Q.shape[-1])
    lT, lQ = logsoftmax(T[..., :V]), logsoftmax(Q[..., :V])
    return (np.exp(lT) * (lT - lQ)).sum(-1)


def stats(a):
    return dict(mean=float(a.mean()), median=float(np.median(a)), q25=float(np.quantile(a, .25)),
                q75=float(np.quantile(a, .75)), frac_under_01=float((a < 0.1).mean()), n=int(len(a)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm", required=True, choices=list(NAME))
    ap.add_argument("--readers", required=True, help="comma list, e.g. 1.5B,3B,7B,14B")
    ap.add_argument("--sample", type=int, default=60)
    ap.add_argument("--dataset", default="hotpotqa")
    ap.add_argument("--lm-device-map", default="cuda:0", help="'auto' for the 72B on two cards")
    ap.add_argument("--dry-run", action="store_true", help="examples + prompts only, no model")
    a = ap.parse_args()
    # ':' as well as ',' — sbatch --export splits its own list on commas, so a comma list passed through
    # --export=ALL,READERS=7B,14B arrives as READERS=7B (the 2026-09-07 first submission ran one reader per block)
    readers = [r.strip() for r in a.readers.replace(":", ",").split(",") if r.strip()]
    order = list(NAME)
    for r in readers:
        assert order.index(r) < order.index(a.lm), f"reader {r} must be smaller than the LM {a.lm}"
    cache = os.environ.get("HF_HOME", "/work/hdd/myproject/anon/hf")
    ns = argparse.Namespace(dataset=a.dataset, doc_number=40, sample=a.sample, sample_seed=42,
                            retriever="BM25", split="validation", pass_number=0, cache_dir=cache,
                            max_length=30000, retrieval_split="train", max_corpus_examples=None,
                            babilong_split="qa2", babilong_length="64k")
    dataset_name, examples, _ = _load_examples(ns)
    print(f"[Info] {len(examples)} {dataset_name} examples; LM {a.lm}; readers {readers}", flush=True)
    if a.dry_run:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(NAME[a.lm], cache_dir=cache)
        ex = examples[0]
        pc, _ = _build_teacher_prompt_with_audit(ex, dataset=dataset_name, tokenizer=tok, max_length=30000)
        exq = copy.copy(ex); exq.documents = []
        pq, _ = _build_teacher_prompt_with_audit(exq, dataset=dataset_name, tokenizer=tok, max_length=30000)
        print(f"[dry-run] ex0 ctx prompt {len(tok(pc).input_ids)} tokens, no-ctx prompt {len(tok(pq).input_ids)} tokens; "
              f"eos={tok.eos_token_id} EOT={EOT}; ok")
        return
    global OUT_NPZ, OUT_JSON
    if a.sample != 60:                 # 2026-09-08: larger runs live beside the 60-example files, never over them
        OUT_NPZ = f"{OUT_NPZ}_n{a.sample}"; OUT_JSON = f"{OUT_JSON}_n{a.sample}"
    os.makedirs(OUT_NPZ, exist_ok=True); os.makedirs(OUT_JSON, exist_ok=True)
    dev = "cuda:0"
    t0 = time.perf_counter()
    lm_cache = f"{OUT_NPZ}/lm{a.lm}_Tpass.npz"      # the LM's own passes (T logits, gen, D_L, seg, prompts)
    store = {}
    if os.path.exists(lm_cache):
        Z = np.load(lm_cache, allow_pickle=True)
        for i in Z["idx"]:
            i = int(i)
            store[i] = dict(gen=[int(t) for t in Z[f"gen_{i}"]], T=Z[f"T_{i}"], D_L=Z[f"DL_{i}"],
                            seg=Z[f"seg_{i}"].astype(object), prompt_ctx=str(Z[f"pc_{i}"]), ended_eot=bool(Z[f"eot_{i}"]))
        print(f"[{a.lm}] LM passes loaded from {lm_cache} ({len(store)} examples)", flush=True)
        examples = []
    else:
        lm, tok = load_causal_lm(NAME[a.lm], device_map=a.lm_device_map, cache_dir=cache)
        lm.eval()
    for i, ex in enumerate(examples):
        prompt_ctx, _ = _build_teacher_prompt_with_audit(ex, dataset=dataset_name, tokenizer=tok, max_length=30000)
        exq = copy.copy(ex); exq.documents = []
        prompt_noctx, _ = _build_teacher_prompt_with_audit(exq, dataset=dataset_name, tokenizer=tok, max_length=30000)
        gen, T = greedy_traj(lm, tok, prompt_ctx, dev)
        if not gen:
            print(f"[skip] ex{i} empty gen", flush=True); continue
        L = forced_logits(lm, tok, prompt_noctx, gen, dev)
        store[i] = dict(gen=gen, T=T.numpy(), D_L=kl_rows(T.numpy(), L.numpy()), seg=seg_of(tok, gen),
                        prompt_ctx=prompt_ctx, ended_eot=(gen[-1] == EOT))
        print(f"[{a.lm}] ex{i} gen={len(gen)} eot={int(gen[-1] == EOT)} "
              f"task={int(((store[i]['seg'] == 'reason') | (store[i]['seg'] == 'answer')).sum())}", flush=True)
    if examples:
        arrs = {"idx": np.array(sorted(store))}
        for i, d in store.items():
            arrs.update({f"gen_{i}": np.array(d["gen"], dtype=np.int32), f"T_{i}": d["T"], f"DL_{i}": d["D_L"],
                         f"seg_{i}": d["seg"].astype("U8"), f"pc_{i}": np.array(d["prompt_ctx"]), f"eot_{i}": np.array(d["ended_eot"])})
        np.savez(lm_cache, **arrs)
        print(f"[{a.lm}] LM passes cached -> {lm_cache}", flush=True)
        del lm
    import gc; gc.collect(); torch.cuda.empty_cache()   # device_map=auto (72B) held 94.9 GiB after a bare del (job 3107257)
    for _d in range(torch.cuda.device_count()):
        with torch.cuda.device(_d):
            torch.cuda.empty_cache()
    print(f"[{a.lm}] LM freed: " + ", ".join(f"cuda:{_d} {torch.cuda.memory_allocated(_d) / 2**30:.1f} GiB" for _d in range(torch.cuda.device_count())), flush=True)
    print(f"[{a.lm}] LM passes done in {time.perf_counter() - t0:.0f}s; "
          f"{sum(len(d['gen']) for d in store.values())} positions, "
          f"{sum(int(d['ended_eot']) for d in store.values())}/{len(store)} ended at <|endoftext|>", flush=True)
    out_json = f"{OUT_JSON}/lm{a.lm}.json"
    prior = json.load(open(out_json))["readers"] if os.path.exists(out_json) else {}   # merge: blocks may add readers
    summary = {"lm": a.lm, "dataset": dataset_name, "n_examples": len(store), "readers": dict(prior),
               "_source": "scripts/branch_kl_grid.py — teacher-forced KL(p_T||p_S), KL(p_T||p_L), min; TASK span "
                          "(reason+answer, clipped at the teacher's <|endoftext|>); same 60 hotpotQA d40 seed-42 "
                          "examples and prompt builders as dump_branch_logits.py / fusion_kl_analysis.py"}
    for s in readers:
        t1 = time.perf_counter()
        slm, stok = load_causal_lm(NAME[s], device_map=dev, cache_dir=cache)
        slm.eval()
        cols = {k: [] for k in ("ex", "t", "seg", "D_S", "D_L")}
        for i, d in store.items():
            S = forced_logits(slm, stok, d["prompt_ctx"], d["gen"], dev).numpy()
            n = len(d["gen"])
            cols["ex"].append(np.full(n, i)); cols["t"].append(np.arange(n)); cols["seg"].append(d["seg"])
            cols["D_S"].append(kl_rows(d["T"], S)); cols["D_L"].append(d["D_L"])
        del slm
        gc.collect(); torch.cuda.empty_cache()
        D = {k: np.concatenate(v) for k, v in cols.items()}
        TASK = (D["seg"] == "reason") | (D["seg"] == "answer")
        orc = np.minimum(D["D_S"], D["D_L"])
        blk = {"reader": stats(D["D_S"][TASK]), "query_lm": stats(D["D_L"][TASK]), "oracle": stats(orc[TASK]),
               "reader_preferred_frac": float((D["D_S"] <= D["D_L"])[TASK].mean()),
               "by_segment": {sg: {"reader": stats(D["D_S"][D["seg"] == sg]), "query_lm": stats(D["D_L"][D["seg"] == sg]),
                                   "oracle": stats(orc[D["seg"] == sg])} for sg in ("reason", "answer")},
               "all_positions": {"reader": stats(D["D_S"]), "query_lm": stats(D["D_L"]), "oracle": stats(orc)},
               "n_task": int(TASK.sum()), "n_all": int(len(orc)), "seconds": round(time.perf_counter() - t1, 1)}
        summary["readers"][s] = blk
        np.savez_compressed(f"{OUT_NPZ}/lm{a.lm}_s{s}.npz", ex=D["ex"], t=D["t"], seg=D["seg"].astype("U8"),
                            D_S=D["D_S"].astype(np.float32), D_L=D["D_L"].astype(np.float32))
        print(f"[{a.lm}+{s}] TASK n={blk['n_task']}  reader {blk['reader']['mean']:.4f}  "
              f"query_lm {blk['query_lm']['mean']:.4f}  oracle {blk['oracle']['mean']:.4f}  "
              f"(oracle vs better branch x{min(blk['reader']['mean'], blk['query_lm']['mean']) / blk['oracle']['mean']:.2f})",
              flush=True)
        if a.lm == "32B" and s == "7B":
            P = PUBLISHED_32B_7B
            first = TASK & (D["ex"] < 60)             # the published basis is the first 60 examples of the seeded order
            c_blk = {"reader": float(D["D_S"][first].mean()), "query_lm": float(D["D_L"][first].mean()),
                     "oracle": float(orc[first].mean()), "n_task": int(first.sum())}
            dev_ = {k: abs(c_blk[k] - P[k]) for k in ("reader", "query_lm", "oracle")}
            same_n = c_blk["n_task"] == P["n_task"]
            print(f"[CANARY 32B+7B, first 60 examples] published reader {P['reader']} / query_lm {P['query_lm']} / oracle {P['oracle']} "
                  f"on n_task {P['n_task']}; here n_task {c_blk['n_task']}; |diff| {dev_}", flush=True)
            ok = same_n and all(v < 0.01 for v in dev_.values())
            summary["canary_32B_7B"] = {"published": P, "ok": bool(ok), "abs_diff": dev_}
            print("[CANARY] " + ("PASS — the published basis reproduces" if ok else
                                  "FAIL — the trajectory or the scope differs from the published dump; STOP and read"),
                  flush=True)
        json.dump(summary, open(out_json, "w"), indent=1)
    print(f"[Done] LM {a.lm} x readers {readers} -> {OUT_JSON}/lm{a.lm}.json  ({time.perf_counter() - t0:.0f}s)")
    if a.lm == "32B" and summary.get("canary_32B_7B", {}).get("ok") is False:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
