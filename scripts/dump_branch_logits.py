#!/usr/bin/env python
"""Teacher-forced branch-logit dump for the alignment analysis (2026-08-23, user-directed).

Records, along the TEACHER'S OWN greedy trajectory (teacher-forcing kills drift), full-vocab raw
logits of five configurations at every generated position:
  T    teacher      = 32B, context prompt (its generation-time logits ARE the forced logits)
  L    LM-base      = 32B, NO-context prompt (documents=[] copy -> instruction+question), forced
  Lft  LM lam085    = 32B + stage2_on_v5reader_lam085 adapter, NO-context, forced
  S    SLM-base     = 7B, context prompt, forced
  Sft  S-solo       = 7B + reader_binding_v5distill adapter, context prompt, forced
Fused distributions (plain=λS+(1-λ)L etc., raw-logit mixture exactly as src fusion does) are
DERIVED OFFLINE from these tensors — nothing else needs the GPU again.

Prompts come from the SAME builders the harnesses use (`_build_teacher_prompt_with_audit`,
ANSWER_PROMPT_VARIANT=reason_v3), examples from the same loader (hotpotqa d40 seed42, first-N of
the scored ordering). Per example also stored: gen token ids, reasoning/answer segmentation
(char offsets of 'Final Answer:' mapped to token positions), and the context-copy vocab id set.

Output: /work/hdd/myproject/anon/analysis/branch_logits/hotpot_ex{idx:03d}.npz  (~350MB each, N=60
~21GB). ANALYSIS PROBE: B=1 forwards, no timing, no benchmark score is produced here.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("ANSWER_PROMPT_VARIANT", "reason_v3")

import copy
import numpy as np
import torch

from scripts.run_h2o_teacher_experiments import _load_examples
from scripts.run_evidence_sketch_experiments import _build_teacher_prompt_with_audit
from src.models import load_causal_lm
from src.lora_load import load_lora_stack

OUT = "/work/hdd/myproject/anon/analysis/branch_logits"
CKPT = "/work/hdd/myproject/anon/kvreuse_ckpts"
MAX_NEW = 200


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
        if nxt == eos:
            break
        gen.append(nxt)
        out = model(torch.tensor([[nxt]], device=dev), past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits.append(out.logits[0, -1].to(torch.float16).cpu())
    # logits[i] is the distribution that PRODUCED gen[i]; drop the trailing one past the last token
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
    # logits at positions lp-1 .. lp+T-2 predict gen tokens 0..T-1
    return out.logits[0, lp - 1: lp - 1 + len(gen_ids)].to(torch.float16).cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=60)
    ap.add_argument("--dataset", default="hotpotqa")
    a = ap.parse_args()
    ns = argparse.Namespace(dataset=a.dataset, doc_number=40, sample=a.sample, sample_seed=42,
                            retriever="BM25", split="validation", pass_number=0,
                            cache_dir=os.environ.get("HF_HOME", "/work/hdd/myproject/anon/hf"), max_length=30000,
                            retrieval_split="train", max_corpus_examples=None,
                            babilong_split="qa2", babilong_length="64k")
    dataset_name, examples, _ = _load_examples(ns)
    print(f"[Info] loaded {len(examples)} {dataset_name} examples")
    os.makedirs(OUT, exist_ok=True)
    dev = "cuda:0"
    cache = os.environ.get("HF_HOME", "/work/hdd/myproject/anon/hf")

    # ---- pass plan: 32B base (T gen + L forced) -> +lam085 (Lft) -> 7B base (S) -> +S-solo (Sft)
    store = {}
    lm, tok = load_causal_lm("Qwen/Qwen2.5-32B-Instruct", device_map=dev, cache_dir=cache)
    lm.eval()
    for i, ex in enumerate(examples):
        prompt_ctx, _ = _build_teacher_prompt_with_audit(ex, dataset=dataset_name, tokenizer=tok, max_length=30000)
        exq = copy.copy(ex); exq.documents = []
        prompt_noctx, _ = _build_teacher_prompt_with_audit(exq, dataset=dataset_name, tokenizer=tok, max_length=30000)
        gen, t_log = greedy_traj(lm, tok, prompt_ctx, dev)
        if not gen:
            print(f"[skip] ex{i} empty gen"); continue
        l_log = forced_logits(lm, tok, prompt_noctx, gen, dev)
        text = tok.decode(gen, skip_special_tokens=False)
        # segmentation: reasoning = [0, fa_start), answer = [fa_end, first newline after)
        fa = text.find("Final Answer:")
        # map char offsets to token index via prefix decode lengths
        offs, acc = [], ""
        for t_id in gen:
            acc += tok.decode([t_id], skip_special_tokens=False)
            offs.append(len(acc))
        def tok_at(char):
            for j, o in enumerate(offs):
                if o > char: return j
            return len(gen)
        seg_reason_end = tok_at(fa) if fa >= 0 else len(gen)
        ans_start = tok_at(fa + len("Final Answer:")) if fa >= 0 else len(gen)
        nl = text.find("\n", fa) if fa >= 0 else -1
        ans_end = tok_at(nl) if nl >= 0 else len(gen)
        ctx_ids = set()
        for d in ex.documents:
            dt = getattr(d, "text", None) or (d.get("text") if isinstance(d, dict) else str(d))
            ctx_ids.update(tok(dt, add_special_tokens=False).input_ids)
        store[i] = dict(gen=np.array(gen, dtype=np.int32), T=t_log.numpy(), L=l_log.numpy(),
                        seg=np.array([seg_reason_end, ans_start, ans_end], dtype=np.int32),
                        copy_mask=np.array([int(t in ctx_ids) for t in gen], dtype=np.int8),
                        prompt_ctx=prompt_ctx, prompt_noctx=prompt_noctx,
                        question=ex.question, gold=str(ex.answer))
        print(f"[32B] ex{i} gen={len(gen)} reason_end={seg_reason_end} ans=[{ans_start},{ans_end})", flush=True)
    lm = load_lora_stack(lm, f"{CKPT}/stage2_on_v5reader_lam085_r16_s600", label="LM-lam085"); lm.eval()
    for i, d in store.items():
        d["Lft"] = forced_logits(lm, tok, d["prompt_noctx"], list(d["gen"]), dev).numpy()
        print(f"[32B+lam085] ex{i}", flush=True)
    del lm
    torch.cuda.empty_cache()
    slm, stok = load_causal_lm("Qwen/Qwen2.5-7B-Instruct", device_map=dev, cache_dir=cache)
    slm.eval()
    for i, d in store.items():
        d["S"] = forced_logits(slm, stok, d["prompt_ctx"], list(d["gen"]), dev).numpy()
        print(f"[7B] ex{i}", flush=True)
    slm = load_lora_stack(slm, f"{CKPT}/reader_binding_v5distill_r16_s900", label="S-solo"); slm.eval()
    for i, d in store.items():
        d["Sft"] = forced_logits(slm, stok, d["prompt_ctx"], list(d["gen"]), dev).numpy()
        np.savez_compressed(f"{OUT}/{dataset_name}_ex{i:03d}.npz",
                            gen=d["gen"], T=d["T"], L=d["L"], Lft=d["Lft"], S=d["S"], Sft=d["Sft"],
                            seg=d["seg"], copy_mask=d["copy_mask"],
                            question=d["question"], gold=d["gold"])
        print(f"[7B+Ssolo] ex{i} SAVED", flush=True)
    print(f"[Done] {len(store)} examples -> {OUT}")


if __name__ == "__main__":
    main()
