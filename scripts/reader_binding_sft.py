#!/usr/bin/env python
"""READER-side binding SFT (2026-08-09): LoRA the 7B *reader* on synthetic multi-session temporal-binding
data (gen_binding_corpus.py) — the skill the Q32B7B_RESULTS §6 diagnosis says cannot be transferred through
logits/selections/digests and must live in the model that READS the context.

Train/eval consistency: the prompt is the CANONICAL locomo instruction (QA_REASON_V3_LOCOMO) and the
question is rendered with the same q_turn() the accumulate harness uses. Loss = token-mean CE over the
target (reasoning+answer) tokens + newline + EOS (terminator supervised — FT_REPORT lesson #1/#2).

PREFLIGHT SAVE: the real save_pretrained() runs BEFORE the optimizer loop (CLAUDE.md hard rule — never
lose a trained adapter to a full disk after the GPU hours are spent).
"""
import argparse, json, os, sys, time, random
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import load_causal_lm
from src.qa_prompts import QA_REASON_V3_LOCOMO
import src.qa_prompts as _QP
from scripts.mab_eval import q_turn
from scripts import chat_wrap as _CWM
from peft import LoraConfig, get_peft_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--max-length", type=int, default=28000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed)

    rows = [json.loads(l) for l in open(a.corpus)]
    print(f"[data] {len(rows)} examples from {a.corpus}", flush=True)
    model, tok = load_causal_lm(a.model, cache_dir=a.cache_dir, device_map="cuda:0")
    lcfg = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lcfg); model.train()
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    dev = next(model.parameters()).device

    # ---- PREFLIGHT SAVE (real call, real path, before any GPU-hours are spent) ----
    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out)
    print(f"[preflight] adapter written to {a.out} — save path verified", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    order = list(range(len(rows))); random.shuffle(order)
    step = micro = 0; t0 = time.perf_counter(); run_loss = 0.0; ntok_acc = 0
    while step < a.max_steps:
        for i in order:
            r = rows[i]
            # v3 corpora carry a per-row `instruction` constant name (mixed-format training); default = locomo.
            _instr = getattr(_QP, r["instruction"]) if r.get("instruction") else QA_REASON_V3_LOCOMO
            # byte-matched to how mtrag_accum builds the reader's input: prefill(INSTRUCTION + "\n\n" + ctx)
            # then the accumulated q/a history, then this turn's question. (The old form had a stray "\n"
            # after the context and no history at all — see build_multiturn_corpus.py.)
            prompt = _instr + "\n\n" + r["context"] + r.get("history", "") + q_turn(r["question"])
            prompt = _CWM.wrap(prompt)      # 2026-09-05: same turn markers as generation and eval (CHAT_WRAP)
            target = r["target"] + "\n"
            pids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
            tids = tok(target, return_tensors="pt", add_special_tokens=False).input_ids
            tids = torch.cat([tids, torch.tensor([[tok.eos_token_id]])], 1)
            if pids.shape[1] + tids.shape[1] > a.max_length:
                pids = pids[:, -(a.max_length - tids.shape[1]):]
            ids = torch.cat([pids, tids], 1).to(dev)
            P, T = pids.shape[1], tids.shape[1]
            out = model(ids, use_cache=False)
            logits = out.logits[:, P - 1: P - 1 + T, :]
            loss_sum = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tids.to(dev).reshape(-1),
                                       reduction="sum")
            (loss_sum / (T * a.grad_accum)).backward()      # token-mean within example, averaged over accum
            run_loss += loss_sum.item(); ntok_acc += T; micro += 1
            if micro % a.grad_accum == 0:
                opt.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % 20 == 0 or step == 1:
                    print(f"[{step}/{a.max_steps}] loss/token={run_loss/max(ntok_acc,1):.4f} "
                          f"({micro} micro, {time.perf_counter()-t0:.0f}s)", flush=True)
                    run_loss = 0.0; ntok_acc = 0
                if step >= a.max_steps: break
        random.shuffle(order)
    model.save_pretrained(a.out)
    print(f"[Done] reader adapter -> {a.out} ({time.perf_counter()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
