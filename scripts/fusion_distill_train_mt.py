#!/usr/bin/env python
"""MULTI-TURN (accumulate/KV-reuse) FUSION DISTILLATION — LoRA-FT the LM (and optionally the SLM) so the FUSED
SLM+LM output reproduces the full-context teacher **in the multi-turn setting the §7.0-V table evaluates**.

WHY: the single-turn musique adapter did NOT transfer to multi-turn QASPER (setting+task+λ all shifted).
This trains natively IN the accumulate setting, on the SAME benchmark, at the SAME λ as eval — removing the shift.

It replays `mtrag_accum.run_ours`'s loop EXACTLY (same INSTRUCTION, q_turn, fmt, prep, history handling), but instead
of decoding it TEACHER-FORCES the teacher's per-turn completion y and backprops the fusion loss:

  per turn:  slm_logits = SLM([...ctx, history, q_turn ++ y])     # SLM branch sees passages
             lm_logits  = LM ([...history,       q_turn ++ y])     # LM branch NEVER sees passages
             fused      = λ·slm + (1−λ)·lm        (identical weighted-sum + λ as inference)
             loss       = CE(fused, y)  over the completion tokens only

GRADIENT SCOPE (documented approximation): the passage/history forwards run under `no_grad` — they only BUILD the
KV cache (the cache still reflects the LoRA-adapted model, since the forward applies LoRA; only the gradient is
truncated to the CURRENT turn). This is truncated-BPTT over turns: tractable and standard. After each turn the
caches are cropped back (exactly like run_ours) AND detached, so no graph is retained across turns.

Targets come from a teacher-mode `mtrag_accum` run over the TRAIN split of the same benchmark (`raw` field =
full reasoning + "Final Answer:"), filtered to teacher-correct turns (acc_f1 >= --f1-threshold). Low-quality turns
are still REPLAYED (so history/passages stay faithful) but contribute NO loss.
"""
import os, sys, json, argparse, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("REASON_FIX", "1")
os.environ.setdefault("REASON_HIST", "ref")
import torch
import torch.nn.functional as F
from collections import defaultdict
from peft import LoraConfig, get_peft_model
from src.models import load_causal_lm
import src.qa_prompts as qa_prompts
import scripts.mtrag_accum as MT           # reuse the EXACT accumulate machinery (train == eval)
from scripts.mab_eval import q_turn as q_turn_terse


def _fwd(st, ids, want_grad, all_logits):
    """Stateful.forward, but optionally with grad and returning ALL positions' logits (for teacher forcing)."""
    n = ids.shape[1]
    ctx = torch.enable_grad() if want_grad else torch.no_grad()
    with ctx:
        out = st.model(ids, past_key_values=st.cache, use_cache=True,
                       logits_to_keep=(0 if all_logits else 1),
                       position_ids=torch.arange(st.pos, st.pos + n, device=st.dev)[None],
                       cache_position=torch.arange(st.cache_len, st.cache_len + n, device=st.dev),
                       attention_mask=torch.ones((1, st.cache_len + n), device=st.dev, dtype=torch.long))
    st.cache = out.past_key_values; st.cache_len += n; st.pos += n
    return out


def _rollback(st, cache_len, pos):
    """Crop the cache back to the pre-question state (same as run_ours) and DETACH so no graph is retained."""
    st.crop(cache_len); st.pos = pos
    for lyr in st.cache.layers:
        if getattr(lyr, "keys", None) is not None:
            lyr.keys = lyr.keys.detach(); lyr.values = lyr.values.detach()


def lr_at(step, n_optim, base_lr, warmup_frac):
    w = max(1, int(warmup_frac * n_optim))
    return base_lr * (step + 1) / w if step < w else base_lr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-log", required=True, help="mtrag_accum --method teacher run over the TRAIN ref")
    ap.add_argument("--ref", required=True, help="the TRAIN reuse-ref jsonl (same one the teacher-log was made from)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--slm-model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--instruction", default="QA_REASON_V3", help="qa_prompts constant (MUST match the eval run)")
    ap.add_argument("--lam", type=float, default=0.85, help="fusion weight on SLM (MUST match eval λ)")
    ap.add_argument("--f1-threshold", type=float, default=0.5)
    ap.add_argument("--max-comp-tokens", type=int, default=200)
    ap.add_argument("--max-conv", type=int, default=250)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--grad-accum", type=int, default=8, help="optimizer step every N scored TURNS")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-frac", type=float, default=0.03)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--train-slm", action="store_true", help="ALSO LoRA-train the SLM (joint)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=0, help="cap optimizer updates (canary)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cache-dir", default=os.environ.get("HF_HOME", "/work/hdd/myproject/anon/hf"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)

    # the instruction/prompt MUST be the eval one — set it on the reused module
    MT.INSTR_OVERRIDE = getattr(qa_prompts, a.instruction)
    print(f"[Info] instruction={a.instruction} reason_fix={MT.REASON_FIX} reason_hist={MT.REASON_HIST} lam={a.lam}", flush=True)

    # 1) teacher targets per (conv,turn), filtered to teacher-correct
    tgt = {}
    for l in open(a.teacher_log):
        if not l.strip(): continue
        d = json.loads(l)
        raw = d.get("raw") or ""
        if float(d.get("acc_f1", 0.0)) >= a.f1_threshold and "Final Answer:" in raw:
            tgt[(d.get("conv"), int(d.get("turn")))] = raw
    print(f"[Info] {len(tgt)} teacher-correct turns (acc_f1>={a.f1_threshold}) from {a.teacher_log}", flush=True)

    # 2) conversations from the TRAIN ref (same grouping as mtrag_accum.main)
    rows = [json.loads(l) for l in open(a.ref) if l.strip()]
    by = defaultdict(list)
    for r in rows: by[r["conversation_id"]].append(r)
    convs = [sorted(v, key=lambda r: int(r["turn"])) for v in by.values()][: a.max_conv]
    convs = [c for c in convs if any((c[0]["conversation_id"], int(t["turn"])) in tgt for t in c)]
    n_scored = sum(1 for c in convs for t in c if (c[0]["conversation_id"], int(t["turn"])) in tgt)
    print(f"[Info] {len(convs)} conversations, {n_scored} scored turns", flush=True)
    if not convs:
        print("❌ no usable conversations — check --ref matches the teacher-log", flush=True); sys.exit(2)

    # 3) models
    slm, slm_tok = load_causal_lm(a.slm_model, cache_dir=a.cache_dir, device_map=a.device)
    lm, lm_tok = load_causal_lm(a.model, cache_dir=a.cache_dir, device_map=a.device)
    lcfg = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    lm = get_peft_model(lm, lcfg); lm.train()
    print("[LM]  ", end=""); lm.print_trainable_parameters()
    if a.train_slm:
        slm = get_peft_model(slm, lcfg); slm.train()
        print("[SLM] ", end=""); slm.print_trainable_parameters()
    else:
        slm.eval()
        for p in slm.parameters(): p.requires_grad_(False)
    dev = next(lm.parameters()).device
    trainable = [p for p in lm.parameters() if p.requires_grad]
    if a.train_slm: trainable += [p for p in slm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=a.lr)

    n_optim = max(1, int(a.epochs * n_scored) // a.grad_accum)
    if a.max_steps > 0: n_optim = min(n_optim, a.max_steps)
    print(f"[Info] target optim_updates={n_optim} grad_accum={a.grad_accum} lr={a.lr} train_slm={a.train_slm}", flush=True)

    step = 0; scored = 0; running = 0.0; nseen = 0
    opt.zero_grad(set_to_none=True)
    done = False
    for epoch in range(int(a.epochs) + 1):
        if done: break
        order = convs[:]; random.shuffle(order)
        for tasks in order:
            if done: break
            turns = MT.prep(tasks)
            S = MT.Stateful(slm, slm_tok, dev, 0.0); L = MT.Stateful(lm, lm_tok, dev, 0.0)
            _fwd(S, S._ids(MT.INSTRUCTION() + "\n\n"), False, False)
            _fwd(L, L._ids(MT.INSTRUCTION() + "\n\n"), False, False)
            for tr in turns:
                if tr["newk"]:
                    _fwd(S, S._ids(MT.fmt(tr["newk"])), False, False)   # LM never sees passages
                y = tgt.get((tr["conv"], int(tr["turn"])))
                qt = q_turn_terse(tr["q"])
                if y and tr["ans"] == "ANSWERABLE" and tr["q"]:
                    sb, sp = S.cache_len, S.pos; lb, lp = L.cache_len, L.pos
                    comp = lm_tok(y, add_special_tokens=False, return_tensors="pt")["input_ids"][:, : a.max_comp_tokens]
                    T = comp.shape[1]
                    if T >= 2:
                        sq = S._ids(qt); lq = L._ids(qt)
                        Qs, Ql = sq.shape[1], lq.shape[1]
                        s_in = torch.cat([sq, comp.to(dev)], dim=1); l_in = torch.cat([lq, comp.to(dev)], dim=1)
                        s_out = _fwd(S, s_in, a.train_slm, True)
                        l_out = _fwd(L, l_in, True, True)
                        sl = s_out.logits[:, Qs - 1: Qs - 1 + T, :]
                        ll = l_out.logits[:, Ql - 1: Ql - 1 + T, :]
                        V = min(sl.shape[-1], ll.shape[-1])
                        fused = a.lam * sl[..., :V].float() + (1 - a.lam) * ll[..., :V].float()
                        loss = F.cross_entropy(fused.reshape(-1, V), comp.reshape(-1).to(dev))
                        (loss / a.grad_accum).backward()
                        running += float(loss.detach()); nseen += 1; scored += 1
                        if scored % a.grad_accum == 0:
                            for g in opt.param_groups: g["lr"] = lr_at(step, n_optim, a.lr, a.warmup_frac)
                            torch.nn.utils.clip_grad_norm_(trainable, a.grad_clip)
                            opt.step(); opt.zero_grad(set_to_none=True); step += 1
                            if step % a.log_every == 0 or step == 1:
                                print(f"[{step}/{n_optim}] loss={running/max(1,nseen):.4f} "
                                      f"lr={opt.param_groups[0]['lr']:.2e} (turns={scored})", flush=True)
                                running = 0.0; nseen = 0
                            if step >= n_optim: done = True
                    _rollback(S, sb, sp); _rollback(L, lb, lp)
                # history into BOTH branches (teacher's reasoning + REFERENCE gold — same as eval's reason_hist=ref)
                hgen = MT._hist_reason(qt, y, tr["aref"]) if y else MT.hist_block(tr["q"], tr["aref"])
                _fwd(S, S._ids(hgen), False, False); _fwd(L, L._ids(hgen), False, False)
                if done: break
            del S, L
            torch.cuda.empty_cache()

    os.makedirs(a.out, exist_ok=True)
    lm.save_pretrained(a.out); lm_tok.save_pretrained(a.out)
    slm_out = None
    if a.train_slm:
        slm_out = a.out.rstrip("/") + "_slmada"; os.makedirs(slm_out, exist_ok=True)
        slm.save_pretrained(slm_out); slm_tok.save_pretrained(slm_out)
        print(f"[Done] SLM adapter -> {slm_out}", flush=True)
    json.dump({"setting": "multi-turn accumulate", "model": a.model, "slm_model": a.slm_model, "lam": a.lam,
               "instruction": a.instruction, "reason_hist": MT.REASON_HIST, "f1_threshold": a.f1_threshold,
               "n_convs": len(convs), "n_scored_turns": n_scored, "optim_updates": step, "lr": a.lr,
               "epochs": a.epochs, "grad_accum": a.grad_accum, "lora_r": a.lora_r, "train_slm": a.train_slm,
               "slm_adapter": slm_out, "teacher_log": a.teacher_log, "ref": a.ref},
              open(os.path.join(a.out, "fusion_distill_meta.json"), "w"), indent=2)
    print(f"[Done] LM adapter -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
