#!/usr/bin/env python
"""STAGE-2 of the staged fusion FT (user design 2026-08-09): the reader is ALREADY binding-SFT'd and
FROZEN; now LoRA-train the LM **inside the fusion** so it learns to work WITH the new reader.

    slm_logits = SLM_ft( instruction ++ context ++ query ++ y )   # frozen (reader adapter loaded)
    lm_logits  = LM    ( instruction ++ query ++ y )              # LoRA — the only thing that updates
    loss       = CE( lam*slm_logits + (1-lam)*lm_logits , y )     # y = the TEACHER's generation

★ NO GOLD (2026-08-10 doctrine). `y` is row["target"], which in binding_corpus_v5/v6.jsonl is the
teacher-32B generation — point this script ONLY at a teacher-target corpus. (The v1-v4 corpora carried
generator gold in the same field and are VOID; do not use them here.)
★ TRAIN-λ MUST EQUAL EVAL-λ. The LM is being trained to supply exactly the (1-λ) residual of this mixture;
training at 0.7 and decoding at 0.85 trains it for the wrong share. λ* is per-benchmark on a distilled
reader (LoCoMo 0.85, CLUTRR 0.7), so pick the λ of the arm this adapter is for and record it.

vs the old 'joint' (both branches moving) this is stable: one branch frozen. vs the LM-only fusion-FT
(style churn, §7) this trains the LM against the reader it will actually decode with.

PREFLIGHT SAVE before the loop (CLAUDE.md hard rule). 32B+7B needs 2 GPUs (--device auto; 76G weights).
"""
import argparse, json, os, sys, time, random
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import load_causal_lm
import src.qa_prompts as _QP
from src.qa_prompts import QA_REASON_V3_LOCOMO
from scripts.mab_eval import q_turn
from scripts import chat_wrap as _CWM
from peft import LoraConfig, get_peft_model, PeftModel
from src.lora_load import load_lora_stack        # aborts on a no-op adapter (see src/lora_load.py)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--slm-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--slm-lora", default=None,
                    help="the stage-1 reader adapter, loaded FROZEN. Omit (or 'none') to train the LM "
                         "branch inside a fusion whose reader is NAIVE — the 'L-fusion first, attach "
                         "S-solo afterwards' variant. The loader at line ~68 already tolerates a missing "
                         "value; this was `required=True` only because every run so far had a reader.")
    ap.add_argument("--lm-model", default="Qwen/Qwen2.5-32B-Instruct")
    ap.add_argument("--lm-lora", default=None,
                    help="load an EXISTING LM adapter before training. Required for round 2+ of the "
                         "alternation: the reader must be fit against the LM it will actually decode with, "
                         "not against the base LM.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lam", type=float, default=0.7)
    ap.add_argument("--defer-weight", type=float, default=0.0,
                    help="weight on the DEFER term: on rows labelled `branch_label=defer` (the reader "
                         "alone already predicts the target, the query-only LM branch does not), pull the "
                         "FUSED distribution back onto the reader's at the ANSWER positions. 0 reproduces "
                         "the original objective exactly.")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--max-length", type=int, default=28000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--train-branch", choices=("lm", "reader", "both"), default="lm",
                    help="which branch gets the LoRA. 'reader' = FUSION-AWARE READER TRAINING: the reader "
                         "learns logits that WIN the mixture at this lambda against a blind LM, which is "
                         "the objective the mechanism actually calls for — v5 trains the reader with plain "
                         "CE, which only sharpens it incidentally.")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed)

    rows = [json.loads(l) for l in open(a.corpus)]
    n_no_target = sum(1 for r in rows if not (r.get("target") or "").strip())
    if n_no_target:
        raise SystemExit(f"{n_no_target} rows have no target — this script requires a TEACHER-target corpus")
    print(f"[data] {len(rows)} teacher-target examples from {a.corpus} (train-lambda={a.lam})", flush=True)
    slm, slm_tok = load_causal_lm(a.slm_model, cache_dir=a.cache_dir, device_map=a.device)
    if a.slm_lora and a.slm_lora.lower() != "none":
        slm = load_lora_stack(slm, a.slm_lora, label="SLM prior")
    lm, lm_tok = load_causal_lm(a.lm_model, cache_dir=a.cache_dir, device_map=a.device)
    if a.lm_lora and a.lm_lora.lower() != "none":
        lm = load_lora_stack(lm, a.lm_lora, label="LM prior")
    # ★ 2026-08-11: if the branch we are about to LoRA already carries an adapter, MERGE it into the
    # weights first. Wrapping PEFT twice saves keys with a doubled 'base_model.model.' prefix, which then
    # loads as a silent NO-OP at eval (that is what made 'alternation round 2' read as 26.9%). After the
    # merge the new adapter has ordinary keys — but it is then only valid ON TOP of the merged parent, so
    # record the parent in the adapter dir and evaluate with --slm-lora <parent>,<this>.
    _prior = a.slm_lora if a.train_branch == "reader" else a.lm_lora
    if _prior and _prior.lower() != "none":
        if a.train_branch == "reader":
            slm = slm.merge_and_unload()
        else:
            lm = lm.merge_and_unload()
        print(f"[merge] prior adapter merged into the {a.train_branch.upper()} weights before the new LoRA "
              f"(eval this adapter stacked: {_prior},{a.out})", flush=True)
    lcfg = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    def _arm(m):
        m = get_peft_model(m, lcfg); m.train(); m.print_trainable_parameters()
        m.gradient_checkpointing_enable(); m.enable_input_require_grads()
        return m
    if a.train_branch == "lm":
        slm.eval(); lm = _arm(lm)
        saves = {"": lm}
    elif a.train_branch == "reader":
        lm.eval(); slm = _arm(slm)
        saves = {"": slm}
    else:                       # SIMULTANEOUS joint: one fused loss, both branches move on the same step
        lm = _arm(lm); slm = _arm(slm)
        saves = {"lm": lm, "reader": slm}
    print(f"[branch] training the {a.train_branch.upper()} inside the fusion at lambda={a.lam}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    for sub, m in saves.items():
        m.save_pretrained(os.path.join(a.out, sub) if sub else a.out)
    print(f"[preflight] adapter(s) written to {a.out}", flush=True)

    params = [p for m in saves.values() for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr)
    sdev = next(slm.parameters()).device; ldev = next(lm.parameters()).device
    order = list(range(len(rows))); random.shuffle(order)
    step = micro = 0; t0 = time.perf_counter(); run_loss = 0.0; ntok = 0
    while step < a.max_steps:
        for i in order:
            r = rows[i]
            # per-row instruction, byte-identical to how the reader was trained and how eval builds the
            # prompt (the corpus is mixed-domain: QA_REASON_V3 for document rows, *_LOCOMO for dialogue)
            instr = getattr(_QP, r.get("instruction") or "QA_REASON_V3_LOCOMO")
            qt = r.get("history", "") + q_turn(r["question"])
            target = r["target"] + "\n"
            if a.defer_weight > 0 and "answer_start_tok" not in r:
                _cut = target.rfind("Final Answer:")
                r["answer_start_tok"] = 0 if _cut < 0 else slm_tok(
                    target[:_cut + len("Final Answer:")], return_tensors="pt",
                    add_special_tokens=False).input_ids.shape[1]
            tids = slm_tok(target, return_tensors="pt", add_special_tokens=False).input_ids
            tids = torch.cat([tids, torch.tensor([[slm_tok.eos_token_id]])], 1)
            T = tids.shape[1]
            # SLM branch: full context (frozen, no grad)
            # byte-matched to mtrag_accum: reader = INSTRUCTION + "\n\n" + ctx + history + q_turn,
            #                               LM     = INSTRUCTION            + history + q_turn (no context)
            sp = slm_tok(_CWM.wrap(instr + "\n\n" + r["context"] + qt), return_tensors="pt", add_special_tokens=False).input_ids
            if sp.shape[1] + T > a.max_length: sp = sp[:, -(a.max_length - T):]
            lp = lm_tok(_CWM.wrap(instr + qt), return_tensors="pt", add_special_tokens=False).input_ids
            _si, _li = torch.cat([sp, tids], 1).to(sdev), torch.cat([lp, tids], 1).to(ldev)
            if a.train_branch == "lm":
                with torch.no_grad():
                    so = slm(_si, use_cache=False)
                lo = lm(_li, use_cache=False)
            elif a.train_branch == "reader":         # fusion-aware READER training: grad flows to the SLM
                so = slm(_si, use_cache=False)
                with torch.no_grad():
                    lo = lm(_li, use_cache=False)
            else:                                    # simultaneous: grad flows to BOTH
                so = slm(_si, use_cache=False)
                lo = lm(_li, use_cache=False)
            sl = so.logits[:, sp.shape[1] - 1: sp.shape[1] - 1 + T, :]
            ll = lo.logits[:, lp.shape[1] - 1: lp.shape[1] - 1 + T, :]
            V = min(sl.shape[-1], ll.shape[-1])
            dev = ldev if a.train_branch == "lm" else sdev
            fused = a.lam * sl[..., :V].to(dev) + (1 - a.lam) * ll[..., :V].to(dev)
            loss_sum = F.cross_entropy(fused.reshape(-1, V).float(), tids.to(dev).reshape(-1), reduction="sum")

            # ★ DEFER TERM. The plain fused cross-entropy above cannot teach the LM branch to hold still.
            #   Where the frozen reader alone already predicts the teacher's tokens, the fused distribution
            #   is already right, the loss is already small, and almost no gradient reaches the branch — so
            #   across v8 / v9 / CHAIN / CHAIN2 / L-fusion-first the share of already-correct CLUTRR turns
            #   the arm BREAKS never moved: 31 / 29 / 28 / 31 / 34 %. Every one of those was a change of row
            #   CONTENT, and content cannot fix which rows carry gradient.
            #   On rows measured (scripts/score_corpus_branches.py) as reader-knows-it / branch-does-not,
            #   pull the FUSED distribution back onto the reader's own. `sl` is frozen and computed under
            #   no_grad, so this gradient reaches ONLY the LM branch, and it says: do not move what the
            #   reader already has right. Rows where the branch's prior is also correct are left alone, so
            #   the contrast between deferring and asserting is what gets learned rather than a flat
            #   "always defer" (which is just lambda=1).
            if a.defer_weight > 0 and r.get("branch_label") == "defer":
                st = min(int(r.get("answer_start_tok") or 0), T - 1)
                f_lp = F.log_softmax(fused[:, st:, :].float(), dim=-1)
                r_lp = F.log_softmax(sl[..., :V].to(dev)[:, st:, :].float(), dim=-1)
                defer = F.kl_div(f_lp, r_lp, reduction="sum", log_target=True)
                loss_sum = loss_sum + a.defer_weight * defer
            (loss_sum / (T * a.grad_accum)).backward()
            run_loss += loss_sum.item(); ntok += T; micro += 1
            if micro % a.grad_accum == 0:
                opt.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % 20 == 0 or step == 1:
                    print(f"[{step}/{a.max_steps}] fused-loss/token={run_loss/max(ntok,1):.4f} "
                          f"({micro} micro, {time.perf_counter()-t0:.0f}s)", flush=True)
                    run_loss = 0.0; ntok = 0
                if step >= a.max_steps: break
        random.shuffle(order)
    for sub, m in saves.items():
        m.save_pretrained(os.path.join(a.out, sub) if sub else a.out)
    print(f"[Done] fusion-trained adapter(s) -> {a.out} ({time.perf_counter()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
