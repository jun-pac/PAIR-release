#!/usr/bin/env python
"""Generate the full-context TEACHER targets for the benchmark-agnostic fusion-FT mix.

Runs teacher-14B (FULL context, the CANONICAL bench_config prompt QA_REASON_V3 / sha c1a237e0 — the same
prompt the eval uses, selected via ANSWER_PROMPT_VARIANT=reason_v3) over the mix corpus from build_ft_mix.py,
and writes the teacher's generation as the distillation target.

★ FILTERING POLICY (rewritten 2026-07-29 — the old one broke the experiment):
  The FT objective is BEHAVIOUR CLONING: only the example INPUTS are used, and whatever the teacher answers is
  the target. **The teacher's behaviour is never curated.** If it ignores the 'reason first' instruction, or
  answers something wrong, that is what it does on that input and the student should reproduce it — the goal is
  "SLM+LM output ≈ teacher output, on any input, under any prompt", nothing more.
  So `keep` rejects only records that are UNUSABLE, never records we dislike:
     * empty generation
     * generation was still running at --max-new (a truncated observation: we never saw how it ends, so there is
       no termination behaviour to clone). Recorded as `hit_max_new`.
  Everything else is recorded as METADATA, not used to filter: `has_reasoning`, `chat_leak`, `gold_gate_pass`
  (the latter only if --f1-threshold > 0). The OLD policy gated QA sources on gold-F1 >= 0.5 while leaving
  summarization ungated — that leaked gold into a gold-free pipeline AND silently reweighted the mix.

Resumable: re-run with the same --out and it skips example_ids already present.
"""
import os, sys, json, argparse, time
os.environ.setdefault("ANSWER_PROMPT_VARIANT", "reason_v3")   # CANONICAL QA_REASON_V3 (c1a237e0) = the eval prompt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from types import SimpleNamespace
from src.models import load_causal_lm, prepare_inputs
from src.eval import compute_best_em_f1, extract_final_answer
import scripts.run_evidence_sketch_experiments as H
import scripts.bench_config as BC

SUMM_SOURCES = {"gov_report", "qmsum", "multi_news"}
MARKER = "Final Answer:"


def rec_to_example(r):
    return SimpleNamespace(example_id=r["example_id"], question=r["question"],
                           documents=list(r["documents"]), answer=(r["golds"] or [""])[0],
                           answers=list(r["golds"]))


@torch.no_grad()
def generate(model, tok, prompt, max_length, max_new, dev):
    """Returns (text, hit_cap): hit_cap=True means generation was still going at --max-new, i.e. the teacher
    never terminated — such a row is NOT a clean behaviour target and is rejected below."""
    ids = prepare_inputs(tok, prompt, max_length=max_length)["input_ids"].to(dev)
    out = model(ids, use_cache=True, logits_to_keep=1)
    past, logits = out.past_key_values, out.logits[:, -1, :]
    gen, K = [], int(ids.shape[1])
    for step in range(max_new):
        t = int(torch.argmax(logits, -1))
        if t == tok.eos_token_id:
            return tok.decode(gen, skip_special_tokens=True), False
        gen.append(t)
        txt = tok.decode(gen, skip_special_tokens=True)
        i = txt.find(MARKER)
        if i >= 0:
            nl = txt.find("\n", i + len(MARKER))
            if nl != -1:
                return txt[:nl], False
            if len(gen) - 0 > 0 and txt.count(MARKER) >= 2:
                return txt[:txt.find(MARKER, i + len(MARKER))].rstrip(), False
        out = model(torch.tensor([[t]], device=dev), past_key_values=past, use_cache=True,
                    cache_position=torch.tensor([K + step], device=dev), logits_to_keep=1)
        past, logits = out.past_key_values, out.logits[:, -1, :]
    return tok.decode(gen, skip_special_tokens=True), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--max-length", type=int, default=30000)
    ap.add_argument("--max-new", type=int, default=320, help="long enough for the summarization targets")
    ap.add_argument("--f1-threshold", type=float, default=0.0,
                    help="GOLD gate, recorded only (does NOT change `keep`) — see the module docstring. 0 = off.")
    ap.add_argument("--require-reasoning", type=int, default=0,
                    help="ABLATION ONLY (default 0 = off). 1 rejects targets that jump straight to 'Final Answer:'. "
                         "Off by default: the teacher ignoring its own instruction is the teacher's behaviour, and "
                         "this pipeline clones behaviour rather than curating it.")
    ap.add_argument("--cache-dir", default=os.environ.get("HF_HOME", "/work/hdd/myproject/anon/hf"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            if l.strip():
                done.add(str(json.loads(l).get("example_id")))
    recs = [json.loads(l) for l in open(a.corpus) if l.strip()]
    todo = [r for r in recs if str(r["example_id"]) not in done]
    print(f"[Info] corpus={len(recs)} already_done={len(done)} todo={len(todo)}", flush=True)

    model, tok = load_causal_lm(a.model, cache_dir=a.cache_dir, device_map=a.device); model.eval()
    dev = next(model.parameters()).device
    commit = BC.code_commit()
    kept = {}
    t0 = time.perf_counter()
    with open(a.out, "a") as f:
        for i, r in enumerate(todo):
            ex = rec_to_example(r)
            prompt = H._build_qa_answer_prompt_preserving_query(ex, include_docs=True, tokenizer=tok,
                                                                max_length=a.max_length)
            text, hit_cap = generate(model, tok, prompt, a.max_length, a.max_new, dev)
            pred = extract_final_answer(text)
            em, f1 = compute_best_em_f1(pred, ex.answers or [ex.answer])
            is_summ = r["source"] in SUMM_SOURCES
            mi = text.find(MARKER)
            has_reasoning = mi > 0 and bool(text[:mi].strip())
            leak = any(m in text for m in ("Human:", "Assistant:", "<|im_start|>", "<|im_end|>"))
            # USABILITY gate only (see module docstring): non-empty, and we actually observed the teacher stop.
            # has_reasoning / leak / gold are RECORDED, not filtered on.
            ok = bool(text.strip()) and (not hit_cap) and (has_reasoning or not a.require_reasoning)
            r2 = dict(r); r2.update({"teacher_text": text, "teacher_pred": pred, "teacher_f1": float(f1),
                                     "is_summarization": is_summ, "keep": bool(ok), "code_commit": commit,
                                     "has_reasoning": bool(has_reasoning), "hit_max_new": bool(hit_cap),
                                     "chat_leak": bool(leak),
                                     "gold_gate_pass": bool(a.f1_threshold <= 0 or f1 >= a.f1_threshold)})
            f.write(json.dumps(r2) + "\n"); f.flush()
            kept[r["source"]] = kept.get(r["source"], 0) + (1 if ok else 0)
            if (i + 1) % 25 == 0:
                el = time.perf_counter() - t0
                print(f"[{i+1}/{len(todo)}] {el/ (i+1):.1f}s/ex kept={sum(kept.values())} by_source={kept}", flush=True)
    print(f"[Done] kept={sum(kept.values())} by_source={kept} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
