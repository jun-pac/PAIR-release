#!/usr/bin/env python
"""
PROFESSOR EXPERIMENT — snapKV's "cheating" advantage disappears under multi-question KV reuse (QASPER).

Motivation (professor): snapKV does a FULL per-query prefill, computes ALL attention, and only THEN evicts —
so even keeping 512 tokens it keeps exactly the tokens that query needs. That is a hidden advantage. The
realistic setting that removes it: a stream of follow-up questions on the SAME document (QASPER assigns several
questions per paper). You prefill the paper ONCE and REUSE its (compressed) KV across questions — you cannot
afford a fresh full prefill per question. Now snapKV must compress the doc-KV anchored on query1 and reuse that
for query2..k, so later queries get a doc-KV that was NOT tailored to them. Ours (a cheap uncompressed SLM
reader) reuses losslessly, so it has no such penalty (and its prefill is cheap anyway).

THE KV MANIPULATION (this is snapKV's own machinery, not cacheblend):
  * snapKV keeps the tokens its "observation window" (the last W tokens) attends to. We set W = exactly the
    length of the query+instruction suffix, so the window IS the query (not an arbitrary last-few-tokens).
  * Prefill [doc | query], run snapKV's scorer -> per KV-head, per layer, score every DOC token by the query's
    attention. Keep top-B doc tokens (sorted by ORIGINAL position), DROP the query window entirely -> a clean
    doc-only compressed KV cache. (Stock SnapKV reorders KV by score and force-keeps the window; we subclass its
    `compress` to keep doc-only in positional order — see DocOnlySnapKVPress.)
  * Decode a question: append only that question's tokens on top of the cached doc-KV. position_ids continue
    from the ORIGINAL doc length |D| (RoPE), cache_position continues from B (the compressed length) — the
    KVPress position/cache divergence. Greedy, stop after "Final Answer:".

METHODS (all one model, default Qwen2.5-14B-Instruct), per paper (doc D fixed order, questions Q1..Qk):
  teacher-full    : context = UNCOMPRESSED D-KV, reused per Qi  (== full prefill [D|Qi]; lossless)   -> CEILING
  snapkv-perquery : context = snapKV(D, window=Qi), RECOMPUTED per Qi (full prefill every question)  -> snapKV best
  snapkv-reuse    : context = snapKV(D, window=Q1), built ONCE, reused for every Qi                   -> THE TARGET
The three share the identical prompt / doc order / budget B and one decode path, so snapkv-reuse minus
snapkv-perquery on Q2..Qk is PURELY the reuse penalty. (ours-14B+3B lossless reuse comes from the standard
fusion pipeline — its reuse is mathematically exact, so ours-per-query == ours-reuse.)

Raw generations are saved to JSONL; score with the canonical EM-strict + F1 afterward. An inline EM/F1 is kept
for live monitoring only. A correctness gate (--smoke) checks teacher-full (cache-reuse decode) == HF generate.
"""
from __future__ import annotations
import argparse, json, os, re, string, sys, time
from dataclasses import dataclass

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import DynamicCache
from kvpress import SnapKVPress

from src.models import load_causal_lm
from src.data import _load_qasper_dataset, _format_qasper_documents, _extract_qasper_questions
from src.qa_prompts import build_full_context_qa_prompt


# ----------------------------- snapKV: keep DOC-ONLY, positional order -----------------------------
@dataclass
class DocOnlySnapKVPress(SnapKVPress):
    """snapKV scoring, but the compressed cache = top-B DOC tokens (positions [0, doc_len)) in ASCENDING
    positional order, with the query window DROPPED. window_size is set to the query-suffix length so the
    observation window is exactly the query."""
    keep_budget: int = 512
    doc_len: int = 0

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)  # [b, n_kv, k_len]
        dl = int(self.doc_len)
        B = min(int(self.keep_budget), dl)
        doc_scores = scores[:, :, :dl]                    # score only the doc positions
        idx = doc_scores.topk(B, dim=-1).indices          # [b, n_kv, B] (score order)
        idx, _ = torch.sort(idx, dim=-1)                  # -> positional (ascending) order
        gidx = idx.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
        k = keys[:, :, :dl, :].gather(2, gidx).contiguous()
        v = values[:, :, :dl, :].gather(2, gidx).contiguous()
        return k, v                                        # length B, query window dropped


# ----------------------------- KV build / decode primitives -----------------------------
@torch.no_grad()
def prefill_uncompressed(model, doc_ids):
    """Full (uncompressed) doc-KV. Reusing this prefix + appending a query == full prefill [D|Q] (exact)."""
    dev = model.device
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    out = model(input_ids=doc_ids.to(dev), use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    legacy = tuple((k.detach(), v.detach()) for (k, v) in out.past_key_values.to_legacy_cache())
    return legacy, time.perf_counter() - t0


@torch.no_grad()
def snapkv_compress_doc(model, doc_ids, win_suffix_ids, keep_budget):
    """Prefill [doc | window-query] under DocOnlySnapKVPress -> doc-only top-B compressed KV (length B)."""
    dev = model.device
    doc_len = int(doc_ids.shape[1])
    full = torch.cat([doc_ids, win_suffix_ids], dim=1).to(dev)
    press = DocOnlySnapKVPress(compression_ratio=0.5, window_size=int(win_suffix_ids.shape[1]),
                               keep_budget=int(keep_budget), doc_len=doc_len)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    with press(model):
        out = model(input_ids=full, use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    legacy = tuple((k.detach(), v.detach()) for (k, v) in out.past_key_values.to_legacy_cache())
    return legacy, time.perf_counter() - t0


@torch.no_grad()
def decode_from_context(model, tok, ctx_legacy, doc_len, suffix_ids, ctx_len, max_new):
    """Append `suffix_ids` (a question) on top of a context cache and greedy-decode. position_ids continue
    from the ORIGINAL doc length (RoPE); cache_position continues from ctx_len (compressed length)."""
    dev = model.device
    eos = tok.eos_token_id
    cache = DynamicCache.from_legacy_cache(tuple((k.clone(), v.clone()) for (k, v) in ctx_legacy))
    Lq = int(suffix_ids.shape[1])
    pos = torch.arange(doc_len, doc_len + Lq, device=dev).unsqueeze(0)
    cp = torch.arange(ctx_len, ctx_len + Lq, device=dev)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    out = model(input_ids=suffix_ids.to(dev), past_key_values=cache, position_ids=pos,
                cache_position=cp, use_cache=True)
    cache = out.past_key_values
    tokn = int(out.logits[:, -1, :].argmax(-1))
    gen = [tokn]
    rp = doc_len + Lq
    cq = ctx_len + Lq
    for _ in range(max_new - 1):
        if eos is not None and tokn == eos:
            break
        txt = tok.decode(gen)
        fa = txt.lower().rfind("final answer:")
        if fa >= 0 and "\n" in txt[fa + len("final answer:"):]:
            break
        out = model(input_ids=torch.tensor([[tokn]], device=dev), past_key_values=cache,
                    position_ids=torch.tensor([[rp]], device=dev),
                    cache_position=torch.tensor([cq], device=dev), use_cache=True)
        cache = out.past_key_values
        tokn = int(out.logits[:, -1, :].argmax(-1))
        gen.append(tokn)
        rp += 1
        cq += 1
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return tok.decode(gen, skip_special_tokens=True), len(gen), time.perf_counter() - t0


# ----------------------------- inline monitoring metric (canonical scorer is source of truth) -----------------------------
def _norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _extract(text):
    low = text.lower()
    i = low.rfind("final answer:")
    if i >= 0:
        return text[i + len("final answer:"):].strip().split("\n")[0].strip()
    return text.strip().split("\n")[-1].strip()


def em_strict(pred, golds):
    p = _norm(pred)
    return float(any(p == _norm(g) for g in golds if g))


def f1_max(pred, golds):
    def f1(p, g):
        pt, gt = _norm(p).split(), _norm(g).split()
        if not pt or not gt:
            return float(pt == gt)
        common = {}
        for w in pt:
            if w in gt:
                common[w] = min(pt.count(w), gt.count(w))
        ncommon = sum(common.values())
        if ncommon == 0:
            return 0.0
        prec, rec = ncommon / len(pt), ncommon / len(gt)
        return 2 * prec * rec / (prec + rec)
    return max((f1(pred, g) for g in golds if g), default=0.0)


# ----------------------------- QASPER grouped by paper (FIXED doc order) -----------------------------
def load_papers(split, cache_dir, sample_papers, min_questions, variant):
    ds = _load_qasper_dataset(split, cache_dir=cache_dir)
    if sample_papers:
        ds = ds.select(range(min(sample_papers, len(ds))))
    papers = []
    for paper in ds:
        docs = _format_qasper_documents(paper)  # FIXED natural order (NOT per-question BM25 rerank)
        qs = _extract_qasper_questions(paper)
        qs = [(qid, q, golds) for (qid, q, _ans, golds) in qs if q and golds]
        if len(qs) < min_questions or not docs:
            continue
        # question-independent reusable prefix: build a full prompt, strip its trailing "Question:" marker
        q0 = qs[0][1]
        full0 = build_full_context_qa_prompt(q0, docs, prompt_variant=variant)
        marker0 = f"\n\nQuestion: {q0}\n"
        assert full0.endswith(marker0), "prompt tail mismatch"
        prefix = full0[: -len(marker0)]  # {instruction}Context:\n{doc_blocks}
        papers.append({"paper_id": paper.get("id", ""), "prefix": prefix, "questions": qs})
    return papers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--cache-dir", default=os.environ.get("HF_HOME", "/work/hdd/myproject/anon/hf"))
    ap.add_argument("--split", default="validation")
    ap.add_argument("--variant", default="reason_then_answer_qasper_full")
    ap.add_argument("--sample-papers", type=int, default=0, help="0 = all papers")
    ap.add_argument("--min-questions", type=int, default=2)
    ap.add_argument("--keep-budget", type=int, default=512)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--max-doc-tokens", type=int, default=16000)
    ap.add_argument("--methods", default="teacher,snapkv_perquery,snapkv_reuse")
    ap.add_argument("--output", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="1 paper + correctness gate vs HF generate")
    args = ap.parse_args()
    methods = args.methods.split(",")

    print(f"[Info] loading {args.model}", flush=True)
    model, tok = load_causal_lm(args.model, device_map="cuda:0", cache_dir=args.cache_dir)
    model.eval()

    done = set()
    if args.resume and os.path.exists(args.output):
        for l in open(args.output):
            try:
                r = json.loads(l)
                done.add((r["example_id"], r["method"]))
            except Exception:
                pass
        print(f"[Info] resume: {len(done)} (qid,method) already done", flush=True)

    papers = load_papers(args.split, args.cache_dir, args.sample_papers or (1 if args.smoke else 0),
                         args.min_questions, args.variant)
    print(f"[Info] {len(papers)} papers (>= {args.min_questions} q), variant={args.variant}, B={args.keep_budget}", flush=True)

    if args.smoke:
        _correctness_gate(model, tok, papers[0], args)

    fout = open(args.output, "a")
    agg = {m: {"em": [], "f1": []} for m in methods}
    n_written = 0
    for pi, paper in enumerate(papers):
        prefix = paper["prefix"]
        doc_ids = tok(prefix, return_tensors="pt", add_special_tokens=True,
                      truncation=True, max_length=args.max_doc_tokens).input_ids
        doc_len = int(doc_ids.shape[1])
        qs = paper["questions"]
        suffix_ids = [tok(f"\n\nQuestion: {q}\n", return_tensors="pt", add_special_tokens=False).input_ids
                      for (_qid, q, _g) in qs]

        # one-time context builds (teacher uncompressed; snapkv-reuse anchored on Q1)
        ctx = {}
        build_s = {}
        if "teacher" in methods:
            ctx["teacher"], build_s["teacher"] = prefill_uncompressed(model, doc_ids)
        if "snapkv_reuse" in methods:
            ctx["snapkv_reuse"], build_s["snapkv_reuse"] = snapkv_compress_doc(model, doc_ids, suffix_ids[0], args.keep_budget)

        for qi, (qid, q, golds) in enumerate(qs):
            for m in methods:
                if (qid, m) in done:
                    continue
                if m == "teacher":
                    text, glen, dec_s = decode_from_context(model, tok, ctx["teacher"], doc_len,
                                                            suffix_ids[qi], doc_len, args.max_new)
                    cbuild = build_s["teacher"] if qi == 0 else 0.0
                    ctx_len = doc_len
                elif m == "snapkv_reuse":
                    ctx_len = int(ctx["snapkv_reuse"][0][0].shape[2])  # true compressed length
                    text, glen, dec_s = decode_from_context(model, tok, ctx["snapkv_reuse"], doc_len,
                                                            suffix_ids[qi], ctx_len, args.max_new)
                    cbuild = build_s["snapkv_reuse"] if qi == 0 else 0.0
                elif m == "snapkv_perquery":
                    pq_ctx, cbuild = snapkv_compress_doc(model, doc_ids, suffix_ids[qi], args.keep_budget)
                    ctx_len = int(pq_ctx[0][0].shape[2])
                    text, glen, dec_s = decode_from_context(model, tok, pq_ctx, doc_len,
                                                            suffix_ids[qi], ctx_len, args.max_new)
                else:
                    continue
                ext = _extract(text)
                em, f1 = em_strict(ext, golds), f1_max(ext, golds)
                agg[m]["em"].append(em)
                agg[m]["f1"].append(f1)
                rec = {"paper_id": paper["paper_id"], "example_id": qid, "q_index": qi,
                       "n_questions": len(qs), "method": m, "question": q, "golds": golds,
                       "raw": text, "extracted": ext, "em": em, "f1": f1,
                       "keep_budget": args.keep_budget, "doc_len": doc_len, "ctx_len": ctx_len,
                       "ctx_build_s": round(cbuild, 4), "decode_s": round(dec_s, 4), "gen_len": glen}
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                n_written += 1
        if (pi + 1) % 10 == 0 or args.smoke:
            msg = " | ".join(f"{m}: em={sum(agg[m]['em'])/max(1,len(agg[m]['em'])):.3f} "
                             f"f1={sum(agg[m]['f1'])/max(1,len(agg[m]['f1'])):.3f} (n={len(agg[m]['em'])})"
                             for m in methods)
            print(f"[{pi+1}/{len(papers)}] {msg}", flush=True)
    fout.close()
    print(f"[Done] wrote {n_written} records to {args.output}", flush=True)


@torch.no_grad()
def _correctness_gate(model, tok, paper, args):
    """teacher-full via cache-reuse decode must match HF model.generate greedy on [D|Q1]."""
    print("[Gate] teacher cache-reuse decode vs HF generate on [D|Q1] ...", flush=True)
    dev = model.device
    prefix = paper["prefix"]
    q1 = paper["questions"][0][1]
    doc_ids = tok(prefix, return_tensors="pt", add_special_tokens=True,
                  truncation=True, max_length=args.max_doc_tokens).input_ids
    doc_len = int(doc_ids.shape[1])
    suffix = tok(f"\n\nQuestion: {q1}\n", return_tensors="pt", add_special_tokens=False).input_ids
    ctx, _ = prefill_uncompressed(model, doc_ids)
    mine, _, _ = decode_from_context(model, tok, ctx, doc_len, suffix, doc_len, 48)
    full = torch.cat([doc_ids, suffix], dim=1).to(dev)
    gen = model.generate(full, max_new_tokens=48, do_sample=False, num_beams=1,
                         pad_token_id=tok.eos_token_id)
    ref = tok.decode(gen[0, full.shape[1]:], skip_special_tokens=True)
    a, b = mine.strip()[:80], ref.strip()[:80]
    print(f"[Gate] MINE : {a!r}")
    print(f"[Gate] HFGEN: {b!r}")
    print(f"[Gate] PASS={a[:60] == b[:60]} (teacher cache-reuse decode vs HF generate, first 60 chars)", flush=True)


if __name__ == "__main__":
    main()
