#!/usr/bin/env python
"""bench_config.py — THE SINGLE SOURCE OF TRUTH for every benchmark's experiment settings.

Why this exists: experiments were run with ad-hoc, per-run prompts/extractors/dimensions, so results were not
comparable and conclusions flipped run-to-run. From now on:
  * Harnesses READ their settings from here (they cannot silently diverge).
  * Every run records a PROVENANCE FINGERPRINT derived from here (prompt sha, extractor, decoding, max_new, ratio,
    doc/truncation, seed/order, model). scripts/build_table.py REFUSES to compare runs whose fingerprints differ.
  * scripts/validate_experiment.py and tests/ check runs AGAINST this config.

To change a benchmark's setting, change it HERE (one place), bump nothing else. Two runs are comparable IFF they
share the same fingerprint fields below (except the model/method being compared).
"""
import hashlib, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.qa_prompts as qp

# ---- GLOBAL invariants (apply to ALL benchmarks unless a benchmark overrides) ----
GLOBAL = dict(
    decoding="manual-greedy",          # ours needs manual; every method in a comparison MUST match this
    reason_fix=True,                    # reasoning is ON (the method) — see scripts/reason_fix.py
    reason_hist="ref",                  # multi-turn history = model reasoning + REFERENCE gold answer (no error-propagation)
    extractor="reason_answer_v1",       # the ONE canonical extractor = scripts.reason_fix.extract_answer
    fusion_lambda=0.7,
)

def _sha(text): return hashlib.sha256((text or "").encode()).hexdigest()[:16]

# ---- per-benchmark canonical config. `instruction`/`direct_instruction` are qa_prompts CONSTANT NAMES. ----
BENCHMARKS = {
    "locomo": dict(
        family="multiturn-accumulate", ref="/work/hdd/myproject/anon/locomo/locomo_reuse_ref_full.jsonl",
        instruction="QA_REASON_V3_LOCOMO", direct_instruction="QA_DIRECT_LOCOMO",
        metric="token_f1", max_new=200, ratio=0.8125, max_conv=15,
        doc_number=None, truncation=128000, order="conversation", seed=None,
    ),
    "qasper": dict(
        family="multiturn-accumulate", ref="/work/hdd/myproject/anon/qasper/qasper_reuse_ref_full.jsonl",
        instruction="QA_REASON_V3", direct_instruction=None,
        metric="qasper_canonical_f1", max_new=200, ratio=0.8125, max_conv=110,
        doc_number=0, truncation=128000, order="paper", seed=None,
    ),
    "eventqa": dict(
        family="mab-mc", episodes="results/mab_episodes/eventqa_65k.jsonl",
        instruction="QA_REASON_V2", direct_instruction=None,
        metric="token_f1", max_new=160, ratio=0.5, n_sets=5,
        doc_number=None, truncation=None, order="canonical", seed=None,
    ),
    "mtrag": dict(
        family="multiturn-accumulate", ref="/work/hdd/myproject/anon/mtrag/reference.jsonl",
        instruction="QA_REASON_V3_MTRAG", direct_instruction=None,   # reason + COMPLETE conversational answer (gold ~25w)
        metric="token_f1", max_new=200, ratio=0.8125, max_conv=110,
        doc_number=None, truncation=128000, order="conversation", seed=None,
    ),
    # CLUTRR multi-question kinship as an ACCUMULATE KV-reuse bench (QASPER-style: one STORY = one conversation, the
    # story is turn-1 context reused across the derived questions). CLEAN single-relation-word gold -> relation EM
    # (no length pathology). Prompt/extractor mirror the existing single-turn CLUTRR run (reason_then_answer_clutrr).
    "clutrr": dict(
        family="multiturn-accumulate", ref="/work/hdd/myproject/anon/clutrr/clutrr_mq_reuse_ref.jsonl",
        instruction="QA_FULL_CONTEXT_INSTRUCTION_REASON_CLUTRR", direct_instruction=None,
        metric="clutrr_relation_em", max_new=200, ratio=0.8125, max_conv=331,
        doc_number=0, truncation=128000, order="story", seed=None,
    ),
    # CLUTRR SINGLE-QUESTION (2026-09-06, user: CLUTRR is entirely synthetic, so a gain on it cannot come
    # from the LM's pretrained knowledge; wanted as a small ablation section, and "accum이 아닌 세팅이 오히려
    # 좋을거같음"). The SAME 331 stories / 1862 questions / story text / golds / instruction / metric as
    # `clutrr`, regrouped by scripts/clutrr_to_mtrag.py --single-question so that EVERY question is its own
    # one-turn conversation: the story is the context of each question, no history, no cross-question KV
    # reuse, and no earlier question's teacher-forced reference answer in the prompt. max_conv 254 = the
    # first 70 stories = the identical question set (same example_ids) to the CLUTRR-accum N=254 cells.
    "clutrr_sq": dict(
        family="multiturn-accumulate", ref="/work/hdd/myproject/anon/clutrr/clutrr_sq_ref.jsonl",
        instruction="QA_FULL_CONTEXT_INSTRUCTION_REASON_CLUTRR", direct_instruction=None,
        metric="clutrr_relation_em", max_new=200, ratio=0.8125, max_conv=254,   # ratio as `clutrr` (harness default; presses only)
        doc_number=0, truncation=128000, order="story", seed=None,
    ),
    # NarrativeQA as an ACCUMULATE bench (2026-08-20, replaces the audit-compromised CLUTRR-accum slot):
    # one STORY (8-25k tok, movie script or short gutenberg text) = one conversation, full text as turn-1
    # context, ~30 official human questions per story asked over it (589 Q / 20 stories; caps user-set).
    # Free-form short answers with TWO references -> token-F1 inline vs ref1 (harness convention), offline
    # multi-gold rescoring via the `golds` field exactly as QASPER-accum. Built by
    # scripts/build_narrativeqa_ref.py (deterministic selection; QuALITY analyzed and rejected: 4-way MC
    # = reader-bound regime + metric break, ctx only ~6k).
    "narrativeqa": dict(
        family="multiturn-accumulate", ref="/work/hdd/myproject/anon/narrativeqa/narrativeqa_reuse_ref.jsonl",
        instruction="QA_REASON_V3_NQA", direct_instruction=None,
        metric="token_f1", max_new=200, ratio=0.8125, max_conv=20,
        doc_number=0, truncation=128000, order="story", seed=None,
    ),
    # LooGLE longdep_qa as an ACCUMULATE bench (2026-08-23; chosen by the corrected filter: REASONING
    # first — 26% compute/count + 8% causal + longdep multihop; free-form short golds; ctx 8-30k window).
    # 355 Q / 70 docs; MC-contaminated rows and >12-word explanation golds dropped at build (recorded).
    # Built by scripts/build_loogle_ref.py. Full candidate-survey verdicts in RESULTS_MASTER 2026-08-23.
    "loogle": dict(
        family="multiturn-accumulate", ref="/work/hdd/myproject/anon/loogle/loogle_reuse_ref.jsonl",
        instruction="QA_REASON_V3", direct_instruction=None,
        metric="token_f1", max_new=200, ratio=0.8125, max_conv=70,
        doc_number=0, truncation=128000, order="doc", seed=None,
    ),
    # ★ single-turn THROUGHPUT bases (2026-08-28, task A4.3): each canonical d40 question as a 1-turn
    # conversation (contexts = its 40 cached docs), so mtrag_accum's batched decode + conv_wall_s
    # total-wall timer + provenance cover the single-turn workload — the one where the method's
    # prefill advantage is NOT amortised. Refs: scripts/build_singleturn_ref.py (docs from the same
    # deterministic docs-caches as the canonical harness; musique first-96, hotpot seed-42 first-96).
    # Passage formatting is the accumulate convention, so these are their OWN comparison bases —
    # never mix with old singleturn-rag logs. λ: musique 0.7 / hotpot 0.85 (established 32B+7B).
    "musique_st40": dict(
        family="multiturn-accumulate", ref="/work/hdd/myproject/anon/singleturn/musique_st40_ref.jsonl",
        instruction="QA_REASON_V3", direct_instruction=None,
        metric="token_f1", max_new=200, ratio=0.78125, max_conv=96,
        doc_number=0, truncation=128000, order="first-N", seed=None,
    ),
    # musique_st40r (2026-08-29): SEED-42 RANDOM 96 of the same docs-cache — musique's validation
    # is hop-ordered, so musique_st40's first-96 = the 2-hop easy band (the sandwich-inversion
    # subset, RESULTS_MASTER 28e addendum v2); the random sample mixes hops like hotpotqa_st40.
    "musique_st40r": dict(
        family="multiturn-accumulate", ref="/work/hdd/myproject/anon/singleturn/musique_st40r_ref.jsonl",
        instruction="QA_REASON_V3", direct_instruction=None,
        metric="token_f1", max_new=200, ratio=0.78125, max_conv=96,
        doc_number=0, truncation=128000, order="random", seed=42,
    ),
    # ★ FULL accuracy bases (2026-08-29, user: accuracy comes from the WHOLE set; only throughput
    # may approximate on a subset). Every cached question: musique 2417, hotpot 600. These are the
    # ONLY sources for the _st40 curves' F1 columns; the 96-question refs are TIMING bases only.
    "musique_st40_full": dict(
        family="multiturn-accumulate", ref="/work/hdd/myproject/anon/singleturn/musique_st40_full_ref.jsonl",
        instruction="QA_REASON_V3", direct_instruction=None,
        metric="token_f1", max_new=200, ratio=0.78125, max_conv=2417,
        doc_number=0, truncation=128000, order="first-N", seed=None,
    ),
    "hotpotqa_st40_full": dict(
        family="multiturn-accumulate", ref="/work/hdd/myproject/anon/singleturn/hotpotqa_st40_full_ref.jsonl",
        instruction="QA_REASON_V3", direct_instruction=None,
        metric="token_f1", max_new=200, ratio=0.78125, max_conv=600,
        doc_number=0, truncation=128000, order="random", seed=42,
    ),
    # ---- the CONTEXT-LENGTH axis (2026-09-02, revised 09-03) -----------------------
    # Identical questions, golds, prompt and metric to the d40 benches above; only the number
    # of retrieved documents differs, so accuracy and throughput are both read against context
    # length with nothing else varying.
    # WHY THE AXIS STOPS AT 160. Qwen2.5-32B and -7B are both max_position_embeddings=32768
    # with no rope scaling, and the 128,000-character truncation exists to keep a context
    # inside that window. Measured raw context: hotpot d160 maxes at 127,470 characters and
    # 0 of 600 examples are cut, while d200 cuts 13.7%, d240 59.3% and d320 96.7%. A d320
    # bench would therefore not be a 320-document bench at all -- the model never sees the
    # documents past its window -- so the d320 entries were removed rather than reported.
    # musique is denser: d80 is its last fully clean point (d120 cuts 13 of 2417, 0.5%).
    "hotpotqa_st120_full": dict(
        family="multiturn-accumulate", ref="/work/hdd/myproject/anon/singleturn/hotpotqa_st120_full_ref.jsonl",
        instruction="QA_REASON_V3", direct_instruction=None,
        metric="token_f1", max_new=200, ratio=0.78125, max_conv=600,
        doc_number=0, truncation=128000, order="random", seed=42,
    ),
    "musique_st120_full": dict(
        family="multiturn-accumulate", ref="/work/hdd/myproject/anon/singleturn/musique_st120_full_ref.jsonl",
        instruction="QA_REASON_V3", direct_instruction=None,
        metric="token_f1", max_new=200, ratio=0.78125, max_conv=2417,
        doc_number=0, truncation=128000, order="first-N", seed=None,
    ),
    # Identical questions, identical golds, identical prompt and metric to the d40
    # benches above; only the number of retrieved documents differs, so accuracy and
    # throughput can both be read against context length with nothing else varying.
    "hotpotqa_st80_full": dict(
        family="multiturn-accumulate", ref="/work/hdd/myproject/anon/singleturn/hotpotqa_st80_full_ref.jsonl",
        instruction="QA_REASON_V3", direct_instruction=None,
        metric="token_f1", max_new=200, ratio=0.78125, max_conv=600,
        doc_number=0, truncation=128000, order="random", seed=42,
    ),
    "musique_st80_full": dict(
        family="multiturn-accumulate", ref="/work/hdd/myproject/anon/singleturn/musique_st80_full_ref.jsonl",
        instruction="QA_REASON_V3", direct_instruction=None,
        metric="token_f1", max_new=200, ratio=0.78125, max_conv=2417,
        doc_number=0, truncation=128000, order="first-N", seed=None,
    ),
    "hotpotqa_st160_full": dict(
        family="multiturn-accumulate", ref="/work/hdd/myproject/anon/singleturn/hotpotqa_st160_full_ref.jsonl",
        instruction="QA_REASON_V3", direct_instruction=None,
        metric="token_f1", max_new=200, ratio=0.78125, max_conv=600,
        doc_number=0, truncation=128000, order="random", seed=42,
    ),
    # ---- the EXTENDED context axis, past the native window (2026-09-11, user: "YaRN이란걸 쓰면 더
    # 되는거 같아서 200, 240, 280, 320까지 더 해보면 좋을거같음"). Same 600 questions, golds, prompt and
    # metric as every depth above; the refs come from the same docs caches (build_singleturn_ref.py,
    # DOCS=200/240/280; d320 was built 2026-09-02). Beyond d160 the context passes the models' 32,768
    # native window, so every run on these benches sets ROPE_YARN_FACTOR=4 (Qwen's documented YaRN
    # setting: factor 4.0 over original_max_position_embeddings 32768 -> 131k), stamped in provenance
    # as axis_rope_yarn_factor. That makes them a different model from the native-window runs — the
    # user's decision is to draw the axis as eight points and say so at the point where YaRN starts.
    # `truncation` is None here because the harness applies no character cap (the 128,000 above is a
    # recorded value that never cut a d40-d160 context); nothing is cut at these depths either.
    "hotpotqa_st200_full": dict(
        family="multiturn-accumulate", ref="/work/hdd/myproject/anon/singleturn/hotpotqa_st200_full_ref.jsonl",
        instruction="QA_REASON_V3", direct_instruction=None,
        metric="token_f1", max_new=200, ratio=0.78125, max_conv=600,
        doc_number=0, truncation=None, order="random", seed=42,
    ),
    "hotpotqa_st240_full": dict(
        family="multiturn-accumulate", ref="/work/hdd/myproject/anon/singleturn/hotpotqa_st240_full_ref.jsonl",
        instruction="QA_REASON_V3", direct_instruction=None,
        metric="token_f1", max_new=200, ratio=0.78125, max_conv=600,
        doc_number=0, truncation=None, order="random", seed=42,
    ),
    "hotpotqa_st280_full": dict(
        family="multiturn-accumulate", ref="/work/hdd/myproject/anon/singleturn/hotpotqa_st280_full_ref.jsonl",
        instruction="QA_REASON_V3", direct_instruction=None,
        metric="token_f1", max_new=200, ratio=0.78125, max_conv=600,
        doc_number=0, truncation=None, order="random", seed=42,
    ),
    "hotpotqa_st320_full": dict(
        family="multiturn-accumulate", ref="/work/hdd/myproject/anon/singleturn/hotpotqa_st320_full_ref.jsonl",
        instruction="QA_REASON_V3", direct_instruction=None,
        metric="token_f1", max_new=200, ratio=0.78125, max_conv=600,
        doc_number=0, truncation=None, order="random", seed=42,
    ),
    "musique_st160_full": dict(
        family="multiturn-accumulate", ref="/work/hdd/myproject/anon/singleturn/musique_st160_full_ref.jsonl",
        instruction="QA_REASON_V3", direct_instruction=None,
        metric="token_f1", max_new=200, ratio=0.78125, max_conv=2417,
        doc_number=0, truncation=128000, order="first-N", seed=None,
    ),
    "hotpotqa_st40": dict(
        family="multiturn-accumulate", ref="/work/hdd/myproject/anon/singleturn/hotpotqa_st40_ref.jsonl",
        instruction="QA_REASON_V3", direct_instruction=None,
        metric="token_f1", max_new=200, ratio=0.78125, max_conv=96,
        doc_number=0, truncation=128000, order="random", seed=42,
    ),
    # single-turn RAG benchmarks (musique/hotpotqa/2wiki): recorded so their DIMENSIONS are pinned + checkable.
    "musique": dict(
        family="singleturn-rag", instruction="QA_REASON_V3", direct_instruction=None,
        metric="token_f1", max_new=200, ratio=None, doc_number=40, truncation=30000, order="first-N", seed=None,
    ),
    "hotpotqa": dict(
        family="singleturn-rag", instruction="QA_REASON_V3", direct_instruction=None,
        metric="token_f1", max_new=200, ratio=None, doc_number=40, truncation=30000,
        order="random", seed=42,   # ★ hotpot is seed=42 RANDOM order, NOT benchmark order — must match to compare
    ),
}

def get(bench):
    if bench not in BENCHMARKS: raise KeyError(f"unknown benchmark {bench!r}; known: {list(BENCHMARKS)}")
    c = dict(GLOBAL); c.update(BENCHMARKS[bench]); c["bench"] = bench
    return c

def prompt_text(name):
    return getattr(qp, name) if name else ""

def fingerprint(bench, model, method, **override):
    """The comparability fingerprint. Two runs go in the SAME table IFF these match (model/method excepted)."""
    c = get(bench); c.update(override)
    return dict(
        bench=bench, method=method, model=model,
        prompt_name=c["instruction"], prompt_sha=_sha(prompt_text(c["instruction"])),
        extractor=c["extractor"], decoding=c["decoding"], reason_fix=c["reason_fix"], reason_hist=c["reason_hist"],
        max_new=c["max_new"], ratio=c["ratio"], doc_number=c["doc_number"], truncation=c["truncation"],
        order=c["order"], seed=c["seed"], metric=c["metric"],
    )

_COMMIT = None
def code_commit():
    """Short git commit of the code at run time — recorded in every fingerprint so a results TABLE can link
    to the EXACT code that produced it (the user's provenance rule). NOT a COMPARE_KEY (cross-commit runs
    may be legitimately identical); it is informational provenance surfaced in tables."""
    global _COMMIT
    if _COMMIT is None:
        import subprocess
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        try:
            h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                                        stderr=subprocess.DEVNULL).decode().strip() or "unknown"
            # ★ MARK A DIRTY TREE (2026-08-16). This used to record HEAD alone, which made the field a LIE
            # whenever the run had uncommitted edits — and in this workflow it usually does. A LoCoMo floor
            # log stamped 08cc529 could not be reproduced by checking out 08cc529 (0.3246 against its
            # recorded 0.4425), while the CURRENT code at the same batch reproduced it exactly: the run had
            # actually used HEAD + uncommitted changes that were committed later. Half a day went into
            # chasing a code difference that never existed. A hash that cannot be checked out is not
            # provenance, so a dirty tree is now stamped as such.
            dirty = subprocess.call(["git", "diff", "--quiet", "HEAD"], cwd=root,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0
            _COMMIT = h + ("-dirty" if dirty else "")
        except Exception:
            _COMMIT = "unknown"
    return _COMMIT

def run_fingerprint(bench, model, method, prompt_name, prompt_text_, max_new, ratio, reason_fix, reason_hist,
                    slm_model=None, lam=None):
    """Fingerprint recorded by a harness from the ACTUAL settings it used (prompt/max_new/ratio) + the
    benchmark-FIXED dims (doc/truncation/order/seed/metric) from config. build_table.py compares these."""
    c = get(bench)
    fp = dict(bench=bench, method=method, model=model, prompt_name=prompt_name, prompt_sha=_sha(prompt_text_),
              extractor=c["extractor"], decoding=c["decoding"], reason_fix=bool(reason_fix), reason_hist=reason_hist,
              max_new=int(max_new), ratio=ratio, doc_number=c["doc_number"], truncation=c["truncation"],
              order=c["order"], seed=c["seed"], metric=c["metric"], code_commit=code_commit())
    if slm_model is not None: fp["slm_model"] = slm_model
    if lam is not None: fp["lam"] = lam
    return fp

def example_ids_sha(ids):
    """Stable hash of the SORTED example-id set — two runs must share it (or its intersection) to be same-N."""
    return hashlib.sha256("\x1f".join(sorted(str(i) for i in ids)).encode()).hexdigest()[:16]

def provenance(bench, model, method, example_ids=None, **override):
    """Full provenance block stored on every result record. build_table.py + validate_experiment.py check it."""
    fp = fingerprint(bench, model, method, **override)
    if example_ids is not None:
        fp["n"] = len(example_ids); fp["example_ids_sha"] = example_ids_sha(example_ids)
    for k in ("slm_model", "lam"):
        if k in override: fp[k] = override[k]
    return fp

# fields that MUST be identical across a comparison set (a table). model/method are the compared variables.
COMPARE_KEYS = ["bench", "prompt_sha", "extractor", "decoding", "reason_fix", "reason_hist",
                "max_new", "ratio", "doc_number", "truncation", "order", "seed", "metric"]

if __name__ == "__main__":
    for b in BENCHMARKS:
        fp = fingerprint(b, "Qwen/Qwen2.5-14B-Instruct", "teacher")
        print(f"{b:10s} prompt={fp['prompt_name']}({fp['prompt_sha']}) metric={fp['metric']} "
              f"max_new={fp['max_new']} ratio={fp['ratio']} doc={fp['doc_number']} trunc={fp['truncation']} "
              f"order={fp['order']} seed={fp['seed']}")
