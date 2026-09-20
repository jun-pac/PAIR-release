#!/usr/bin/env python
"""
Build a MULTI-QUESTION KV-reuse benchmark from CLUTRR (clean single-word metric — the QASPER replacement).

WHY: QASPER's token-F1 is length-broken (a verbose teacher scores BELOW a terse student despite higher recall).
We need shared-context + multiple-questions with a CLEAN metric. CLUTRR gives it: one kinship graph is shared
context, and CLUTRR's own symbolic solver output `proof_state` yields several ground-truth (A, relation, B)
DERIVED facts — each a multi-hop question, answered by ONE relation word -> exact match, no length pathology.

★ CRITICAL DESIGN CHOICE (2026-07-05): we do NOT use CLUTRR's narrative `clean_story`. It is directionally
INCONSISTENT with its own proof graph for some edges (e.g. story 147 text "Pedro got his son Harold" = Harold is
Pedro's son, but the proof needs Pedro to be Harold's son to make Pedro==Antonio's brother). A model reading the
narrative would get a DIFFERENT answer than the proof-derived gold -> invalid benchmark. So we GENERATE the story
ourselves from the proof's LEAF facts, which are self-consistent with the derived-fact golds by construction.

CONSTRUCTION (documented in CLUTRR_MULTIQ.md):
  * proof_state = list of steps {derived_fact: [support1, support2]}. KEYS = derived facts; RHS facts that are
    never a key = LEAVES (base edges). Convention (A, r, B) == "B is A's r" (verified: proof is self-consistent
    under it AND the top key == the dataset's labeled (query, target)).
  * STORY = the leaf facts realized as clean atomic sentences "B is A's r." (shuffled). This IS the context.
  * QUESTIONS = the DERIVED facts (keys), dedup by (A,B) pair; the labeled target is Q1 (anchor), the rest Q2+.
    Each is multi-hop (needs >=2 leaves to compose). gold = the relation word. Metric = relation-vocabulary EM.
  * CONTEXT PADDING: pad with leaf-sentences from OTHER stories whose entities are DISJOINT (cannot inject false
    relations) up to ~CLUTRR_PAD_CHARS, target block placed mid-context. Identical context for all methods.
VALIDATION: (1) top derived key == labeled (query[0], target_text, query[1]) [100% => parse correct]; (2) every
question's two entities appear in the leaf sentences [answerable from the generated story].
"""
import ast, json, os, random, re, sys
from collections import Counter
from datasets import load_dataset

CACHE = os.environ.get("HF_HOME", "/work/hdd/myproject/anon/hf")
CONFIG = os.environ.get("CLUTRR_CONFIG", "gen_train234_test2to10")
MIN_Q = int(os.environ.get("CLUTRR_MIN_Q", "3"))
PAD_CHARS = int(os.environ.get("CLUTRR_PAD_CHARS", "8000"))
MAX_K = int(os.environ.get("CLUTRR_MAX_K", "100"))
# ★ Keep only stories whose LEAF facts are all BLOOD relations. Two ambiguities are thereby removed: (a) a
# grandfather/aunt/uncle GIVEN as a base fact is side-ambiguous (Nicole's grandfather = paternal or maternal?);
# (b) a spouse (wife/husband) leaf triggers CLUTRR's in-law-COLLAPSING convention — e.g. it labels the wife's father
# as your "father" (not father-in-law), which a correctly-reasoning model would dispute. With blood-only leaves every
# derived relation is UNIQUELY determined AND matches real kinship -> a fair, solvable ceiling. (331 stories, 1862 Q.)
DIRECT = {"father", "mother", "son", "daughter", "brother", "sister"}


def parse_proof(p):
    """-> (leaves[list of (A,r,B)], derived_questions[list of {a,b,rel}], target_key)."""
    proof = ast.literal_eval(p) if isinstance(p, str) else p
    keys, supports = [], []
    for step in proof:
        for k, v in step.items():
            keys.append(tuple(k))
            supports.extend(tuple(x) for x in v)
    keyset = set(keys)
    leaves, seen = [], set()
    for s in supports:
        if s not in keyset and s not in seen:
            seen.add(s); leaves.append(s)
    qseen, qs = set(), []
    for (a, r, b) in keys:
        if (a, b) in qseen:
            continue
        qseen.add((a, b)); qs.append({"a": a, "b": b, "rel": r})
    return leaves, qs, keys[0]


def sentences(leaves):
    return [f"{b} is {a}'s {r}." for (a, r, b) in leaves]   # (A,r,B) == "B is A's r"


def leaf_entities(leaves):
    return {x for (a, r, b) in leaves for x in (a, b)}


def main():
    ds = load_dataset("CLUTRR/v1", CONFIG, split="test", cache_dir=CACHE)
    rows = list(ds)
    parsed = []
    for i, row in enumerate(rows):
        try:
            leaves, qs, top = parse_proof(row["proof_state"])
        except Exception:
            parsed.append(None); continue
        parsed.append((leaves, qs, top))
    # distractor pool: (index, entity-set, sentences) for every parseable story
    pool = [(i, leaf_entities(p[0]), sentences(p[0])) for i, p in enumerate(parsed) if p]

    out, match, ansok, total_q = [], 0, 0, 0
    hop_hist, rel_hist, ctx_chars = Counter(), Counter(), []
    for i, row in enumerate(rows):
        if not parsed[i]:
            continue
        leaves, qs, top = parsed[i]
        q = ast.literal_eval(row["query"]) if isinstance(row["query"], str) else row["query"]
        tgt = str(row["target_text"]).strip()
        if top[0] == q[0] and top[2] == q[1] and top[1] == tgt:
            match += 1
        if len(qs) < MIN_Q:
            continue
        se = ast.literal_eval(row["story_edges"]) if isinstance(row["story_edges"], str) else row["story_edges"]
        if len(se) > MAX_K:
            continue
        if not all(r in DIRECT for (a, r, b) in leaves):   # unambiguous direct-relation leaves only
            continue
        ents = leaf_entities(leaves)
        # answerability: every question entity must appear in the leaves
        if all(x["a"] in ents and x["b"] in ents for x in qs):
            ansok += 1
        else:
            continue
        rng = random.Random(i)
        tgt_sents = sentences(leaves); rng.shuffle(tgt_sents)
        tgt_block = " ".join(tgt_sents)
        # pad: disjoint-entity distractor blocks, target placed mid-context
        before, after, used = [], [], set(ents)
        budget = max(0, PAD_CHARS - len(tgt_block)); acc = 0
        for (j, dents, dsents) in pool:
            if j == i or acc >= budget or (dents & used):
                continue
            blk = " ".join(dsents)
            (before if len(before) <= len(after) else after).append(blk)
            used |= dents; acc += len(blk) + 1
        context = "\n".join(before + [tgt_block] + after)
        se = ast.literal_eval(row["story_edges"]) if isinstance(row["story_edges"], str) else row["story_edges"]
        out.append({"story_id": i, "target_story": tgt_block, "context": context,
                    "n_edges": len(se), "n_leaves": len(leaves), "n_distractors": len(before) + len(after),
                    "questions": qs, "n_questions": len(qs)})
        total_q += len(qs); hop_hist[len(se)] += 1; ctx_chars.append(len(context))
        for x in qs:
            rel_hist[x["rel"]] += 1

    import statistics
    N = sum(1 for p in parsed if p)
    print(f"parseable stories={N}  target-validation match={match}/{N} ({100*match/N:.1f}%)  answerable-check pass={ansok}")
    print(f"stories kept (>={MIN_Q} Qs, answerable)={len(out)}  questions={total_q}  mean Q/story={total_q/max(len(out),1):.1f}")
    print(f"padded-context chars: median={statistics.median(ctx_chars):.0f} (~{int(statistics.median(ctx_chars))//4} tok)  "
          f"mean distractors/story={statistics.mean(o['n_distractors'] for o in out):.1f}")
    print(f"edges/story: {dict(sorted(hop_hist.items()))}")
    print(f"relations ({len(rel_hist)}): {dict(rel_hist.most_common())}")
    outp = "results/clutrr_multiq/clutrr_multiq.json"
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    json.dump(out, open(outp, "w"))
    print(f"wrote {outp}")
    e = out[0]
    print(f"\n=== EXAMPLE story {e['story_id']} ({e['n_questions']} Qs, {e['n_leaves']} leaf-facts, {e['n_distractors']} distractors) ===")
    print("target story (generated):", e["target_story"])
    for x in e["questions"]:
        print(f"  Q: In one word, {x['b']} is {x['a']}'s what?  -> {x['rel']}")


if __name__ == "__main__":
    main()
