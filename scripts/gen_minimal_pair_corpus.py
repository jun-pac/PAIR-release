#!/usr/bin/env python
"""MINIMAL-PAIR corpus (v14) — the S-side half of the adversarial fix.

WHAT THE LOCOMO LOGS SAY THE TRAP ACTUALLY IS (260813_LOCOMO10_DIAGNOSIS §3b, N=600). LoCoMo's adversarial
questions are SINGLE-SLOT PERTURBATIONS of real answerable questions in the same conversation:

    adversarial  "What happened to CAROLINE's son on their road trip?"      gold: Not mentioned
    answerable   "What happened to MELANIE's son on their road trip?"       gold: He got into an accident

50 of the 148 adversarial turns still have a >=0.55-overlap answerable twin inside the 600-turn sample
alone, and that sample holds only 452 of a conversation's 105-260 answerable questions. The reader fails by
TOPIC-MATCHING the neighbouring fact and re-attributing it: it answers "the importance of self-care",
"a rainbow sidewalk", "Sweden" — vivid details that are in the transcript, bound to someone else.

WHY THE EXISTING `unans_corpus.jsonl` DID NOT TEACH THIS. v12 trained the reader on 560 absent rows and
moved its LoCoMo abstention from 10% to 12%. Three defects, and only the first is about structure:
  1. THE QUESTION CARRIES A HINT — "Name the release that Hana shipped, according to these records. If the
     records do not say, reply that they do not say." A reader told that declining is on the menu learns
     nothing about noticing absence when nobody offers.
  2. NO PAIRED TWIN. `absent` and `distractor` are separate rows over separate contexts, so the corpus
     never presents the SAME context and the SAME wording with one slot changed. Without the pair the task
     is absence-detection; with it, the task is DISCRIMINATION, which is the capability actually missing.
  3. Container mix is memo-heavy. LoCoMo is dated multi-session dialogue.

This generator fixes all three. Every unanswerable row is emitted TOGETHER WITH its answerable twin over
the SAME context, differing in exactly one slot, with identical wording and no hint. Container defaults to
`transcript` (dated, one speaker per block) as the closest thing in the v8 grid to LoCoMo's sessions.

SLOT FAMILIES (each mirrors a class read off the LoCoMo logs):
  speaker   the predicate is bound to a DIFFERENT person who is present in the documents
            ("Which release did <other owner> ship?")            <- Caroline<-Melanie, Sam<-Evan
  locative  the fact exists, at a different site that is present ("Where was ... shipped from <site>?")
  scope     the fact exists, under a different scope/version that is present
  item      the queried item exists in the domain vocabulary but was never planted in THIS context

The twin's substituted value is ALWAYS present in the context, so a string scan for it succeeds and does
not settle the question — the model has to check the BINDING. That is the whole point.

★ NO GOLD SUPERVISION. This writes `target: ""`. Targets come from a teacher pass
(`binding_teacher_gen.py`) and the v6 gate is run BEFORE any training: the teacher must decline on the
unanswerable twins and answer correctly on the answerable ones. v1-v4 were deleted for interpolating the
gold into the target; `scripts/audit_corpus_supervision.py` checks this file's output.

★ GENERATE THE TEACHER TARGETS UNDER AN INSTRUCTION THAT PERMITS DECLINING. QA_REASON_V3 and
QA_REASON_V3_LOCOMO both say "NEVER reply 'Not mentioned'", so a teacher pass under either will confabulate
on every unanswerable twin and fail its own gate. Use QA_REASON_V3_LOCOMO_ABSTAIN, which is also what the
eval A/B runs, so training and eval agree on what the task is.

Usage:
  gen_minimal_pair_corpus.py --pairs-per-cell 6 --out results/fusionft/minpair_probe.jsonl --seed 20260813
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_binding_corpus_v8 import DOMAINS, NAMES, plant, render, F  # noqa: E402

ABSENT_GOLD = "Not mentioned in the conversation"   # LoCoMo's exact category-5 gold string
FAMILIES = ("speaker", "locative", "scope", "item")
# v8's DOMAINS carry the PAST tense ("shipped", "ran"), which is right for the body sentence "I shipped the
# parser release." but ungrammatical after an auxiliary — the first generation emitted "Which unit did Rosa
# purchased?" and "Which survey did Ryo ran?". Questions built with `did` use the base form from here.
# Explicit rather than inferred: "ran" -> "run" is not a rule any suffix strip gets right.
VERB_BASE = {"shipped": "ship", "handled": "handle", "purchased": "purchase", "ran": "run"}


SMALLTALK = ["Nice, that sounds good.", "Oh really? Tell me more.", "That's great to hear!",
             "Ha, I know the feeling.", "Makes sense.", "Good to know, thanks.",
             "How's everything else going?", "Same here, honestly."]


def render_dialogue(f, rng, D):
    """LoCoMo-SHAPED context: dated sessions of two-speaker dialogue, matching the benchmark byte-for-byte
    in layout:

        [Session 1 | 1:10 pm on 27 March, 2023]
        Audrey: Hey Andrew! ...
        Andrew: ...

    WHY THIS EXISTS. v12 put 560 synthetic absent rows into the reader and moved its LoCoMo abstention from
    10% to 12%. Those rows are MEMOS. The v8 grid's other containers are documents too — headers plus
    numbered entries — and LoCoMo is dated multi-session chat between two named people. Training the
    discrimination on one layout and testing it on the other adds a domain gap on top of the capability
    gap, for no reason: the layout is free to match, so it should match."""
    people = f["people"]
    a, b = people[0], people[1]
    blocks = []
    for u in f["docs"]:
        hh = rng.randrange(1, 12); mm = rng.randrange(0, 60); ap = rng.choice(["am", "pm"])
        lines = [f"[Session {u['idx'] + 1} | {hh}:{mm:02d} {ap} on {F(u['date'])}]"]
        # the block's OWNER is the speaker who reports the facts; the other speaker only reacts, so the
        # binding (who did what) is carried by the speaker label exactly as it is in LoCoMo
        other = b if u["owner"] == a else a
        lines.append(f"{u['owner']}: Hey {other}! Good to catch up.")
        for e in u["entries"]:
            lines.append(f"{u['owner']}: I {D['verb']} the {e['item']} {D['noun']}.")
            lines.append(f"{other}: {rng.choice(SMALLTALK)}")
        if not u["entries"]:
            lines.append(f"{u['owner']}: {rng.choice(SMALLTALK)}")
            lines.append(f"{other}: {rng.choice(SMALLTALK)}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _present(f, field):
    """Values of `field` that actually appear in the rendered documents."""
    return sorted({u[field] for u in f["docs"]})


def _holdout_item(f):
    """v8's plant() assigns EVERY item in the domain vocabulary to some document, so nothing is ever
    missing and the `item` family produced zero rows on the first generation. Remove one planted item here
    rather than editing the shared v8 generator, which every existing corpus is reproduced from.

    Note this family is the EASIER one: the held-out item is genuinely absent from the context, so a string
    scan settles it. It is kept because LoCoMo contains that class too ("What country is Melanie's grandma
    from?" — the grandma is present, the country is nowhere), but the speaker / locative / scope families
    are the ones that carry the cross-binding the reader actually fails."""
    live = [u for u in f["docs"] if u["entries"]]
    donors = [u for u in live if len(u["entries"]) >= 2] or live
    if not donors:
        return None
    u = donors[0]
    held = u["entries"].pop()["item"]
    f["live"] = [x for x in f["docs"] if x["entries"]]
    f["multi"] = [x for x in f["docs"] if len(x["entries"]) >= 2]
    return held if f["live"] else None


def build_pair(rng, dom, container, family, ci, instruction="QA_REASON_V3_LOCOMO_ABSTAIN"):
    """Return [answerable, unanswerable-twin] over ONE context, differing in exactly one slot."""
    f = plant(rng, dom, rng.randrange(12, 26))
    if f is None:
        return []
    D = DOMAINS[dom]
    if container == "dialogue" and family in ("locative", "scope"):
        # a dialogue block has no site/scope header to bind against; the LoCoMo traps this corpus is
        # modelled on are overwhelmingly SPEAKER swaps anyway (Caroline<-Melanie, Sam<-Evan, Joanna<-Nate)
        return []
    held = _holdout_item(f) if family == "item" else None
    if family == "item" and not held:
        return []
    # `render` with ref='plain' emits every entry as "I <verb> the <item> <noun>." — the actor, site, date
    # and scope live in the block HEADER only, so every question below is genuinely non-local.
    ctx = render_dialogue(f, rng, D) if container == "dialogue" else render(container, f, rng, "plain")
    u = rng.choice(f["live"])
    e = rng.choice(u["entries"])

    if family == "speaker":
        owners_with_entries = {uu["owner"] for uu in f["live"]}
        others = [p for p in _present(f, "owner") if p not in owners_with_entries]
        if not others:
            return []
        # the answerable member must have exactly ONE right answer. An owner can hold several documents and
        # a document several entries, so without this the gold is one of two or three equally correct items
        # and the teacher gate fails a row the teacher actually got right.
        mine = [x["item"] for uu in f["live"] if uu["owner"] == u["owner"] for x in uu["entries"]]
        if len(mine) != 1:
            return []
        vb = VERB_BASE[D['verb']]
        q_ok = f"Which {D['noun']} did {u['owner']} {vb}?"
        q_no = f"Which {D['noun']} did {rng.choice(others)} {vb}?"
        gold_ok = mine[0]
    elif family in ("locative", "scope"):
        # TWO-SLOT BINDING, one slot swapped. The first generation asked "Where was the cache release
        # shipped?" and paired it with "Which release was shipped from Oslo by Viktor?" — a different
        # QUESTION FORM, which destroys the minimal pair the whole design rests on. Both members now share
        # the wording and differ in exactly the site / scope token.
        field, prep = ("site", "from") if family == "locative" else ("scope", "under")
        # the answer must be UNIQUE for (owner, value), or the answerable member has several right answers
        bound = [x["item"] for uu in f["live"] if uu["owner"] == u["owner"] and uu[field] == u[field]
                 for x in uu["entries"]]
        if len(bound) != 1:
            return []
        # and the swapped value must give the SAME owner nothing, or the twin is answerable after all
        others = [v for v in _present(f, field) if v != u[field]
                  and not any(uu["owner"] == u["owner"] and uu[field] == v and uu["entries"]
                              for uu in f["docs"])]
        if not others:
            return []
        vb = VERB_BASE[D['verb']]
        q_ok = f"Which {D['noun']} did {u['owner']} {vb} {prep} {u[field]}?"
        q_no = f"Which {D['noun']} did {u['owner']} {vb} {prep} {rng.choice(others)}?"
        gold_ok = bound[0]
    else:                                             # item: held out of THIS context before rendering
        q_ok = f"Who {D['verb']} the {e['item']} {D['noun']}?"
        q_no = f"Who {D['verb']} the {held} {D['noun']}?"
        gold_ok = u["owner"]

    base = dict(context=ctx, target="", instruction=instruction)
    meta = dict(container=container, domain=dom, family=family)
    return [
        dict(example_id=f"mp-{family}-{dom}-{container}-{ci}-ans", question=q_ok, gold=gold_ok,
             meta=dict(meta, kind="answerable"), **base),
        dict(example_id=f"mp-{family}-{dom}-{container}-{ci}-abs", question=q_no, gold=ABSENT_GOLD,
             meta=dict(meta, kind="unanswerable"), **base),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs-per-cell", type=int, default=6)
    ap.add_argument("--containers", default="dialogue,transcript",
                    help="`dialogue` reproduces LoCoMo's own layout ([Session N | time on date] + two-speaker "
                         "turns) and is the default; the document containers are kept so the corpus is not "
                         "a single layout, which is how v8 was built")
    ap.add_argument("--families", default=",".join(FAMILIES),
                    help="which trap families to emit. ★ THE `item` FAMILY CARRIES A LEXICAL SHORTCUT: it "
                         "holds its item OUT of the context, so 'is every content word of the question in "
                         "the context?' separates its two sides 100.0%% vs 0.0%% — a perfect giveaway, on "
                         "half the corpus (144 of 289 pairs in v14). speaker/locative/scope keep the swapped "
                         "token PRESENT and separate 0.0 vs 0.0, which is the intended binding task. "
                         "Measured in 260813_LOCOMO10_DIAGNOSIS §7n; the default is left at all four so v14 "
                         "stays reproducible.")
    ap.add_argument("--seed", type=int, default=20260813)
    ap.add_argument("--instruction", default="QA_REASON_V3_LOCOMO_ABSTAIN",
                    help="the qa_prompts constant the TEACHER generates under, and therefore the prompt the "
                         "trained reader must be evaluated with. QA_REASON_V3 and QA_REASON_V3_LOCOMO both "
                         "forbid declining, so a teacher pass under either confabulates on every twin and "
                         "fails its own v6 gate.")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    containers = [c.strip() for c in a.containers.split(",") if c.strip()]
    families = [f.strip() for f in a.families.split(",") if f.strip()]
    bad = [f for f in families if f not in FAMILIES]
    if bad:
        raise SystemExit(f"[ABORT] unknown families {bad}; known: {list(FAMILIES)}")

    rows, ci = [], 0
    for dom in DOMAINS:
        for container in containers:
            for family in families:
                made = 0
                for _ in range(a.pairs_per_cell * 6):     # retries: plant() can refuse a layout
                    if made >= a.pairs_per_cell:
                        break
                    p = build_pair(rng, dom, container, family, ci, a.instruction)
                    if p:
                        rows.extend(p); ci += 1; made += 1
    rng.shuffle(rows)
    with open(a.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    n_abs = sum(1 for r in rows if r["meta"]["kind"] == "unanswerable")
    fam = {f: sum(1 for r in rows if r["meta"]["family"] == f) for f in families}
    print(f"[minimal-pair] {len(rows)} rows -> {a.out}")
    print(f"  answerable {len(rows) - n_abs} / unanswerable {n_abs}   (paired, same context, one slot)")
    print(f"  by family: {fam}")
    print(f"  containers: {containers}   domains: {list(DOMAINS)}")
    print("\nNEXT, in order — do not skip the gate:")
    print("  1. teacher targets UNDER THE ABSTAIN PROMPT:")
    print(f"       binding_teacher_gen.py --corpus {a.out} --instruction QA_REASON_V3_LOCOMO_ABSTAIN")
    print("  2. v6 GATE: the teacher must DECLINE on the unanswerable twins and be CORRECT on their")
    print("     answerable partners. A corpus of tasks the teacher cannot do regressed all four benchmarks.")
    print("  3. only then train, and evaluate under the same instruction the corpus was generated with.")


if __name__ == "__main__":
    main()
