#!/usr/bin/env python
"""v8 STRUCTURAL-REFERENT corpus (2026-08-10) — built from a CAPABILITY definition, not from a benchmark.

THE SENTENCE THAT JUSTIFIES THIS CORPUS (it names no benchmark, and the corpus is the grid it implies):

    In a context made of MULTIPLE DELIMITED DOCUMENTS, each carrying a metadata header, resolve a LOCAL
    expression in a document's body against a referent that lives in that document's STRUCTURE — its
    header fields, its position, or a neighbouring document — rather than in the sentence itself.

A small reader reads sentences locally and returns the expression unresolved ("the day before", "there",
"the same as above", "I"). The fusion LM cannot supply the referent because it never sees the context.
The gap is general and appears in any multi-document setting: memos, changelogs, tickets, lab notebooks,
minutes, spec revisions, call transcripts.

    GRID = 7 reference types x 7 document containers x 4 domains

    reference types   reltime  "the day before"          -> the header's date field
                      speaker  "I / we shipped it"       -> the header's author/chair/assignee field
                      locative "there / on site"         -> the header's location field
                      unit     "it came to 42"           -> the unit declared once in the bundle preamble
                      ordinal  "entry 2"                 -> position within the document
                      ellipsis "the same as the gateway" -> a value stated in a neighbouring document
                      scope    "this revision"           -> the version/period named in the header

    containers        memo · changelog · notebook · ticket · minutes · spec · transcript

WHY THE GRID, AND WHAT IS DELIBERATELY NOT IN IT. v5-v7 rendered everything as `[Session N | date]`
two-friends chat — the target benchmark's own serialization — and the generality ablation showed that slice
was load-bearing, so the corpus could fairly be read as "the benchmark with the names changed". Here the
conversational container is `transcript`, ONE of seven, formatted as a record with a metadata block rather
than as a chat session; relative time is ONE reference type of seven and is not special-cased. Every cell is
filled independently of any benchmark, which is what makes "why is this in your training set?" answerable
without naming one.

`--holdout-container` writes the named container's rows to a separate file instead of the training file, so
the same generator produces the leave-one-container-out training corpus AND a probe set that measures the
capability directly, in a container the reader never trained on.

NO GOLD IS EVER TRAINED ON — `gold` is diagnostic metadata; targets come from the pair's own teacher via
scripts/binding_teacher_gen.py.

Usage:
  gen_binding_corpus_v8.py --per-cell 12 --out results/fusionft/binding_corpus_v8_struct.jsonl
  gen_binding_corpus_v8.py --per-cell 12 --holdout-container spec \
      --out .../v8_train_no_spec.jsonl --holdout-out .../v8_probe_spec.jsonl
"""
import argparse
import datetime
import json
import random

REF_TYPES = ("reltime", "speaker", "locative", "unit", "ordinal", "ellipsis", "scope")
CONTAINERS = ("memo", "changelog", "notebook", "ticket", "minutes", "spec", "transcript")

NAMES = ["Priya", "Marcus", "Elif", "Tomas", "Nadia", "Ryo", "Camille", "Dario", "Ingrid", "Femi",
         "Lucia", "Anders", "Zainab", "Mateo", "Hana", "Viktor", "Amara", "Jonas", "Selin", "Owen",
         "Kwame", "Sofia", "Aleks", "Mira", "Tariq", "Yuki", "Bea", "Noor", "Emil", "Rosa"]

DOMAINS = {
    "releases":  dict(items=["parser", "scheduler", "indexer", "gateway", "renderer", "cache"],
                      noun="release", verb="shipped",
                      sites=["Frankfurt", "Oslo", "Toronto", "Pune"],
                      unit="MB", scope_name="version", scopes=["v3.1", "v3.2", "v4.0", "v4.1"]),
    "incidents": dict(items=["disk", "network", "auth", "queue", "billing", "search"],
                      noun="incident", verb="handled",
                      sites=["the primary region", "the failover region", "the edge tier", "the batch tier"],
                      unit="minutes", scope_name="sprint", scopes=["sprint 8", "sprint 9", "sprint 10"]),
    "holdings":  dict(items=["lathe", "microscope", "plotter", "centrifuge", "kiln", "scanner"],
                      noun="unit", verb="purchased",
                      sites=["the north depot", "the annex store", "the main workshop", "the field office"],
                      unit="EUR", scope_name="budget round", scopes=["round A", "round B", "round C"]),
    "studies":   dict(items=["soil", "canopy", "aquifer", "sediment", "pollen", "runoff"],
                      noun="survey", verb="ran",
                      sites=["site 4", "the upper transect", "the delta plot", "the control plot"],
                      unit="samples", scope_name="season", scopes=["the spring season", "the autumn season"]),
}
FILLER = ["No blockers to report.", "Follow-ups carried over.", "Checklist reviewed, nothing outstanding.",
          "Notes circulated afterwards.", "Distribution list unchanged."]

HEAD = {
    "memo": lambda D, u: (f"INTERNAL MEMO {u['idx']+1:03d}\nTo: distribution\nFrom: {u['owner']}\n"
                          f"Date: {F(u['date'])}\nSite: {u['site']}\n"
                          f"{D['scope_name'].title()}: {u['scope']}"),
    "changelog": lambda D, u: (f"## {D['scope_name'].title()} {u['scope']} — released {F(u['date'])}\n"
                               f"Maintainer: {u['owner']} · Environment: {u['site']}"),
    "notebook": lambda D, u: (f"Notebook page {u['idx']+1}\nDate: {F(u['date'])}\nStation: {u['site']}\n"
                              f"Recorded by: {u['owner']}\n{D['scope_name'].title()}: {u['scope']}"),
    "ticket": lambda D, u: (f"[TICKET-{1000+u['idx']}] opened {F(u['date'])}\n"
                            f"Assignee: {u['owner']} | Component location: {u['site']} | "
                            f"{D['scope_name'].title()}: {u['scope']}"),
    "minutes": lambda D, u: (f"MINUTES OF MEETING No. {u['idx']+1}\nHeld: {F(u['date'])}\n"
                             f"Venue: {u['site']}\nChair: {u['owner']}\n"
                             f"{D['scope_name'].title()} under review: {u['scope']}"),
    "spec": lambda D, u: (f"SPECIFICATION SHEET — revision {u['scope']}\nEffective: {F(u['date'])}\n"
                          f"Owner: {u['owner']}\nApplies at: {u['site']}"),
    "transcript": lambda D, u: (f"TRANSCRIPT OF RECORDED CALL {u['idx']+1}\n"
                                f"Recorded: {F(u['date'])} · From: {u['site']}\n"
                                f"Speaking: {u['owner']}\n{D['scope_name'].title()}: {u['scope']}"),
}


def F(d):
    return f"{d.day} {d.strftime('%B')} {d.year}"


def plant(rng, dom, n_docs):
    """Each ITEM appears in exactly ONE document, so every structural question has a unique answer.
    Difficulty is the number of documents to scan, never ambiguity."""
    D = DOMAINS[dom]
    people = rng.sample(NAMES, 4)
    d = datetime.date(2021, 1, 1) + datetime.timedelta(days=rng.randrange(0, 900))
    docs = []
    for i in range(n_docs):
        d += datetime.timedelta(days=rng.randrange(3, 25))
        docs.append(dict(idx=i, date=d, site=rng.choice(D["sites"]), scope=rng.choice(D["scopes"]),
                         owner=rng.choice(people), entries=[]))
    items = list(D["items"])
    rng.shuffle(items)
    n_hosts = rng.randrange(3, 5)
    hosts = rng.sample(range(n_docs), n_hosts)
    for k, it in enumerate(items):
        docs[hosts[k % n_hosts]]["entries"].append(dict(item=it, num=rng.randrange(12, 400)))
    live = [u for u in docs if u["entries"]]
    multi = [u for u in docs if len(u["entries"]) >= 2]
    if len(live) < 3 or not multi:
        return None
    return dict(dom=dom, people=people, docs=docs, live=live, multi=multi)


def entry_text(D, e, kind, rel_phrase=None, ell_item=None):
    """The body sentence. The actor is ALWAYS first person and the referent is ALWAYS non-local: if the
    body named the actor / place / date, the question would be answerable off the line itself."""
    base = f"I {D['verb']} the {e['item']} {D['noun']}"
    if kind == "locative":
        return base + " there."
    if kind == "unit":
        return base + f"; it came to {e['num']}."
    if kind == "scope":
        return base + f" as part of this {D['scope_name']}."
    if kind == "reltime":
        return base + f" {rel_phrase}."
    if kind == "ellipsis":
        return base + f"; the same as the {ell_item}."
    return base + "."


def render(container, f, rng, ref, target_entry=None, rel_phrase=None, ell_item=None):
    D = DOMAINS[f["dom"]]
    blocks = []
    for u in f["docs"]:
        lines = []
        for i, e in enumerate(u["entries"]):
            if ref == "reltime":
                kind = "reltime" if e is target_entry else "plain"
            elif ref == "ellipsis":
                kind = "ellipsis" if e is target_entry else "unit"
            else:
                kind = ref
            lines.append(f"{i+1}. " + entry_text(D, e, kind, rel_phrase, ell_item))
        if not lines:
            lines = [rng.choice(FILLER)]
        blocks.append(HEAD[container](D, u) + "\n" + "\n".join(lines) + "\n———")
    pre = ""
    if ref in ("unit", "ellipsis"):
        pre = (f"Bundle note: throughout this file, the figure quoted after an entry is that entry's value "
               f"in {D['unit']}.\n\n")
    return pre + "\n\n".join(blocks)


def make(rng, dom, container, ref, ci):
    f = plant(rng, dom, rng.randrange(12, 26))
    if f is None:
        return []
    D = DOMAINS[dom]
    u = rng.choice(f["live"])
    e = rng.choice(u["entries"])
    off = rel_phrase = ell_src = ell_item = None
    if ref == "reltime":
        off, rel_phrase = rng.choice([(-1, "the day before"), (-7, "the week before"),
                                      (0, "that same morning"), (-30, "the month before")])
    if ref == "ellipsis":
        others = [x for uu in f["live"] for x in uu["entries"] if x is not e]
        if not others:
            return []
        ell_src = rng.choice(others)
        ell_item = ell_src["item"]
    ctx = render(container, f, rng, ref, target_entry=e, rel_phrase=rel_phrase, ell_item=ell_item)

    if ref == "reltime":
        q = f"On what date did the {e['item']} {D['noun']} take place?"
        gold = F(u["date"] + datetime.timedelta(days=off))
    elif ref == "speaker":
        q = f"Who {D['verb']} the {e['item']} {D['noun']}?"
        gold = u["owner"]
    elif ref == "locative":
        q = f"Where was the {e['item']} {D['noun']} {D['verb']}?"
        gold = u["site"]
    elif ref == "unit":
        q = f"What value is recorded for the {e['item']} {D['noun']}, including its unit?"
        gold = f"{e['num']} {D['unit']}"
    elif ref == "scope":
        q = f"Which {D['scope_name']} does the {e['item']} {D['noun']} belong to?"
        gold = u["scope"]
    elif ref == "ordinal":
        um = rng.choice(f["multi"])
        k = rng.randrange(len(um["entries"]))
        q = f"In the document dated {F(um['date'])}, which {D['noun']} is described in entry {k+1}?"
        gold = um["entries"][k]["item"]
    else:
        q = f"What value applies to the {e['item']} {D['noun']}?"
        gold = str(ell_src["num"])
    return [dict(example_id=f"v8-{ref}-{dom}-{container}-{ci}", context=ctx, question=q, gold=gold,
                 target="", instruction="QA_REASON_V3",
                 meta=dict(container=container, domain=dom, reftype=ref))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=12)
    ap.add_argument("--seed", type=int, default=880)
    ap.add_argument("--out", required=True)
    ap.add_argument("--holdout-container", default=None,
                    help="write this container's rows to --holdout-out instead of --out")
    ap.add_argument("--holdout-out", default=None)
    a = ap.parse_args()
    if a.holdout_container and not a.holdout_out:
        raise SystemExit("--holdout-container requires --holdout-out")
    rng = random.Random(a.seed)
    n = h = 0
    fh = open(a.holdout_out, "w") if a.holdout_out else None
    with open(a.out, "w") as fo:
        for dom in DOMAINS:
            for container in CONTAINERS:
                for ref in REF_TYPES:
                    for ci in range(a.per_cell):
                        for r in make(rng, dom, container, ref, ci):
                            if container == a.holdout_container:
                                fh.write(json.dumps(r) + "\n"); h += 1
                            else:
                                fo.write(json.dumps(r) + "\n"); n += 1
    if fh:
        fh.close()
        print(f"[gen-v8] HELD OUT container '{a.holdout_container}': {h} rows -> {a.holdout_out}")
    print(f"[gen-v8] {n} rows -> {a.out}  "
          f"({len(DOMAINS)} domains x {len(CONTAINERS)} containers x {len(REF_TYPES)} reference types)")


if __name__ == "__main__":
    main()
