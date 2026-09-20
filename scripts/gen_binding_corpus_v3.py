#!/usr/bin/env python
"""v3 GENERIC accumulate-reading corpus (2026-08-10) — user requirement: remove the LoCoMo smell entirely
(no single context TEMPLATE), and mix with existing general training data.

v2 failed the no-regression test (CLUTRR −0.06 trend, musique −0.018) because its every context was a
LoCoMo-shaped dated-session dialogue → domain-specialized reader. v3 trains the same UNDERLYING reading
policy (locate the right instance among near-duplicates; derive from anchors; compose/aggregate across
units; respect recency) across FIVE context formats that share nothing but the skill:

  A dialogue_sessions — multi-speaker chats, 3 header style variants (the only LoCoMo-adjacent slice, 1/5)
  B meeting_notes     — dated bulletin lists ("Meeting notes — 12 March 2023 / Attendees / bullets")
  C email_thread      — dated emails (From/To/Date/Subject/body)
  D doc_collection    — titled dated mini-reports, NO dialogue (single-turn-QA-shaped)
  E journal           — first-person dated diary entries

  question ops (uniform across formats): when(derive from relative ref), where(extract),
  order(compose 2 units), count(aggregate same-family mentions), total-cost(numeric arithmetic across
  2 facts), recency-update(later unit supersedes earlier).

  + REPLAY: --replay <ftmix teacher-gen jsonl> mixes real general-QA training rows (documents+question,
  target = the teacher generation) so the reader's general QA behavior is anchored (user: "기존 학습데이터랑
  섞어서"). Replay rows carry instruction=QA_REASON_V3; synthetic dialogue rows QA_REASON_V3_LOCOMO;
  synthetic document-ish rows QA_REASON_V3. reader_binding_sft reads the per-row `instruction` field.

Pre-registered success criteria (BEFORE any run): LoCoMo gain retained (≥ well above plain 19%),
CLUTRR and musique NO-LOSS — and if the policy is truly generic, CLUTRR (floor 0.264, weak reading)
should IMPROVE. A LoCoMo-only gain again = still dataset-specific → reject the approach.
"""
import argparse, json, random, datetime

NAMES = ["Priya", "Marcus", "Elif", "Tomas", "Nadia", "Ryo", "Camille", "Dario", "Ingrid", "Femi",
         "Lucia", "Anders", "Zainab", "Mateo", "Hana", "Viktor", "Amara", "Jonas", "Selin", "Owen"]
EVENTS = [
    ("attended a {} class", ["pottery", "salsa", "watercolor", "fencing", "improv"], ["the community center", "the riverside studio", "the annex hall"]),
    ("booked a {} workshop", ["photography", "woodworking", "baking", "calligraphy"], ["the maker space", "the culinary school", "the arts collective"]),
    ("toured the {} museum", ["maritime", "railway", "aviation", "textile"], ["the north district", "the waterfront", "the old town"]),
    ("presented a {} demo", ["robotics", "mapping", "sensor", "drone"], ["the tech hub", "the main lab", "the client office"]),
    ("organized a {} fundraiser", ["bake-sale", "fun-run", "auction"], ["the school gym", "the church hall", "the park pavilion"]),
]
CHAT_FILLER = ["How have you been?", "Busy week, honestly.", "Did you see the news today?", "I missed it entirely.",
               "We should catch up properly soon.", "Definitely."]
NOTE_FILLER = ["Budget review deferred to next cycle.", "Facilities reported no incidents.",
               "Recruiting update: two interviews scheduled.", "No further business."]
def F(d): return f"{d.day} {d.strftime('%B')} {d.year}"
def FM(d): return f"{d.strftime('%B')} {d.year}"
REFS = [("yesterday", -1, F), ("last week", -7, FM), ("last month", -30, FM), ("earlier this morning", 0, F)]
_INF = [("attended", "attend"), ("booked", "book"), ("toured", "tour"), ("presented", "present"), ("organized", "organize")]
def inf(ev):
    for p, b in _INF: ev = ev.replace(p, b)
    return ev

def plant_facts(rng, n_units, n_people=3):
    ppl = rng.sample(NAMES, n_people)
    start = datetime.date(2021, 1, 1) + datetime.timedelta(days=rng.randrange(0, 900))
    dates, d = [], start
    for _ in range(n_units):
        d += datetime.timedelta(days=rng.randrange(3, 25)); dates.append(d)
    tpl, objs, locs = rng.choice(EVENTS)
    spk = rng.choice(ppl); obj = rng.choice(objs); loc = rng.choice(locs); cost = rng.randrange(20, 90) * 5
    gsi = rng.randrange(n_units // 2, n_units)
    ref, off, fmt = rng.choice(REFS)
    facts = dict(ppl=ppl, dates=dates, gold=dict(spk=spk, ev=tpl.format(obj), loc=loc, cost=cost, si=gsi,
                 ref=ref, when=fmt(dates[gsi] + datetime.timedelta(days=off))))
    dis = []
    for o in rng.sample([x for x in objs if x != obj], min(2, len(objs) - 1)):
        dsi = rng.choice([s for s in range(n_units) if s != gsi])
        dis.append(dict(spk=spk, ev=tpl.format(o), loc=rng.choice([l for l in locs if l != loc]),
                        cost=rng.randrange(20, 90) * 5, si=dsi))
    ev2tpl, ev2o, ev2l = rng.choice([e for e in EVENTS if e[0] != tpl])
    e2 = dict(spk=spk, ev=ev2tpl.format(rng.choice(ev2o)), loc=rng.choice(ev2l), cost=rng.randrange(20, 90) * 5,
              si=rng.choice([s for s in range(n_units) if s != gsi]))
    facts["dis"] = dis; facts["e2"] = e2
    return facts

def mention(fact, style, rng):
    s = "I" if style in ('journal', 'chat') else f"{fact['spk']}"
    base = f"{s} {fact['ev']} at {fact['loc']}"
    if 'ref' in fact: base += f" {fact['ref']}"
    if rng.random() < 0.5: base += f" (about ${fact['cost']} all in)"
    return base + "."

def render(fmt, facts, rng):
    units = []
    HDRS = [lambda i, d: f"[Session {i+1} | {F(d)}]", lambda i, d: f"=== Day {i+1} — {F(d)} ===",
            lambda i, d: f"Chat log {i+1} ({F(d)})"]
    hdr = rng.choice(HDRS)
    for si, d in enumerate(facts['dates']):
        lines = []
        mns = [dict(f, ref=facts['gold']['ref']) if f is facts['gold'] else f
               for f in [facts['gold']] + facts['dis'] + [facts['e2']] if f['si'] == si]
        if fmt == 'dialogue':
            a, b = facts['ppl'][0], facts['ppl'][1]
            for q in rng.sample(CHAT_FILLER[::2], 2):
                lines.append(f"{a}: {q}"); lines.append(f"{b}: {rng.choice(CHAT_FILLER[1::2])}")
            for m in mns: lines.insert(rng.randrange(len(lines) + 1), f"{m['spk']}: " + mention(m, 'chat', rng))
            units.append(hdr(si, d) + "\n" + "\n".join(lines))
        elif fmt == 'meeting':
            lines = [f"Meeting notes — {F(d)}", f"Attendees: {', '.join(facts['ppl'])}"]
            for m in mns: lines.append(f"- {mention(m, 'note', rng)}")
            lines += [f"- {x}" for x in rng.sample(NOTE_FILLER, 2)]
            units.append("\n".join(lines))
        elif fmt == 'email':
            frm, to = rng.sample(facts['ppl'], 2)
            body = " ".join(mention(m, 'mail', rng) for m in mns) or rng.choice(NOTE_FILLER)
            units.append(f"From: {frm}\nTo: {to}\nDate: {F(d)}\nSubject: quick update\n\n{body}")
        elif fmt == 'doc':
            body = " ".join(mention(m, 'doc', rng) for m in mns) or rng.choice(NOTE_FILLER)
            units.append(f"### Field report #{si+1} ({F(d)})\n{body}")
        else:  # journal
            body = " ".join(mention(m, 'journal', rng) for m in mns) or "Quiet day."
            units.append(f"Diary — {F(d)}\n{body}")
    out = "\n\n".join(units)
    if fmt == 'journal':
        out = f"Personal journal of {facts['gold']['spk']}.\n\n" + out
    return out

def questions(facts, rng):
    g = facts['gold']; e2 = facts['e2']; d = facts['dates']
    qs = [
        (f"When did {g['spk']} {inf(g['ev'])}?", g['when'],
         f"Step 1 (Reasoning): The unit dated {F(d[g['si']])} says {g['spk']} {g['ev']} '{g['ref']}', so it happened {g['when']}.\nStep 2 (Answer):\nFinal Answer: {g['when']}"),
        (f"Where did {g['spk']} {inf(g['ev'])}?", g['loc'],
         f"Step 1 (Reasoning): The entry dated {F(d[g['si']])} places it at {g['loc']}.\nStep 2 (Answer):\nFinal Answer: {g['loc']}"),
        (f"Which came first: {g['spk']} {inf(g['ev'])}, or {g['spk']} {inf(e2['ev'])}?",
         ("the former" if g['si'] < e2['si'] else "the latter"),
         f"Step 1 (Reasoning): The first is recorded on {F(d[g['si']])} and the second on {F(d[e2['si']])}.\nStep 2 (Answer):\nFinal Answer: {'the former' if g['si'] < e2['si'] else 'the latter'}"),
        (f"How much did {g['spk']} spend on {inf(g['ev'])} and {inf(e2['ev'])} combined?", f"${g['cost'] + e2['cost']}",
         f"Step 1 (Reasoning): The two amounts are ${g['cost']} and ${e2['cost']}, which sum to ${g['cost'] + e2['cost']}.\nStep 2 (Answer):\nFinal Answer: ${g['cost'] + e2['cost']}"),
    ]
    fam = g['ev'].split()[0]
    n_fam = 1 + sum(1 for x in facts['dis'] if x['ev'].split()[0] == fam)
    qs.append((f"How many different times did {g['spk']} {fam} something, according to these records?", str(n_fam),
               f"Step 1 (Reasoning): Counting every distinct '{fam} ...' mention by {g['spk']} across the records gives {n_fam}.\nStep 2 (Answer):\nFinal Answer: {n_fam}"))
    return qs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-format", type=int, default=120)
    ap.add_argument("--replay", default="")
    ap.add_argument("--replay-n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=55)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    n = 0
    with open(a.out, "w") as f:
        for fmt in ("dialogue", "meeting", "email", "doc", "journal"):
            for ci in range(a.per_format):
                facts = plant_facts(rng, rng.randrange(10, 22))
                ctx = render(fmt, facts, rng)
                instr = "QA_REASON_V3_LOCOMO" if fmt == "dialogue" else "QA_REASON_V3"
                for qi, (q, gold, tgt) in enumerate(questions(facts, rng)):
                    f.write(json.dumps(dict(example_id=f"v3-{fmt}-{ci}-{qi}", context=ctx, question=q,
                                            gold=gold, target=tgt, instruction=instr,
                                            meta=dict(fmt=fmt))) + "\n"); n += 1
        if a.replay:
            rows = [json.loads(l) for l in open(a.replay)]
            rows = [r for r in rows if r.get("keep")]
            rng.shuffle(rows)
            for r in rows[: a.replay_n]:
                f.write(json.dumps(dict(example_id=f"replay-{r['example_id']}", context="\n\n".join(r["documents"]),
                                        question=r["question"], gold=(r.get("golds") or [""])[0],
                                        target=r["teacher_text"], instruction="QA_REASON_V3",
                                        meta=dict(fmt="replay", source=r.get("source")))) + "\n"); n += 1
    print(f"[gen-v3] wrote {n} examples -> {a.out}")

if __name__ == "__main__":
    main()
