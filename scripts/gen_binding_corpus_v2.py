#!/usr/bin/env python
"""v2 GENERAL conversational-memory corpus (2026-08-09) — the generalization §9a demands before any claim.

v1 was a single-shape probe (100% "When did X ...?", LoCoMo-matched gold phrasing) — enough for the
feasibility claim, not for a method claim, and fully learnable by the 7B (fresh-seed loss 0.001 → stage-2
vacuous). v2 samples the GENERAL task family with difficulty the reader cannot template away:

  question types per conversation (10 Qs):
    4x temporal-when   (relative refs resolved against session dates; PLAIN answer formats, NOT
                        LoCoMo-phrased: "7 May 2023" / "May 2023" / "2022")
    2x order/multi-hop ("What did X do first/last, E1 or E2?", "Where was X's E mentioned?")
    2x entity/list     ("Which <attr> did X mention for E?"; "List every event X mentioned" — union)
    1x yes/no          ("Did both A and B mention E-type events?")
    1x recency-update  ("What is X's current <state>?" — a later session SUPERSEDES an earlier one)
  difficulty: same event TEMPLATE reused across sessions with different objects/attributes (binding must
  use the full qualifier), state-update pairs, 16–26 sessions, varied reasoning-target wording (4 variants).

Held-out: all names/templates are this file's own; no LoCoMo/eval text. Deterministic via --seed.
Row schema identical to v1: {example_id, context, question, gold, target, meta}.
"""
import argparse, json, random, datetime

NAMES = ["Priya", "Marcus", "Elif", "Tomas", "Nadia", "Ryo", "Camille", "Dario", "Ingrid", "Femi",
         "Lucia", "Anders", "Zainab", "Mateo", "Hana", "Viktor", "Amara", "Jonas", "Selin", "Owen"]
# (template, objects, locations) — location doubles as the entity attribute
EVENTS = [
    ("went to a {} class", ["pottery", "salsa", "watercolor", "fencing", "improv"],
     ["the community center", "the riverside studio", "the old library annex"]),
    ("signed up for a {} workshop", ["photography", "woodworking", "baking", "calligraphy"],
     ["the maker space", "the culinary school", "the arts collective"]),
    ("ran a charity {}", ["5K", "10K", "half-marathon"],
     ["the harbor park", "the botanical gardens", "the university track"]),
    ("visited the {} museum", ["maritime", "railway", "aviation", "textile"],
     ["the north district", "the waterfront", "the capital"]),
    ("performed a {} song at the open mic", ["folk", "jazz", "blues"],
     ["the corner cafe", "the jazz cellar", "the student union"]),
    ("baked a {} cake for the bake sale", ["carrot", "lemon", "marble"],
     ["the school fair", "the church hall", "the office party"]),
]
# recency-update states: (topic, [state progression])
STATES = [
    ("instrument", ["learning the guitar", "switching to the ukulele", "back to the guitar seriously"]),
    ("diet", ["trying a vegetarian diet", "doing a strict vegan month", "settling on pescatarian"]),
    ("job project", ["leading the onboarding project", "moved to the billing migration", "heading the analytics revamp"]),
    ("apartment", ["repainting the kitchen", "renovating the balcony", "redoing the whole living room"]),
]
FILLER = [
    "How's work been treating you lately?", "Pretty hectic, but I'm managing.",
    "Did you catch the game last night?", "No, I completely forgot it was on!",
    "The weather has been so strange this week.", "Tell me about it, I never know what to wear.",
    "Have you talked to your sister recently?", "Yes, she says hi by the way.",
    "I've been trying to cook more at home.", "That's a great habit, honestly.",
    "My commute was terrible this morning.", "Again? You should try the other route.",
    "I'm thinking about repainting the living room.", "Ooh, what color?",
    "We should grab coffee sometime soon.", "Definitely, let's plan for it.",
]
def _fmt_full(d): return f"{d.day} {d.strftime('%B')} {d.year}"
def _fmt_month(d): return f"{d.strftime('%B')} {d.year}"
REFS = [("yesterday", -1, _fmt_full), ("last week", -7, _fmt_full), ("last month", -30, _fmt_month),
        ("last year", -365, lambda d: str(d.year)), ("this morning", 0, _fmt_full)]
_INF = [("went to", "go to"), ("visited", "visit"), ("ran", "run"), ("signed up", "sign up"),
        ("performed", "perform"), ("baked", "bake")]
def inf(ev):
    for p, b in _INF: ev = ev.replace(p, b)
    return ev
R_TPL = [
    "In Session {k} dated {d}, {s} mentioned this saying '{ref}', which resolves to {g}.",
    "The mention is in Session {k} (dated {d}); '{ref}' relative to that date gives {g}.",
    "{s} brought this up in Session {k} on {d} using '{ref}', so the date is {g}.",
    "Session {k} is dated {d} and {s} said '{ref}' there, so this happened {g}.",
]

def gen_conversation(rng, n_sessions):
    a, b = rng.sample(NAMES, 2)
    start = datetime.date(2021, 1, 1) + datetime.timedelta(days=rng.randrange(0, 900))
    dates, d = [], start
    for _ in range(n_sessions):
        d += datetime.timedelta(days=rng.randrange(3, 35)); dates.append(d)
    mentions, states, blocks, used = [], [], [], set()
    # plan one state-progression for one speaker across the conversation
    st_topic, st_seq = rng.choice(STATES); st_spk = rng.choice([a, b])
    st_sessions = sorted(rng.sample(range(n_sessions), min(len(st_seq), max(2, n_sessions // 7))))
    for si, sd in enumerate(dates):
        hh, mm = rng.randrange(8, 21), rng.choice([0, 15, 30, 45])
        ampm = "am" if hh < 12 else "pm"; h12 = hh if hh <= 12 else hh - 12
        header = f"[Session {si+1} | {h12}:{mm:02d} {ampm} on {sd.day} {sd.strftime('%B')}, {sd.year}]"
        lines = []
        for pi in rng.sample(range(0, len(FILLER) - 1, 2), rng.randrange(5, 8)):
            lines.append(f"{a}: {FILLER[pi]}"); lines.append(f"{b}: {FILLER[pi+1]}")
        for _ in range(rng.randrange(1, 3)):
            tpl, objs, locs = rng.choice(EVENTS); obj = rng.choice(objs); loc = rng.choice(locs)
            ev = tpl.format(obj); spk = rng.choice([a, b])
            if (spk, ev) in used: continue
            used.add((spk, ev))
            ref, off, fmt = rng.choice(REFS)
            when = sd + datetime.timedelta(days=off)
            lines.insert(rng.randrange(0, len(lines) + 1),
                         f"{spk}: I {ev} at {loc} {ref} — honestly a highlight of my week.")
            mentions.append(dict(speaker=spk, event=ev, obj=obj, loc=loc, session=si + 1,
                                 sdate=sd, ref=ref, gold_date=fmt(when), order_key=(si, len(mentions))))
        if si in st_sessions:
            stage = st_sessions.index(si)
            if stage < len(st_seq):
                lines.insert(rng.randrange(0, len(lines) + 1),
                             f"{st_spk}: Update on my {st_topic}: I'm {st_seq[stage]} these days.")
                states.append(dict(speaker=st_spk, topic=st_topic, state=st_seq[stage], session=si + 1, sdate=sd))
        blocks.append(header + "\n" + "\n".join(lines))
    return a, b, mentions, states, "\n\n".join(blocks)

def make_questions(rng, a, b, mentions, states):
    qs = []
    def rt(m, g):
        return rng.choice(R_TPL).format(k=m['session'], d=_fmt_full(m['sdate']), s=m['speaker'], ref=m['ref'], g=g)
    # temporal-when (plain formats, no LoCoMo phrasing): 'last week' resolves to the session date's week -> answer the session date month/day form
    for m in rng.sample(mentions, min(4, len(mentions))):
        g = m['gold_date'] if m['ref'] != 'last week' else _fmt_month(m['sdate'])
        qs.append((f"When did {m['speaker']} {inf(m['event'])}?", g,
                   f"Step 1 (Reasoning): {rt(m, g)}\nStep 2 (Answer):\nFinal Answer: {g}"))
    # order multi-hop
    spk_ms = {}
    for m in mentions: spk_ms.setdefault(m['speaker'], []).append(m)
    for spk, ms in list(spk_ms.items()):
        if len(ms) >= 2 and len(qs) < 6:
            m1, m2 = ms[0], ms[-1]
            if m1['session'] == m2['session']: continue
            q = f"Which did {spk} mention first: the time they {inf(m1['event'])}, or the time they {inf(m2['event'])}?"
            g = f"they {inf(m1['event'])}"
            tgt = (f"Step 1 (Reasoning): {spk} mentioned the first in Session {m1['session']} and the second in "
                   f"Session {m2['session']}; Session {m1['session']} comes earlier.\nStep 2 (Answer):\nFinal Answer: {g}")
            qs.append((q, g, tgt))
    # entity (location attribute)
    for m in rng.sample(mentions, min(2, len(mentions))):
        q = f"Where did {m['speaker']} say they {m['event']}?"
        g = m['loc']
        qs.append((q, g, f"Step 1 (Reasoning): In Session {m['session']}, {m['speaker']} said they {m['event']} "
                         f"at {g}.\nStep 2 (Answer):\nFinal Answer: {g}"))
    # yes/no: did both speakers mention any event at all of a given template family?
    if mentions:
        fam = rng.choice(mentions)['event'].split()[0]
        got = {m['speaker'] for m in mentions if m['event'].startswith(fam)}
        g = "Yes" if {a, b} <= got else "No"
        qs.append((f"Did both {a} and {b} mention that they {fam} somewhere in these conversations?", g,
                   f"Step 1 (Reasoning): Checking every session, the speakers who mention '{fam} ...' are "
                   f"{', '.join(sorted(got)) or 'neither'}.\nStep 2 (Answer):\nFinal Answer: {g}"))
    # recency-update
    if len(states) >= 2:
        s_last = states[-1]
        q = f"What is {s_last['speaker']}'s most recent update about their {s_last['topic']}?"
        g = s_last['state']
        tgt = (f"Step 1 (Reasoning): {s_last['speaker']} gave updates in sessions "
               f"{', '.join(str(s['session']) for s in states)}; the LATEST is Session {s_last['session']} "
               f"dated {_fmt_full(s_last['sdate'])}, which supersedes the earlier ones.\n"
               f"Step 2 (Answer):\nFinal Answer: {g}")
        qs.append((q, g, tgt))
    return qs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-conversations", type=int, default=300)
    ap.add_argument("--min-sessions", type=int, default=16)
    ap.add_argument("--max-sessions", type=int, default=26)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--out", required=True)
    a_ = ap.parse_args()
    rng = random.Random(a_.seed)
    n = 0
    with open(a_.out, "w") as f:
        for ci in range(a_.n_conversations):
            a, b, mentions, states, text = gen_conversation(rng, rng.randrange(a_.min_sessions, a_.max_sessions + 1))
            for qi, (q, g, tgt) in enumerate(make_questions(rng, a, b, mentions, states)):
                f.write(json.dumps(dict(example_id=f"bindv2-{ci}-{qi}", context=text, question=q, gold=g,
                                        target=tgt, meta=dict(n_sessions=text.count("[Session")))) + "\n")
                n += 1
    print(f"[gen-v2] wrote {n} examples -> {a_.out}")

if __name__ == "__main__":
    main()
