#!/usr/bin/env python
"""
Score the CLUTRR-MultiQ experiment: teacher-14B / ours-14B+3B / floor-3B / snapKV(per-query,reuse), split by
Q1 (anchor) vs Q2+ (reused), with the relation-vocabulary EM (clean, no length pathology). Prints the paired
table on the SHARED example_ids (`<story_id>_<qi>`). Usage:
  python scripts/clutrr_multiq_score.py <snapkv.jsonl> <ours.jsonl> <floor3b.jsonl>
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clutrr_multiq_snapkv import rel_em, _extract


def qi_of(eid):
    return int(str(eid).split("_")[-1])


def load_snapkv(fp):
    """harness output has extracted + golds per (example_id, method)."""
    out = {}
    for l in open(fp):
        if not l.strip():
            continue
        r = json.loads(l)
        out.setdefault(r["method"], {})[str(r["example_id"])] = rel_em(r["extracted"], r["golds"])
    return out


def load_re(fp, field):
    """run_evidence output: generation in r[field]['text'], gold in r['answer']."""
    out = {}
    for l in open(fp):
        if not l.strip():
            continue
        r = json.loads(l)
        if not isinstance(r, dict) or "example_id" not in r:
            continue
        d = r.get(field) or {}
        ext = _extract(d.get("text", "") if isinstance(d, dict) else "")
        out[str(r["example_id"])] = rel_em(ext, [r.get("answer")])
    return out


def main(ra_reuse_fp, old_snapkv_fp, ours_fp, floor_fp):
    """RANDOM-ANCHOR design: snapkv_reuse (random anchor per story) ran ONLY on the non-anchor questions -> the
    reuse file's example_ids ARE the non-anchor (reused) set. Score every method on THAT set, so the anchor is no
    longer systematically the hardest question (fixes the Q1=target difficulty confound)."""
    reuse = load_snapkv(ra_reuse_fp).get("snapkv_reuse", {})
    nonanchor = set(reuse)                       # reused (non-anchor) questions only
    old = load_snapkv(old_snapkv_fp)             # teacher + per-query are anchor-independent (reuse them)
    methods = {
        "teacher-14B": old.get("teacher", {}),
        "ours 14B+3B": load_re(ours_fp, "slm_lm"),
        "floor-3B": load_re(floor_fp, "teacher"),
        "snapKV per-query": old.get("snapkv_perquery", {}),
        "snapKV reuse (rand-anchor)": reuse,
    }
    sets = [set(m) for m in methods.values() if m]
    shared = sorted(nonanchor & set.intersection(*sets)) if sets else []
    print(f"### CLUTRR-MultiQ (RANDOM ANCHOR) — REUSED (non-anchor) questions · shared-N={len(shared)} · relation-EM\n")
    print("| method | EM |")
    print("|---|--:|")
    def avg(m):
        return sum(m[e] for e in shared) / len(shared) if shared else 0.0
    rows = {name: avg(m) for name, m in methods.items()}
    for name in methods:
        print(f"| {name} | {rows[name]:.3f} |")
    pq, ru, tch, fl = rows["snapKV per-query"], rows["snapKV reuse (rand-anchor)"], rows["teacher-14B"], rows["floor-3B"]
    print(f"\nREUSE PENALTY: snapKV per-query {pq:.3f} -> reuse {ru:.3f}  (Δ {ru-pq:+.3f}); teacher {tch:.3f} (no penalty).")
    if tch - fl > 1e-6:
        print(f"ours closeness-to-teacher = {100*(rows['ours 14B+3B']-fl)/(tch-fl):.0f}%  (ours {rows['ours 14B+3B']:.3f}, floor {fl:.3f}, teacher {tch:.3f})")


def avg_main(ra_reuse_fp, old_snapkv_fp, ours_fp, floor_fp):
    """LoCoMo-format: Q1 anchor | Q2+ reused | AVG (weighted over ALL questions). The reuse file holds the
    NON-anchor (Q2+) questions; the anchor (Q1) set = the complement. teacher/ours/floor/snapKV-fresh are
    anchor-independent (their logs hold all questions). snapKV-FROZEN compresses ON the anchor, so its Q1 =
    snapKV-fresh's anchor score, its Q2+ = the reuse score; AVG = |Q1|·Q1 + |Q2+|·Q2+ all over the total."""
    import statistics as st
    ra = load_snapkv(ra_reuse_fp)                # memory-matched budget: snapkv_perquery + snapkv_reuse
    reuse = ra.get("snapkv_reuse", {})
    old = load_snapkv(old_snapkv_fp)             # b64 file: has teacher (budget-independent)
    teacher = old.get("teacher", {})
    perq = ra.get("snapkv_perquery") or old.get("snapkv_perquery", {})   # prefer the MATCHED-budget per-query
    ours, floor = load_re(ours_fp, "slm_lm"), load_re(floor_fp, "teacher")
    # question universe = ids present in the anchor-independent methods; anchor = universe − reused
    universe = set(teacher) & set(perq) & set(ours) & set(floor)
    reused = set(reuse) & universe
    anchor = universe - reused
    m = lambda d, ids: (st.mean(d[e] for e in ids if e in d) if any(e in d for e in ids) else float("nan"))
    def indep(d):  # anchor-independent method: Q1 on anchor, Q2+ on reused, AVG over all
        return (m(d, anchor), m(d, reused), m(d, universe))
    rows = {"teacher-14B": indep(teacher), "ours 14B+3B": indep(ours), "floor-3B": indep(floor),
            "snapKV fresh": indep(perq)}
    # snapKV FROZEN: Q1 = fresh-on-anchor, Q2+ = reuse-on-reused, AVG = weighted
    q1f, q2f = m(perq, anchor), m(reuse, reused)
    navg = (len(anchor) * q1f + len(reused) * q2f) / (len(anchor) + len(reused))
    rows["snapKV frozen"] = (q1f, q2f, navg)
    print(f"### CLUTRR-MultiQ · Q1 anchor vs Q2+ reused · relation-EM · "
          f"N_Q1(anchor)={len(anchor)} N_Q2+(reused)={len(reused)} N_all={len(universe)}\n")
    print(f"| method | Q1 anchor | Q2+ reused | AVG |\n|---|--:|--:|--:|")
    for name, (a, r, v) in rows.items():
        print(f"| {name} | {a:.3f} | {r:.3f} | {v:.3f} |")
    t, o, f = rows["teacher-14B"], rows["ours 14B+3B"], rows["floor-3B"]
    for j, lab in [(0, "Q1"), (1, "Q2+"), (2, "AVG")]:
        cl = 100 * (o[j] - f[j]) / (t[j] - f[j]) if abs(t[j] - f[j]) > 1e-9 else float("nan")
        print(f"closeness[{lab}] = {cl:.0f}%")


if __name__ == "__main__":
    DEFAULT = [
        "results/clutrr_multiq/clutrr_snapkv_reuseRA_s200_b64.jsonl",   # random-anchor reuse (non-anchor questions)
        "results/clutrr_multiq/clutrr_snapkv_s200_b64.jsonl",           # existing teacher + per-query (anchor-independent)
        "results/clutrr_multiqd0_s200_ours_q14b3B_l07_n1300.jsonl",
        "results/clutrr_multiqd0_s200_t_q3b_n1300.jsonl",
    ]
    a = [x for x in sys.argv[1:] if x != "--avg"]
    a = a if a else DEFAULT
    (avg_main if "--avg" in sys.argv else main)(*a)
