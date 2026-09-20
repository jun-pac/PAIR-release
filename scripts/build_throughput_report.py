#!/usr/bin/env python
"""The throughput study, rebuilt 2026-08-28 on the TOTAL wall (task A4.1, HANDOFF_260827).

Everything the previous version of this page showed in §2/§2a/§2b/§5 was WITHDRAWN: those rates
divided generated tokens by the answer wall only — the context prefill sat in a separate field and
the per-turn history commit sat outside every timer — deleting exactly the axis the method wins on.
This rebuild reads every rate through `scripts/throughput_eval.py`'s own `measure()` (the single
sanctioned timing scorer, which refuses logs without the one-timer `conv_wall_s`) and renders BOTH
axes with their basis in the header. Sections whose measurement has not landed yet say so instead of
carrying an old number.

Inputs: results/fusionft/{lcw,lgw,must,host}_*_b*.jsonl (gate-validated one-node runs),
results/timing/lcf_accuracy.json + loogle_accuracy.json (canonical shared-N F1),
results/timing/flash_kvcache_probe_*.json, results/timing/kernels_*.json.
Output: reports/throughput.html
"""
from __future__ import annotations

import glob
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from throughput_eval import load, measure                                   # noqa: E402

P = []
A = P.append

E2E_TOK = ("gen tok/s END-TO-END<br><span style='text-transform:none;letter-spacing:0'>"
           "batch-aggregate, total wall</span>")
DEC_TOK = ("gen tok/s DECODE<br><span style='text-transform:none;letter-spacing:0'>"
           "batch-aggregate, answer wall</span>")

CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Chivo:wght@500;600;700&family=JetBrains+Mono:wght@400;500;700&family=Literata:opsz,wght@7..72,400;7..72,600&display=swap');
:root{
  --bg:#f3f5f7; --panel:#fff; --ink:#14181d; --mut:#5a656f; --line:#dde3e8;
  --meas:#26597f; --roof:#b06a12; --bad:#9b2c39; --good:#2b7150; --chip:#e8eef3;
  --fsnap:#153e63; --fexp:#c8891f; --fspec:#b05585; --fours:#22754a;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#101418; --panel:#171d23; --ink:#e6ebef; --mut:#94a2ad; --line:#28313a;
  --meas:#6fb2e6; --roof:#e0a355; --bad:#e0737f; --good:#5fbf90; --chip:#1d262e;
  --fsnap:#6fb2e6; --fexp:#e0a355; --fspec:#e08ab8; --fours:#57c48c;
}}
:root[data-theme="dark"]{
  --bg:#101418; --panel:#171d23; --ink:#e6ebef; --mut:#94a2ad; --line:#28313a;
  --meas:#6fb2e6; --roof:#e0a355; --bad:#e0737f; --good:#5fbf90; --chip:#1d262e;
  --fsnap:#6fb2e6; --fexp:#e0a355; --fspec:#e08ab8; --fours:#57c48c;
}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);margin:0;font-family:Literata,Georgia,serif;
     font-size:17px;line-height:1.6}
main{max-width:1020px;margin:0 auto;padding:56px 24px 96px;display:flex;flex-direction:column;gap:24px}
h1{font-family:Chivo,system-ui,sans-serif;font-weight:700;font-size:2.05rem;line-height:1.14;margin:0;
   letter-spacing:-.02em;text-wrap:balance}
h2{font-family:Chivo,sans-serif;font-weight:600;font-size:1.2rem;margin:26px 0 0;padding-bottom:7px;
   border-bottom:2px solid var(--line)}
p{margin:0;max-width:70ch}
.lede{color:var(--mut);font-size:1.05rem;max-width:72ch}
.box{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--meas);
     padding:14px 18px;border-radius:0 6px 6px 0}
.box.warn{border-left-color:var(--bad)}
.box.win{border-left-color:var(--good)}
.box b.tag{font-family:Chivo,sans-serif;font-size:.7rem;letter-spacing:.09em;text-transform:uppercase;
       display:block;color:var(--mut);margin-bottom:5px}
.wrap{overflow-x:auto;background:var(--panel);border:1px solid var(--line);border-radius:8px}
table{border-collapse:collapse;width:100%;font-family:"JetBrains Mono",ui-monospace,monospace;font-size:.82rem}
th,td{padding:7px 12px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}
th{font-weight:600;color:var(--mut);font-size:.71rem;letter-spacing:.05em;text-transform:uppercase}
td.n{text-align:right;font-variant-numeric:tabular-nums}
td.m{text-align:right;font-variant-numeric:tabular-nums;color:var(--meas);font-weight:600}
td.r{text-align:right;font-variant-numeric:tabular-nums;color:var(--roof)}
td.b{text-align:right;font-variant-numeric:tabular-nums;color:var(--bad);font-weight:600}
td.g{text-align:right;font-variant-numeric:tabular-nums;color:var(--good);font-weight:600}
tr:last-child td{border-bottom:none}
tr.sep td{border-top:2px solid var(--line)}
tr.hero td{background:var(--chip)}
code{font-family:"JetBrains Mono",monospace;font-size:.86em;background:var(--chip);padding:1px 5px;border-radius:3px}
.foot{color:var(--mut);font-size:.82rem;border-top:1px solid var(--line);padding-top:16px;max-width:none}
ol{max-width:70ch;padding-left:22px}
li{margin-bottom:10px}
li b{font-family:Chivo,sans-serif}
</style>"""

_TOK = None



def _queue_line():
    """What is ACTUALLY queued or running right now, asked of SLURM at build time.

    Added 2026-08-31: the page said a family "needs the same re-run" with no way for a reader to
    tell whether that re-run existed. Naming job IDs in prose does not survive a requeue — seven
    IDs quoted in the sibling ledger were cancelled within a day of being written — so the state is
    read rather than written."""
    import subprocess
    try:
        q = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%j %t"],
                           capture_output=True, text=True, timeout=20)
        rows = [l.split() for l in (q.stdout or "").strip().split("\n") if l.strip()]
    except Exception:
        return ""
    if not rows:
        return ('<br><b>Nothing is queued for these re-runs at the moment.</b> They are sequenced '
                'one benchmark at a time rather than run in parallel, so a bench is finished before '
                'the next is started.')
    names = ", ".join(f"<code>{n}</code> ({'running' if t == 'R' else 'queued'})" for n, t in rows)
    return ('<br><b>In the queue as this page was built:</b> ' + names +
            '. Re-runs are sequenced one benchmark at a time, so a family not named here is waiting '
            'on the bench ahead of it, not abandoned.')

def tok():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct",
                                             cache_dir=os.environ.get("HF_HOME"))
    return _TOK


def parse_tag(tag):
    """tag -> (family, kept-fraction, display) for both tag generations.
    New sweep tags: snap400/600/781/900/950 (ratio*1000, kept=1-ratio), expected*, spec40000/21875/
    10000/05000 (keep*100000). Old tags: snap40/snap60/snap78125..., spec40/spec21875/spec05,
    snap219 (kept-based, LooGLE/singleturn)."""
    import re as _re
    for fam, pat in (("snap", r"snap(\d+)"), ("expected", r"expected(\d+)"), ("spec", r"spec(\d+)")):
        m = _re.fullmatch(pat, tag)
        if not m:
            continue
        d = m.group(1)
        if fam == "spec":
            keep = int(d) / (10 ** len(d))                 # spec40000->0.40, spec05000->0.05
            kept = keep
        else:
            v = int(d) / (10 ** len(d))                    # snap400->0.40 snap78125->0.78125
            # kept-based legacy tags: snap219 (=21.9% kept), snap40-on-loogle... ratio tags are the
            # sweep generation; legacy kept-tags only appear with the legacy prefixes, which pass
            # kept=True below via the bench spec.
            kept = 1.0 - v
        lab = {"snap": "snapKV", "expected": "ExpectedAttn", "spec": "SpecPrefill"}[fam]
        return fam, kept, f"{lab} {100*kept:.4g}% kept" if fam != "spec" else f"{lab} keep {kept:.4g}"
    return {"ours": ("ours", None, "ours 32B+7B"), "teacher": ("teacher", None, "teacher-32B"),
            "floor7": ("floor", None, "floor-7B")}.get(tag, (None, None, tag))


def collect(prefix, tags=None):
    """Gate-measure every {prefix}_{tag}[_b*].jsonl (skipping *_ACC), parse tags automatically."""
    out = {}
    for f in sorted(glob.glob(f"results/fusionft/{prefix}_*.jsonl")):
        if not os.path.getsize(f) or f.endswith("_ACC.jsonl") or f.endswith("_curve.md"):
            continue
        tag = os.path.basename(f)[len(prefix) + 1:-6]
        tag = tag.split("_b")[0] if "_b" in tag and tag.split("_b")[-1].isdigit() else tag
        if tags is not None and tag not in tags:
            continue
        rows, pv = load(f)
        d = measure(rows, tok())
        if d["total"] is None:
            continue
        fam, kept, disp = parse_tag(tag)
        d.update(batch=pv.get("axis_batch_size"), node=pv.get("slurm_node"),
                 job=pv.get("slurm_job_id"), disp=disp, fam=fam, kept=kept,
                 file=os.path.basename(f),
                 inline_f1=statistics.mean(r["acc_f1"] for r in rows),
                 # share of the run's OWN questions whose F1 clears a threshold (2026-09-14, user:
                 # a throughput counted only on the questions that came out right, the way a DB reports
                 # successful transactions). Measured on the same rows as the rate, so goodput = rate x
                 # share multiplies two measurements of ONE run; it is labelled DERIVED wherever drawn.
                 # F1 = 1.0 is exact token overlap with the gold.
                 correct_share={t: round(sum(1 for r in rows if r["acc_f1"] >= t) / len(rows), 4)
                                for t in (1.0, 0.8, 0.5)})
        out[tag] = d
    # ★ ONE NODE PER TABLE — the gate refuses cross-node mixes; this collector must too, or a
    # backfilled arm from another node silently enters a published curve. Keep the majority node.
    if out:
        nodes = [d["node"] for d in out.values()]
        keep_node = max(set(nodes), key=nodes.count)
        dropped = [t for t, d in out.items() if d["node"] != keep_node]
        for t in dropped:
            print(f"[collect:{prefix}] DROPPED {t} (node {out[t]['node']} != table node {keep_node})")
            del out[t]
    return out or None


def curve_svg(arms, acc, xlab):
    """THE curve: press families as connected lines in (answers/s, F1); ours/teacher/floor as
    points. This is the figure the study exists to produce (RESULTS_MASTER 2026-08-27b): does our
    POINT lie above the compression CURVE."""
    pts = {t: (d["ans_s"], acc.get(t)) for t, d in arms.items() if acc.get(t) is not None}
    if len(pts) < 4:
        return ""
    xs = [v[0] for v in pts.values()]
    ys = [v[1] for v in pts.values()]
    x0, x1 = 0, max(xs) * 1.08
    y0, y1 = min(ys) - 0.03, max(ys) + 0.03
    W, Hh, PL, PB, PT, PR = 900, 430, 62, 52, 18, 168

    def px(x):
        return PL + (x - x0) / (x1 - x0) * (W - PL - PR)

    def py(y):
        return PT + (y1 - y) / (y1 - y0) * (Hh - PT - PB)

    FAM = dict(snap=("var(--fsnap)", "", "circle"), expected=("var(--fexp)", "7 4", "square"),
               spec=("var(--fspec)", "2 4", "triangle"))
    s = [f'<figure><svg viewBox="0 0 {W} {Hh}" width="100%" role="img" '
         f'aria-label="accuracy against end-to-end answers per second">']
    yt = y0
    while yt <= y1 + 1e-9:
        v = round(yt, 2)
        s.append(f'<line x1="{PL}" y1="{py(v):.1f}" x2="{W-PR}" y2="{py(v):.1f}" '
                 f'stroke="var(--line)" stroke-width="1"/>')
        s.append(f'<text x="{PL-8}" y="{py(v)+4:.1f}" text-anchor="end" fill="var(--mut)" '
                 f'font-size="11" font-family="IBM Plex Mono,monospace">{v:.2f}</text>')
        yt += 0.05
    xt = 0.0
    step = 0.25 if x1 < 1.5 else 0.5
    while xt <= x1:
        s.append(f'<text x="{px(xt):.1f}" y="{Hh-PB+20}" text-anchor="middle" fill="var(--mut)" '
                 f'font-size="11" font-family="IBM Plex Mono,monospace">{xt:g}</text>')
        xt += step
    s.append(f'<text x="{(PL+W-PR)/2:.0f}" y="{Hh-8}" text-anchor="middle" fill="var(--mut)" '
             f'font-size="11.5" font-family="Archivo,sans-serif">{xlab}</text>')
    s.append(f'<text x="20" y="{(PT+Hh-PB)/2:.0f}" fill="var(--mut)" font-size="11.5" '
             f'font-family="Archivo,sans-serif" transform="rotate(-90 20 {(PT+Hh-PB)/2:.0f})" '
             f'text-anchor="middle">F1 →</text>')

    def marker(x, y, shape, col, r=5, sw=2):
        if shape == "circle":
            return (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{col}" '
                    f'stroke="var(--panel)" stroke-width="{sw}">')
        if shape == "square":
            return (f'<rect x="{x-r:.1f}" y="{y-r:.1f}" width="{2*r}" height="{2*r}" fill="{col}" '
                    f'stroke="var(--panel)" stroke-width="{sw}">')
        return (f'<path d="M {x:.1f} {y-r-1:.1f} L {x+r+1:.1f} {y+r:.1f} L {x-r-1:.1f} {y+r:.1f} Z" '
                f'fill="{col}" stroke="var(--panel)" stroke-width="{sw}">')

    for fam, (col, dash, shape) in FAM.items():
        fam_pts = sorted(((arms[t]["kept"], t) for t in pts if arms[t]["fam"] == fam))
        if len(fam_pts) < 2:
            continue
        line = " ".join(f"{px(pts[t][0]):.1f},{py(pts[t][1]):.1f}" for _, t in fam_pts)
        dash_attr = f'stroke-dasharray="{dash}" ' if dash else ""
        s.append(f'<polyline points="{line}" fill="none" stroke="{col}" stroke-width="2" '
                 f'{dash_attr}stroke-linejoin="round"/>')
        for kept, t in fam_pts:
            x, y = px(pts[t][0]), py(pts[t][1])
            s.append(marker(x, y, shape, col) + f'<title>{arms[t]["disp"]} · B={arms[t]["batch"]} · '
                     f'{pts[t][0]:.3f} answers/s · F1 {pts[t][1]:.4f}</title>'
                     + ("</circle>" if shape == "circle" else "</rect>" if shape == "square" else "</path>"))
        _, tl = fam_pts[-1]
        s.append(f'<text x="{px(pts[tl][0])+9:.1f}" y="{py(pts[tl][1])+4:.1f}" fill="{col}" '
                 f'font-size="11.5" font-family="Archivo,sans-serif" font-weight="600">'
                 f'{ {"snap":"snapKV","expected":"ExpectedAttn","spec":"SpecPrefill"}[fam] }</text>')
    for t, lab, col, big in (("teacher", "teacher-32B", "var(--ink)", 5),
                             ("floor7", "floor-7B", "var(--mut)", 5), ("ours", "OURS", "var(--fours)", 8)):
        if t not in pts:
            continue
        x, y = px(pts[t][0]), py(pts[t][1])
        s.append(f'<path d="M {x:.1f} {y-big-2:.1f} L {x+big+2:.1f} {y:.1f} L {x:.1f} {y+big+2:.1f} '
                 f'L {x-big-2:.1f} {y:.1f} Z" fill="{col}" stroke="var(--panel)" stroke-width="2">'
                 f'<title>{arms[t]["disp"]} · B={arms[t]["batch"]} · {pts[t][0]:.3f} answers/s · '
                 f'F1 {pts[t][1]:.4f}</title></path>')
        s.append(f'<text x="{x:.1f}" y="{y-12:.1f}" text-anchor="middle" fill="{col}" '
                 f'font-size="12" font-family="Archivo,sans-serif" font-weight="700">{lab}</text>')
    s.append("</svg>")
    s.append('<figcaption>Compression baselines have a retention knob and trace a CURVE; ours is a '
             'POINT. The study\'s question is whether the point lies above the curve. Hover any '
             'mark for its arm, batch and values; line style and marker shape also carry family '
             'identity. <b>Why the curves hook back (the "C"):</b> compression buys speed only '
             'through a larger batch, and at the high-removal end the batch hits a cap — the '
             'benchmark\'s conversation/document count, or the sweep ladder\'s top (capped points '
             'share the same B; hover shows it). Past the cap retention cannot raise the decode '
             'rate (dispatch-bound; RESULTS_MASTER 2026-08-27c), while the ruined context makes '
             'generations LONGER on the single-turn benches (measured: hotpot snapKV 76 → 99 '
             'tokens/answer from 60% to 5% kept), so answers/s slides backwards as F1 keeps '
             'falling — the hook is the press\'s quality collapse showing up in its own wall '
             'clock.</figcaption></figure>')
    return "".join(s)


def arms_for(prefix, names):
    """Collect gate-measured rows for the given arm tags; None if nothing has landed."""
    out = {}
    for tag, disp in names.items():
        hits = sorted(glob.glob(f"results/fusionft/{prefix}_{tag}_b*.jsonl"))
        hits = [h for h in hits if os.path.getsize(h)]
        if not hits:
            continue
        f = hits[-1]
        rows, pv = load(f)
        d = measure(rows, tok())
        if d["total"] is None:            # pre-fix log: refuse silently, the gate refuses loudly
            continue
        d["batch"] = pv.get("axis_batch_size")
        d["node"] = pv.get("slurm_node")
        d["job"] = pv.get("slurm_job_id")
        d["disp"] = disp
        d["file"] = os.path.basename(f)
        d["inline_f1"] = statistics.mean(r["acc_f1"] for r in rows)
        out[tag] = d
    return out or None


def bench_table(arms, acc, acc_label, hero="ours"):
    nodes = sorted({d["node"] for d in arms.values()})
    jobs = sorted({str(d["job"]) for d in arms.values()})
    A(f'<div class="wrap"><table><tr><th>arm</th><th>batch (its B<sub>max</sub>)</th>'
      f'<th>{DEC_TOK}</th><th>{E2E_TOK}</th><th>answers/s<br>'
      f'<span style="text-transform:none;letter-spacing:0">END-TO-END</span></th>'
      f'<th>gen tok per answer</th><th>prefill s per seq</th><th>idle slots</th>'
      f'<th>{acc_label}</th></tr>')
    for tag in sorted(arms, key=lambda t: -arms[t]["ans_s"]):
        d = arms[tag]
        cls = ' class="hero"' if tag == hero else ""
        a = acc.get(tag)
        A(f'<tr{cls}><td>{d["disp"]}</td><td class="n">{d["batch"]}</td>'
          f'<td class="n">{d["tok_s"]:.2f}</td><td class="{"m" if tag==hero else "n"}">'
          f'{d["tok_s_e2e"]:.2f}</td><td class="{"m" if tag==hero else "n"}">{d["ans_s"]:.3f}</td>'
          f'<td class="n">{d["mean_row"]:.1f}</td><td class="n">{d["prefill"]/max(d["batch"],1):.2f}</td>'
          f'<td class="n">{d["ragged"]:.1%}</td>'
          f'<td class="{"m" if tag==hero else "n"}">{a:.4f}</td></tr>' if a is not None else
          f'<tr{cls}><td>{d["disp"]}</td><td class="n">{d["batch"]}</td>'
          f'<td class="n">{d["tok_s"]:.2f}</td><td class="n">{d["tok_s_e2e"]:.2f}</td>'
          f'<td class="n">{d["ans_s"]:.3f}</td><td class="n">{d["mean_row"]:.1f}</td>'
          f'<td class="n">{d["prefill"]/max(d["batch"],1):.2f}</td>'
          f'<td class="n">{d["ragged"]:.1%}</td><td class="n">—</td></tr>')
    A("</table></div>")
    # ★ every eager bench table carries its own standing, because a banner in §0 is a standing a
    # reader loses the moment they scroll to a number.
    A('<p style="color:var(--bad);font-size:.85rem"><b>Standing — the teacher / ours / floor rows '
      'here are DIAGNOSTIC (symmetric-eager), not canonical; see §0. The press rows ARE canonical: '
      'kvpress is their published implementation.</b></p>')
    A(f'<p style="color:var(--mut);font-size:.85rem">node {"/".join(nodes)} · job(s) '
      f'{", ".join(jobs)} · every rate divides by <code>conv_wall_s</code> — one timer around '
      f'prefill + decode + history commits, nothing outside it. DECODE divides by the answer wall '
      f'alone and is shown so the two axes can be compared; the END-TO-END columns are the '
      f'headline. answers/s is the axis an arm cannot inflate by writing longer answers.</p>')


def main():
    # renamed from "Decode Throughput Study" (user, 2026-08-28): the whole point of the rebuild is
    # that the measured quantity is the TOTAL wall, not decode.
    A("<title>End-to-End Throughput Study</title>")
    A(CSS)
    A("<main>")
    A("<h1>What one GH200 actually delivers, measured on the whole wall</h1>")
    A('<p class="lede">One GH200 with 95 GiB. A 32B reading a long conversation, against a 7B reader '
      'plus a question-only 32B mixed at the logit level. Every rate on this page divides by a '
      'single timer around everything — context prefill, decode, per-turn history commits. Nothing '
      'here is about accuracy except the accuracy column beside each rate.</p>')

    A('<div class="box warn"><b class="tag">what this page replaced, 2026-08-28</b>Every throughput '
      'table previously on this page was withdrawn: the rates divided generated tokens by the '
      'answer-generation wall only. The context prefill sat in a separate field that was never '
      'added, and the per-turn history commit sat outside every timer — and both run against a '
      'long-context cache for the teacher and every compression baseline, but only against the 7B '
      'reader for us. Decode-only measurement kept the axis the method pays on and deleted the axis '
      'it wins on. The harness now records <code>conv_wall_s</code> — one timer, nothing can fall '
      'outside it by construction — and <code>scripts/throughput_eval.py</code> refuses any log '
      'without it.</div>')

    A('<div class="box warn"><b class="tag">where the current frontier numbers live, 2026-09-04</b>'
      'Sections 2–4 below are the per-arm-B<sub>max</sub> runs of 2026-08-27/28 (<code>lct</code>, '
      '<code>lgt</code>, <code>mur</code>, <code>ho3</code>). They are valid measurements of their own '
      'workloads and stay here with their batches, but on the two accumulating benchmarks each arm '
      'was given only as many CONVERSATIONS as its batch, so the arms of one table did not run the '
      'same work. The paper figure and the ledger (§2 and §2b of <code>reports/ledger.html</code>) '
      'are now on the unified retention grid measured with every arm on the identical workload — '
      'LoCoMo 8 conversations, LooGLE 24, musique and hotpot 96 questions — one job and one node per '
      'benchmark, the complete-batch rate, and the batch printed beside every rate. Where a number '
      'here and a number there differ, the ledger is the published one.</div>')

    # ---- 1 · the claim ---------------------------------------------------------------------------
    A("<h2>0 · Which rows on this page are canonical</h2>")
    A('<div class="box warn"><b class="tag">read this before any table below</b>'
      'Rows on this page are at three different standings and they are NOT interchangeable.'
      '<br><br><b>CANONICAL</b> — teacher / ours / floor on the FKV backend with the '
      '<b>batch-local allocator</b> (hotpot only so far), and the press families on their own '
      'kvpress implementation, which IS their published one.'
      '<br><b>PROVISIONAL</b> — every FKV row for musique and LoCoMo. They were measured while the '
      'backend reserved a DATASET-WIDE <code>STATIC_MAXLEN</code> in every batch slot, which '
      'reserved storage no sequence in the batch would use and so understated OUR OWN B_max. '
      'Re-measured on hotpot this changed ours from B=12 to B=14 and 1.770 to 1.937 answers/s, '
      'teacher from B=8 to B=10 and 0.777 to 0.810, moving ours-over-teacher from ×2.28 to ×2.39. '
      'musique and LoCoMo need the same re-run.'
      '<br><br><b>On _grow(), which was briefly reported here as BLOCKING and is not (resolved 2026-09-01).</b> ''<code>_grow()</code> reallocates the KV buffer when a batch needs more than was reserved, and it had never executed in this project. Forced on LooGLE it fired twice and changed 11 of 46 answers, which looked like a defect. It is not: with the FINAL buffer length held equal and only the path varied — one leg growing 8192 to 32768, one starting at 32768 — the two are <b>46/46 identical, raw generations included</b>. <b>Growing is exact.</b> What moves the answers is the reserved LENGTH itself: the kernel attends to the same positions either way, but its arithmetic depends on the allocated size. That is the same class of fact as FKV generations already differing from eager ones, which is why FKV is a TIMING arm whose accuracy column comes from the eager runs — so it does not block a throughput measurement on any bench.'
      + _queue_line() +
      '<br><b>DIAGNOSTIC, NOT A RESULT</b> — the symmetric-eager teacher / ours / floor rows in §2–§4. '
      'The eager <code>DynamicCache</code> path recopies the whole cache every step; against the '
      'fixed path the ours-over-teacher ratio moves from ×1.69–1.99 to ×1.96–2.39, so those rows '
      'describe an implementation rather than the system. They are kept as a conservative lower '
      'bound and as the canary reference, not as the headline.'
      '<br><br><b>Quantized arms appear only in \u00a76d, and only under the name of the '
      'implementation that produced them.</b> (Until 2026-09-01 this line read "quantized '
      'arms appear nowhere on this page", which stopped being true the moment \u00a76d '
      'landed and was left standing.) HuggingFace\'s generic '
      '<code>QuantizedCache</code> dequantizes the whole cache on every decode step, so its wall '
      'measures that container rather than low-bit KV serving; our quantize-once cache fixes '
      'ACCURACY and leaves that untouched. A quantized throughput row needs a fused '
      'dequant-in-attention path, and until one exists quantized arms are reported on the '
      'memory–accuracy frontier only.</div>')
    A("<h2>1 · The claim, which has two mechanisms</h2>")
    A('<div class="box"><b class="tag">the claim</b>Never "each token is cheaper." Two things are '
      'cheaper per sequence. <b>Capacity:</b> the reader stores 56 KiB of KV per token against the '
      '32B\'s 256, so more sequences fit on one card and the card serves more answers per second. '
      '<b>Prefill:</b> the 32B branch never ingests the context at all — the measured per-sequence '
      'prefill is ~2.1 s against the teacher\'s ~8.0 s on LooGLE\'s ~37k contexts, and a '
      'compression baseline does not escape this axis: it must prefill the full context before it '
      'can compress anything.</div>')
    # PAPER FIGURES (added 2026-09-01). Until today the accuracy-throughput trade-off existed only
    # as the inline SVG curves further down this page, and the two-card result only as tables in
    # §6e - there was nothing a paper could use. Both are embedded here as data URIs so the page and
    # the figure can never drift apart; the PDFs sit beside the PNGs in figures/.
    for _fp, _cap in (("figures/ledger/paper_throughput.png",
                       "<b>Accuracy vs end-to-end throughput.</b> Every arm at its own "
                       "B<sub>max</sub> on the IDENTICAL eager decode path, so the comparison is "
                       "like-for-like. The shaded box is the region every point of which is slower "
                       "AND less accurate than ours; the count in each panel is computed from the "
                       "data. Source: <code>scripts/plot_paper_throughput.py</code>."),
                      ("figures/ledger/paper_2gpu.png",
                       "<b>The two-card overlap.</b> (a) is a matched-batch pair whose generations "
                       "are byte-identical, so it isolates the overlap; (b) is the comparison "
                       "against the alternative anyone would deploy, with ours carrying the larger "
                       "batch — the direction unfavourable to ours; (c) is why. Source: "
                       "<code>scripts/plot_paper_2gpu.py</code>.")):
        if not os.path.exists(_fp):
            continue
        import base64 as _b64
        with open(_fp, "rb") as _f:
            _d = _b64.b64encode(_f.read()).decode()
        A(f'<img src="data:image/png;base64,{_d}" alt="{_fp}" '
          'style="width:100%;max-width:1200px;margin:14px 0 4px">')
        A(f'<p class="mut" style="margin-top:0">{_cap}</p>')

    A('<p>The capacity mechanism at the bandwidth roofline (CALC, from the byte counts at each '
      'arm\'s own B<sub>max</sub> and 3.6 TB/s): teacher-32B at batch 3 moves 87.2 GB per step for '
      '3 tokens; ours at batch 6 moves 90.9 GB for 6. Nearly the same bytes, twice the tokens. '
      'Everything below is about how much of that the implementation delivers, and on which '
      'workload shape.</p>')

    # ---- 2 · LoCoMo ------------------------------------------------------------------------------
    A("<h2>2 · LoCoMo-30 — 30 questions amortise one prefill</h2>")
    # tag convention is IDENTICAL in the accuracy json and the lcw timing logs: snapNN / expectedNN
    # name the press RATIO (snap40 = ratio .40 = 60% kept), specNN names the keep fraction.
    lc_names = dict(ours="ours 32B+7B", teacher="teacher-32B", floor7="floor-7B",
                    snap40="snapKV 60% kept", snap60="snapKV 40% kept",
                    snap78125="snapKV 21.9% kept", snap90="snapKV 10% kept",
                    snap95="snapKV 5% kept",
                    expected40="ExpectedAttn 60% kept", expected78125="ExpectedAttn 21.9% kept",
                    expected95="ExpectedAttn 5% kept", spec40="SpecPrefill keep .40",
                    spec21875="SpecPrefill keep .219", spec05="SpecPrefill keep .05")
    # PREFER the sweep-generation run (lcs_, >=4 points per press family, one node); fall back to
    # the 14-arm lcw_ run. Sweep tags map onto the lcf_accuracy keys explicitly:
    LCS2ACC = dict(teacher="teacher", ours="ours", floor7="floor7",
                   snap400="snap40", snap500="snap50", snap600="snap60", snap700="snap70",
                   snap781="snap78125", snap850="snap85", snap900="snap90", snap950="snap95",
                   expected400="expected40", expected500="expected50", expected600="expected60", expected700="expected70",
                   expected781="expected78125", expected850="expected85",
                   expected900="expected90", expected950="expected95",
                   spec40000="spec40", spec30000="spec30", spec21875="spec21875",
                   spec15000="spec15", spec10000="spec10", spec05000="spec05")
    _acc_raw = (json.load(open("results/timing/lcf_accuracy.json"))
                if os.path.exists("results/timing/lcf_accuracy.json") else {})
    lcs = collect("lct") or collect("lcs")
    if lcs:
        lc = lcs
        lc_acc = {t: _acc_raw.get(LCS2ACC.get(t, t)) for t in lc}
    else:
        lc = arms_for("lcw", lc_names)
        lc_acc = _acc_raw
    if lc:
        A(curve_svg(lc, lc_acc,
                    "completed answers per second, END-TO-END (total wall), each arm at its own "
                    "B_max →") if lcs else "")
        A('<p>LoCoMo-30: ~27.5k-token conversations, 30 questions each, so one context prefill is '
          'amortised over thirty decodes — the workload shape LEAST favourable to the prefill '
          'mechanism. Each arm at its own B<sub>max</sub> (an OOM boundary is a memory fact), all '
          'arms one job one node. Accuracy: canonical F1, shared-N=300 '
          '(<code>results/timing/lcf_accuracy.json</code>).</p>')
        bench_table(lc, lc_acc, "F1<br><span style='text-transform:none;letter-spacing:0'>"
                                "shared-N=300</span>")
        if all(k in lc for k in ("ours", "snap40", "snap60", "teacher")):
            o, s4, s6, t = lc["ours"], lc["snap40"], lc["snap60"], lc["teacher"]
            A(f'<div class="box win"><b class="tag">the corrected wall reverses the withdrawn '
              f'verdict</b>Decode-only measurement had ours below the snapKV curve. On the total '
              f'wall <b>no arm dominates ours on (answers/s, F1)</b>: snapKV at 60% kept is '
              f'+{lc_acc["snap40"]-lc_acc["ours"]:.4f} F1 but {(s4["ans_s"]/o["ans_s"]-1)*100:.0f}% '
              f'on answers/s ({s4["ans_s"]:.3f} vs {o["ans_s"]:.3f}); snapKV at 40% kept is '
              f'+{(s6["ans_s"]/o["ans_s"]-1)*100:.0f}% answers/s but '
              f'{lc_acc["snap60"]-lc_acc["ours"]:+.4f} F1. Ours sits on the Pareto frontier '
              f'between them, at ×{o["ans_s"]/t["ans_s"]:.2f} the teacher\'s answers/s for '
              f'{lc_acc["ours"]-lc_acc["teacher"]:+.4f} F1. The mover is the axis the decode-only '
              f'bug deleted: our per-sequence prefill is {o["prefill"]/o["batch"]:.2f} s — '
              f'essentially the 7B floor\'s — against 3.9–4.9 s for the teacher and every press, '
              f'and even amortised over 30 turns it reorders the frontier.</div>')
    else:
        A('<div class="box"><b class="tag">not yet on this page</b>The LoCoMo-30 total-wall re-run '
          '(job 3041153, 13 arms one node) had not finished when this page was built. Its decode-only '
          'predecessor is withdrawn and is deliberately not shown. Rebuild this page when '
          '<code>results/fusionft/lcw_*_b*.jsonl</code> is complete.</div>')

    # ---- 3 · LooGLE ------------------------------------------------------------------------------
    A("<h2>3 · LooGLE-accum — ~37k contexts, a median of 5 questions per document</h2>")
    lg_names = dict(ours="ours 32B+7B", teacher="teacher-32B", floor7="floor-7B",
                    snap40="snapKV 40% kept", snap219="snapKV 21.9% kept",
                    spec219="SpecPrefill keep .219")
    # PREFER the sweep-generation run (lgs_, >=4 points per family). Sweep tags -> kept-based
    # loogle_accuracy keys (snap600 = ratio .60 = 40% kept -> "snap40", etc.):
    LGS2ACC = dict(teacher="teacher", ours="ours", floor7="floor7",
                   snap500="snap50", snap600="snap40", snap700="snap30",
                   snap781="snap219", snap850="snap15", snap900="snap10", snap950="snap05",
                   expected500="expected50", expected600="expected40", expected700="expected30",
                   expected781="expected219", expected850="expected15",
                   expected900="expected10", expected950="expected05",
                   spec40000="spec40", spec30000="spec30", spec21875="spec219",
                   spec15000="spec15", spec10000="spec10", spec05000="spec05")
    _gacc_raw = (json.load(open("results/timing/loogle_accuracy.json"))
                 if os.path.exists("results/timing/loogle_accuracy.json") else {})
    lgs = collect("lgt") or collect("lgs")
    if lgs:
        lg = lgs
        gacc = {t: _gacc_raw.get(LGS2ACC.get(t, t)) for t in lg}
    else:
        lg = arms_for("lgw", lg_names)
        gacc = _gacc_raw
    if lg:
        if lgs:
            A(curve_svg(lg, gacc, "completed answers per second, END-TO-END (total wall), each arm "
                                  "at its own B_max →"))
        A('<p>The 8 longest documents (batch context ~37k tokens at turn 1 — a deliberate worst '
          'case, not the corpus median). Fewer turns per conversation than LoCoMo, so the prefill '
          'axis carries more weight. Accuracy: canonical F1 at shared-N=355 from the one-node '
          'accuracy runs (rescore_headline_tables.py) — the per-arm inline F1 of these timing logs '
          'is NOT comparable across arms because each arm holds a different first-B document '
          'subset by design.</p>')
        bench_table(lg, gacc, "F1<br><span style='text-transform:none;letter-spacing:0'>"
                              "shared-N=355</span>")
        if "snap219" in lg and "ours" in lg:
            s, o = lg["snap219"], lg["ours"]
            A(f'<div class="box win"><b class="tag">the axis the decode-only bug deleted flips an '
              f'ordering here</b>snapKV at 21.9% kept out-decodes ours on the answer wall '
              f'({s["tok_s"]:.2f} against {o["tok_s"]:.2f}) — but it must prefill the full context '
              f'before compressing ({s["prefill"]/s["batch"]:.2f} s per sequence against our '
              f'{o["prefill"]/o["batch"]:.2f}), and on the end-to-end rate the order reverses: '
              f'ours {o["tok_s_e2e"]:.2f} over {s["tok_s_e2e"]:.2f}, at +{gacc["ours"]-gacc["snap219"]:.4f} '
              f'F1. Ours also dominates snapKV@40% kept and SpecPrefill on both axes at once. '
              f'The one cell snapKV@21.9% still wins is answers/s ({s["ans_s"]:.3f} vs '
              f'{o["ans_s"]:.3f}) — and it writes {s["mean_row"]:.0f} tokens per answer against our '
              f'{o["mean_row"]:.0f}, so that is the axis an arm wins by writing less.</div>')

    # ---- 4 · single-turn -------------------------------------------------------------------------
    A("<h2>4 · Single-turn RAG (musique · hotpotQA) — where nothing amortises the prefill</h2>")
    A('<p class="lede">One question per context, so the context prefill is paid in full on every '
      'answer — the workload shape where the query-only 32B branch matters most. Each benchmark '
      'gets its own subsection below.</p>')
    st_names = dict(ours="ours 32B+7B", teacher="teacher-32B", floor7="floor-7B",
                    snap40="snapKV 40% kept", snap219="snapKV 21.9% kept",
                    spec219="SpecPrefill keep .219")
    shown = False
    for pfx, bench, lam in (("must", "musique_st40", 0.7), ("host", "hotpotqa_st40", 0.85)):
        # PREFER the sweep-generation run (mu2_/ho2_): full retention curves, one node
        st = (collect({"must": "mur", "host": "ho3"}[pfx])
              or collect({"must": "mu3", "host": "ho3"}[pfx])
              or collect({"must": "mu2", "host": "ho2"}[pfx]))
        st_is_new = st is not None
        if not st:
            st = arms_for(pfx[:2] + "st", st_names)
        if not st:
            continue
        shown = True
        # F1 SOURCE: musique = the user-approved FULL-2417 basis. hotpot = inline F1 over the
        # random-96 subset — the FULL-600 rerun was UNREQUESTED and DISCARDED on the user's order
        # (2026-08-30 incident, RESULTS_MASTER); the user deemed hotpot's random subset acceptable.
        full_p = {"must": "results/timing/musique_st40_acc_full.json",
                  "host": "results/timing/hotpotqa_st40_acc_full.json"}[pfx]
        st_full = json.load(open(full_p)) if os.path.exists(full_p) else {}
        # A store cell is a dict {"f1":..,"em":..,"n":..} for most arms and a BARE FLOAT for the
        # ones promoted by merge_appendonly_cells. Reading .get("f1") off a float raises, which is
        # how this page silently stopped rebuilding; read both shapes.
        def _f1(v):
            return v.get("f1") if isinstance(v, dict) else v
        st_acc = {t: _f1(st_full.get(t)) for t in st}
        acc_lbl = ("F1<br><span style='text-transform:none;letter-spacing:0'>FULL set "
                   f"({'2417' if pfx == 'must' else '600'} Qs)</span>")
        A(f"<h3>4{'a' if pfx == 'must' else 'b'} · "
          f"{'musique d40 — FULL-2417 accuracy basis' if pfx == 'must' else 'hotpotQA d40 s42 — FULL-600 accuracy basis'}"
          f" (λ{lam})</h3>")
        if st_is_new:
            A(curve_svg(st, st_acc,
                        "completed answers per second, END-TO-END (total wall), each arm at its "
                        "own B_max →"))
        acc_desc = ("the FULL validation set (all 2417 questions, hop-unbiased)" if pfx == "must"
                    else "the FULL 600-question set (canonical λ0.7 ours arm, job 3049836)")
        A(f'<p><b>{bench}</b> (fusion arm at λ{lam}): the 96 canonical d40 questions as 1-turn conversations, '
          f'~4.5–7k-token contexts, every arm answering the SAME 96 questions batched at its own '
          f'B<sub>max</sub>. Accuracy: {acc_desc}. This bench is its own basis (accumulate '
          f'passage formatting); it is never compared to the old singleturn-rag logs.</p>')
        if abs(lam - 0.7) > 1e-9:
            # ★ LABEL THE MISMATCH RATHER THAN LET A READER ASSUME ONE λ (2026-08-31).
            # hotpot's timing `ours` row was produced by the λ0.85 stage-2 adapter
            # (stage2_on_v5reader_lam085), while the accuracy beside it is the CANONICAL
            # λ0.7 FULL-600 arm — the λ0.85 hotpot arm is the one the record marks VOID.
            # The rate itself is not a function of λ: both branches run the same shapes,
            # the LoRA is merged into the weights (MERGE_LORA=1), and λ enters only as a
            # scalar in the logit combine. So the number is comparable — but a row whose
            # speed and accuracy come from different adapters must say so out loud.
            A(f'<div class="box warn"><b class="tag">this row\'s speed and its accuracy '
              f'come from different adapters</b>The <b>ours</b> wall here was measured with the λ{lam} stage-2 adapter, while the F1 column is the CANONICAL λ0.7 FULL-600 arm (job 3049836) — λ0.85 on hotpot is the configuration the record marks void. The rate is still readable, because it is not a function of λ: both branches run identical shapes, the LoRA is merged into the weights, and λ enters only as a scalar in the logit combine. The basis (b) table in §6c has no such split — its ours row is λ0.7 throughout.</div>')
        bench_table(st, st_acc, acc_lbl)
        if all(k in st for k in ("ours", "teacher", "snap219")):
            o, t, s = st["ours"], st["teacher"], st["snap219"]
            warn = ("" if st["ours"]["inline_f1"] <= t["inline_f1"] + 1e-9 else
                    " <b>Caveat, stated not smoothed:</b> ours scores ABOVE the raw teacher here — "
                    "a sandwich violation, diagnosed at example level (the zero-shot teacher misses "
                    "the final hop; ours' adapters are distilled from teacher traces). The teacher "
                    "prompt in this basis must be strengthened before the accuracy column is quoted "
                    "as a fusion result; the timing columns do not depend on it.")
            A(f'<div class="box win"><b class="tag">single-turn: the prefill is the wall</b>'
              f'The presses out-DECODE ours by 1.5–2× and still lose end-to-end: a press must '
              f'prefill the full context before it can compress anything '
              f'({s["prefill"]/s["batch"]:.1f} s/seq for snapKV@21.9% against ours\' '
              f'{o["prefill"]/o["batch"]:.1f}), so ours delivers {o["ans_s"]:.3f} answers/s against '
              f'its {s["ans_s"]:.3f} and the teacher\'s {t["ans_s"]:.3f} '
              f'(×{o["ans_s"]/t["ans_s"]:.2f}).{warn}</div>')
    if not shown:
        A('<div class="box"><b class="tag">in flight</b>musique_st40 and hotpotqa_st40 (jobs '
          '3041315 / 3041316) run every arm over the same 96 canonical d40 questions through the '
          'batched harness with the total-wall timer. A single-question workload weights the '
          'prefill fully — the shape most favourable to the method — and had no valid timing '
          'harness at all until 2026-08-28 (the old singleturn logs carry no provenance and no '
          'timing fields). This section fills in when they land.</div>')

    # ---- 5 · why B_max is the axis + fusion glue --------------------------------------------------
    A("<h2>5 · Why each arm runs at its own B<sub>max</sub>, and what the fusion glue costs</h2>")
    A('<p>At a fixed batch, KV retention does not move the decode rate: the eager loop is '
      'dispatch/kernel-bound — per-step cost tracks layer count, and every press runs the same '
      '64-layer 32B (kernel decomposition below; RESULTS_MASTER 2026-08-27c). A compression '
      'baseline\'s only throughput channel is the larger batch its smaller cache admits — the same '
      'channel our smaller reader cache uses. So capacity is the fair x-axis for every arm.</p>')
    A('<p>The fusion step itself adds <b>+0.9%</b> over the sum of its two branches (reader step '
      '27.4 + LM step 63.8 = 91.1 against fusion 92.0 ms/step, batch 6 eager; jobs '
      '3030432/3030472/3030499). The per-step sync costs −1.2 ms and the stop-condition callback '
      '0.08 ms — there is no hidden overhead in the fusion path.</p>')

    # ---- 6 · kernels + the probe -----------------------------------------------------------------
    A("<h2>6 · Where a decode step's time actually goes, and the fix that is now proven</h2>")
    prof = {}
    for t in ("eager_flash", "compiled_sdpa", "compiled_flash"):
        p = f"results/timing/kernels_{t}.json"
        if os.path.exists(p):
            prof[t] = json.load(open(p))
    if prof:
        A('<p>Per-kernel GPU time for one reader-7B decode step (batch 6, 27.5k, job 3035559):</p>')
        # ★ the batch goes in the TABLE, not only in the sentence above it (user, 2026-09-03):
        # every published speed number carries the batch it was measured at, so a reader can check
        # it without hunting for the prose.
        A('<div class="wrap"><table><tr><th>configuration</th><th>batch</th>'
          '<th>KV write / copy</th>'
          '<th>attention</th><th>matmul</th><th>total GPU ms/step</th><th>launches/step</th></tr>')
        for t, lab in (("eager_flash", "eager + flash + DynamicCache (the harness path)"),
                       ("compiled_sdpa", "compiled + sdpa + StaticCache"),
                       ("compiled_flash", "compiled + flash + StaticCache")):
            if t not in prof:
                continue
            d = prof[t]
            bk = d["buckets"]
            A(f'<tr><td>{lab}</td><td class="n">6</td>'
              f'<td class="n">{bk.get("KV write / copy",0):.2f}</td>'
              f'<td class="n">{bk.get("attention",0):.2f}</td>'
              f'<td class="n">{bk.get("matmul (projections + MLP)",0):.2f}</td>'
              f'<td class="n">{d["total_gpu_ms_per_step"]:.2f}</td>'
              f'<td class="n">{d["launches_per_step"]:.0f}</td></tr>')
        A("</table></div>")
        A('<p>The eager path spends a third of its GPU time recopying the whole KV cache every step '
          '(<code>DynamicCache</code> concatenation); a static cache removes that but, through HF, '
          'forces sdpa — whose decode attention costs 13× flash\'s. The two fixes never combined '
          'inside HF because its flash path unpads the cache with host-side indices, which CUDA '
          'graphs freeze at capture. The kernel that combines them exists: '
          '<code>flash_attn_with_kvcache</code> — preallocated cache, device-side lengths, '
          'in-kernel append, no cache copy at all.</p>')
    probes = {os.path.basename(p).split("probe_")[1][:-5]: json.load(open(p))
              for p in glob.glob("results/timing/flash_kvcache_probe_*.json")}
    if probes:
        A('<div class="wrap"><table><tr><th>probe (200 teacher-forced steps vs the harness path)'
          '</th><th>argmax match</th><th>wall ms/step</th><th>GPU ms/step</th></tr>')
        for tag, d in sorted(probes.items()):
            arms = d["arms"]
            aA = arms.get("A_eager_flash_dynamic", {})
            A(f'<tr class="sep"><td>{tag} B={d["B"]} ctx {d["ctx"]:,} — harness path</td>'
              f'<td class="n">reference</td><td class="n">{aA.get("wall",0):.2f}</td>'
              f'<td class="n">{aA.get("gpu",0):.2f}</td></tr>')
            for k, lab in (("B_flash_kvcache_eager", "flash_attn_with_kvcache, eager loop"),
                           ("C_flash_kvcache_cudagraph", "same, one CUDA graph")):
                if k not in arms:
                    continue
                m = arms[k]
                if "error" in m:
                    A(f'<tr><td>&nbsp;&nbsp;{lab}</td><td class="b" colspan="3">failed: '
                      f'{m["error"][:80]}</td></tr>')
                    continue
                mm = m.get("argmax_match")
                A(f'<tr><td>&nbsp;&nbsp;{lab}</td>'
                  f'<td class="g">{mm:.2%}</td>'
                  f'<td class="{"g" if k.startswith("C") else "n"}">{m["wall"]:.2f}</td>'
                  f'<td class="n">{m["gpu"]:.2f}</td></tr>')
        A("</table></div>")
        r7 = probes.get("reader7b")
        t32 = probes.get("teacher32b")

        def ratio(d):
            arms = (d or {}).get("arms", {})
            if "C_flash_kvcache_cudagraph" in arms and "wall" in arms["C_flash_kvcache_cudagraph"]:
                return arms["A_eager_flash_dynamic"]["wall"], arms["C_flash_kvcache_cudagraph"]["wall"]
            return None
        rr, rt = ratio(r7), ratio(t32)
        if rr:
            tteach = (f' The teacher measured on the same path gains ×{rt[0]/rt[1]:.2f} '
                      f'({rt[0]:.1f} → {rt[1]:.1f} ms/step at its batch 3) — the 2026-08-27g '
                      f'concern that removing the KV-copy tax would favour the teacher MORE is '
                      f'answered by measurement: both gain, the reader gains more.' if rt else
                      ' The teacher on the same path is still unmeasured.')
            A(f'<div class="box win"><b class="tag">the blocker is broken, and the fix is exact</b>'
              f'The record said flash attention on a static cache produces degenerate output; the '
              f'logs say the flash arms crashed on unrelated bugs and only a batch-1 smoke — the '
              f'regime this project itself bans — degenerated. Measured now: the '
              f'<code>flash_attn_with_kvcache</code> step matches the deployed path on <b>every '
              f'teacher-forced position, with a first-step max|Δlogit| of 0.0000</b>, and under one '
              f'CUDA graph the reader step goes <b>{rr[0]:.1f} → {rr[1]:.1f} ms/step '
              f'(×{rr[0]/rr[1]:.2f})</b>.{tteach} This is a probe at equal row lengths, not the '
              f'harness: integration needs per-row lengths (<code>cache_leftpad</code>) and the '
              f'fused two-branch step, and every press must be re-measured on the same path before '
              f'any comparison table changes.</div>')

    # ---- 6b · the fixed path on the harness ------------------------------------------------------
    fkv_names = dict(ours="ours 32B+7B", teacher="teacher-32B", floor7="floor-7B")
    fkv = arms_for("lcg", fkv_names)   # LoCoMo: dataset-wide reservation, so PROVISIONAL
    if fkv:
        A("<h2>6b · The three ported arms on the fixed decode path — LoCoMo (PROVISIONAL)</h2>")
        A('<div class="box warn"><b class="tag">provisional</b>Measured while the backend reserved a '
          'DATASET-WIDE <code>STATIC_MAXLEN</code> per batch slot, which understated B_max. On '
          'hotpot, re-measuring with the batch-local allocator moved ours from B=12 to B=14 and '
          '1.770 to 1.937 answers/s. LoCoMo needs the same re-run before these are canonical. (An earlier note here called that re-run BLOCKED by <code>_grow()</code>; that was wrong and is retracted &mdash; see section 0.)</div>')
        A('<p>The same LoCoMo-30 protocol as §2, decoding on the kvcache-kernel backend with the '
          'step in one CUDA graph (<code>FKV_DECODE=1 FKV_GRAPH=1</code>). This backend is a '
          '<b>timing arm</b>: on the harness its generations are 69/90 identical to the eager '
          'path\'s (ΔF1 −0.007 at n=90; the graph is byte-identical to the kernel\'s eager loop), '
          'so the F1 column stays the canonical eager shared-N=300. '
          '<b>The presses are not ported yet and are not here</b> — their decode is the same 32B '
          'the teacher runs, so porting is expected to scale them by roughly the teacher\'s gain; '
          'the frontier on this path is open until then.</p>')
        bench_table(fkv, lc_acc, "canonical F1<br><span style='text-transform:none;"
                                 "letter-spacing:0'>eager runs, N=300</span>")
        if lc and all(k in d for d in (fkv, lc) for k in ("ours", "teacher")):
            A(f'<div class="box win"><b class="tag">the fix helps ours more than the teacher, on '
              f'the harness</b>Against each arm\'s own eager row: ours '
              f'×{fkv["ours"]["ans_s"]/lc["ours"]["ans_s"]:.2f} on answers/s '
              f'({lc["ours"]["ans_s"]:.3f} → {fkv["ours"]["ans_s"]:.3f}), teacher '
              f'×{fkv["teacher"]["ans_s"]/lc["teacher"]["ans_s"]:.2f} '
              f'({lc["teacher"]["ans_s"]:.3f} → {fkv["teacher"]["ans_s"]:.3f}) — the probe-predicted '
              f'asymmetry reproduces, and the ours-vs-teacher ratio widens from '
              f'×{lc["ours"]["ans_s"]/lc["teacher"]["ans_s"]:.2f} to '
              f'×{fkv["ours"]["ans_s"]/fkv["teacher"]["ans_s"]:.2f} at the same F1 gap. Ours\' '
              f'B_max moved 8 → {fkv["ours"]["batch"]} (the preallocated buffers).</div>')

    # ---- 6c · the fixed path on the two SINGLE-TURN refs, with an in-job eager canary ------------
    # The LoCoMo block above compares FKV to eager rows measured in a DIFFERENT job on a DIFFERENT
    # node. That is unavoidable for the presses (the backend refuses them) but it is avoidable for
    # the arms FKV does support, so job 3055479 re-measured one eager arm INSIDE itself. Without
    # that canary a reader cannot tell a backend gain from an uncontended node.
    def _full_acc(path):
        try:
            raw = json.load(open(path))
        except Exception:
            return {}
        return {k: (v["f1"] if isinstance(v, dict) else v) for k, v in raw.items()}
    mu_acc = _full_acc("results/timing/musique_st40_acc_full.json")
    ho_acc = _full_acc("results/timing/hotpotqa_st40_acc_full.json")
    for _pfx, _epfx, _bench, _acc, _acc_lab, _pub in (
            ("fkvho2", "fkvho2_eager", "hotpotQA d40 s42 single-turn (96-question ref) — CANONICAL, "
             "batch-local allocator",
             {k: (ho_acc or {}).get(k) for k in fkv_names},
             "canonical F1<br><span style='text-transform:none;letter-spacing:0'>eager runs, "
             "FULL-600</span>", ("teacher", 0.543)),
            ("fkvmu", "eagmu", "musique d40 single-turn (96-question ref) — PROVISIONAL, "
             "dataset-wide reservation",
             {k: (mu_acc or {}).get(k) for k in fkv_names},
             "canonical F1<br><span style='text-transform:none;letter-spacing:0'>eager runs, "
             "FULL-2417</span>", ("teacher", 0.414)),
            ("fkvho", "eagho", "hotpotQA d40 s42 single-turn (96-question ref) — SUPERSEDED, "
             "dataset-wide reservation",
             {k: (ho_acc or {}).get(k) for k in fkv_names},
             "canonical F1<br><span style='text-transform:none;letter-spacing:0'>eager runs, "
             "FULL-600</span>", ("teacher", 0.543))):
        _f = arms_for(_pfx, fkv_names)
        if not _f:
            continue
        _e = arms_for(_epfx, dict(teacher="teacher-32B (eager CANARY)")) or {}
        A(f"<h2>6c · The fixed decode path on {_bench}</h2>")
        A('<p>Same backend and same protocol as §6b, on the single-turn refs. Still a <b>timing '
          'arm</b> — the 2026-08-31 equivalence check found FKV generations are not the eager '
          'generations (paraphrase forks introduced by the kernel swap; the flips are symmetric, '
          'teacher 10W/12L/68T), so the F1 column stays the canonical eager full-set score. '
          '<b>Presses are absent because the backend refuses them</b>, so their rows remain on the '
          'eager tables above.</p>')
        bench_table({**_f, **{f"canary_{k}": v for k, v in _e.items()}}, 
                    {**_acc, **{f"canary_{k}": _acc.get(k) for k in _e}}, _acc_lab)
        _tag, _pub_ans = _pub
        if _tag in _e:
            A(f'<div class="box win"><b class="tag">the canary says the node was not contended'
              f'</b>The eager {_tag}-32B re-measured inside this job reads '
              f'<b>{_e[_tag]["ans_s"]:.3f}</b> answers/s against <b>{_pub_ans:.3f}</b> on the eager '
              f'table above. Both numbers are printed; neither is called equal to the other. It '
              f'matters because basis (b) mixes jobs by construction — the presses cannot leave the '
              f'eager path — so without an in-job anchor a backend gain and a quiet node look '
              f'identical.</div>')
        if _tag in _e and "ours" in _f and "teacher" in _f:
            A(f'<div class="box win"><b class="tag">what the fixed path buys, and what it costs'
              f'</b>Measured inside this one job: teacher {_e[_tag]["ans_s"]:.3f} → '
              f'{_f["teacher"]["ans_s"]:.3f} answers/s. ours reaches '
              f'{_f["ours"]["ans_s"]:.3f} at B={_f["ours"]["batch"]}, so ours-over-teacher on this '
              f'backend is ×{_f["ours"]["ans_s"]/_f["teacher"]["ans_s"]:.2f}. The cost is '
              f'occupancy: FKV PREALLOCATES <code>STATIC_MAXLEN</code> rather than growing the '
              f'cache to what a batch uses, so ours holds {_f["ours"]["batch"]} sequences here '
              f'against 16 on the eager path. And the CUDA graph is captured once per BATCH, not '
              f'once per process, with the capture inside <code>conv_wall_s</code> — these rows are '
              f'conservative.</div>')

    # ---- 7 · what is left ------------------------------------------------------------------------
    # ── 6d · QUANTIZATION THROUGHPUT (job 3058489, 2026-08-31) ───────────────────────────────
    A("<h2>6d · Quantization throughput — hotpot, one job, one node, every arm laddered "
      "<span style='color:#2e7d32'>CANONICAL</span></h2>")
    A('<div class="box"><b class="tag">what this is</b>Job <b>3058489</b> on node <b>gh011</b>: '
      'seven arms over the same 96-question hotpotQA d40 ref, each descending its own batch ladder '
      'so <b>B<sub>max</sub> is measured, not assumed</b>, all in ONE job so node and job are held '
      'fixed as <code>throughput_eval</code> requires. Six arms produced; <code>eag_int4</code> '
      'OOM\'d at all three of its rungs (28/20/16) — my ladder floor was too high, so <b>there is no '
      'int4 throughput row</b> rather than a guessed one.</div>')

    A('<div class="box warn"><b class="tag">read the implementation, not "quantization"</b>'
      'The quantized arms run <b>HQQ append-only</b>, which is a real published quantized-cache '
      'implementation and the best one we can run — but it is <b>not</b> what a fused low-bit kernel '
      '(KIVI, KVQuant, vLLM) does. HQQ\'s <code>update()</code> must return the full K/V that '
      'attention reads, so it dequantizes every committed block to fp16 and concatenates on every '
      'step. Every number below is <b>HQQ append-only\'s</b> throughput, never "quantization\'s". '
      'And the penalty is <b>not</b> a baseline-only penalty: it hits our own composition arm in the '
      'same direction and nearly the same size.</div>')

    A("<h3>(a) Symmetric eager — every arm on the same <code>DynamicCache</code> path, each at its own "
      "B<sub>max</sub></h3>")
    A("<table><tr><th>arm</th><th>B<sub>max</sub></th><th>answers/s END-TO-END<br>96 questions / total "
      "wall</th><th>prefill s per batch</th><th>total wall s</th></tr>"
      "<tr><td>ours 32B+7B</td><td>20</td><td><b>1.135</b></td><td>4.56</td><td>84.6</td></tr>"
      "<tr><td>ours + reader-int8</td><td>20</td><td>0.792</td><td>7.17</td><td>121.3</td></tr>"
      "<tr><td>teacher-32B fp16</td><td>12</td><td>0.549</td><td>10.43</td><td>174.9</td></tr>"
      "<tr><td>teacher-32B + int8</td><td>12</td><td>0.333</td><td>16.59</td><td>288.2</td></tr>"
      "<tr><td>teacher-32B + int4</td><td colspan=4>— no row: OOM at every rung of the ladder "
      "(28/20/16)</td></tr></table>")
    A("<p>ours over teacher on this basis: <b>1.135 / 0.549 = ×2.07</b>.</p>")

    A('<div class="box"><b class="tag">the decisive pair is SAME-BATCH</b>'
      'B<sub>max</sub> differs between ours and the teacher, so those two are compared at their own '
      'capacities. The quantization comparisons are not: each quantized arm is compared with its own '
      'fp16 twin <b>at the identical batch</b>, so nothing but the cache changes.'
      '<br><br><b>teacher+int8 vs teacher fp16, both B=12: 0.333 vs 0.549 = ×0.61.</b>'
      '<br><b>ours+reader-int8 vs ours fp16, both B=20: 0.792 vs 1.135 = ×0.70.</b>'
      '<br><br><b>And int8 bought NO capacity.</b> The fp16 teacher\'s B<sub>max</sub> is 12 and the '
      'int8 teacher\'s B<sub>max</sub> is also 12, although int8 stores 53.6% of the bytes '
      '(measured). Stored bytes fall; <b>peak</b> memory does not, because the per-step dequantize '
      'materialises a full fp16 copy of the cache as a transient. The premise of KV quantization for '
      'serving — smaller KV, bigger batch, more throughput — does not hold for this implementation on '
      'this bench: no batch gain, and 40% of the throughput gone.</div>')

    A("<h3>(b) Each arm at its best available backend — teacher and ours on FKV</h3>")
    A("<table><tr><th>arm</th><th>backend</th><th>B<sub>max</sub></th><th>answers/s END-TO-END</th>"
      "<th>prefill s per batch</th></tr>"
      "<tr><td>ours 32B+7B</td><td>FKV</td><td>14</td><td><b>1.928</b></td><td>3.16</td></tr>"
      "<tr><td>teacher-32B</td><td>FKV</td><td>10</td><td>0.806</td><td>7.55</td></tr></table>")
    A("<p>ours over teacher: <b>1.928 / 0.806 = ×2.39</b>. Quantized arms cannot appear on this basis "
      "at all — the FKV backend preallocates a dense fp buffer and cannot hold a quantized cache.</p>")

    A('<div class="box ok"><b class="tag">independent replication</b>'
      'These FKV rows were measured on a DIFFERENT node from job 3057268, which measured the same '
      'two arms after the batch-local allocator fix. B<sub>max</sub> is identical in both '
      '(ours 14, teacher 10) and the rates agree to within half a percent: ours <b>1.928 vs 1.937</b> '
      '(−0.46%), teacher <b>0.806 vs 0.810</b> (−0.49%), ratio <b>×2.392 vs ×2.391</b>.</div>')

    _nc = "results/timing/node_conditions/qthrho_3058489_gh011.txt"
    if os.path.exists(_nc):
        _l = [x for x in open(_nc).read().split("\n") if x.strip()]
        A('<div class="box"><b class="tag">node conditions during the measurement</b>'
          'gh011 is a shared node and its other tenants were sampled every two minutes while this job '
          f'ran ({len(_l)} samples, <code>{_nc}</code>). Two other users held the remaining three GPUs '
          'throughout, and the set changed once mid-run. This is recorded rather than controlled: '
          '<code>--exclusive</code> would charge the whole 4-GPU node. Every previously published '
          'throughput number in this project has the same exposure and no such record.</div>')

    # 6e is prose+tables that were hand-written into the HTML and therefore were NOT in this
    # generator - regenerating the page on 2026-09-01 deleted them. Recovered from git and kept in
    # reports/_throughput_6e.html, included verbatim here so a rebuild can never drop it again.
    _6e = "reports/_throughput_6e.html"
    if os.path.exists(_6e):
        A(open(_6e, encoding="utf-8").read())
    else:
        A('<div class="box"><b class="tag">missing</b>reports/_throughput_6e.html is not present, '
          'so the 2-GPU section is not on this page.</div>')

    A("<h2>7 · What is left, sized</h2>")
    A("<ol>")
    A('<li><b>Port the press ingest to the FKV backend</b> (compress_append / SpecPrefill selection '
      'on compact per-row storage), then restate the WHOLE curve on the fixed path — §6b/§6c show '
      'ours/teacher/floor there on three benches already, and no press comparison is final until '
      'the presses run the same path.</li>')
    A('<li><b>FKV score re-validation at full N.</b> The 2026-08-31 check CHARACTERISED the gap '
      'rather than closing it: the generations are not the eager generations (teacher raw-identical '
      '9/90, ours 69/90) because the kernel swap changes bf16 reduction order and greedy forks; the '
      'flips are symmetric in both arms (10W/12L/68T and 3W/4L/83T) and re-converge on the same '
      'answer, and <code>FKV_GRAPH</code> is byte-identical to FKV-eager, so the graph capture is '
      'not the cause. It stays a TIMING arm until that is bounded at N=300 rather than n=90.</li>')
    A('<li><b>Continuous batching.</b> 21–39% of slots idle across arms (each table\'s idle '
      'column); an idle-corrected rate is a CALC, so the honest version of this claim needs a real '
      'continuous-batching loop.</li>')
    A('<li><b>Not worth pursuing, on evidence:</b> removing the per-step sync (−1.2 ms), a 3B '
      'reader (more layers, measured slower), λ schedules (accuracy result: the control wins).</li>')
    A("</ol>")

    A('<div class="foot">Timing: one node per table, every rate through '
      '<code>scripts/throughput_eval.py</code> (refuses cross-node, mixed batch where it matters, '
      'batch 1, and any log without <code>conv_wall_s</code>). LoCoMo job 3041153 · LooGLE job '
      '3041217 · single-turn jobs 3041315/3041316 · kernel profiles job 3035559 · probe jobs '
      '3041327/3041349 · decomposition jobs 3030432/3030472/3030499. Roofline: 3.6 TB/s, per-token '
      'KV 7B 56 KiB / 32B 256 KiB. Page: scripts/build_throughput_report.py; every table also in '
      'RESULTS_MASTER.md (2026-08-28 entries). The accuracy/mechanism analysis is a separate '
      'document and deliberately not here.</div>')
    A("</main>")
    os.makedirs("reports", exist_ok=True)
    open("reports/throughput.html", "w").write("\n".join(P))
    print("wrote reports/throughput.html")


if __name__ == "__main__":
    main()
