#!/usr/bin/env python
"""PREFLIGHT — run this before EVERY sbatch. It refuses the job rather than reminding you.

Every check here exists because the rule it enforces was stated, written down, and then broken anyway. A
guard inside one harness does not survive: the next session uses a different script and the rule is gone.
This is the one entry point, and `.claude/skills/preflight` points every session at it.

  preflight.py eval  --harness scripts/run_mtrag_accum.slurm --bench locomo --batch 1 ...
  preflight.py train --out /work/.../adapter --corpus results/fusionft/binding_corpus_v11.jsonl --steps 1150

Exit 0 = clear to submit. Exit 1 = do not submit. Every failure prints the rule and how to satisfy it.
`--force <reason>` records an override in preflight_overrides.log; it never silences a check.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

FAIL, WARN = [], []


def fail(rule, detail, fix):
    FAIL.append((rule, detail, fix))


def warn(rule, detail):
    WARN.append((rule, detail))


# ---------------------------------------------------------------- checks

def check_batch(batch):
    """BATCH >= 2 on every experiment. At batch 1 the flash-attention graph does not compile, so the run is
    slower AND any timing from it is an artefact — and it wastes GPU on every single experiment."""
    if batch is None:
        fail("batch", "no --batch given", "pass --batch N (>= 2); if the harness cannot batch, say so with "
                                          "--force and fix the harness")
    elif batch < 2:
        fail("batch", f"batch={batch}", "run at batch >= 2. GPU is the scarce resource and batch 1 also "
                                        "disables the flash-attention graph compile")


def check_disk(out, need_gib=8.0):
    """A long training job must PROVE it can write its artifact before it spends the GPU (the 2026-08-03
    job that trained 2000/2000 steps and then died in save_pretrained on a full quota)."""
    if not out:
        return
    d = out if os.path.isdir(out) else os.path.dirname(out) or "."
    while d and not os.path.isdir(d):
        d = os.path.dirname(d)
    try:
        free = shutil.disk_usage(d).free / 2 ** 30
    except Exception as e:
        warn("disk", f"could not stat {d}: {e}")
        return
    if free < need_gib:
        fail("disk", f"{free:.1f} GiB free at {d}", f"need >= {need_gib} GiB. Run bash scripts/check_disk.sh "
                                                    "— df lies here, the limit is a PROJECT quota")
    # the project quota is what actually bites; df shows the 8.5P filesystem
    try:
        q = subprocess.run(["lfs", "quota", "-p", "-h", d], capture_output=True, text=True, timeout=20).stdout
        if q.strip():
            warn("disk", "project quota reported: " + " ".join(q.split()[:12]))
    except Exception:
        pass
    check_quotas()


def check_quotas(head_gib=15.0):
    """★ EVERY quota the run writes through, not just the one holding the artifact (2026-08-12).

    Eleven jobs died at once on `OSError: [Errno 122] Disk quota exceeded` — mid-decode, appending result
    rows. The artifact directory was fine; **HOME** was at its hard limit (103G of 103G) because result
    logs and a stray 29 GiB HuggingFace cache live there. `df` showed 1.8P free and the project-quota check
    passed, because neither looks at the user quota on /u. A run writes to home whenever its --out is in
    the repo, which is every eval in this project."""
    try:
        out = subprocess.run(["quota", "-s"], capture_output=True, text=True, timeout=25).stdout
    except Exception as e:
        warn("quota", f"could not read quotas: {e}")
        return
    for line in out.splitlines():
        if not line.startswith("|") or "Used" in line:
            continue
        col = [c.strip() for c in line.strip("|").split("|")]
        if len(col) < 4 or not col[0].startswith(("/u/", "/work/", "/projects/")):
            continue
        path, used, soft, hard = col[0], col[1], col[2], col[3]

        def gib(v):
            v = v.rstrip("*")
            try:
                n = float(re.sub(r"[A-Za-z]", "", v) or 0)
            except ValueError:
                return None
            return n * {"K": 1 / 2 ** 20, "M": 1 / 1024, "G": 1.0, "T": 1024.0}.get(v[-1:].upper(), 1 / 2 ** 30)

        u, h = gib(used), gib(hard)
        if u is None or h is None or h <= 0:
            continue
        head = h - u
        if head < head_gib:
            fail("quota", f"{path}: {used} used of {hard} hard — only {head:.1f} GiB of headroom",
                 "a run that appends result rows will die MID-DECODE with OSError 122 and lose the GPU "
                 "hours already spent. Free space first (compress or relocate old logs; keep model caches "
                 "off /u — ~/.cache/huggingface belongs on /work).")
        elif head < head_gib * 3:
            warn("quota", f"{path}: {used} of {hard} hard, {head:.1f} GiB headroom — tight")


def check_corpus_supervision(corpus):
    """No gold, ever. v1-v4 were generator f-strings with the gold interpolated and their results were VOID;
    one of them resurfaced as the project's best LoCoMo number before it was caught."""
    if not corpus:
        return
    if not os.path.exists(corpus):
        fail("corpus", f"missing: {corpus}", "check the path")
        return
    r = subprocess.run([sys.executable, os.path.join(REPO, "scripts/audit_corpus_supervision.py"), corpus],
                       capture_output=True, text=True)
    if r.returncode != 0:
        fail("supervision", f"{os.path.basename(corpus)} contains a GOLD-SUPERVISED slice",
             "supervision must be the teacher's own generation; see the audit output:\n" + r.stdout.strip())


def check_teacher_targets(corpus):
    """Behaviour cloning: every training row's target must exist and be a model generation."""
    if not corpus or not os.path.exists(corpus):
        return
    n = miss = 0
    for line in open(corpus):
        r = json.loads(line)
        n += 1
        if not str(r.get("target", "")).strip():
            miss += 1
        if n >= 5000:
            break
    if miss:
        fail("targets", f"{miss}/{n} sampled rows have an empty target",
             "merge the teacher shards into `target` before training (usability gates only)")


def check_already_exists(bench, model, slm_model, adapters, lam, method):
    """★ REFUSE a run whose result ALREADY EXISTS (2026-08-15).

    The single largest waste of GPU in this project was not a bad experiment — it was RE-RUNNING one that was
    already on disk. On 2026-08-15 the LoCoMo-30 32B+7B accumulate block (teacher, floor, plain fusion,
    S-solo, and the lambda0.90/0.95 sweep) was regenerated from scratch although
    results/mab/q32b7b_locomo_{teacher32b,floor7b,ours_lam085,ours_readerftv5_lam085,_lam9,_lam95}.jsonl
    already held every one of them. Worse, the re-run used different settings (ratio 0.8125 vs 0.78125,
    batch 3 vs 1), so the new floor was 0.4216 against the old 0.4425 and every closeness in the block came
    out ~10 points higher for no reason but a changed denominator.

    EXPERIMENT_MANIFEST.md exists precisely to prevent this and was not consulted. A guard that lives in a
    document is not a guard, so it lives here: scan the result logs for one whose provenance matches on
    (bench, model, slm_model, slm_lora, lm_lora, lam, method) and refuse if one is found.
    """
    import glob
    want = (str(bench or ""), _base(model), _base(slm_model), _base((adapters or [None])[0] if adapters else None),
            None if lam is None else round(float(lam), 4), str(method or ""))
    hits = []
    for path in glob.glob(os.path.join(REPO, "results", "**", "*.jsonl"), recursive=True):
        try:
            with open(path) as fh:
                row = json.loads(fh.readline())
        except Exception:
            continue
        if not isinstance(row, dict):
            continue
        pr = row.get("_provenance")
        if not isinstance(pr, dict) or not pr:
            continue
        got = (str(pr.get("bench") or ""), _base(pr.get("model")), _base(pr.get("slm_model")),
               _base(pr.get("slm_lora")), None if pr.get("lam") is None else round(float(pr["lam"]), 4),
               str(pr.get("method") or ""))
        if got == want:
            hits.append(os.path.relpath(path, REPO))
    if hits:
        fail("duplicate", f"this exact run already exists: {hits[0]}"
             + (f" (+{len(hits)-1} more)" if len(hits) > 1 else ""),
             "read the existing log instead of spending the GPU; if the settings genuinely differ, say which "
             "and re-run with --force '<what differs and why it matters>'")


def _base(x):
    import os as _o
    return _o.path.basename(str(x).rstrip("/")) if x and str(x).lower() != "none" else None


def check_adapters(adapters):
    """An adapter that loads as a no-op silently evaluates the BASE model, and the number gets attributed to
    the training method. That is exactly what made 'alternation round 2' read as 26.9%."""
    from scripts.audit_corpus_supervision import VOID_ADAPTERS
    for a in adapters or []:
        for part in str(a).replace("+", ",").split(","):
            part = part.strip()
            if not part or part.lower() == "none":
                continue
            base = os.path.basename(part.rstrip("/"))
            if any(base.startswith(v) for v in VOID_ADAPTERS):
                fail("adapter", f"{base} is GOLD-SUPERVISED (void)",
                     "it may not appear in any table; pick a v5+ adapter")
            if not os.path.exists(os.path.join(part, "adapter_model.safetensors")):
                fail("adapter", f"{part} has no adapter_model.safetensors", "check the path")
                continue
            try:
                from safetensors import safe_open
                with safe_open(os.path.join(part, "adapter_model.safetensors"), "pt") as h:
                    keys = list(h.keys())
                if any(k.startswith("base_model.model.base_model.model.") for k in keys):
                    fail("adapter", f"{base} has DOUBLE-PREFIXED keys (nested PEFT wrap)",
                         "it will load as a no-op; evaluate it stacked (`parent+child`) or strip the prefix")
                # ★ ALL-ZERO lora_B (2026-08-13). The check above only caught ONE way an adapter can be a
                # no-op. PEFT initialises lora_B to zero, so a checkpoint that never completed training is
                # a perfect no-op with perfectly normal keys — and this function passed one.
                # reader_binding_v15minpair_r16_s1150 died inside save_pretrained on a quota error after
                # 1150 steps; what survived on disk was the PREFLIGHT SAVE written at startup, 196 lora_B
                # tensors all zero. preflight said "clear". Evaluating it would have produced "the new
                # corpus changes nothing" — a false negative that kills a live direction.
                zb = []
                with safe_open(os.path.join(part, "adapter_model.safetensors"), "pt") as h:
                    for k in keys:
                        if "lora_B" in k:
                            zb.append(bool(h.get_tensor(k).abs().max().item() > 0))
                if zb and not any(zb):
                    fail("adapter", f"{base} has {len(zb)} lora_B tensors and ALL are ZERO",
                         "this is an untrained checkpoint (PEFT zero-inits lora_B) — it evaluates the BASE "
                         "model. Check whether training died at save time, and re-train")
            except Exception as e:
                warn("adapter", f"could not inspect {base}: {e}")


def check_bench_settings(bench, max_new, reason_hist):
    """bench_config is the source of truth. A launcher default (MAXNEW=128) silently overrode LoCoMo's
    canonical 200 and produced a whole second, non-comparable family of logs."""
    if not bench:
        fail("bench", "no --bench", "pass --bench <key>; without it provenance is incomplete and "
                                    "build_table.py refuses the run")
        return
    try:
        import scripts.bench_config as BC
        c = BC.get(bench)
    except Exception as e:
        warn("bench", f"could not read bench_config for {bench}: {e}")
        return
    if max_new is not None and int(max_new) != int(c["max_new"]):
        fail("settings", f"max_new={max_new} but bench_config says {c['max_new']}",
             "use the canonical value, or the run is not comparable to any other")
    if reason_hist and c.get("reason_hist") and reason_hist != c["reason_hist"]:
        fail("settings", f"reason_hist={reason_hist} but bench_config says {c['reason_hist']}",
             "use the canonical value")


def check_lambda(train_lam, eval_lam):
    """A branch trained inside the fusion is fitted to supply the (1-lambda) residual of THAT mixture."""
    if train_lam is not None and eval_lam is not None and abs(float(train_lam) - float(eval_lam)) > 1e-9:
        fail("lambda", f"train-λ {train_lam} != eval-λ {eval_lam}",
             "match them, or state explicitly that this is an off-design probe")


def check_docs_cache(bench, doc, split="validation"):
    """A retrieval benchmark must read a PRECOMPUTED docs cache. Building the BM25 corpus inside the GPU job
    holds the card idle for hours: a hotpot d40 run loaded 39 GiB of weights in 51 seconds and then sat for
    3 hours building the index before its time limit killed it, with zero output. The cache and the loader
    hook (HOTPOT_DOCS_CACHE_DIR / MUSIQUE_DOCS_CACHE_DIR) already existed — the job simply did not set it."""
    if bench not in ("hotpotqa", "musique") or not doc:
        return
    f = os.path.join(REPO, "results/_docs_cache", f"{bench}_{split}_d{doc}.json")
    if not os.path.exists(f):
        fail("docs-cache", f"no precomputed docs for {bench} d{doc}",
             f"run scripts/precompute_{'hotpot' if bench=='hotpotqa' else 'musique'}_docs.py on CPU first "
             f"(no --gpus), then pass {bench.upper().replace('QA','')}_DOCS_CACHE_DIR=results/_docs_cache")
    elif not os.environ.get("HOTPOT_DOCS_CACHE_DIR") and not os.environ.get("MUSIQUE_DOCS_CACHE_DIR"):
        fail("docs-cache", f"{os.path.basename(f)} exists but no *_DOCS_CACHE_DIR is set",
             "export HOTPOT_DOCS_CACHE_DIR=results/_docs_cache (or MUSIQUE_...) or the job rebuilds BM25 "
             "on the GPU node with the card idle")


def check_gpus(gpus, needs_parallel_streams=False, fusion_train=False):
    """GPUs are the scarce resource; a sequential fusion decode gains nothing from a second card.

    ★ 2026-08-13: the rule was written for the DECODE harnesses and had nothing to say about training, so
    it refused the 2 GPUs that fusion training actually needs while waving through the 1 GPU that OOMs.
    `v12fused` (job 2937597) died at step 2 of 1150 — 91.3 of 95.1 GiB in use, trying to allocate 5.10 GiB
    more — because `fusion_stage2_lm_sft.py` holds BOTH models resident (32B 61 GiB + 7B 15 GiB = 76 GiB of
    weights) plus grad activations for the trained branch and a full logits tensor per branch at
    max-length 28000. Inference-only decode has none of that, which is why one card is right there and
    wrong here."""
    if fusion_train:
        if gpus and int(gpus) < 2:
            fail("gpus", f"--gpus-per-node={gpus} for FUSION TRAINING (both models resident + grad "
                         f"activations)",
                 "use 2 GPUs — 1 card is a measured OOM (job 2937597 died at step 2/1150 with 91.3/95.1 "
                 "GiB in use); this is not the sequential-decode case the 1-GPU rule is about")
        return
    if gpus and int(gpus) > 1 and not needs_parallel_streams:
        fail("gpus", f"--gpus-per-node={gpus} for a sequentially-decoding job",
             "use 1 GPU unless the code overlaps the two forwards with real parallel CUDA streams")


# ---------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=("eval", "train"))
    ap.add_argument("--batch", type=int)
    ap.add_argument("--bench")
    ap.add_argument("--max-new", type=int)
    ap.add_argument("--reason-hist")
    ap.add_argument("--corpus")
    ap.add_argument("--out")
    ap.add_argument("--adapter", action="append")
    ap.add_argument("--train-lam", type=float)
    ap.add_argument("--eval-lam", type=float)
    ap.add_argument("--gpus", type=int)
    ap.add_argument("--doc", type=int, help="doc-number for a retrieval benchmark")
    ap.add_argument("--parallel-streams", action="store_true")
    ap.add_argument("--fusion-train", action="store_true",
                    help="fusion_stage2_lm_sft.py: BOTH models resident + grad activations, so "
                         "1 GPU is a measured OOM and >=2 is required")
    ap.add_argument("--model", help="LM branch (for the duplicate-run check)")
    ap.add_argument("--slm-model", help="reader branch (for the duplicate-run check)")
    ap.add_argument("--method", help="ours / teacher / snapkv_frozen / ... (for the duplicate-run check)")
    ap.add_argument("--force", help="record an override WITH A REASON; it does not silence the check")
    a = ap.parse_args()

    check_batch(a.batch)
    check_gpus(a.gpus, a.parallel_streams, a.fusion_train)
    check_adapters(a.adapter)
    # ★ the cheapest check there is: has this exact run already been done?
    if a.kind == "eval" and a.model:
        check_already_exists(a.bench, a.model, a.slm_model, a.adapter, a.eval_lam, a.method)
    if a.kind == "train":
        check_disk(a.out, need_gib=8.0)
        check_corpus_supervision(a.corpus)
        check_teacher_targets(a.corpus)
    else:
        check_bench_settings(a.bench, a.max_new, a.reason_hist)
    check_docs_cache(a.bench, a.doc)
    check_lambda(a.train_lam, a.eval_lam)

    for rule, detail in WARN:
        print(f"⚠️  [{rule}] {detail}")
    if not FAIL:
        print("✅ preflight clear — submit.")
        return 0
    print(f"\n🚫 PREFLIGHT REFUSED ({len(FAIL)} failing):")
    for rule, detail, fix in FAIL:
        print(f"  ✗ [{rule}] {detail}\n      → {fix}")
    if a.force:
        with open(os.path.join(REPO, "preflight_overrides.log"), "a") as f:
            f.write(json.dumps(dict(argv=sys.argv[1:], reason=a.force,
                                    failed=[r for r, _, _ in FAIL])) + "\n")
        print(f"\n⚠️  OVERRIDDEN with reason: {a.force}  (recorded in preflight_overrides.log)")
        return 0
    print("\nFix these, or re-run with --force '<why this is acceptable>' — the override is logged.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
