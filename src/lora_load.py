"""Adapter loading with a HARD no-op guard (2026-08-11).

WHY THIS FILE EXISTS. `round2_reader_on_stage2LM_lam085` was trained by wrapping an already-PEFT model a
second time (`PeftModel.from_pretrained` -> `get_peft_model`), so its checkpoint keys carry a DOUBLE prefix
(`base_model.model.base_model.model.model.layers...`). At eval it was loaded onto a plain 7B, every one of
its 392 keys was "missing", PEFT kept the fresh init (lora_B = 0) and the adapter became an exact NO-OP.
peft only printed a UserWarning, which scrolled past inside a 60KB log. The run scored 26.9% and was
recorded as "alternation collapses at round 2" — a conclusion about a *training method* that was really a
silent load failure. Never again: a LoRA that does not change the model must ABORT the job.

Two things are enforced here:
1. `load_lora_stack(model, "a,b")` wraps adapters SEQUENTIALLY, exactly mirroring how nested training
   builds them, so a round-2 adapter can be evaluated on top of its round-1 parent.
2. every adapter must be ACTIVE after loading: at least one `lora_B` of that adapter is non-zero. A
   trained LoRA always satisfies this (B starts at 0 and only training moves it), so this catches
   key-name mismatches, empty checkpoints, and "I forgot to train it" in one check.
"""
from __future__ import annotations


def _live_lora_b(model) -> int:
    """number of lora_B tensors with any non-zero weight"""
    n = 0
    for name, p in model.named_parameters():
        if "lora_B" in name and p.detach().abs().sum().item() > 0:
            n += 1
    return n


def load_lora_stack(model, spec: str, label: str = "adapter", merge_final: bool = False):
    """Load one or more LoRA adapters onto `model`, stacked in order. Aborts on a no-op adapter.

    Every adapter but the last is MERGED into the weights before the next is applied — the same thing the
    trainer does for round 2+, so a child adapter's ordinary (single-prefix) keys line up.

    merge_final=True also merges the LAST adapter (merge_and_unload) so eval runs on plain weights —
    removes the unmerged-LoRA decode overhead (measured +5–35 %/adapter). In-memory only, seconds, no
    disk cost. bf16 rounding can flip a few greedy paths vs unmerged, so the flag is recorded in
    provenance by the callers.
    """
    from peft import PeftModel

    # '+' is accepted as well as ',' because sbatch --export splits on commas
    parts = str(spec).replace("+", ",").split(",")
    paths = [p.strip() for p in parts if p.strip() and p.strip().lower() != "none"]
    for k, path in enumerate(paths):
        if k:                                   # bake the parent in before stacking the child
            model = model.merge_and_unload()
        before = _live_lora_b(model)
        model = PeftModel.from_pretrained(model, path)
        after = _live_lora_b(model)
        if after <= before:
            raise SystemExit(
                f"❌ {label} adapter is a NO-OP: {path}\n"
                f"   every lora_B loaded as zero -> the checkpoint's keys do not match this model.\n"
                f"   Most likely cause: the adapter was trained on top of ANOTHER adapter (nested PEFT wrap,\n"
                f"   keys prefixed 'base_model.model.base_model.model...'). Pass the full stack in order,\n"
                f"   e.g. --slm-lora <round1>,<round2>. Refusing to run — an inactive adapter silently\n"
                f"   evaluates the BASE model and the number would be attributed to the training method."
            )
        print(f"[Info] {label} adapter: {path}  ({after - before} live lora_B tensors)", flush=True)
    if merge_final and paths:
        model = model.merge_and_unload()
        print(f"[Info] {label}: final adapter MERGED into weights (merge_final)", flush=True)
    return model
