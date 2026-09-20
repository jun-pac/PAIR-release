# PAIR, a large query LM fused with a small full-context reader

Two models decode one answer together. The small reader holds the retrieved context and the large
query LM sees only the question, and at every step the next token is drawn from a fixed linear pool of
their raw logits, `softmax(lambda * z_reader + (1 - lambda) * z_queryLM)`. Neither model is a router
and neither one vetoes the other. The large model never builds a KV cache over the context, which is
where the memory and the throughput come from, and the reader supplies the facts the question needs.
The released pair is Qwen2.5 32B-Instruct with Qwen2.5 7B-Instruct, both carrying a LoRA adapter, at
`lambda = 0.7` on HotpotQA, MuSiQue and LooGLE and `0.85` on LoCoMo. Both adapters are trained only
against the 32B teacher's own outputs, never against dataset answers.

This tree holds the code that produced the paper's experiments, the prompts, the training corpus in
the two views described in `data/README.md`, and the figure PDFs. It is the paper's code, not a
library, so the layout follows the experiments rather than an install.

## The tree

| Path | Contents |
|---|---|
| `src/` | The method. `fusion.py` is the fused decoder, `qa_prompts.py` the prompts, `flash_kvcache_decode.py` and `batched_stateful.py` the batched decode path, `models.py` the loading and placement, `eval.py` the metrics |
| `scripts/` | Experiment entry points and the SLURM launchers that ran them |
| `data/` | The training corpus, two views. See `data/README.md` |
| `figures/` | The figure PDFs as they appear in the paper |

## One harness runs every accuracy and throughput cell

`scripts/mtrag_accum.py` runs every arm of every accuracy and throughput table. The method is chosen
by `--method`, the benchmark by `--bench`, and a launcher differs from another only in its
environment. The canonical PAIR arm on HotpotQA, taken from `scripts/run_hotpot_ours_lam07.slurm`,
is

```bash
BATCH_SIZE=16 MERGE_LORA=1 python scripts/mtrag_accum.py --bench hotpotqa_st40_full --max-conv 3000 \
  --model Qwen/Qwen2.5-32B-Instruct --max-new 200 --method ours --lam 0.7 --lm-no-accum \
  --slm-model Qwen/Qwen2.5-7B-Instruct --slm-lora $CKPT/reader_binding_v5distill_r16_s900 \
  --lm-lora $CKPT/stage2_on_v5reader_lam07_r16_s600 \
  --out results/fusionft/hofa_ours.jsonl
```

The two baselines of every table are the same harness with `--method teacher` at 32B (the full
context ceiling) and at 7B (the reader alone). The compression arms are `--method quant_int8`,
`quant_int4`, `snapkv_frozen`, `pyramidkv_frozen`, `specprefill` and `h2o`, and that last flag name is
a legacy alias in our harness which selects kvpress's expected-attention eviction. The H2O method
itself was never run here, and the paper names that arm expected attention.

Per-benchmark settings live in `scripts/bench_config.py`, which is the single source of truth for the
prompt, the extractor, the metric, the generation length, the document count, the truncation, the
order and the seed. `python scripts/bench_config.py` prints them. Every run records a fingerprint of
those settings, and `scripts/build_table.py` refuses to put two runs in one table when their
fingerprints differ.

## The prompts

Every prompt is a named constant in `src/qa_prompts.py`, and `scripts/bench_config.py` names the one
each benchmark uses. All four benchmarks use the same instruction, `QA_REASON_V3`, except LoCoMo,
which uses `QA_REASON_V3_LOCOMO`, and the difference is that the LoCoMo variant describes a dated
conversation instead of documents. The reader branch, the query LM branch and the fused decode all
receive byte-identical prompts inside a run, which is what makes the per-branch divergence
measurable. `scripts/dump_prompts.py` writes out the exact assembled prompts, and
`scripts/chat_wrap.py` holds the per-family chat template used for the OLMo-3 and Gemma-3 pairs.

## Which code produced which figure

The plotting code is not part of this release. This table names the launcher that produced each
figure's measurements, which is the part that can be re-run.

| Figure | Launcher | What it measures |
|---|---|---|
| `main_efficiency` | `run_kv_resident_probe.slurm`, `run_timing_grid_bmax.slurm` | Retained KV bytes at answer time, and throughput at each arm's own largest batch |
| `paper_context_yarn8`, the two goodput variants | `run_context_axis.slurm`, `run_context_axis_yarn.slurm`, `run_context_ttft_b1.slurm` | F1 and throughput against context depth, and time to first token at batch 1 |
| `paper_throughput_vs_batch`, the two goodput variants | `run_batch_sweep_fkv.slurm` | Throughput against batch size for the three arms, with each arm's F1 |
| `paper_xfam_context`, `paper_xfam_context_kv` | `run_xfamily_context.slurm`, `run_kv_resident_xfam.slurm` | The same transfer measured on Qwen2.5 32B+3B, OLMo-3 32B+7B and Gemma-3 12B+4B |
| `paper_branch_kl_grid` | `run_branch_kl_grid.slurm` | Per-branch KL to the full-context 32B over a grid of reader sizes |
| `paper_decisive_token`, `paper_example_topk` | `run_branch_kl_fused.slurm`, `run_decisive_token_probe.slurm`, `run_teacher_own_fork.slurm` | One answer position by position, and the next-token distribution of each branch |
| `paper_fusion_intuition` | `scripts/fusion_intuition.py` over the dump from `scripts/dump_divergence_logits.py` | How often each branch is the wrong one, and whether the pool follows the wrong branch |
| `paper_turn_position` | `scripts/turn_position_analysis.py` over the accuracy logs | F1 against a question's position in its conversation |
| `fig1_banner` | `run_kv_resident_probe.slurm`, `run_decisive_token_probe.slurm` | The banner's KV figures and its worked example |

Two scripts in that table both compute and draw. They are included because the measurement is
defined inside them.

## Training the pair

Stage 1 fine-tunes the reader on the teacher's targets, stage 2 fine-tunes the query LM with the
reader frozen and `lambda` fixed at the value the arm will be evaluated at. Train-time `lambda` and
eval-time `lambda` must be equal.

```bash
sbatch scripts/run_v5seed777_train.slurm     # stage 1, the 7B reader, 900 steps, LoRA r16
sbatch scripts/run_stage2_sft.slurm          # stage 2, the 32B query LM, 600 steps, LoRA r16
```

`scripts/reader_binding_sft.py` and `scripts/fusion_stage2_lm_sft.py` are the two trainers.
`scripts/fusion_distill_train.py` is the variant that trains against the fused distribution rather
than one branch, launched by `scripts/run_fusion_distill_2gpu.slurm`, and
`scripts/run_ft_ablation.slurm` evaluates the resulting adapters through the same accuracy harness.
The corpus build is described in `data/README.md`.

The adapter weights are not in this release.

## Environment

Python 3.11.9, torch 2.11.0 with CUDA 12.8, transformers 4.57.6, peft 0.19.1, accelerate 1.10.1,
datasets 2.21.0, flash-attn 2.8.3, hqq 0.2.8 for the quantized-cache arm and kvpress for the
eviction arms. The measurements were taken on NVIDIA GH200 cards with 95 GB each, one card per run,
and every pair in the paper fits on one card. The fused decode needs flash attention compiled,
because a batch of one takes a different decode path and its timings are not comparable.

## The measurements behind the figures

`results/timing/` holds the stores the figures are drawn from, as JSON. Each figure PDF in
`figures/` has its numbers here, so every value the paper reports can be read back without running
anything.

| Store | What it holds |
|---|---|
| `kv_resident.json`, `kv_resident_xfam.json` | Retained key and value bytes at answer time, per arm and per context depth |
| `context_axis.json`, `context_ttft_b1.json`, `context_throughput_yarn8.json` | F1 and throughput against context depth, and time to first token at batch 1 |
| `batch_sweep_fkv.json` | Throughput against batch size for the three arms |
| `xfam_context.json` | The same transfer on the three other model pairs |
| `branch_kl_grid_n240/lm*.json`, `branch_kl_n240_headline.json` | Per-branch KL to the full-context 32B over the 240 example basis |
| `decisive_token.json`, `example91_kl.json` | One answer position by position, with each branch's next-token distribution |
| `fusion_intuition.json`, `turn_position.json` | How often each branch is the wrong one, and F1 against a question's position |
| `prompts.json` | The assembled prompts of all four benchmarks, regenerated by `scripts/dump_prompts.py` |

The other stores in that directory are measurements from the same campaign that no figure uses.

## Paths in these scripts

Every absolute path has been rewritten for release. `/u/anon/SLM_LM` is this tree,
`/u/anon/.venvs/slm_lm` is the virtual environment, `/work/hdd/myproject/anon` is the scratch space that
holds the HuggingFace cache and the adapter checkpoints, and `my-slurm-account` is the SLURM account.
Point those four at your own before running a launcher.

## What is not in this tree

The per-example result logs, the analysis and report documents, the plotting code, the dashboard
pages, the session handoffs and the adapter weights. The measured values those logs were reduced to
are in `results/timing/`. The corpus ships in the two views of `data/README.md`
rather than in full, because its context field is 96% of its bytes.
