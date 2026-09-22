# Changelog

Cuttlefish was developed as a research line inside a fork of the `nano-trm` project. That fork's
granular history — **282 commits from 2026-08-22 to 2026-09-12** — plus two stretches of uncommitted
work have been collapsed into the single initial commit of this repository.

This file reconstructs that development record from the old repository so the reasoning path is not
lost. Entries are grouped by phase; the dates are those of the original work.

> Provenance note: the phases marked **(uncommitted)** were present as working-tree files in the old
> repository but were never committed there. They were recovered into this repository and are listed
> here from the file inventory rather than from git history.

---

## 2026-08-22 — Repository bootstrap

- `Init`: snapshot of the `nano-trm` Tiny-Recursive-Model codebase (Sudoku-Extreme / Maze) used as
  scaffolding for the graph-model experiments.

## 2026-08-23 – 2026-08-27 — BDH "Theseus" architecture search (~195 commits)

Starting from the public BDH block, major components were replaced one at a time while preserving a
causal all-position language model, an explicit dynamic-graph interpretation and a recurrent /
streaming potential. Most branches were recorded as `experiment: …` followed by
`result: accept/reject …`, and rejected branches were reverted.

- **Graph routing**: relative graph kernel (rejected), signed sparse graph routing (accepted),
  dense signed transport, implicit latent graph token mixer, continuous coordinate state,
  hierarchical local-global graph mixer.
- **Latent routes**: bounded evolving latent routes, independent relation evolution (rejected),
  compressed route rank, learned evolution strength (rejected), refinement-only evolution.
- **Association memory**: replace the BDH Gram graph with association slots (accepted, surpassed
  BDH); wider association memory; tied association-dictionary transport.
- **Position**: replace BDH RoPE with coordinate phases (deferred); absorb positional relations into
  the evolving graph state (removed RoPE there).
- **Recurrence & depth**: one shared graph call per reasoning update; fused graph + channel evolution;
  front-loaded graph evolution; pre-norm graph residual path (fixed gradient collapse across depth);
  removed redundant graph-state normalization; initial-only input injection.
- **Operator fusion & speed**: fuse graph write / evolve / read; lean graph; low-rank token routing
  (selected rank 64); latent-node budget trade-off; persistent latent graph state; batched-inference
  memory tests; long-context graph/Transformer speed crossover.
- **Relations & heads**: split latent propagation into heads (rejected); fused multi-relation routes;
  dynamic relation heads for causal BDH; gated graph memory replacing Hebbian accumulation;
  differential and cheap additive sparse binding; direct graph state (rejected); SwiGLU state
  readout / binding; tied encoder-decoder dictionary; grouped decoder compression + fused grouped
  decoder.
- **Baselines & tasks**: size-matched causal Transformer; original BDH causal baseline; Shakespeare
  depth scaling and autoregressive generation checkpoints; Sudoku-9 comparison of original vs
  compressed BDH; causal Delta-rule memory reference; cosine causal relation.
- `docs: add BDH Theseus experiment handoff`.

## 2026-08-29 – 2026-08-30 — Cuttlefish blocks, graph attention/transformer, compile & linear modes

- Baseline: cached RoPE cos/sin; 10M mixed-corpus and module benchmark harness.
- **Block knobs**: `use_rope` / `normalize_gram` / `feature_multiplier`; `rel_bias` position mode;
  `value_proj` / `head_metric`; binding toggle; per-token SwiGLU FFN sub-block; GAT-style softmax
  edges (with a self-loop fix for empty causal neighborhoods).
- **GraphAttentionBlock**: multi-head causal Gram graph with learned value / head-scale and per-head
  binding; separate query projection; separate QK projections (`qk` matrix instead of `gram`).
- **Linear-attention forms**: `linear_attn` + `--linear-chunk`; chunked mathematically identical
  Gram; parallel-scan (cumsum) linear mode that removes the serial chunk dependency and is
  torch.compile-friendly.
- **GraphTransformerBlock**: proper gated graph with zero-init ReZero output, RoPE on graph keys.
- `lean_graph` block + `gram_dim`; `AttnGraphBlock` (Transformer softmax attention + Gram message).
- `perf`: torch.compile gives L5R4 h232 **65 → 34.6 ms/step (118k tok/s)**.
- `tool`: `sample_model.py` byte-level sampling from checkpoints.
- `bench`: context sweep O(t²) vs O(t) — quadratic wins to ~10k.

## 2026-08-31 – 2026-09-10 — Matched baselines & official evaluation **(uncommitted)**

- **PK baselines**: `transformer.py`, `transformer_rope.py`, `transformer_cosine.py`, `hyena.py`
  (+ `hyena_rope`, `hyena_dilated`, `hyena_fish`), `gated_fish.py`, `liv.py`, `gdn2_sudoku.py`.
- **Cuttlefish family**: `gram_norm.py` (degree-normalized CutDeep-Norm), `rope_group_gate.py`
  (RoPEGate8), `linear_fish.py` (exact O(T) numerator), `linear_rope_group_gate.py`,
  `shared_gram.py`, `cut_noemb.py`, `mix_model.py`.
- **Sudoku**: `train_sudoku9_compare.py`, `train_sudoku9_bdhbest.py`, `train_sudoku9_gdn2.py`,
  `eval_sudoku9_official.py` (4096-example protocol), `eval_sudoku9_extreme.py`,
  `eval_sudoku9_gdn2.py`; `train_maze_compare.py`.
- **Small-scale LM comparison**: `train_orig.py`, `train_compare.py`, `sample_compare.py`.
- **Analysis / figures**: `plot_*`, `visualize_cuttlefish.py`, `scale_sweep.py`,
  `bench_long_bdhbest.py`.

## 2026-09-11 — 10B-token formal pretraining, eval harness, SFT data

- `train`: stream ten shards for one 512-token epoch; checkpoint formal runs every 10,000 steps.
- `bench`: steady training throughput and peak memory; 136M model benchmarked at 512 context.
- `sample`: sliding-window repetition penalty; explicit CPU inference.
- `eval`: zero-shot removal of signed exponential shaping.
- `data`: reproducible ChatML SFT mixture builder; select compact complete UltraChat turns; retain
  full conversations for context extension; defer SFT tokenization to training preparation; finalize
  the 105k ChatML SFT cherry mixture.

## 2026-09-12 — ChatML SFT pipeline & four-benchmark baseline

- `train`: ChatML assistant-only SFT pipeline (`train_sft.py`).
- `eval`: parameterize the four-benchmark evaluation by checkpoint and width (`eval_easy_bench.py`).
- `result`: record the 10B pre-SFT four-benchmark baseline (PIQA / HellaSwag / Winogrande / ARC-Easy).

## 2026-09-13 – 2026-09-15 — Post-training **(uncommitted)**

- **Checkpoint merging / task vectors**: `mix_checkpoints.py` (lerp and `a + b − base` task-vector
  merge); dialogue × arithmetic capability mixing; evaluation on the four benchmarks.
- **Arithmetic**: continued training and OOD template probes; GSM8K preparation;
  `eval_final_arithmetic_qa.py`, `eval_arithmetic_template_robustness.py`.
- **GRPO**: `train_math_grpo_probe.py` — verifiable-reward GRPO on two-digit addition
  (low-KL and adaptive-KL variants), lifting addition accuracy on the pretrain base.
- **Continued pretraining**: `train_ultrafineweb_continue.py` on English Ultra-FineWeb shards.

---

## This repository

- Extracted the Cuttlefish line out of the `nano-trm` fork into a standalone repository.
- Scrubbed environment-specific absolute paths; Hugging Face caches are now resolved through the
  `HF_HOME` environment variable (default `~/.cache/huggingface`).
- Migrated large local artifacts (`experiments/runs/`, `experiments/data/`) in place and gitignored
  them.
- Collapsed all of the above into a single initial commit.
