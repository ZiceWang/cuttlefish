# `experiments/` — code index

The code is split into themed subdirectories. Two shared library directories
(`models/`, `baselines/`) hold importable modules; the other directories hold
runnable entry points.

```text
experiments/
├── models/       # Cuttlefish core (signed-graph gated message passing)
├── baselines/    # competing architectures (PK baselines)
├── sudoku/       # Sudoku-9 / Maze training and evaluation
├── lm/           # large-scale language-model training + small-scale comparison
├── posttrain/    # SFT / arithmetic / GRPO / checkpoint merging
├── datasets/     # data builders and tokenizer-budget utilities
└── analysis/     # probes, audits, plotting / visualization
```

## How scripts resolve imports and paths

Every script in `sudoku/ lm/ posttrain/ datasets/ analysis/` starts with a small
**path bootstrap** that adds `experiments/` and all of its subdirectories to
`sys.path` and points `HERE` at the repository root. That keeps the original
convention working unchanged:

- flat sibling imports (`import bdh_best`, `from train_sft import ...`) resolve
  against `models/`, `baselines/` and the other theme directories;
- `HERE / "runs"`, `HERE / "data"`, `HERE / "bpe4096.json"` resolve against the
  repository root, where `runs/`, `data/` and `bpe4096.json` live.

`models/` and `baselines/` are pure library modules and carry no bootstrap.

Run any entry point directly from the repository root, e.g.:

```bash
python experiments/lm/train_big.py --help
python experiments/sudoku/train_sudoku9_compare.py --help
python experiments/posttrain/train_sft.py --help
```

## `models/` — Cuttlefish core

| File | Description |
|---|---|
| `bdh_orig.py` | Original BDH: single shared encoder + unnormalized Gram graph + product binding. |
| `bdh_best.py` | Best BDH variant: `gram_swiglu`, `separate_qk` + `relu_qk`, `normalize_gram`, RoPE. |
| `gram_norm.py` | `GramAttentionNorm` — degree-normalized Gram graph for stable linear aggregation. |
| `rope_group_gate.py` | `CutDeep-Norm` with a low-cost nonlinear gate aligned to RoPE frequency pairs. |
| `linear_fish.py` | `LinearFish` — exact linearization of the unnormalized numerator, O(T²) → O(T). |
| `linear_rope_group_gate.py` | Linear-time RoPEGate8 with signed messages and separable scale normalization. |
| `shared_gram.py` | `SharedGram` — relation-graph operator weights shared across all layers. |
| `cut_noemb.py` | CutDeep-Norm without the learned input position embedding. |
| `mix_model.py` | Mixed BDH + Transformer stacking. |

## `baselines/` — competing architectures (PK)

| File | Description |
|---|---|
| `transformer.py` | Standard pre-norm Transformer with scaled dot-product attention + SwiGLU FFN. |
| `transformer_rope.py` | Transformer without learned position embedding, RoPE on Q/K (theta = 2^16). |
| `transformer_cosine.py` | TF-RoPE but softmax attention replaced by degree-normalized cosine graph messages. |
| `hyena.py` | Hyena operator + block, based on the official `HazyResearch/safari` implementation. |
| `hyena_rope.py` | Hyena with RoPE instead of learned position embeddings. |
| `hyena_dilated.py` | `HyenaDilated` — FFT long convolution replaced by a dilated-convolution pyramid. |
| `hyena_fish.py` | `HyenaFish` — Hyena skeleton + Cuttlefish-style graph operator. |
| `gated_fish.py` | `GatedFish` — Hyena recurrent state evolution × Cuttlefish dynamic graph projection. |
| `liv.py` | LIV convolution block (Gated-Short-Conv) + aligned language-model wrapper. |
| `gdn2_sudoku.py` | Small autoregressive GDN-2 model for the Sudoku-9 comparison (needs `lit_gpt`). |

## `sudoku/`

| File | Description |
|---|---|
| `train_sudoku9_compare.py` | Matched autoregressive Sudoku-9 training across architectures. |
| `train_sudoku9_bdhbest.py` | Sudoku-9 training for the widened-GLU `bdh_best` architecture. |
| `train_sudoku9_gdn2.py` | Train official GDN-2 on the autoregressive Sudoku-9 task. |
| `eval_sudoku9_official.py` | Official 4096-example Sudoku-9 evaluation using each model's `best.pt`. |
| `eval_sudoku9_extreme.py` | Evaluate four architectures on real Sudoku-Extreme puzzles. |
| `eval_sudoku9_gdn2.py` | Evaluate the official GDN-2 Sudoku checkpoint on a fixed held-out set. |
| `train_maze_compare.py` | Maze-30x30 four-architecture comparison (matched params, path-masked loss). |

## `lm/` — large-scale language model

| File | Description |
|---|---|
| `train_big.py` | Large-corpus training: FineWeb/DCLM mix, byte-level vocab 256, ~10M–100M params. |
| `sample_big.py` | Byte-level sampling from a large-corpus `bdh_best` checkpoint. |
| `eval_easy_bench.py` | Four simple benchmarks: PIQA / HellaSwag / Winogrande / ARC-Easy (likelihood). |
| `eval_mmlu.py` | MMLU four-option zero-shot log-likelihood evaluation. |
| `eval_remove_signedexp.py` | Zero-shot ablation removing SignedExp shaping from a trained checkpoint. |
| `train_ultrafineweb_continue.py` | Continue the 4096-token Cuttlefish pretrain on English Ultra-FineWeb shards. |
| `bench_long_bdhbest.py` | Long-sequence check on `bdh_best`: numerical stability, memory, speed. |
| `scale_sweep.py` | Width/depth sweep at fixed params; bf16 + compile throughput measurement. |
| `train_orig.py` | Small-corpus baseline training run. |
| `train_compare.py` | Unified 5-layer, ~10M-param comparison of five modules on the same pipeline. |
| `sample_compare.py` | Rolling text sampling from four trained checkpoints for side-by-side comparison. |

## `posttrain/` — SFT / GRPO / checkpoint merging

| File | Description |
|---|---|
| `train_sft.py` | ChatML SFT for the 136M SignedExp-noabs model (assistant-only loss). |
| `sample_chatml.py` | Sample an SFT checkpoint with native ChatML control tokens. |
| `eval_sft_checkpoints.py` | Evaluate SFT state-dict checkpoints on one fixed ChatML validation stream. |
| `test_sft_pipeline.py` | Smoke test for the packed ChatML SFT pipeline. |
| `train_synthetic_arithmetic_qa.py` | Short continued pretraining on deterministic synthetic arithmetic QA. |
| `train_diverse_arithmetic_qa.py` | Answer-masked training on diverse symbolic and natural-language arithmetic QA. |
| `eval_diverse_arithmetic_checkpoint.py` | Evaluate the diverse-arithmetic checkpoint. |
| `eval_final_arithmetic_qa.py` | Probe arithmetic ability of the final continued checkpoint in QA format. |
| `eval_arithmetic_template_robustness.py` | Arithmetic generalization under unseen question phrasings. |
| `train_math_grpo_probe.py` | Short verifiable-reward GRPO probe on two-digit addition candidates. |
| `mix_checkpoints.py` | Linearly merge two architecture-identical state dicts (lerp / task-vector). |

## `datasets/`

| File | Description |
|---|---|
| `build_sft_mixture.py` | Build an English-first ChatML SFT mixture without a context-length cutoff. |
| `build_sft_additional.py` | Build a balanced 150k-example SmolTalk increment in native ChatML form. |
| `build_sft_expansion.py` | Cross-deduplicated expansion to ~5× v1 of the SFT pool. |
| `diverse_arithmetic_generator.py` | Programmatic arithmetic-expression and word-problem generator. |
| `prepare_gsm8k.py` | Download and normalize the complete GSM8K train split (non-streaming). |
| `compare_tokenizer_budgets.py` | Token-budget conversion on identical pretraining documents. |
| `count_probe_tokens.py` | Count tokens of the Ultra-FineWeb probe shards. |

## `analysis/`

| File | Description |
|---|---|
| `probe_ood_arithmetic_formats.py` | Probe out-of-distribution arithmetic formats. |
| `probe_apple_word_problem.py` | Probe Apple-style word problems. |
| `inspect_math_grpo.py` | Diagnose whether the addition GRPO probe learned arithmetic or answer bias. |
| `inspect_poly3_errors.py` | Inspect polynomial-degree-3 arithmetic errors. |
| `score_suffixes.py` | Score candidate continuations by conditional causal-LM log likelihood. |
| `sample_long_ropegate.py` | Generate past the training context, printing boundary-aligned segments. |
| `compare_continue_samples.py` | Identical-seed generation comparison for base and continued checkpoints. |
| `plot_frequency_river.py` | Flow-style view of RoPE frequency routing across Cuttlefish depth. |
| `plot_head_constellation.py` | Circular signed-relation constellations for final-layer Cuttlefish heads. |
| `plot_layer_fingerprints.py` | Radial glyph fingerprints for each canonical Cuttlefish layer. |
| `plot_retrieval_storyboard.py` | Sentence-level storyboard of signed retrieval for several query positions. |
| `visualize_cuttlefish.py` | Publication-style mechanism visualizations for canonical RoPEGate8NoAbs. |
