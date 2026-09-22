# Cuttlefish — Signed-Graph Gated Message-Passing Language Models

Cuttlefish is an autoregressive sequence model built from **causal, content-dependent graph
message passing**. Each layer builds its own directed relation graph between earlier and current
tokens, aggregates **signed** messages over that graph, and combines the message with the current
state through a wide SwiGLU readout.

This repository contains the model, the matched baselines it is compared against, the training and
evaluation entry points for both the **Sudoku reasoning** and the **large-scale language-model**
experiments, plus the post-training tooling (SFT / verifiable-reward GRPO / checkpoint merging).

The full technical write-up is in [`docs/CUTTLEFISH.md`](docs/CUTTLEFISH.md).

## Benchmarks

Sudoku-9 / Sudoku-Extreme reasoning, Maze-30x30, the 100M language model, the ~10B-token large-scale
pretraining run, the simple zero-shot likelihood benchmarks, and the post-training (GRPO / checkpoint
merging) results.

All numbers are the tracked records in `results/` (aggregate: `results/CUTTLEFISH_RESULTS.json`;
raw per-run metrics: `results/*.json`; selected logs: `results/logs/`). The full write-up is
`docs/CUTTLEFISH.md` §8. See [Limitations](#limitations) for known caveats.

### Sudoku-9 — autoregressive, 30k steps, 4096 held-out boards

| Model | Token acc | Exact-board acc |
|---|---:|---:|
| Transformer-RoPE (no learned position) | 0.9682 | 0.7676 |
| Cuttlefish RoPEGate8 (2,884,476 params) | 0.9660 | 0.7246 |
| Hyena | 0.9536 | 0.6252 |
| Transformer (learned position) | 0.9276 | 0.6072 |
| CutDeep-Norm | 0.9360 | 0.5981 |
| CutNoEmb | 0.9258 | 0.5691 |
| Shared relation graph (112112) + degree norm | 0.9088 | 0.5168 |
| Transformer-RoPE + cosine degree-norm kernel | 0.9194 | 0.4856 |
| LIV | 0.1117 | 0.0000 |
| LinearFish (unnormalized O(T) variant) | — | 0.3398 |

Evidence: `results/logs/eval_*.log`, `results/sudoku9_*.json`. A later RoPEGate8-SignedExp run reached
`0.7407` exact (see `results/sudoku9_official.json`, which the evaluator overwrites each run).

### Sudoku-Extreme — real puzzles, teacher-forcing token accuracy, n = 1000

| Difficulty filter | bdhbest | transformer | hyena | liv |
|---|---:|---:|---:|---:|
| ≥ 0 | 0.7964 | 0.7566 | 0.8518 | 0.1249 |
| ≥ 30 | 0.7956 | 0.7545 | 0.8475 | 0.1206 |
| ≥ 100 | 0.7918 | 0.7566 | 0.8440 | 0.1216 |

Autoregressive exact-board accuracy is `0.0` for every architecture at this training budget.

### Maze-30x30 — 1500 steps, matched ~2.87M parameters

| Arch | Params | Path F1 | Row exact | Auto-connected |
|---|---:|---:|---:|---:|
| hyena | 2,936,252 | 0.1458 | 0.0 | 0.0 |
| liv | 2,881,072 | 0.1386 | 0.0 | 0.0 |
| transformer | 2,865,744 | 0.1331 | 0.0 | 0.0 |
| bdhbest | 2,869,248 | 0.1259 | 0.0 | 0.0 |

### 100M language model — 80k steps, context 256, BPE-4096

| Model | Parameters | Best val loss | PPL | Token acc |
|---|---:|---:|---:|---:|
| Transformer, L12 d832 | 106,890,368 | 2.7243 | 15.2459 | 45.10% |
| Cuttlefish, L12 d450 | 105,861,600 | 2.8344 | 17.0209 | 43.59% |

Evidence: `results/big_night2_norm.json`, `results/big_night_tf2.json`, `results/logs/night2_norm.log`,
`results/logs/night_tf2.log`.

### Large-scale pretraining — 136M parameters, ~10B tokens

| Item | Value |
|---|---|
| Model | Cuttlefish RoPEGate8-SignedExp-noabs, R = 1 |
| Parameters | 136,511,488 |
| Corpus | 10 training shards (FinePDFs-Edu / DCLM / FineWeb-Edu), BPE-4096 |
| Context / tokens per step | 512 / 8,192 |
| Steps | 1,220,000 |
| Training tokens | 9,994,240,000 (≈ 7.73B SmolLM2-equivalent tokens) |
| Best eval loss / PPL | 2.4993 / 12.18 |
| Final eval loss / PPL / token acc | 2.5159 / 12.38 / 0.4805 |
| Throughput (512 ctx, batch 16) | ≈ 4.6×10⁴ tok/s eager; ≈ 1.25×10⁵ tok/s with `torch.compile` |
| Peak GPU memory | 15.85 GiB eager / 10.64 GiB compiled |

Four-benchmark evaluation of this checkpoint (pre-SFT):

| Benchmark | Random | 10B checkpoint | Transformer (reference) | n |
|---|---:|---:|---:|---:|
| PIQA | 0.50 | 0.6288 | 0.6421 | 299 |
| HellaSwag | 0.25 | 0.3033 | 0.3000 | 300 |
| Winogrande | 0.50 | 0.5100 | 0.4767 | 300 |
| ARC-Easy | 0.25 | 0.3993 | 0.3758 | 298 |

This checkpoint is also the base of the GRPO probe below, and appears as the "pre-SFT 10B baseline"
reference row in the checkpoint-merging table. Evidence: `results/logs/formal512_10shard_1ep.log`,
`results/logs/easy_bench_formal10b_pre_sft.log`, `results/easy_bench_formal10b_pre_sft.json`,
`results/tokenizer_budget_comparison.json`, `results/big_bench512b16*.json`.

### Simple zero-shot likelihood benchmarks

| Benchmark | Random | Cuttlefish | Transformer | n |
|---|---:|---:|---:|---:|
| PIQA | 0.50 | 0.6120 | 0.6421 | 299 |
| HellaSwag | 0.25 | 0.2767 | 0.3000 | 300 |
| Winogrande | 0.50 | 0.4900 | 0.4767 | 300 |
| ARC-Easy | 0.25 | 0.3826 | 0.3758 | 298 |

After continued pretraining on English Ultra-FineWeb (SignedExp-noabs checkpoint), the same four
benchmarks give PIQA 0.6120 / HellaSwag 0.2933 / Winogrande 0.5267 / ARC-Easy 0.3893
(`results/easy_bench_ultrafineweb_en_final.json`).

### Post-training — GRPO on two-digit addition (1000 steps)

| Base checkpoint | LR | KL | Before acc | After acc | Before P(correct) | After P(correct) |
|---|---:|---|---:|---:|---:|---:|
| 10B pretrain | 1e-6 | β = 0.03 | 0.0820 | 0.1660 | 0.0989 | 0.1341 |
| arithmetic SFT | 5e-7 | β = 0.10 | 0.0449 | 0.1504 | 0.0979 | 0.1222 |
| arithmetic SFT (adaptive KL, target 0.04) | 5e-7 | β = 0.10 → | 0.0449 | 0.1250 | 0.0979 | 0.1157 |

Evidence: `results/pretrain_add_grpo_1000.json`, `results/grpo_lowkl_arithmetic_1000.json`,
`results/grpo_adaptivekl_arithmetic_1000.json`.

### Post-training — checkpoint merging (four benchmarks, merged-model column)

| Variant | PIQA | HellaSwag | Winogrande | ARC-Easy |
|---|---:|---:|---:|---:|
| dialog 75 / arithmetic 25 | 0.6355 | 0.2933 | 0.5100 | 0.3926 |
| dialog 50 / arithmetic 50 | 0.6355 | 0.2767 | 0.5000 | 0.3725 |
| dialog 25 / arithmetic 75 | 0.6288 | 0.2767 | 0.4933 | 0.3960 |
| task-vector (dialog + arithmetic − base) | 0.6154 | 0.2967 | 0.5000 | 0.3490 |
| pre-SFT 10B baseline (reference) | 0.6288 | 0.3033 | 0.5100 | 0.3993 |

Evidence: `results/easy_bench_mix_*.json`, `results/easy_bench_taskvec_dialog_plus_arithmetic.json`,
`results/easy_bench_formal10b_pre_sft.json`.

### LinearFish parity

The unnormalized Gram numerator has an exact O(T) prefix form; with identical weights the maximum
logit difference versus the quadratic path is `0.0` (`results/logs/eval_linearfish.log`).

## Repository layout

```text
cuttlefish/
├── CHANGELOG.md               # reconstructed development history (from the old repo)
├── docs/
│   ├── CUTTLEFISH.md          # architecture, equations, design findings, results
│   ├── EXPERIMENT_MAP.md      # script -> result -> report section
│   └── REPRODUCIBILITY.md     # how each result was produced
├── experiments/               # all code, split by theme
│   ├── README.md              # per-directory file index
│   ├── models/                # Cuttlefish core modules
│   ├── baselines/             # competing architectures (Transformer / Hyena / LIV / GDN-2)
│   ├── sudoku/                # Sudoku-9 & Maze training/evaluation
│   ├── lm/                    # large-scale LM training + small-scale comparison
│   ├── posttrain/             # SFT / arithmetic / GRPO / checkpoint merging
│   ├── datasets/              # data builders & tokenizer-budget utilities
│   └── analysis/              # probes, audits, plotting / visualization
├── results/                   # small, tracked evidence
│   ├── *.json                 # per-run metric records
│   └── logs/                  # selected independent-eval / training logs
├── scripts/                   # shell helpers (sampling / data download)
├── runs/                      # checkpoints + full logs      (gitignored, ~87 GB)
├── data/                      # SFT / arithmetic datasets     (gitignored, ~3 GB)
├── bpe4096.json               # byte/BPE tokenizer used by the LM runs
├── pyproject.toml
└── LICENSE
```

> **How imports and paths stay working after the split.**
> `models/` and `baselines/` are importable libraries; every runnable script under
> `sudoku/ lm/ posttrain/ datasets/ analysis/` starts with a small path bootstrap that adds
> `experiments/` and its subdirectories to `sys.path` and points `HERE` at the repository root.
> Flat sibling imports (`import bdh_best`, `from train_sft import ...`) and root-relative paths
> (`HERE / "runs"`, `HERE / "data"`, `HERE / "bpe4096.json"`) therefore keep working unchanged.
> See [`experiments/README.md`](experiments/README.md).

## Install

```bash
cd cuttlefish
uv venv && uv pip install -e .      # or: pip install -r requirements
```

The original experiments ran on CUDA (H100) with `torch>=2.11`, `tokenizers>=0.23`,
`datasets>=4.0`, `transformers>=5.17`, `pyarrow`, `numpy`, `pandas`, `matplotlib`.

## Quickstart

```bash
# Sudoku-9, matched architecture comparison
python experiments/sudoku/train_sudoku9_compare.py --help
python experiments/sudoku/eval_sudoku9_official.py --help

# 100M language model
python experiments/lm/train_big.py --help
python experiments/lm/eval_easy_bench.py --help

# post-training
python experiments/posttrain/train_sft.py --help
python experiments/posttrain/train_math_grpo_probe.py --help
python experiments/posttrain/mix_checkpoints.py --help
```

Large artifacts (`runs/`, `data/`) are **not** in git. Datasets and checkpoints must be
regenerated or restored locally; see `docs/REPRODUCIBILITY.md`.

## Limitations

- **Head-to-head results.** On the tested matched settings Cuttlefish does not outperform the strongest
  Transformer baseline: Sudoku-9 exact-board 0.7246 vs 0.7676, and 100M-LM validation loss 2.8344 vs
  2.7243. On the simple zero-shot likelihood benchmarks the two are close, and MMLU stays near chance.
- **Generation quality.** The best 100M Cuttlefish checkpoint shows more semantic drift and repetition
  than the matched Transformer.
- **Complexity claim.** The literal signed absolute-degree divisor is not exactly prefix-factorizable
  and remains quadratic. The clean linear deployment form drops that divisor (it is largely cancelled
  by the following per-head LayerNorm) and uses the exact prefix numerator; finite-epsilon parity
  should be verified per checkpoint.
- **Sudoku-Extreme.** Autoregressive exact-board accuracy is 0.0 at the tested training budget.
- **Baselines.** `gdn2_sudoku` requires an external `lit_gpt` package that is not vendored here.

## Attribution

The baseline implementations build on public reference code: the original BDH block, Hyena
(`HazyResearch/safari`), LIV / Gated-Short-Conv, GDN-2, and a standard pre-norm Transformer.
The repository scaffolding was separated out of a fork of `nano-trm` (MIT).
