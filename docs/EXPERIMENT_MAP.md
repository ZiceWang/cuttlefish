# Experiment map

How the code, the tracked evidence in `results/`, and the report map onto each other.

Scripts are named below without their directory; they live under the matching theme
subdirectory of `experiments/` (`models/`, `baselines/`, `sudoku/`, `lm/`, `posttrain/`,
`datasets/`, `analysis/`) — see [`experiments/README.md`](../experiments/README.md).

## 1. Sudoku-9 (reasoning)

| Model | Train | Evaluate | Evidence |
|---|---|---|---|
| Cuttlefish CutDeep-Norm | `train_sudoku9_compare.py` | `eval_sudoku9_official.py` | `results/logs/sudoku9_deepnorm.log`, `results/logs/eval_deepnorm.log` |
| Cuttlefish RoPEGate8 | `train_sudoku9_compare.py` (+ `rope_group_gate.py`) | `eval_sudoku9_official.py` | `results/logs/sudoku9_ropegate8.log`, `results/logs/eval_ropegate8.log`, `results/sudoku9_official.json` |
| Cuttlefish CutNoEmb | `train_sudoku9_compare.py` (+ `cut_noemb.py`) | `eval_sudoku9_official.py` | `results/logs/sudoku9_cutnoemb.log`, `results/logs/eval_cutnoemb.log` |
| Shared-graph + degree norm | `shared_gram.py` | `eval_sudoku9_official.py` | `results/logs/sudoku9_pat112norm.log`, `results/logs/eval_pat112norm.log` |
| Transformer-RoPE | `transformer_rope.py` | `eval_sudoku9_official.py` | `results/logs/sudoku9_tfrope.log`, `results/logs/eval_tfrope.log` |
| Transformer cosine kernel | `transformer_cosine.py` | `eval_sudoku9_official.py` | `results/logs/sudoku9_tfcos.log`, `results/logs/eval_tfcos_valid.log` |
| Hyena | `hyena.py` | `eval_sudoku9_official.py` | `results/logs/sudoku9_hyena30k.log`, `results/logs/eval_hyena30k.log` |
| LIV | `liv.py` | `eval_sudoku9_official.py` | `results/logs/sudoku9_liv30k.log`, `results/logs/eval_liv30k.log` |
| LinearFish parity | `linear_fish.py` | `eval_sudoku9_official.py` | `results/logs/eval_linearfish.log` |
| GDN-2 | `train_sudoku9_gdn2.py` / `gdn2_sudoku.py` | `eval_sudoku9_gdn2.py` | `results/sudoku9_gdn2.json` |

Additional Sudoku/Maze entry points: `train_sudoku9_bdhbest.py`, `eval_sudoku9_extreme.py`,
`train_maze_compare.py`.

## 2. 100M language model (large-scale training)

| Model | Train | Evaluate | Evidence |
|---|---|---|---|
| Cuttlefish L12 d450 | `train_big.py` | `eval_easy_bench.py`, `eval_mmlu.py` | `results/big_night2_norm.json`, `results/logs/night2_norm.log`, `results/logs/easy_bench.json` |
| Transformer L12 d832 | `train_big.py` | `eval_easy_bench.py`, `eval_mmlu.py` | `results/big_night_tf2.json`, `results/logs/night_tf2.log` |
| Four-benchmark baseline | — | `eval_easy_bench.py` | `results/easy_bench.json`, `results/easy_bench_formal10b_pre_sft.json` |
| Throughput / memory | `scale_sweep.py`, `bench_long_bdhbest.py` | — | `results/big_bench512b16*.json` |
| Continued pretrain (Ultra-FineWeb) | `train_ultrafineweb_continue.py` | `eval_easy_bench.py` | `results/easy_bench_ultrafineweb_en_final.json` |

Data notes: `results/tokenizer_budget_comparison.json`,
`results/ultrafineweb_probe_token_counts.json`.

## 3. Post-training (SFT → arithmetic → GRPO → merge)

| Stage | Code | Evidence |
|---|---|---|
| Build ChatML SFT mixture | `build_sft_mixture.py`, `build_sft_additional.py`, `build_sft_expansion.py` | `results/logs/sft_cherry_build*.log` |
| ChatML SFT | `train_sft.py`, `test_sft_pipeline.py` | `results/logs/sft_chatml_2048_cos1ep.log`, `results/logs/sft_chatml2048_cos1ep_v2.log`, `results/logs/sft_chatml_stage2_5x.log` |
| SFT checkpoint eval | `eval_sft_checkpoints.py` | `results/logs/sft_*.log` |
| Arithmetic continued training | `train_synthetic_arithmetic_qa.py`, `train_diverse_arithmetic_qa.py`, `diverse_arithmetic_generator.py`, `prepare_gsm8k.py` | `results/logs/arithmetic_*.log` |
| Arithmetic probes | `eval_final_arithmetic_qa.py`, `eval_arithmetic_template_robustness.py`, `probe_ood_arithmetic_formats.py`, `probe_apple_word_problem.py` | `results/logs/*arithmetic*.log`, `results/*.json` |
| GRPO (verifiable reward) | `train_math_grpo_probe.py`, `inspect_math_grpo.py` | `results/pretrain_add_grpo_probe.json`, `results/pretrain_add_grpo_1000.json`, `results/grpo_lowkl_arithmetic_1000.json`, `results/grpo_adaptivekl_arithmetic_1000.json`, `results/logs/grpo_*.log` |
| Checkpoint merge / task vector | `mix_checkpoints.py` | `results/easy_bench_mix_*.json`, `results/easy_bench_taskvec_dialog_plus_arithmetic.json` |

## 4. Small-scale LM comparison

`train_orig.py`, `train_compare.py`, `train_big.py`; sampled with `sample_compare.py`,
`sample_big.py`. Evidence: `results/compare_5L_10M.json`, `results/big_*.json`.

## 5. Visualization

`visualize_cuttlefish.py`, `plot_frequency_river.py`, `plot_head_constellation.py`,
`plot_layer_fingerprints.py`, `plot_retrieval_storyboard.py`, `plot_layer_fingerprints.py`.
