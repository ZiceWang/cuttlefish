# Cuttlefish Results Package

This package contains the single-file Cuttlefish technical report, core model code, evaluation/training entry points, and selected evidence logs. Large model checkpoints and datasets are deliberately excluded.

## Start Here

- `CUTTLEFISH.md`: architecture, equations, design findings, final results and limitations.
- `CUTTLEFISH_RESULTS.json`: machine-readable aggregate reconstructed from independent evidence logs.
- The release ZIP is tested with `unzip -t`; its SHA-256 digest is reported alongside the package path.

## Complexity Caveat

`linear_fish.py` is an exact O(T) prefix implementation of the Gram numerator. CutDeep-Norm in `gram_norm.py` divides by `sum(abs(score))`, but the block immediately applies per-head LayerNorm. Since LayerNorm nearly cancels positive scalar rescaling, the preferred clean linear form drops the explicit degree divisor and applies the existing LayerNorm to the exact prefix numerator. The literal absolute-degree code path remains quadratic; finite-epsilon checkpoint parity between the two forms must still be measured.

## Main Implementations

- `bdh_best.py`: base causal Gram model and gated readout.
- `gram_norm.py`: signed absolute-degree-normalized CutDeep-Norm operator.
- `rope_group_gate.py`: low-cost eight-frequency-group Q/K gate aligned to RoPE pairs.
- `linear_fish.py`: exact unnormalized LinearFish variant.
- `transformer.py`, `transformer_rope.py`, `transformer_cosine.py`: matched Transformer baselines and kernel ablation.
- `hyena.py`, `liv.py`: sequence-model baselines used in the Sudoku comparison.
- `shared_gram.py`, `cut_noemb.py`: relation-sharing and position-embedding ablations.

## Entry Points

- `train_sudoku9_compare.py`: matched autoregressive Sudoku training.
- `eval_sudoku9_official.py`: 4096-example Sudoku evaluation.
- `train_big.py`: 100M-scale language-model training.
- `eval_easy_bench.py`: simple zero-shot likelihood benchmarks.
- `eval_mmlu.py`: MMLU likelihood probe.

Run `python <script> --help` for the supported arguments. The original experiments used local datasets and CUDA environments, so dataset paths and dependency versions may need to be adapted on another machine.

## Evidence Layout

- `runs/eval_*.log`: independent 4096-example Sudoku results.
- `runs/sudoku9_*.log`: selected 30k-step training trajectories.
- `runs/night2_norm.log`, `runs/night_tf2.log`: full 80k-step language-model trajectories.
- `runs/big_night2_norm.json`, `runs/big_night_tf2.json`: final language-model metrics.
- `runs/easy_bench.json`: simple benchmark results.

`runs/sudoku9_official.json` is not included because the evaluator overwrites it on each run. Use `CUTTLEFISH_RESULTS.json` and the independent `runs/eval_*.log` files instead.
