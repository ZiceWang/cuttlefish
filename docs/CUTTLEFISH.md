# Cuttlefish: Degree-Normalized Gated Graph Message Passing

## 1. Scope

Cuttlefish is an autoregressive sequence model built from causal, content-dependent graph message passing. Each layer constructs a directed relation graph between earlier and current tokens, aggregates signed messages over that graph, and combines the message with the current state through a wide SwiGLU readout.

The final evaluated family has four defining properties:

1. **Content-dependent relation graph.** Every layer recomputes its own graph from the current hidden state.
2. **Wide relation space.** Q and K are projected to four heads of width `2d`, for a total relation width of `8d` per projection.
3. **Signed graph aggregation with normalized readout.** The evaluated CutDeep-Norm code divides by absolute row degree and then applies per-head LayerNorm; because LayerNorm is nearly invariant to positive scalar rescaling, the clean linear form can omit the explicit degree divisor while retaining the normalized readout.
4. **Independent depth.** The strongest models use `R=1`; layers do not share Q/K or GLU parameters.

The architecture descends from the graph-message and gated-readout ideas explored in BDH, but the implementation and results documented here refer to this concrete PyTorch architecture, not to every mechanism proposed in the BDH paper.

## 2. Notation

For a sequence of length `T` and hidden width `d`, let

- `X ∈ R^(B×T×d)` be the normalized layer input,
- `H=4` be the number of relation heads,
- `d_k=2d` be the width of each relation head,
- `W_Q,W_K ∈ R^(d×Hd_k)` be independent per-layer projections,
- `R_p` be the RoPE rotation at position `p`,
- `V=X` be the value/source state used by the graph operator.

All masks are strictly causal: token `t` may receive messages only from positions `s<t`.

## 3. Relation Graph

For every layer and head:

```math
Q = \operatorname{ReLU}(XW_Q), \qquad K = \operatorname{ReLU}(XW_K).
```

After splitting into heads, applying RoPE and L2 normalization:

```math
\widetilde q_{t,h}
= \frac{R_t q_{t,h}}{\lVert R_t q_{t,h}\rVert_2},
\qquad
\widetilde k_{s,h}
= \frac{R_s k_{s,h}}{\lVert R_s k_{s,h}\rVert_2}.
```

The signed causal relation weight is

```math
a_{t,s,h} =
\begin{cases}
\widetilde q_{t,h}^{\mathsf T}\widetilde k_{s,h}, & s<t,\\
0, & s\ge t.
\end{cases}
```

Thus `a ∈ [-1,1]`; negative edges are inhibitory rather than discarded.

## 4. Degree-Normalized Message

For each head:

```math
m_{t,h}
= \frac{\sum_{s<t} a_{t,s,h}v_s}
       {\max\left(\varepsilon,\sum_{s<t}|a_{t,s,h}|\right)}.
```

The denominator normalizes the absolute row degree without removing negative edges. In the implemented block, however, heads are immediately normalized and then averaged:

```math
g_t = \frac{1}{H}\sum_{h=1}^{H}\operatorname{LN}(m_{t,h}).
```

For positive scalar `d` and negligible LayerNorm epsilon,

```math
\operatorname{LN}(n/d)=\operatorname{LN}(n).
```

Consequently, the explicit absolute-degree divisor is largely redundant in this exact placement. Its remaining effects come from finite epsilon, zero/small-message behavior and optimization numerics, rather than a materially different normalized forward direction.

The positive-only ablation replaces `a` by `max(a,0)`. It underperformed the signed version, indicating that inhibitory edges carry useful information.

## 5. Wide Gated Readout

The state and graph message are concatenated:

```math
z_t=[x_t;g_t]\in\mathbb R^{2d}.
```

With internal width `m=5d`:

```math
[u_t,r_t] = W_{\mathrm{up}}z_t,
\qquad
y_t = W_{\mathrm{down}}
\bigl(\operatorname{SiLU}(u_t)\odot r_t\bigr),
```

where `W_up : 2d -> 2m` and `W_down : m -> d`.

The block has an outer residual bypassing the relation graph and readout:

```math
\operatorname{out}
= x_{\mathrm{block}}
+W_{\mathrm{out}}\operatorname{LN}(x_{\mathrm{inner}}).
```

### 5.1 Sudoku / CutDeep-Norm block

The strongest Sudoku model uses the graph message as an explicit intermediate residual before the GLU:

```math
x_1=\operatorname{LN}(x_0+g),
```

```math
x_2=\operatorname{LN}
\left(x_1+\operatorname{LN}\bigl(\operatorname{GLU}([x_1;g])\bigr)\right),
```

```math
\operatorname{out}=x_{\mathrm{block}}+W_{\mathrm{out}}\operatorname{LN}(x_2).
```

Configuration: `L=8`, `R=1`, `d=92`, `H=4`, `d_k=184`, GLU width `460`.

### 5.2 100M language-model block

The 100M run uses `R=1` with the GLU update applied directly from `[x_0;g]`:

```math
x_1=\operatorname{LN}
\left(x_0+\operatorname{LN}\bigl(\operatorname{GLU}([x_0;g])\bigr)\right),
```

```math
\operatorname{out}=x_{\mathrm{block}}+W_{\mathrm{out}}\operatorname{LN}(x_1).
```

Configuration: `L=12`, `R=1`, `d=450`, `H=4`, `d_k=900`, GLU width `2250`.

## 6. Position Encoding

The default Cuttlefish language model adds a learned absolute position embedding to the token embedding and also applies RoPE inside the relation graph. The no-learned-position Sudoku ablation retained only graph-internal RoPE.

```text
Default Cuttlefish: token embedding + learned position embedding + graph RoPE
CutNoEmb ablation:  token embedding + graph RoPE
```

The final reported 100M and CutDeep-Norm models use the default form unless explicitly labeled `CutNoEmb`.

## 7. Complexity and the LinearFish Variant

The explicit degree-normalized relation graph has quadratic sequence complexity:

```math
O\bigl(HT^2(d_k+d)\bigr)
```

plus projection/readout cost `O(Td^2)`.

The **unnormalized** Gram numerator has an exact associative form:

```math
n_t
=\sum_{s<t}(\widetilde q_t^{\mathsf T}\widetilde k_s)v_s
=\widetilde q_t^{\mathsf T}
\left(\sum_{s<t}\widetilde k_sv_s^{\mathsf T}\right).
```

`experiments/models/linear_fish.py` implements this exact prefix-state form and is state-dict compatible with the unnormalized Cuttlefish block. Identical weights produced a maximum logit difference of zero in the verification run.

The evaluated degree denominator

```math
\sum_{s<t}|\widetilde q_t^{\mathsf T}\widetilde k_s|
```

does **not** share the same exact low-rank prefix factorization because of the absolute value after the pairwise dot product. However, because the resulting message is immediately LayerNormed, the clean deployment form can remove this redundant scalar divisor and linearize the signed numerator exactly. Therefore:

- the code path with explicit signed absolute-degree computation remains quadratic;
- the clean `LinearFish + per-head LayerNorm` form is exactly linearizable and preserves the normalized message direction up to LayerNorm epsilon effects;
- if absolute degree itself must remain semantically observable before a non-scale-invariant operation, obtaining it exactly in strict linear time remains open.

## 8. Main Results

### 8.1 Sudoku-9, 30k steps, 4096 held-out puzzles

All models are autoregressive and approximately parameter matched around 2.7-2.9M parameters.

| Model | Token accuracy | Exact-board accuracy |
|---|---:|---:|
| Transformer-RoPE, no learned position embedding | 0.9682 | **0.7676** |
| **Cuttlefish RoPEGate8** | **0.9660** | **0.7246** |
| Hyena, 30k | 0.9536 | 0.6252 |
| Transformer, learned position embedding, 30k | 0.9276 | 0.6072 |
| **CutDeep-Norm, 30k** | **0.9360** | **0.5981** |
| CutNoEmb, 30k | 0.9258 | 0.5691 |
| 112112 shared relation graph + degree norm, 30k | 0.9088 | 0.5168 |
| Transformer-RoPE with cosine degree-normalized kernel | 0.9194 | 0.4856 |
| LIV, 30k | 0.1117 | 0.0000 |

`RoPEGate8` is a controlled replacement of coordinate-wise Q/K ReLU. Each relation head divides its RoPE pairs into eight frequency groups and applies a shared identity-centered gate `2 sigmoid(W_g x)` to Q and K before RoPE. The gate adds 23,552 parameters (`+0.82%`) and improves exact-board accuracy from `0.5981` to `0.7246` (`+0.1265`, a 21.1% relative increase).

Evaluation protocol: training selects a checkpoint using recurring 32-puzzle exact accuracy; the selected checkpoint is then evaluated on 4096 puzzles. The 4096 score is the reported result because the 32-puzzle trajectory is noisy.

### 8.2 100M language modeling, 80k steps

Dataset: BPE-4096 encoding of seven training shards and one validation shard from the local FinePDFs-Edu / DCLM / FineWeb-Edu mixture. Context 256, batch 64, AdamW, cosine schedule, peak learning rate `3e-4`.

| Model | Parameters | Best validation loss | PPL | Token accuracy |
|---|---:|---:|---:|---:|
| Transformer, L12 d832 | 106,890,368 | **2.7243** | **15.2459** | **45.10%** |
| **Cuttlefish, L12 d450** | 105,861,600 | **2.8344** | **17.0209** | **43.59%** |

### 8.3 Simple zero-shot likelihood benchmarks

Each option is scored by length-normalized conditional log-likelihood. Roughly 300 examples per task.

| Benchmark | Random baseline | Cuttlefish | Transformer |
|---|---:|---:|---:|
| PIQA | 0.50 | 0.6120 | **0.6421** |
| HellaSwag | 0.25 | 0.2767 | **0.3000** |
| Winogrande | 0.50 | **0.4900** | 0.4767 |
| ARC-Easy | 0.25 | **0.3826** | 0.3758 |

These probes use roughly 300 examples each, so differences at this scale are small.

### 8.4 Large-scale pretraining, 136M parameters, ~10B tokens

The scaled run uses the RoPEGate8-SignedExp-noabs block with `R = 1` and `136,511,488` parameters. It streamed ten shards of a BPE-4096 FinePDFs-Edu / DCLM / FineWeb-Edu mixture for one epoch at context 512 and 8,192 tokens per step.

| Item | Value |
|---|---:|
| Parameters | 136,511,488 |
| Steps | 1,220,000 |
| Training tokens | 9,994,240,000 |
| Best eval loss | 2.4993 |
| Best-eval PPL | 12.18 |
| Final eval loss | 2.5159 |
| Final eval PPL | 12.38 |
| Final token accuracy | 0.4805 |

Throughput at context 512 / batch 16 was about `4.6e4` tokens/s eagerly and `1.25e5` tokens/s with `torch.compile`, at `15.85` GiB and `10.64` GiB peak GPU memory respectively.

Four-benchmark evaluation of this checkpoint (before any SFT):

| Benchmark | Random baseline | 10B checkpoint | Transformer (reference) | n |
|---|---:|---:|---:|---:|
| PIQA | 0.50 | 0.6288 | 0.6421 | 299 |
| HellaSwag | 0.25 | 0.3033 | 0.3000 | 300 |
| Winogrande | 0.50 | 0.5100 | 0.4767 | 300 |
| ARC-Easy | 0.25 | 0.3993 | 0.3758 | 298 |

This checkpoint is the base of the GRPO probe and of the SFT / checkpoint-merging work summarised in `README.md`.

## 9. Empirically Supported Design Observations

1. **Normalized graph readout is useful, but the explicit degree divisor is not cleanly identified as the cause.** In the evaluated implementation it is followed immediately by LayerNorm, which largely cancels positive scalar degree rescaling. The cleaner interpretation is signed aggregation followed by per-head normalization.
2. **Negative edges are useful.** Clamping all negative edges reduced Sudoku performance relative to absolute-degree normalization.
3. **Independent layer depth is stronger than repeated application of a shared block in the tested Sudoku regime.** The final architecture uses `R=1`; this is an empirical design choice, not a claim that recurrent computation is universally invalid.
4. **Wide gated readout is useful.** Earlier byte-level language-model ablations favored a `5d` state-message SwiGLU over linear/additive fusion and narrower gates.
5. **The relation operator and readout are co-designed.** Replacing Transformer softmax with cosine degree normalization while keeping the Transformer skeleton substantially reduced Sudoku exact accuracy.
6. **Pure RoPE improves the tested Transformer more than the tested graph model.** Position encoding is therefore treated as a controlled experimental variable, not as a Cuttlefish contribution.
7. **RoPE-aligned nonlinear gating is substantially better than coordinate-wise ReLU in this setting.** An eight-group shared Q/K gate improved 4096-board exact accuracy from `0.5981` to `0.7246` under the same width, depth and training protocol.

## 10. Limitations and Positioning

Cuttlefish is a distinct signed-graph autoregressive architecture with competitive small-model reasoning and language-model performance, but it does not currently outperform the strongest matched Transformer baselines. Its supported contributions are:

- a concrete signed relation graph with normalized per-head readout;
- a wide state-message gated readout;
- evidence that inhibitory edges and graph/readout co-design matter;
- an exactly linearizable unnormalized variant sharing the same relation numerator;
- a reproducible set of matched Sudoku, language-model and commonsense experiments.

The literal absolute-degree code path should not be advertised as strictly linear-time. The preferred linear deployment form removes that nearly redundant divisor and uses the exact prefix numerator followed by the existing per-head LayerNorm; its numerical parity should be reported explicitly for each checkpoint.

Additional limitations:

- **100M language model.** Under matched training the Cuttlefish model trails the Transformer baseline by `0.1101` validation loss, `1.775` PPL and `1.51` token-accuracy points, and its best checkpoint shows more semantic drift and repetition.
- **Simple benchmarks.** At this scale MMLU stays near chance and is not used as a headline result.
- **Sudoku-Extreme.** Autoregressive exact-board accuracy is `0.0` at the tested training budget.

## 11. Source and Evidence Map

| Artifact | Purpose |
|---|---|
| `experiments/models/bdh_best.py` | Base causal Gram graph and wide GLU block |
| `experiments/models/gram_norm.py` | Signed absolute-degree normalization used by CutDeep-Norm |
| `experiments/models/linear_fish.py` | Exact prefix-state implementation of the unnormalized numerator |
| `experiments/models/rope_group_gate.py` | Eight-group identity-centered Q/K gate aligned to RoPE pairs |
| `experiments/lm/train_big.py` | 100M language-model training and best/last checkpoints |
| `experiments/sudoku/train_sudoku9_compare.py` | Parameter-matched autoregressive Sudoku experiments |
| `experiments/sudoku/eval_sudoku9_official.py` | 4096-puzzle exact evaluation |
| `experiments/lm/eval_easy_bench.py` | PIQA, HellaSwag, Winogrande, ARC-Easy likelihood evaluation |
| `results/logs/night2_norm.log` | Full 100M Cuttlefish training trajectory |
| `results/logs/night_tf2.log` | Full matched Transformer trajectory |
| `results/big_night2_norm.json` | Final Cuttlefish LM metric record |
| `results/big_night_tf2.json` | Final Transformer LM metric record |
| `results/logs/sudoku9_deepnorm.log` / `results/logs/eval_deepnorm.log` | CutDeep-Norm Sudoku trajectory/result |
| `results/logs/sudoku9_tfrope.log` / `results/logs/eval_tfrope.log` | strongest Transformer-RoPE Sudoku result |
| `results/logs/sudoku9_ropegate8.log` / `results/logs/eval_ropegate8.log` | RoPE-aligned gated Cuttlefish result |
| `results/easy_bench.json` | simple benchmark scores |
| `results/logs/formal512_10shard_1ep.log` | full ~10B-token large-scale pretraining trajectory |
| `results/easy_bench_formal10b_pre_sft.json` | four-benchmark evaluation of the ~10B-token checkpoint |
