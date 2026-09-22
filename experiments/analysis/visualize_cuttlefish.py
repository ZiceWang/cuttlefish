"""Publication-style mechanism visualizations for canonical RoPEGate8NoAbs."""
# --- repo path bootstrap: flat sibling imports + runs/data/tokenizer paths ---
import sys as _sys
from pathlib import Path as _Path
_EXP = _Path(__file__).resolve().parents[1]
for _d in [_EXP, *sorted(_p for _p in _EXP.iterdir() if _p.is_dir() and _p.name != "__pycache__")]:
    if str(_d) not in _sys.path:
        _sys.path.insert(0, str(_d))
HERE = _EXP.parent
# ---------------------------------------------------------------------------
import argparse
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

import bdh_best
import rope_group_gate

warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)


BLUE = "#0F4D92"
RED = "#B64342"
GRAY = "#767676"
GOLD = "#D99A21"
TEAL = "#42949E"
VIOLET = "#9A4D8E"


def apply_style():
    plt.rcParams.update({
        "font.family": "Lato",
        "font.size": 11,
        "axes.linewidth": 1.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })


def build_model(device):
    config = bdh_best.Config(
        hidden_size=450, heads=4, mlp_mult=16, layers=12, recurrences=1,
        vocab_size=4096, context_length=256, dropout=0.0,
        mixer_mode="swiglu", swiglu_in_loop=False, mixer_width=2250)
    model = rope_group_gate.RoPEGroupGateNoAbs(config, groups=8).to(device)
    state = torch.load(HERE / "runs" / "big_ropegate8noabs_r1_full.best.pt",
                       map_location=device)
    model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in state.items()})
    model.eval()
    return model


@torch.inference_mode()
def extract(model, ids):
    hidden = model.input_norm(model.token_embedding(ids)).unsqueeze(1)
    all_scores, all_gates = [], []
    for block in model.blocks:
        source = block.ln(hidden)
        enc = source.squeeze(1)
        batch, length, _ = enc.shape
        q, k = block._project_qk(enc)
        q = q.view(batch, length, block.heads, block.qk_width).permute(0, 2, 1, 3)
        k = k.view(batch, length, block.heads, block.qk_width).permute(0, 2, 1, 3)
        positions = torch.arange(length, device=ids.device, dtype=block.attn.freqs.dtype)
        phases = positions.view(1, 1, -1, 1) * block.attn.freqs
        q = F.normalize(bdh_best.rope(phases, q), dim=-1)
        k = F.normalize(bdh_best.rope(phases, k), dim=-1)
        scores = q @ k.mT
        mask = torch.ones(length, length, dtype=torch.bool, device=ids.device).tril(-1)
        scores = scores.masked_fill(~mask, torch.nan)
        gates = 2.0 * block.gate_proj(enc).sigmoid()
        gates = gates.view(batch, length, block.heads, block.groups)
        all_scores.append(scores[0].float().cpu())
        all_gates.append(gates[0].float().cpu())
        hidden = block(hidden)
    return torch.stack(all_scores).numpy(), torch.stack(all_gates).numpy()


def distance_profiles(scores):
    layers, heads, length, _ = scores.shape
    positive = np.full((layers, length - 1), np.nan)
    negative = np.full((layers, length - 1), np.nan)
    for lag in range(1, length):
        values = np.diagonal(scores, offset=-lag, axis1=-2, axis2=-1)
        positive[:, lag - 1] = np.nanmean(np.where(values > 0, values, np.nan), axis=(1, 2))
        negative[:, lag - 1] = np.nanmean(np.where(values < 0, -values, np.nan), axis=(1, 2))
    return positive, negative


def save_overview(scores, gates, output):
    selected = [0, 3, 7, 11]
    fig = plt.figure(figsize=(15.5, 8.5))
    grid = fig.add_gridspec(2, 4, height_ratios=[1.05, 0.9], hspace=0.38, wspace=0.28)
    vmax = np.nanpercentile(np.abs(scores[selected]), 99)
    image = None
    for column, layer in enumerate(selected):
        ax = fig.add_subplot(grid[0, column])
        matrix = np.nanmean(scores[layer], axis=0)
        image = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                          origin="lower", interpolation="nearest", rasterized=True)
        ax.set_title(f"Layer {layer + 1}")
        ax.set_xlabel("Source token")
        if column == 0:
            ax.set_ylabel("Query token")
        ax.spines[:].set_visible(False)
    cbar = fig.colorbar(image, ax=[fig.axes[i] for i in range(4)], fraction=0.018, pad=0.015)
    cbar.set_label("Signed cosine relation")

    ax_gate = fig.add_subplot(grid[1, :2])
    gate_map = gates.mean(axis=(1, 2))
    gate_image = ax_gate.imshow(gate_map, aspect="auto", cmap="Blues", vmin=0.6, vmax=1.4)
    ax_gate.set_title("RoPE frequency-group gate activation")
    ax_gate.set_xlabel("Frequency group (high to low index)")
    ax_gate.set_ylabel("Layer")
    ax_gate.set_xticks(range(8), [str(i + 1) for i in range(8)])
    ax_gate.set_yticks(range(12), [str(i + 1) for i in range(12)])
    fig.colorbar(gate_image, ax=ax_gate, fraction=0.045, pad=0.03, label="Mean gate")

    ax_dist = fig.add_subplot(grid[1, 2:])
    positive, negative = distance_profiles(scores)
    distances = np.arange(1, scores.shape[-1])
    for layer, alpha in zip(selected, [0.35, 0.5, 0.72, 1.0]):
        ax_dist.plot(distances, positive[layer], color=BLUE, alpha=alpha, lw=2)
        ax_dist.plot(distances, negative[layer], color=RED, alpha=alpha, lw=2)
    ax_dist.plot([], [], color=BLUE, lw=2, label="Positive relation")
    ax_dist.plot([], [], color=RED, lw=2, label="Negative relation magnitude")
    ax_dist.set_title("Relation strength by causal distance")
    ax_dist.set_xlabel("Token distance")
    ax_dist.set_ylabel("Mean relation magnitude")
    ax_dist.legend(loc="upper right")
    ax_dist.grid(axis="y", color="#E5E5E5", lw=0.7)

    fig.suptitle("Canonical Cuttlefish: signed relations and RoPE-aligned gates",
                 fontsize=17, fontweight="bold", y=0.99)
    for suffix in ("png", "pdf"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_heads(scores, labels, output):
    layer = scores[-1]
    vmax = np.nanpercentile(np.abs(layer), 99)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
    image = None
    for head, ax in enumerate(axes.flat):
        image = ax.imshow(layer[head], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                          origin="lower", interpolation="nearest", rasterized=True)
        ax.set_title(f"Head {head + 1}")
        ax.spines[:].set_visible(False)
    tick_positions = np.linspace(0, len(labels) - 1, min(8, len(labels)), dtype=int)
    tick_labels = [labels[i][:12] or "_" for i in tick_positions]
    for ax in axes[-1]:
        ax.set_xticks(tick_positions, tick_labels, rotation=45, ha="right")
        ax.set_xlabel("Source token")
    for ax in axes[:, 0]:
        ax.set_ylabel("Query token")
    fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02, label="Signed cosine relation")
    fig.suptitle("Final-layer Cuttlefish head specialization", fontsize=17,
                 fontweight="bold", y=0.98)
    for suffix in ("png", "pdf"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_relation_arcs(scores, text, encoding, output):
    """Highlight signed source relations directly on the original sentence."""
    layer = scores[-1]
    length = layer.shape[-1]
    words = [(match.group(), match.start(), match.end())
             for match in re.finditer(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*|[^\w\s]", text)]
    offsets = encoding.offsets[:length]
    queries = [length // 2, length - 1]
    fig, axes = plt.subplots(2, 1, figsize=(15, 6.8), gridspec_kw={"hspace": 0.55})
    for ax, query in zip(axes, queries):
        relation = np.nanmean(layer[:, query, :], axis=0)
        query_start, query_end = offsets[query]
        query_word = next((index for index, (_, start, end) in enumerate(words)
                           if start < query_end and end > query_start), len(words) - 1)
        word_weights = {}
        for word_index, (_, start, end) in enumerate(words):
            token_indices = [index for index, (left, right) in enumerate(offsets)
                             if index < query and left < end and right > start]
            values = [(index, relation[index]) for index in token_indices
                      if np.isfinite(relation[index])]
            if values:
                _, weight = max(values, key=lambda item: abs(item[1]))
                word_weights[word_index] = float(weight)
        strongest = dict(sorted(word_weights.items(), key=lambda item: abs(item[1]),
                                reverse=True)[:10])
        scale = max((abs(weight) for weight in strongest.values()), default=1.0)
        x, y = 0.02, 0.72
        for word_index, (word, _, _) in enumerate(words):
            width = 0.018 + 0.0105 * len(word)
            if x + width > 0.98:
                x, y = 0.02, y - 0.25
            if word_index == query_word:
                color, alpha = "#FFE45E", 0.95
            elif word_index in strongest:
                weight = strongest[word_index]
                color = "#29B6F6" if weight > 0 else "#FF5252"
                alpha = 0.28 + 0.67 * abs(weight) / scale
            else:
                color, alpha = "white", 0.0
            ax.text(x, y, word, transform=ax.transAxes, fontsize=13, va="center",
                    bbox=dict(boxstyle="round,pad=0.24", facecolor=color,
                              edgecolor="none", alpha=alpha))
            x += width
        ax.set_axis_off()
        ax.set_title(f"Layer 12 query token {query}: “{words[query_word][0]}”", fontsize=14)
    axes[0].text(0.99, 1.08, "Positive source", transform=axes[0].transAxes,
                 ha="right", bbox=dict(facecolor="#29B6F6", alpha=0.75, edgecolor="none"))
    axes[0].text(0.84, 1.08, "Negative source", transform=axes[0].transAxes,
                 ha="right", bbox=dict(facecolor="#FF5252", alpha=0.75, edgecolor="none"))
    axes[0].text(0.68, 1.08, "Current query", transform=axes[0].transAxes,
                 ha="right", bbox=dict(facecolor="#FFE45E", edgecolor="none"))
    fig.suptitle("Cuttlefish signed retrieval highlighted in the original sentence",
                 fontsize=17, fontweight="bold", y=0.99)
    for suffix in ("png", "pdf"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def head_metrics(scores):
    layer = scores[-1]
    metrics = []
    for head in layer:
        rows, cols = np.where(np.isfinite(head))
        values = head[rows, cols]
        distances = rows - cols
        magnitude = np.abs(values)
        total = magnitude.sum() + 1e-12
        row_selectivity = []
        for row in range(head.shape[0]):
            current = np.abs(head[row, np.isfinite(head[row])])
            if current.size:
                probability = current / (current.sum() + 1e-12)
                entropy = -(probability * np.log(probability + 1e-12)).sum()
                row_selectivity.append(1 - entropy / np.log(max(current.size, 2)))
        metrics.append([
            (values > 0).mean(),
            magnitude[distances <= 8].sum() / total,
            magnitude[distances >= 20].sum() / total,
            np.mean(row_selectivity),
            magnitude.mean(),
        ])
    metrics = np.asarray(metrics)
    normalized = metrics.copy()
    for column in range(metrics.shape[1]):
        low, high = metrics[:, column].min(), metrics[:, column].max()
        normalized[:, column] = 0.2 + 0.8 * (metrics[:, column] - low) / (high - low + 1e-12)
    return metrics, normalized


def save_head_roles(scores, output):
    raw, metrics = head_metrics(scores)
    names = ["Positive", "Local", "Long-range", "Selective", "Strength"]
    angles = np.linspace(0, 2 * np.pi, len(names), endpoint=False)
    angles = np.r_[angles, angles[0]]
    colors = [BLUE, RED, TEAL, VIOLET]
    fig = plt.figure(figsize=(12.5, 5.8))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.15], wspace=0.28)
    ax = fig.add_subplot(grid[0, 0], polar=True)
    for head, color in enumerate(colors):
        values = np.r_[metrics[head], metrics[head, 0]]
        ax.plot(angles, values, color=color, lw=2.5, label=f"Head {head + 1}")
        ax.fill(angles, values, color=color, alpha=0.07)
    ax.set_xticks(angles[:-1], names)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0], ["", "", "", ""])
    ax.set_ylim(0, 1.05)
    ax.set_title("Final-layer functional roles", pad=22, fontweight="bold")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=4)

    ax_bar = fig.add_subplot(grid[0, 1])
    y = np.arange(4)
    signed_balance = 2 * raw[:, 0] - 1
    local_minus_long = raw[:, 1] - raw[:, 2]
    ax_bar.axvline(0, color="#BDBDBD", lw=1)
    ax_bar.barh(y + 0.16, signed_balance, height=0.28, color=colors, alpha=0.9,
                label="Positive vs. negative")
    ax_bar.barh(y - 0.16, local_minus_long, height=0.28, color=colors, alpha=0.38,
                hatch="//", label="Local vs. long-range")
    ax_bar.set_yticks(y, [f"Head {i + 1}" for i in y])
    ax_bar.set_xlabel("Role balance")
    ax_bar.set_xlim(-0.55, 0.75)
    ax_bar.set_title("Signed and distance preferences", fontweight="bold")
    ax_bar.legend(loc="lower right")
    ax_bar.grid(axis="x", color="#E5E5E5", lw=0.7)
    fig.suptitle("Cuttlefish heads specialize rather than collapse",
                 fontsize=17, fontweight="bold", y=0.99)
    for suffix in ("png", "pdf"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_gate_trajectories(gates, output):
    means = gates.mean(axis=(1, 2))
    low = (gates < 0.05).mean(axis=(1, 2, 3))
    high = (gates > 1.95).mean(axis=(1, 2, 3))
    layers = np.arange(1, gates.shape[0] + 1)
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, means.shape[1]))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.45, 1]})
    ax = axes[0]
    for group, color in enumerate(colors):
        ax.plot(layers, means[:, group], marker="o", ms=4, lw=2,
                color=color, label=f"Group {group + 1}")
    ax.axhline(1, color="#333333", lw=1.2, ls="--", label="Identity gate")
    ax.set_xticks(layers)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean gate value")
    ax.set_ylim(0, 2.08)
    ax.set_title("Frequency routing across depth", fontweight="bold")
    ax.grid(axis="y", color="#E5E5E5", lw=0.7)
    ax.legend(ncol=3, fontsize=9)

    ax = axes[1]
    ax.bar(layers, low, color=RED, label="Closed: g < 0.05")
    ax.bar(layers, high, bottom=low, color=BLUE, label="Open: g > 1.95")
    ax.set_xticks(layers)
    ax.set_ylim(0, 0.72)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Fraction of gates")
    ax.set_title("Near-binary gate saturation", fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="y", color="#E5E5E5", lw=0.7)
    fig.suptitle("Cuttlefish learns depth-dependent RoPE frequency routing",
                 fontsize=17, fontweight="bold", y=1.01)
    for suffix in ("png", "pdf"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_layer_evolution(scores, gates, output):
    points = []
    for layer in range(scores.shape[0]):
        values = scores[layer][np.isfinite(scores[layer])]
        head_ranks = []
        for head in scores[layer]:
            singular = np.linalg.svd(np.nan_to_num(head), compute_uv=False)
            probability = singular / (singular.sum() + 1e-12)
            head_ranks.append(np.exp(-(probability * np.log(probability + 1e-12)).sum()))
        points.append([
            values.mean(),
            np.abs(values).mean(),
            np.mean(head_ranks),
            ((gates[layer] < 0.05) | (gates[layer] > 1.95)).mean(),
        ])
    points = np.asarray(points)
    fig, ax = plt.subplots(figsize=(9.2, 6.5))
    ax.plot(points[:, 0], points[:, 1], color="#B7B7B7", lw=2, zorder=1)
    scatter = ax.scatter(points[:, 0], points[:, 1], s=12 * points[:, 2],
                         c=points[:, 3], cmap="plasma", vmin=0, vmax=0.65,
                         edgecolor="black", linewidth=0.7, zorder=3)
    for layer, (x, y) in enumerate(points[:, :2], 1):
        ax.text(x, y, str(layer), ha="center", va="center", fontsize=9,
                color="white" if points[layer - 1, 3] > 0.28 else "black",
                fontweight="bold", zorder=4)
    ax.axvline(0, color="#BDBDBD", lw=1, ls="--")
    ax.set_xlabel("Signed relation bias (negative ← 0 → positive)")
    ax.set_ylabel("Mean absolute relation strength")
    ax.set_title("Layer-state trajectory", fontweight="bold")
    ax.grid(color="#ECECEC", lw=0.7)
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Gate saturation fraction")
    ax.text(0.02, 0.98, "Bubble size = effective relation rank",
            transform=ax.transAxes, va="top", color=GRAY)
    fig.suptitle("Cuttlefish changes computational regime across depth",
                 fontsize=17, fontweight="bold", y=0.98)
    for suffix in ("png", "pdf"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def contextual_token_labels(text, encoding):
    """Map BPE pieces to the complete source word containing each piece."""
    labels = []
    for token_id, (start, end) in zip(encoding.ids, encoding.offsets):
        piece = text[start:end].strip()
        if piece and any(character.isalnum() for character in piece):
            left, right = start, end
            while left > 0 and (text[left - 1].isalnum() or text[left - 1] in "'-"):
                left -= 1
            while right < len(text) and (text[right].isalnum() or text[right] in "'-"):
                right += 1
            label = text[left:right]
        else:
            label = piece or Tokenizer.from_file(str(HERE / "bpe4096.json")).decode([token_id]).strip()
        labels.append(label.replace("\n", "\\n") or "_")
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--output-dir", type=Path, default=HERE / "runs" / "visualizations")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    apply_style()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = Tokenizer.from_file(str(HERE / "bpe4096.json"))
    text = (
        "The history of science is a history of questions. Observations become "
        "measurements, measurements become models, and models are tested against "
        "new evidence. A useful theory must explain what is already known while "
        "making precise predictions about what has not yet been observed."
    )
    encoding = tokenizer.encode(text)
    token_ids = encoding.ids[:args.max_tokens]
    ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    model = build_model(device)
    scores, gates = extract(model, ids)
    labels = contextual_token_labels(text, encoding)[:args.max_tokens]
    np.savez_compressed(args.output_dir / "cuttlefish_mechanism_data.npz",
                        token_ids=np.array(token_ids), labels=np.array(labels),
                        scores=scores, gates=gates)
    save_overview(scores, gates, args.output_dir / "cuttlefish_mechanism_overview")
    save_heads(scores, labels, args.output_dir / "cuttlefish_final_layer_heads")
    save_relation_arcs(scores, text, encoding, args.output_dir / "cuttlefish_relation_arcs")
    save_head_roles(scores, args.output_dir / "cuttlefish_head_roles")
    save_gate_trajectories(gates, args.output_dir / "cuttlefish_gate_trajectories")
    save_layer_evolution(scores, gates, args.output_dir / "cuttlefish_layer_evolution")
    print(f"tokens={len(token_ids)} layers={scores.shape[0]} heads={scores.shape[1]}")
    print(f"saved={args.output_dir}")


if __name__ == "__main__":
    main()
