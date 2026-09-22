"""Radial glyph fingerprints for each canonical Cuttlefish layer."""
# --- repo path bootstrap: flat sibling imports + runs/data/tokenizer paths ---
import sys as _sys
from pathlib import Path as _Path
_EXP = _Path(__file__).resolve().parents[1]
for _d in [_EXP, *sorted(_p for _p in _EXP.iterdir() if _p.is_dir() and _p.name != "__pycache__")]:
    if str(_d) not in _sys.path:
        _sys.path.insert(0, str(_d))
HERE = _EXP.parent
# ---------------------------------------------------------------------------
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
import numpy as np


OUT = HERE / "runs" / "visualizations"


def main():
    data = np.load(OUT / "cuttlefish_mechanism_data.npz", allow_pickle=True)
    scores, gates = data["scores"], data["gates"]
    rows = []
    for layer in range(12):
        values = scores[layer][np.isfinite(scores[layer])]
        ranks = []
        for head in scores[layer]:
            singular = np.linalg.svd(np.nan_to_num(head), compute_uv=False)
            p = singular / (singular.sum() + 1e-12)
            ranks.append(np.exp(-(p * np.log(p + 1e-12)).sum()))
        rows.append((values.mean(), np.abs(values).mean(), np.mean(ranks),
                     ((gates[layer] < .05) | (gates[layer] > 1.95)).mean()))
    rows = np.asarray(rows)
    strength = rows[:, 1] / rows[:, 1].max()
    rank = rows[:, 2] / rows[:, 2].max()
    saturation = rows[:, 3]
    bias = rows[:, 0]

    plt.rcParams.update({"font.family": "Lato", "font.size": 11, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(14, 6.2))
    for index in range(12):
        x, y = index + 1, 0
        radius = 0.32 + 0.36 * strength[index]
        color = plt.cm.coolwarm(0.5 + np.clip(bias[index] / 0.08, -0.5, 0.5))
        opening = 360 * saturation[index]
        ax.add_patch(Wedge((x, y), radius, 90 + opening / 2, 450 - opening / 2,
                           width=0.12 + 0.18 * rank[index], facecolor=color,
                           edgecolor="#222222", linewidth=1.2))
        ax.add_patch(Circle((x, y), 0.09, color="#202020"))
        ax.text(x, y - radius - 0.16, f"L{index + 1}", ha="center", fontweight="bold")
        ax.text(x, y + radius + 0.12, f"{rows[index,1]:.2f}", ha="center",
                color="#555555", fontsize=9)
    ax.set_xlim(0.25, 12.75)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0.3, 0.86, "Larger ring = stronger relations", fontsize=11)
    ax.text(0.3, 0.68, "Thicker ring = higher effective rank", fontsize=11)
    ax.text(0.3, 0.50, "Larger opening = more gate saturation", fontsize=11)
    ax.text(8.7, 0.86, "Blue = negative bias", color="#3775BA", fontsize=11)
    ax.text(10.7, 0.86, "Red = positive bias", color="#B64342", fontsize=11)
    fig.suptitle("Each Cuttlefish layer develops a distinct computational fingerprint",
                 fontsize=19, fontweight="bold", y=0.98)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"cuttlefish_layer_fingerprints.{suffix}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
