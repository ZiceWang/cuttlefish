"""Circular signed-relation constellations for final-layer Cuttlefish heads."""
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
import numpy as np


OUT = HERE / "runs" / "visualizations"
BLUE, RED = "#29B6F6", "#FF5252"


def main():
    data = np.load(OUT / "cuttlefish_mechanism_data.npz", allow_pickle=True)
    scores = data["scores"][-1]
    labels = data["labels"].tolist()
    length = scores.shape[-1]
    selected = np.unique(np.linspace(0, length - 1, 18, dtype=int))
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, len(selected), endpoint=False)
    positions = np.c_[np.cos(angles), np.sin(angles)]

    plt.rcParams.update({"font.family": "Lato", "font.size": 11,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    for head, ax in enumerate(axes.flat):
        candidates = []
        for target_index, target in enumerate(selected):
            for source_index, source in enumerate(selected):
                if source >= target:
                    continue
                weight = scores[head, target, source]
                if np.isfinite(weight):
                    candidates.append((abs(weight), weight, source_index, target_index))
        strongest = sorted(candidates, reverse=True)[:34]
        scale = strongest[0][0] if strongest else 1.0
        for magnitude, weight, source, target in strongest:
            p0, p1 = positions[source], positions[target]
            midpoint = (p0 + p1) / 2
            control = midpoint * 0.18
            t = np.linspace(0, 1, 70)[:, None]
            curve = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * control + t ** 2 * p1
            ax.plot(curve[:, 0], curve[:, 1], color=BLUE if weight > 0 else RED,
                    lw=0.45 + 2.8 * magnitude / scale,
                    alpha=0.18 + 0.62 * magnitude / scale, zorder=1)
        ax.scatter(positions[:, 0], positions[:, 1], s=55, color="#202020",
                   edgecolor="white", linewidth=1, zorder=3)
        for index, (x, y) in enumerate(positions):
            label = labels[selected[index]] or "_"
            ax.text(1.16 * x, 1.16 * y, label[:13], ha="center", va="center",
                    fontsize=8.5, rotation=np.degrees(angles[index]) - 90,
                    rotation_mode="anchor")
        valid_values = scores[head][np.isfinite(scores[head])]
        positive = (valid_values > 0).mean()
        ax.set_title(f"Head {head + 1}  |  positive-edge share {positive:.2f}",
                     fontweight="bold", pad=14)
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-1.35, 1.35)
        ax.set_aspect("equal")
        ax.axis("off")

    axes[0, 0].plot([], [], color=BLUE, lw=3, label="Positive relation")
    axes[0, 0].plot([], [], color=RED, lw=3, label="Negative relation")
    axes[0, 0].legend(loc="upper center", bbox_to_anchor=(1.08, 1.30), ncol=2)
    fig.suptitle("Final-layer Cuttlefish heads form distinct signed constellations",
                 fontsize=19, fontweight="bold", y=0.985)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"cuttlefish_head_constellation.{suffix}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
