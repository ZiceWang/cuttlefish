"""Flow-style view of RoPE frequency routing across Cuttlefish depth."""
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


def main():
    data = np.load(OUT / "cuttlefish_mechanism_data.npz", allow_pickle=True)
    means = data["gates"].mean(axis=(1, 2))
    layers = np.arange(1, means.shape[0] + 1)
    weights = means / means.sum(axis=1, keepdims=True)
    boundaries = np.cumsum(weights, axis=1)
    lower = np.c_[np.zeros(len(layers)), boundaries[:, :-1]]

    plt.rcParams.update({"font.family": "Lato", "font.size": 12,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(13, 6.8))
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, 8))
    dense_x = np.linspace(1, 12, 500)
    for group in range(8):
        lo = np.interp(dense_x, layers, lower[:, group])
        hi = np.interp(dense_x, layers, boundaries[:, group])
        ax.fill_between(dense_x, lo, hi, color=colors[group], alpha=0.9,
                        label=f"Frequency group {group + 1}")
        center = (lo + hi) / 2
        ax.plot(dense_x, center, color="white", lw=0.7, alpha=0.4)

    for layer in layers:
        ax.axvline(layer, color="white", lw=0.8, alpha=0.45)
    ax.set_xlim(1, 12)
    ax.set_ylim(0, 1)
    ax.set_xticks(layers)
    ax.set_yticks([])
    ax.set_xlabel("Cuttlefish layer")
    ax.set_title("RoPE frequency routing forms a depth-wise information river",
                 fontsize=18, fontweight="bold", pad=18)
    ax.text(1.05, 1.025, "Relative gate allocation", color="#555555")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=4,
              frameon=False)
    ax.spines[["left", "bottom"]].set_visible(False)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"cuttlefish_frequency_river.{suffix}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
