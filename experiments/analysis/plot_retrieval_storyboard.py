"""Sentence-level storyboard of signed retrieval for several query positions."""
# --- repo path bootstrap: flat sibling imports + runs/data/tokenizer paths ---
import sys as _sys
from pathlib import Path as _Path
_EXP = _Path(__file__).resolve().parents[1]
for _d in [_EXP, *sorted(_p for _p in _EXP.iterdir() if _p.is_dir() and _p.name != "__pycache__")]:
    if str(_d) not in _sys.path:
        _sys.path.insert(0, str(_d))
HERE = _EXP.parent
# ---------------------------------------------------------------------------
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tokenizers import Tokenizer


OUT = HERE / "runs" / "visualizations"
TEXT = (
    "The history of science is a history of questions. Observations become "
    "measurements, measurements become models, and models are tested against "
    "new evidence. A useful theory must explain what is already known while "
    "making precise predictions about what has not yet been observed."
)


def main():
    data = np.load(OUT / "cuttlefish_mechanism_data.npz", allow_pickle=True)
    scores = data["scores"][-1].mean(axis=0)
    tokenizer = Tokenizer.from_file(str(HERE / "bpe4096.json"))
    encoding = tokenizer.encode(TEXT)
    offsets = encoding.offsets[:scores.shape[0]]
    words = [(match.group(), match.start(), match.end())
             for match in re.finditer(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*|[^\w\s]", TEXT)]
    queries = [8, 18, 29, 39, 49, 58]

    plt.rcParams.update({"font.family": "Lato", "font.size": 11, "pdf.fonttype": 42})
    fig, axes = plt.subplots(len(queries), 1, figsize=(15, 12),
                             gridspec_kw={"hspace": 0.42})
    for ax, query in zip(axes, queries):
        q_start, q_end = offsets[query]
        q_word = next((i for i, (_, start, end) in enumerate(words)
                       if start < q_end and end > q_start), len(words) - 1)
        relation = scores[query]
        word_weights = {}
        for word_index, (_, start, end) in enumerate(words):
            values = [relation[index] for index, (left, right) in enumerate(offsets)
                      if index < query and left < end and right > start and np.isfinite(relation[index])]
            if values:
                word_weights[word_index] = max(values, key=abs)
        strongest = dict(sorted(word_weights.items(), key=lambda item: abs(item[1]),
                                reverse=True)[:8])
        scale = max((abs(value) for value in strongest.values()), default=1.0)
        x, y = 0.015, 0.56
        for index, (word, _, _) in enumerate(words):
            width = 0.016 + 0.0096 * len(word)
            if x + width > 0.99:
                x, y = 0.015, y - 0.34
            if index == q_word:
                color, alpha = "#FFE45E", 1.0
            elif index in strongest:
                value = strongest[index]
                color = "#29B6F6" if value > 0 else "#FF5252"
                alpha = 0.3 + 0.65 * abs(value) / scale
            else:
                color, alpha = "white", 0.0
            ax.text(x, y, word, transform=ax.transAxes, fontsize=12.2, va="center",
                    bbox=dict(boxstyle="round,pad=0.20", facecolor=color,
                              edgecolor="none", alpha=alpha))
            x += width
        ax.text(0.005, 0.98, f"Query {query}: {words[q_word][0]}", transform=ax.transAxes,
                va="top", fontweight="bold", color="#333333")
        ax.axis("off")

    fig.text(0.72, 0.965, "Current query", bbox=dict(facecolor="#FFE45E", edgecolor="none"))
    fig.text(0.82, 0.965, "Positive source", bbox=dict(facecolor="#29B6F6", edgecolor="none"))
    fig.text(0.92, 0.965, "Negative source", bbox=dict(facecolor="#FF5252", edgecolor="none"))
    fig.suptitle("Cuttlefish retrieval changes as the sentence unfolds",
                 fontsize=19, fontweight="bold", y=0.99)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"cuttlefish_retrieval_storyboard.{suffix}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
