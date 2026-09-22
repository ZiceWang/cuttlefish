"""Estimate token-budget conversion on identical pretraining documents."""
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
import json
import os
import random
import statistics
from pathlib import Path

import pyarrow.parquet as pq
from tokenizers import Tokenizer
from transformers import AutoTokenizer

# Hugging Face cache root; override with HF_HOME (defaults to ~/.cache/huggingface).
HF_HOME = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
ROOT = (
    HF_HOME / "hub"
    / "datasets--HuggingFaceFW--finepdfs_edu_50BT-dclm_30BT-fineweb_edu_20BT-shuffled"
    / "snapshots" / "8904a95879538b9e7db6cf8636b2cd16b5e86a76" / "data"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=10)
    ap.add_argument("--row-groups", type=int, default=5)
    ap.add_argument("--documents-per-group", type=int, default=40)
    ap.add_argument("--trained-tokens", type=int, default=9_994_240_000)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    ours = Tokenizer.from_file(str(HERE / "bpe4096.json"))
    smol = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", use_fast=True)
    rng = random.Random(20260912)
    pairs = []
    for shard in range(args.shards):
        parquet = pq.ParquetFile(ROOT / f"train-{shard:05d}-of-00100.parquet")
        groups = rng.sample(range(parquet.num_row_groups),
                            min(args.row_groups, parquet.num_row_groups))
        texts = []
        for group in groups:
            column = parquet.read_row_group(group, columns=["text"]).column(0)
            indices = rng.sample(range(len(column)), min(args.documents_per_group, len(column)))
            texts.extend(column[index].as_py() for index in indices)
        our_lengths = [len(encoding.ids) for encoding in ours.encode_batch(texts)]
        smol_ids = smol(texts, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        pairs.extend((a, len(b)) for a, b in zip(our_lengths, smol_ids))
        print(f"shard={shard} documents={len(texts)}", flush=True)
    our_total = sum(a for a, _ in pairs)
    smol_total = sum(b for _, b in pairs)
    ratio = smol_total / our_total
    result = {
        "documents": len(pairs), "our_bpe4096_tokens": our_total,
        "smollm2_tokens": smol_total, "smollm2_per_our_token_ratio": ratio,
        "our_per_smollm2_ratio": our_total / smol_total,
        "median_document_ratio": statistics.median(b / a for a, b in pairs if a),
        "trained_our_tokens": args.trained_tokens,
        "equivalent_smollm2_tokens": round(args.trained_tokens * ratio),
        "smollm2_vocab_size": len(smol),
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text)


if __name__ == "__main__":
    main()
