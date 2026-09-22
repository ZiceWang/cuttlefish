"""Evaluate SFT state-dict checkpoints on one fixed ChatML validation stream."""
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
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train_sft import DATA_DIR, PackedChatDataset, evaluate, make_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    ap.add_argument("--validation", type=Path, default=DATA_DIR / "validation.jsonl")
    ap.add_argument("--context-length", type=int, default=2048)
    ap.add_argument("--batches", type=int, default=64)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for checkpoint in args.checkpoint:
        model = make_model(4099, args.context_length).to(device)
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state)
        loader = DataLoader(PackedChatDataset(args.validation,
                            DATA_DIR / "bpe4099_chatml.json", args.context_length),
                            batch_size=1, num_workers=0)
        results[str(checkpoint)] = evaluate(model, loader, device, args.batches)
        print(json.dumps(results, indent=2), flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
