"""Evaluate the official GDN-2 Sudoku checkpoint on a fixed held-out set."""
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
import random
from pathlib import Path

import torch

from gdn2_sudoku import GDN2Sudoku
from train_sudoku9_compare import SEP, generate_solution, make_example




@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=555)
    args = parser.parse_args()

    device = "cuda"
    model = GDN2Sudoku(11, hidden_size=192, layers=5, heads=6, head_dim=32).to(device)
    state = torch.load(HERE / "runs" / "sudoku9_gdn2.best.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()

    rng = random.Random(args.seed)
    correct_tokens = 0
    correct_boards = 0
    total_tokens = 0
    total_boards = 0
    for start in range(0, args.examples, args.batch_size):
        size = min(args.batch_size, args.examples - start)
        batch = [make_example(rng) for _ in range(size)]
        prompts = torch.tensor([p + [SEP] for p, _, _ in batch], dtype=torch.long, device=device)
        solutions = torch.tensor([s for _, s, _ in batch], dtype=torch.long, device=device)
        predictions = generate_solution(model, prompts)
        matches = predictions == solutions
        correct_tokens += matches.sum().item()
        correct_boards += matches.all(dim=1).sum().item()
        total_tokens += matches.numel()
        total_boards += size
        print(f"evaluated={total_boards}/{args.examples}", flush=True)

    result = {
        "model": "gdn2",
        "parameters": sum(p.numel() for p in model.parameters()),
        "examples": total_boards,
        "seed": args.seed,
        "token_accuracy": correct_tokens / total_tokens,
        "exact_accuracy": correct_boards / total_boards,
    }
    output = HERE / "runs" / "eval_gdn2_4096.json"
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    print(f"saved={output}", flush=True)


if __name__ == "__main__":
    main()
