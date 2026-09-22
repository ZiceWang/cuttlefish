"""Train official GDN-2 on the autoregressive Sudoku-9 task."""
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
import torch.nn.functional as F

from gdn2_sudoku import GDN2Sudoku
from train_sudoku9_compare import get_batch, make_example, generate_solution, PUZZLE_LEN, SEP



@torch.no_grad()
def evaluate(model, rng, device, examples):
    model.eval()
    batch = [make_example(rng) for _ in range(examples)]
    prompts = torch.tensor([p + [SEP] for p, _, _ in batch], dtype=torch.long, device=device)
    solutions = torch.tensor([s for _, s, _ in batch], dtype=torch.long, device=device)
    predictions = generate_solution(model, prompts)
    token_accuracy = (predictions == solutions).float().mean().item()
    exact_accuracy = (predictions == solutions).all(dim=1).float().mean().item()
    model.train()
    return token_accuracy, exact_accuracy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-examples", type=int, default=32)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=160)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--heads", type=int, default=5)
    p.add_argument("--head-dim", type=int, default=32)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda"
    train_rng = random.Random(args.seed)
    eval_rng = random.Random(args.seed + 1)
    model = GDN2Sudoku(11, args.hidden, args.layers, args.heads, args.head_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    params = sum(p.numel() for p in model.parameters())
    print(f"[gdn2] d={args.hidden} L={args.layers} H={args.heads} params={params:,}", flush=True)

    best = None
    best_state = None
    for step in range(1, args.steps + 1):
        x, y = get_batch(args.batch_size, train_rng, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-100)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            tok, exact = evaluate(model, eval_rng, device, args.eval_examples)
            if best is None or exact > best["exact_accuracy"]:
                best = {"token_accuracy": tok, "exact_accuracy": exact, "step": step}
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            print(f"step={step}/{args.steps} loss={loss.item():.4f} token_acc={tok:.4f} "
                  f"exact_acc={exact:.4f} best_exact={best['exact_accuracy']:.4f}", flush=True)

    out = HERE / "runs" / "sudoku9_gdn2.json"
    out.write_text(json.dumps({"params": params, "config": vars(args), "best": best}, indent=2))
    torch.save(best_state, out.with_suffix(".best.pt"))
    print(f"[gdn2] best={best}", flush=True)


if __name__ == "__main__":
    main()
