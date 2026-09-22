"""sudoku9 (9x9) 用 bdh_best 加宽 GLU 架构训练。

任务：puzzle(81) + SEP + solution(81)，VOCAB=11，loss 只算 solution 部分。
"""
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
import random
from pathlib import Path

import torch
from torch.nn import functional as F

import bdh_best

SIDE = 9
BOX = 3
PUZZLE_LEN = SIDE * SIDE
SEP = 10
VOCAB_SIZE = 11
CTX = PUZZLE_LEN + 1 + PUZZLE_LEN  # 163


def shuffled(values, rng):
    values = list(values)
    rng.shuffle(values)
    return values


def make_solution(rng):
    pattern = lambda row, col: (BOX * (row % BOX) + row // BOX + col) % SIDE
    rows = [g * BOX + r for g in shuffled(range(BOX), rng) for r in shuffled(range(BOX), rng)]
    cols = [g * BOX + c for g in shuffled(range(BOX), rng) for c in shuffled(range(BOX), rng)]
    digits = shuffled(range(1, SIDE + 1), rng)
    board = [digits[pattern(row, col)] for row in rows for col in cols]
    if rng.random() < 0.5:
        board = [board[col * SIDE + row] for row in range(SIDE) for col in range(SIDE)]
    return board


def make_example(rng, min_clues=28, max_clues=45):
    solution = make_solution(rng)
    clue_count = rng.randint(min_clues, max_clues)
    clue_positions = set(rng.sample(range(PUZZLE_LEN), clue_count))
    puzzle = [value if i in clue_positions else 0 for i, value in enumerate(solution)]
    return puzzle, solution, puzzle + [SEP] + solution


def get_batch(batch_size, rng, device):
    sequences = [make_example(rng)[2] for _ in range(batch_size)]
    sequence = torch.tensor(sequences, dtype=torch.long, device=device)
    inputs, targets = sequence[:, :-1], sequence[:, 1:].clone()
    targets[:, :PUZZLE_LEN] = -100
    return inputs, targets


@torch.no_grad()
def generate_solution(model, prompts):
    sequence = prompts
    for _ in range(PUZZLE_LEN):
        logits = model(sequence)
        next_logits = logits[:, -1, :]
        next_logits[:, 0] = float("-inf")
        next_logits[:, SEP] = float("-inf")
        next_token = next_logits.argmax(dim=-1, keepdim=True)
        sequence = torch.cat((sequence, next_token), dim=1)
    return sequence[:, -PUZZLE_LEN:]


@torch.no_grad()
def evaluate(model, rng, device, examples):
    model.eval()
    batch = [make_example(rng) for _ in range(examples)]
    prompts = torch.tensor([puzzle + [SEP] for puzzle, _, _ in batch], dtype=torch.long, device=device)
    solutions = torch.tensor([solution for _, solution, _ in batch], dtype=torch.long, device=device)
    predictions = generate_solution(model, prompts)
    token_accuracy = (predictions == solutions).float().mean().item()
    exact_accuracy = (predictions == solutions).all(dim=1).float().mean().item()
    inputs, targets = get_batch(examples, rng, device)
    logits = model(inputs)
    loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-100)
    model.train()
    return {"loss": loss.item(), "token_accuracy": token_accuracy, "exact_accuracy": exact_accuracy}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-examples", type=int, default=32)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--recurrences", type=int, default=2)
    p.add_argument("--mixer-mult", type=int, default=5, help="GLU 内部宽 = mixer_mult*d")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--output", type=Path, default=HERE / "runs" / "sudoku9_bdhbest.json")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_rng = random.Random(args.seed)
    eval_rng = random.Random(args.seed + 1)
    d = args.n_embd
    config = bdh_best.Config(
        hidden_size=d, heads=4, mlp_mult=16, layers=args.n_layer,
        recurrences=args.recurrences, vocab_size=VOCAB_SIZE, context_length=CTX,
        dropout=0.0, mixer_mode="swiglu", swiglu_in_loop=False,
        mixer_width=args.mixer_mult * d)
    model = bdh_best.BDHBest(config).to(device)
    if args.compile:
        model = torch.compile(model, mode="reduce-overhead")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} parameters={n_params:,} config={config}", flush=True)

    best = None
    best_state = None
    for step in range(1, args.steps + 1):
        inputs, targets = get_batch(args.batch_size, train_rng, device)
        logits = model(inputs)
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-100)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            m = evaluate(model, eval_rng, device, args.eval_examples)
            if best is None or m["exact_accuracy"] > best["exact_accuracy"]:
                best = m
                best_state = {
                    k.removeprefix("_orig_mod."): v.detach().cpu()
                    for k, v in model.state_dict().items()
                }
            print(f"step={step}/{args.steps} train_loss={loss.item():.4f} "
                  f"eval_loss={m['loss']:.4f} token_acc={m['token_accuracy']:.4f} "
                  f"exact_acc={m['exact_accuracy']:.4f} best_exact={best['exact_accuracy']:.4f}",
                  flush=True)

    cfg = dict(vars(args))
    cfg["output"] = str(cfg["output"])
    result = {"parameters": n_params, "steps": args.steps, "config": cfg,
              "best": best}
    args.output.write_text(__import__("json").dumps(result, indent=2))
    torch.save(best_state, args.output.with_suffix(".best.pt"))
    print(f"\nbest: token_acc={best['token_accuracy']:.4f} exact_acc={best['exact_accuracy']:.4f}")


if __name__ == "__main__":
    main()
