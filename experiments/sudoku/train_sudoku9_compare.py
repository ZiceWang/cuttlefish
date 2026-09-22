"""Sudoku9 对比训练：Transformer / LIV / Hyena（与 train_sudoku9_bdhbest.py 同构）。

同参数量对齐 Cuttlefish(d128,L4,R2)=2.78M：
  TF d168 L8 / LIV d184 L5 / Hyena d208 L5
任务：puzzle(81)+SEP+solution(81)，VOCAB=11，loss 只算 solution 部分。
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

import transformer
import liv
import hyena
import hyena_fish
import bdh_best
import linear_fish
import gated_fish
import hyena_rope
import hyena_dilated
import transformer_rope
import transformer_cosine
import cut_noemb
import shared_gram
import rope_group_gate
import linear_rope_group_gate
import gram_norm
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


DEFAULT = {
    "tfcos": dict(hidden=168, layers=8),
    "tfrope": dict(hidden=168, layers=8),
    "cutnoemb": dict(hidden=92, layers=8),
    "pat112norm": dict(hidden=120, layers=6),
    "deepnorm": dict(hidden=92, layers=8),
    "deepnormpos": dict(hidden=92, layers=8),
    "ropegate8": dict(hidden=92, layers=8),
    "ropegate8signedexp": dict(hidden=92, layers=8),
    "ropegate8noabs": dict(hidden=92, layers=8),
    "linearropegate8noabs": dict(hidden=92, layers=8),
    "linearropegate8abs": dict(hidden=92, layers=8),
    "linearropegate8rawabs": dict(hidden=92, layers=8),
    "linearropegate8rawnoabs": dict(hidden=92, layers=8),
    "cutpat112": dict(hidden=120, layers=6, pattern=(0, 0, 1, 0, 0, 1)),
    "cutshared4": dict(hidden=108, layers=8, groups=4),
    "cutshared2": dict(hidden=108, layers=8, groups=2),
    "cutshared": dict(hidden=112, layers=8),
    "cutdeep": dict(hidden=92, layers=8),
    "cuttr1": dict(hidden=128, layers=4, recurrences=1),
    "hyenadil": dict(hidden=208, layers=5),
    "hyenarope": dict(hidden=208, layers=5),
    "gatedfish": dict(hidden=118, layers=4),
    "linearfish": dict(hidden=128, layers=4),
    "hyenafish": dict(hidden=148, layers=4),
    "transformer": dict(hidden=168, layers=8),
    "liv": dict(hidden=184, layers=5),
    "hyena": dict(hidden=208, layers=5),
}


def build_model(arch, hidden, layers):
    if arch == "tfcos":
        return transformer_cosine.TransformerCosine(transformer.Config(
            hidden_size=hidden, heads=4, layers=layers, vocab_size=VOCAB_SIZE,
            context_length=CTX))
    if arch == "tfrope":
        return transformer_rope.TransformerRoPE(transformer.Config(
            hidden_size=hidden, heads=4, layers=layers, vocab_size=VOCAB_SIZE,
            context_length=CTX))
    if arch == "cutnoemb":
        return cut_noemb.CutNoEmb(bdh_best.Config(
            hidden_size=hidden, heads=4, mlp_mult=16, layers=layers, recurrences=1,
            vocab_size=VOCAB_SIZE, context_length=CTX, dropout=0.0,
            mixer_mode="swiglu", swiglu_in_loop=False, mixer_width=5 * hidden))
    if arch == "pat112norm":
        return shared_gram.SharedGramModel(shared_gram.Config(
            hidden_size=hidden, layers=6, group_pattern=(0, 0, 1, 0, 0, 1),
            norm_attn=True, posify=False, vocab_size=VOCAB_SIZE, context_length=CTX))
    if arch == "deepnorm":
        return gram_norm.NormDeep(bdh_best.Config(
            hidden_size=hidden, heads=4, mlp_mult=16, layers=layers, recurrences=1,
            vocab_size=VOCAB_SIZE, context_length=CTX, dropout=0.0,
            mixer_mode="swiglu", swiglu_in_loop=False, mixer_width=5 * hidden), posify=False)
    if arch == "deepnormpos":
        return gram_norm.NormDeep(bdh_best.Config(
            hidden_size=hidden, heads=4, mlp_mult=16, layers=layers, recurrences=1,
            vocab_size=VOCAB_SIZE, context_length=CTX, dropout=0.0,
            mixer_mode="swiglu", swiglu_in_loop=False, mixer_width=5 * hidden), posify=True)
    if arch == "ropegate8":
        return rope_group_gate.RoPEGroupGate(bdh_best.Config(
            hidden_size=hidden, heads=4, mlp_mult=16, layers=layers, recurrences=1,
            vocab_size=VOCAB_SIZE, context_length=CTX, dropout=0.0,
            mixer_mode="swiglu", swiglu_in_loop=False, mixer_width=5 * hidden), groups=8)
    if arch == "ropegate8signedexp":
        return rope_group_gate.RoPEGroupGateSignedExp(bdh_best.Config(
            hidden_size=hidden, heads=4, mlp_mult=16, layers=layers, recurrences=1,
            vocab_size=VOCAB_SIZE, context_length=CTX, dropout=0.0,
            mixer_mode="swiglu", swiglu_in_loop=False, mixer_width=5 * hidden), groups=8)
    if arch == "ropegate8noabs":
        return rope_group_gate.RoPEGroupGateNoAbs(bdh_best.Config(
            hidden_size=hidden, heads=4, mlp_mult=16, layers=layers, recurrences=1,
            vocab_size=VOCAB_SIZE, context_length=CTX, dropout=0.0,
            mixer_mode="swiglu", swiglu_in_loop=False, mixer_width=5 * hidden), groups=8)
    if arch == "linearropegate8noabs":
        return linear_rope_group_gate.LinearRoPEGroupGate(bdh_best.Config(
            hidden_size=hidden, heads=4, mlp_mult=16, layers=layers, recurrences=1,
            vocab_size=VOCAB_SIZE, context_length=CTX, dropout=0.0,
            mixer_mode="swiglu", swiglu_in_loop=False, mixer_width=5 * hidden),
            groups=8, learned_absolute=False)
    if arch == "linearropegate8abs":
        return linear_rope_group_gate.LinearRoPEGroupGate(bdh_best.Config(
            hidden_size=hidden, heads=4, mlp_mult=16, layers=layers, recurrences=1,
            vocab_size=VOCAB_SIZE, context_length=CTX, dropout=0.0,
            mixer_mode="swiglu", swiglu_in_loop=False, mixer_width=5 * hidden),
            groups=8, learned_absolute=True)
    if arch == "linearropegate8rawabs":
        return rope_group_gate.RoPEGroupGateRaw(bdh_best.Config(
            hidden_size=hidden, heads=4, mlp_mult=16, layers=layers, recurrences=1,
            vocab_size=VOCAB_SIZE, context_length=CTX, dropout=0.0,
            mixer_mode="swiglu", swiglu_in_loop=False, mixer_width=5 * hidden),
            groups=8, learned_absolute=True)
    if arch == "linearropegate8rawnoabs":
        return rope_group_gate.RoPEGroupGateRaw(bdh_best.Config(
            hidden_size=hidden, heads=4, mlp_mult=16, layers=layers, recurrences=1,
            vocab_size=VOCAB_SIZE, context_length=CTX, dropout=0.0,
            mixer_mode="swiglu", swiglu_in_loop=False, mixer_width=5 * hidden),
            groups=8, learned_absolute=False)
    if arch == "cutpat112":
        return shared_gram.SharedGramModel(shared_gram.Config(
            hidden_size=hidden, layers=layers, group_pattern=(0, 0, 1, 0, 0, 1),
            vocab_size=VOCAB_SIZE, context_length=CTX))
    if arch == "cutshared4":
        return shared_gram.SharedGramModel(shared_gram.Config(
            hidden_size=hidden, layers=layers, n_groups=4, vocab_size=VOCAB_SIZE,
            context_length=CTX))
    if arch == "cutshared2":
        return shared_gram.SharedGramModel(shared_gram.Config(
            hidden_size=hidden, layers=layers, n_groups=2, vocab_size=VOCAB_SIZE,
            context_length=CTX))
    if arch == "cutshared":
        return shared_gram.SharedGramModel(shared_gram.Config(
            hidden_size=hidden, layers=layers, vocab_size=VOCAB_SIZE,
            context_length=CTX))
    if arch == "cutdeep":
        return bdh_best.BDHBest(bdh_best.Config(
            hidden_size=hidden, heads=4, mlp_mult=16, layers=layers, recurrences=1,
            vocab_size=VOCAB_SIZE, context_length=CTX, dropout=0.0,
            mixer_mode="swiglu", swiglu_in_loop=False, mixer_width=5 * hidden))
    if arch == "cuttr1":
        c = dict(DEFAULT["cuttr1"])
        return bdh_best.BDHBest(bdh_best.Config(
            hidden_size=hidden, heads=4, mlp_mult=16, layers=layers,
            recurrences=c.get("recurrences", 2), vocab_size=VOCAB_SIZE,
            context_length=CTX, dropout=0.0, mixer_mode="swiglu",
            swiglu_in_loop=False, mixer_width=5 * hidden))
    if arch == "hyenadil":
        return hyena_dilated.HyenaDilated(hyena.Config(
            hidden_size=hidden, layers=layers, order=2, filter_order=64,
            ffn_mult=4, vocab_size=VOCAB_SIZE, context_length=CTX))
    if arch == "hyenarope":
        return hyena_rope.HyenaRoPE(hyena.Config(
            hidden_size=hidden, layers=layers, order=2, filter_order=64,
            ffn_mult=4, vocab_size=VOCAB_SIZE, context_length=CTX))
    if arch == "gatedfish":
        return gated_fish.GatedFish(gated_fish.Config(
            hidden_size=hidden, layers=layers, recurrences=2,
            vocab_size=VOCAB_SIZE, context_length=CTX))
    if arch == "linearfish":
        return linear_fish.LinearFish(bdh_best.Config(
            hidden_size=hidden, heads=4, mlp_mult=16, layers=layers, recurrences=2,
            vocab_size=VOCAB_SIZE, context_length=CTX, dropout=0.0,
            mixer_mode="swiglu", swiglu_in_loop=False, mixer_width=5 * hidden))
    if arch == "hyenafish":
        return hyena_fish.HyenaFish(hyena_fish.Config(
            hidden_size=hidden, layers=layers, recurrences=2, vocab_size=VOCAB_SIZE,
            context_length=CTX))
    if arch == "transformer":
        return transformer.Transformer(transformer.Config(
            hidden_size=hidden, heads=4, layers=layers, vocab_size=VOCAB_SIZE,
            context_length=CTX))
    if arch == "liv":
        return liv.LIV(liv.Config(hidden_size=hidden, intermediate_mult=4, layers=layers,
                                  vocab_size=VOCAB_SIZE, context_length=CTX))
    if arch == "hyena":
        return hyena.Hyena(hyena.Config(hidden_size=hidden, layers=layers, order=2,
                                        filter_order=64, ffn_mult=4, vocab_size=VOCAB_SIZE,
                                        context_length=CTX))
    raise ValueError(arch)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["ropegate8signedexp", "linearropegate8rawabs", "linearropegate8rawnoabs", "linearropegate8abs", "linearropegate8noabs", "ropegate8noabs", "ropegate8", "pat112norm", "tfcos", "tfrope", "cutnoemb", "deepnorm", "deepnormpos", "cutpat112", "cutshared4", "cutshared2", "cutshared", "cutdeep", "cuttr1", "hyenadil", "hyenarope", "gatedfish", "linearfish", "hyenafish", "transformer", "liv", "hyena"], required=True)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-examples", type=int, default=32)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--output", type=Path)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_rng = random.Random(args.seed)
    eval_rng = random.Random(args.seed + 1)
    d = DEFAULT[args.arch]["hidden"]
    L = DEFAULT[args.arch]["layers"]
    model = build_model(args.arch, d, L).to(device)
    if args.compile:
        model = torch.compile(model, mode="reduce-overhead")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{args.arch}] d={d} L={L} params={n_params:,} device={device}", flush=True)

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

    out = args.output or HERE / "runs" / f"sudoku9_{args.arch}.json"
    result = {"arch": args.arch, "parameters": n_params, "hidden": d, "layers": L,
              "steps": args.steps, "best": best}
    out.write_text(__import__("json").dumps(result, indent=2))
    torch.save(best_state, out.with_suffix(".best.pt"))
    print(f"\n[{args.arch}] best: token_acc={best['token_accuracy']:.4f} exact_acc={best['exact_accuracy']:.4f}")


if __name__ == "__main__":
    main()
