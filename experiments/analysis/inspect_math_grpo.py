"""Diagnose whether the addition GRPO probe learned arithmetic or answer bias."""
# --- repo path bootstrap: flat sibling imports + runs/data/tokenizer paths ---
import sys as _sys
from pathlib import Path as _Path
_EXP = _Path(__file__).resolve().parents[1]
for _d in [_EXP, *sorted(_p for _p in _EXP.iterdir() if _p.is_dir() and _p.name != "__pycache__")]:
    if str(_d) not in _sys.path:
        _sys.path.insert(0, str(_d))
HERE = _EXP.parent
# ---------------------------------------------------------------------------
import random
from pathlib import Path

import torch
from tokenizers import Tokenizer

from train_math_grpo_probe import candidate_logits
from train_sft import HERE, make_model


def make_problem(rng, lo, hi, carry=None):
    while True:
        a, b = rng.randint(lo, hi), rng.randint(lo, hi)
        if carry is None or ((a % 10 + b % 10 >= 10) == carry):
            break
    answer = a + b
    pool = {answer}
    while len(pool) < 10:
        pool.add(max(0, answer + rng.randint(-18, 18)))
    choices = list(pool); rng.shuffle(choices)
    return f"{a} + {b} =", choices, choices.index(answer)


@torch.inference_mode()
def evaluate(model, tok, lo, hi, carry, seed=42, count=2048):
    rng = random.Random(seed); hits = prob = 0.0
    for offset in range(0, count, 64):
        batch = [make_problem(rng, lo, hi, carry) for _ in range(min(64, count-offset))]
        scores = candidate_logits(model, tok, batch, "cuda")
        target = torch.tensor([x[2] for x in batch], device="cuda")
        hits += (scores.argmax(-1) == target).sum().item()
        prob += scores.softmax(-1).gather(1, target[:, None]).sum().item()
    return hits/count, prob/count


@torch.inference_mode()
def greedy(model, tok, prompt, new_tokens=16):
    ids = tok.encode(prompt, add_special_tokens=False).ids
    for _ in range(new_tokens):
        x = torch.tensor([ids], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            token = model(x)[0, -1].argmax().item()
        ids.append(token)
    return tok.decode(ids)


def main():
    tok = Tokenizer.from_file(str(HERE / "bpe4096.json"))
    checkpoints = {
        "pretrain": HERE / "runs/big_formal512_10shard_1ep.best.pt",
        "grpo1000": HERE / "runs/pretrain_add_grpo_1000.pt",
    }
    buckets = [
        ("train_range_all", 10, 89, None),
        ("train_range_no_carry", 10, 89, False),
        ("train_range_carry", 10, 89, True),
        ("ood_single_digit", 1, 9, None),
        ("ood_large", 90, 150, None),
    ]
    prompts = ["37 + 58 =", "23 + 41 =", "8 + 7 =", "112 + 79 =",
               "The development of scientific knowledge depends on"]
    for name, checkpoint in checkpoints.items():
        model = make_model(4096, 128).cuda()
        model.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True))
        model.eval(); print(f"\n=== {name} ===", flush=True)
        for bucket, lo, hi, carry in buckets:
            acc, p = evaluate(model, tok, lo, hi, carry)
            print(f"{bucket}: accuracy={acc:.4%} correct_probability={p:.4%}", flush=True)
        for prompt in prompts:
            print("GREEDY", repr(prompt), "=>", repr(greedy(model, tok, prompt)), flush=True)
        del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
