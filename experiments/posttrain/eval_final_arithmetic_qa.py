"""Probe arithmetic ability of the final continued checkpoint in QA format."""
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
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from compare_continue_samples import load


CKPT = HERE / "runs/ultrafineweb_en_1ep/final.pt"
TESTS = [
    ("What is 2 + 3?", "5", ["3", "4", "5", "6"]),
    ("What is 7 + 8?", "15", ["13", "14", "15", "16"]),
    ("What is 37 + 58?", "95", ["85", "94", "95", "96"]),
    ("What is 91 - 46?", "45", ["35", "44", "45", "55"]),
    ("What is 6 times 7?", "42", ["36", "40", "42", "48"]),
    ("What is 12 times 8?", "96", ["86", "92", "96", "108"]),
    ("What is 100 divided by 4?", "25", ["20", "24", "25", "40"]),
    ("Alice has 12 apples and buys 9 more. How many apples does she have?", "21", ["3", "20", "21", "108"]),
]


def greedy(model, tok, prompt, device, length=16):
    ids = tok.encode(prompt, add_special_tokens=False).ids
    new = []
    with torch.inference_mode():
        for _ in range(length):
            x = torch.tensor([ids[-512:]], device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                nxt = model(x)[0, -1].argmax().item()
            ids.append(nxt); new.append(nxt)
    return tok.decode(new)


def rank(model, tok, prompt, candidates, device):
    pre = tok.encode(prompt, add_special_tokens=False).ids
    rows = []
    with torch.inference_mode():
        for candidate in candidates:
            suffix = tok.encode(" " + candidate, add_special_tokens=False).ids
            ids = pre + suffix
            x = torch.tensor([ids[:-1]], device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lp = F.log_softmax(model(x)[0].float(), -1)
            vals = [lp[len(pre) - 1 + i, token].item() for i, token in enumerate(suffix)]
            rows.append((sum(vals) / len(vals), candidate))
    return [candidate for _, candidate in sorted(rows, reverse=True)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=CKPT)
    args = ap.parse_args()
    device = "cuda"
    tok = Tokenizer.from_file(str(HERE / "bpe4096.json"))
    model = load(args.checkpoint, device)
    free_correct = choice_correct = 0
    for question, gold, choices in TESTS:
        prompt = f"Question: {question}\nAnswer:"
        output = greedy(model, tok, prompt, device)
        first_number = (re.search(r"-?\d+", output) or [None])[0]
        ranking = rank(model, tok, prompt, choices, device)
        free_correct += first_number == gold
        choice_correct += ranking[0] == gold
        print(f"Q: {question}\ngold={gold} greedy={output!r} first_number={first_number} "
              f"ranking={ranking}\n", flush=True)
    print(f"SUMMARY free={free_correct}/{len(TESTS)} choice={choice_correct}/{len(TESTS)}", flush=True)


if __name__ == "__main__":
    main()
