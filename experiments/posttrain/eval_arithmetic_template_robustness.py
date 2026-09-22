"""Evaluate arithmetic generalization under unseen question phrasings."""
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
import re
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(HERE))
from compare_continue_samples import load


TEMPLATES = {
    "trained": lambda e: f"Question: What is {e}?\nAnswer:",
    "calculate": lambda e: f"Question: Calculate {e}.\nAnswer:",
    "compute": lambda e: f"Compute {e}.\nAnswer:",
    "bare": lambda e: f"{e} =",
    "please": lambda e: f"Please solve the following arithmetic problem: {e}.\nThe answer is",
}


def generate(model, tok, prompt):
    ids = tok.encode(prompt, add_special_tokens=False).ids
    new = []
    with torch.inference_mode():
        for _ in range(32):
            x = torch.tensor([ids[-512:]], device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                nxt = model(x)[0, -1].argmax().item()
            ids.append(nxt); new.append(nxt)
            if "\n" in tok.decode(new):
                break
    return tok.decode(new)


def cases(rng, n=30):
    result = []
    for _ in range(n):
        a, b, c = rng.randint(0, 30), rng.randint(0, 30), rng.randint(0, 99)
        result.append((f"{a} x {b} + {c}", a * b + c))
    return result


def main():
    tok = Tokenizer.from_file(str(HERE / "bpe4096.json"))
    model = load(HERE / "runs/arithmetic_qa_5k/final.pt", "cuda")
    test = cases(random.Random(99173))
    for name, template in TEMPLATES.items():
        correct = 0
        examples = []
        for expression, gold in test:
            text = generate(model, tok, template(expression))
            nums = re.findall(r"-?\d+", text)
            ok = bool(nums and int(nums[-1]) == gold)
            correct += ok
            if len(examples) < 3:
                examples.append((expression, gold, text.strip(), ok))
        print(f"{name}: {correct}/{len(test)} = {correct/len(test):.1%}")
        for row in examples:
            print(" ", row)


if __name__ == "__main__":
    main()
