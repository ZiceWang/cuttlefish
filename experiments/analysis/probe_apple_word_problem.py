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
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(HERE))
from compare_continue_samples import load

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", type=Path, default=HERE / "runs/arithmetic_expr_poly_20k/step_16000.pt")
args = ap.parse_args()
checkpoint = args.checkpoint
tokenizer = Tokenizer.from_file(str(HERE / "bpe4096.json"))
model = load(checkpoint, "cuda").eval()

prompts = {
    "direct": "Question: Alice has 5 apples and eats 2 of them. How many apples remain?\nAnswer:",
    "solve": "Solve the following arithmetic word problem.\nQuestion: Alice has 5 apples and eats 2 of them. How many apples remain?\nAnswer:",
    "colloquial": "Alice has 5 apples. She eats 2. How many are left?\nAnswer:",
}

for name, prompt in prompts.items():
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    new = []
    with torch.inference_mode():
        for _ in range(64):
            x = torch.tensor([ids[-512:]], device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                nxt = model(x)[0, -1].argmax().item()
            ids.append(nxt)
            new.append(nxt)
            if "\n\n" in tokenizer.decode(new):
                break
    print(f"{name}\nPROMPT: {prompt!r}\nOUTPUT: {tokenizer.decode(new)!r}\n", flush=True)
