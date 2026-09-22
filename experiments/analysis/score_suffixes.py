"""Score candidate continuations by conditional causal-LM log likelihood."""
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
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from train_sft import DATA_DIR, make_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--candidate", nargs="+", required=True)
    ap.add_argument("--raw", action="store_true",
                    help="Score candidates directly after prompt without ChatML wrapping")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    tok = Tokenizer.from_file(str(DATA_DIR / "bpe4099_chatml.json"))
    model = make_model(tok.get_vocab_size(), 2048).to(args.device)
    state = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    if "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    prefix = args.prompt if args.raw else (
        "<|im_start|>system\nYou are a helpful and accurate assistant.<|im_end|>\n"
        f"<|im_start|>user\n{args.prompt}<|im_end|>\n"
        "<|im_start|>assistant\nThe answer is")
    prefix_ids = tok.encode(prefix, add_special_tokens=False).ids
    rows = []
    with torch.inference_mode():
        for candidate in args.candidate:
            suffix_ids = tok.encode(" " + candidate, add_special_tokens=False).ids
            ids = prefix_ids + suffix_ids
            x = torch.tensor([ids[:-1]], device=args.device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)[0].float()
            logp = F.log_softmax(logits, -1)
            start = len(prefix_ids) - 1
            values = [logp[start + i, token].item() for i, token in enumerate(suffix_ids)]
            rows.append((candidate, sum(values), sum(values) / len(values), suffix_ids))
    # Length-normalized suffix likelihood is the standard fair comparison
    # when candidates do not share an identical tokenizer length.
    normalizer = torch.logsumexp(torch.tensor([r[2] for r in rows]), 0).item()
    for rank, row in enumerate(sorted(rows, key=lambda r: r[2], reverse=True), 1):
        candidate, total, mean, tokens = row
        print(f"{rank:2d}. {candidate:>4}  mean_logp={mean:8.4f}  "
              f"normalized_prob={math.exp(mean-normalizer):.4%}  "
              f"sum_logp={total:8.4f} tokens={tokens}")


if __name__ == "__main__":
    main()
