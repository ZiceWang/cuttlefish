"""Zero-shot ablation: remove SignedExp shaping from a trained checkpoint."""
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
import math
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import Dataset
from tokenizers import Tokenizer

import bdh_best
from gram_norm import GramAttentionNorm
from rope_group_gate import RoPEGroupGateSignedExp

# Hugging Face cache root; override with HF_HOME (defaults to ~/.cache/huggingface).
HF_HOME = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
DATASET_DIR = (
    HF_HOME / "datasets" / "HuggingFaceFW___finepdfs_edu_50_bt-"
    "dclm_30_bt-fineweb_edu_20_bt-shuffled" / "default-fdc261053abb4d32" / "0.0.0"
    / "8904a95879538b9e7db6cf8636b2cd16b5e86a76"
)
PREFIX = "finepdfs_edu_50_bt-dclm_30_bt-fineweb_edu_20_bt-shuffled-train"


def batches(dataset, tok, count, batch_size, context, seed):
    rng = random.Random(seed)
    for _ in range(count):
        rows = []
        while len(rows) < batch_size:
            ids = tok.encode(dataset[rng.randrange(len(dataset))]["text"]).ids
            if len(ids) <= context:
                continue
            start = rng.randrange(len(ids) - context)
            seq = ids[start:start + context + 1]
            rows.append(seq)
        batch = torch.tensor(rows, dtype=torch.long)
        yield batch[:, :-1], batch[:, 1:]


@torch.inference_mode()
def evaluate(model, cached_batches, device):
    loss_sum = correct = tokens = 0
    for x, y in cached_batches:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss_sum += F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="sum").item()
        correct += (logits.argmax(-1) == y).sum().item()
        tokens += y.numel()
    loss = loss_sum / tokens
    return {"loss": loss, "perplexity": math.exp(loss), "accuracy": correct / tokens,
            "tokens": tokens}


@torch.inference_mode()
def generate(model, tok, prompt, new_tokens, device):
    ids = tok.encode(prompt).ids
    for _ in range(new_tokens):
        x = torch.tensor([ids[-512:]], device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
        ids.append(logits[0, -1].argmax().item())
    return tok.decode(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--new-tokens", type=int, default=32)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    device = "cuda"
    tok = Tokenizer.from_file(str(HERE / "bpe4096.json"))
    config = bdh_best.Config(hidden_size=512, heads=4, mlp_mult=16, layers=12,
                             recurrences=1, vocab_size=4096, context_length=512,
                             dropout=0.1, mixer_mode="swiglu", swiglu_in_loop=False,
                             mixer_width=2560)
    model = RoPEGroupGateSignedExp(config, groups=8, learned_absolute=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()
    val = Dataset.from_file(str(DATASET_DIR / f"{PREFIX}-00007-of-00008.arrow"))
    fixed = list(batches(val, tok, args.batches, args.batch_size, 512, 20260911))
    prompt = "Artificial intelligence is"
    baseline = evaluate(model, fixed, device)
    baseline_text = generate(model, tok, prompt, args.new_tokens, device)
    for block in model.blocks:
        block.attn = GramAttentionNorm(block.qk_width, posify=False).to(device)
    ablated = evaluate(model, fixed, device)
    ablated_text = generate(model, tok, prompt, args.new_tokens, device)
    result = {"checkpoint": args.checkpoint, "baseline": baseline,
              "remove_signed_exp": ablated, "prompt": prompt,
              "baseline_text": baseline_text, "remove_signed_exp_text": ablated_text}
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
