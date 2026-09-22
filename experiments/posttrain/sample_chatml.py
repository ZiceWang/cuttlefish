"""Sample an SFT checkpoint with native ChatML control tokens."""
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
from pathlib import Path

import torch
from tokenizers import Tokenizer

from train_sft import DATA_DIR, HERE, make_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--prompt", default="Why is the sky blue?")
    ap.add_argument("--system", default="You are a helpful and accurate assistant.")
    ap.add_argument("--demo-json", type=Path)
    ap.add_argument("--context-length", type=int, default=2048)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--min-p", type=float, default=0.0)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--no-repeat-ngram", type=int, default=0)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    tokenizer = Tokenizer.from_file(str(DATA_DIR / "bpe4099_chatml.json"))
    model = make_model(tokenizer.get_vocab_size(), args.context_length).to(args.device)
    state = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    if "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    messages = [{"role": "system", "content": args.system}]
    if args.demo_json:
        demos = json.loads(args.demo_json.read_text())
        if any(m.get("role") not in ("user", "assistant") for m in demos):
            raise ValueError("demonstrations must be user/assistant messages")
        messages.extend(demos)
    messages.append({"role": "user", "content": args.prompt})
    text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
                   for m in messages) + "<|im_start|>assistant\n"
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    end = tokenizer.token_to_id("<|im_end|>")
    with torch.inference_mode():
        for _ in range(args.max_new_tokens):
            x = torch.tensor([ids[-args.context_length:]], device=args.device)
            with torch.autocast(device_type=args.device, dtype=torch.bfloat16,
                                enabled=args.device == "cuda"):
                logits = model(x)[0, -1] / args.temperature
            if args.repetition_penalty != 1.0:
                for previous in set(ids):
                    logits[previous] = (logits[previous] * args.repetition_penalty
                                        if logits[previous] < 0 else
                                        logits[previous] / args.repetition_penalty)
            if args.no_repeat_ngram > 0 and len(ids) >= args.no_repeat_ngram - 1:
                prefix_len = args.no_repeat_ngram - 1
                prefix = tuple(ids[-prefix_len:]) if prefix_len else ()
                banned = set()
                for i in range(len(ids) - args.no_repeat_ngram + 1):
                    if tuple(ids[i:i + prefix_len]) == prefix:
                        banned.add(ids[i + prefix_len])
                if banned:
                    logits[list(banned)] = -torch.inf
            if args.top_k:
                cutoff = torch.topk(logits, args.top_k).values[-1]
                logits[logits < cutoff] = -torch.inf
            probs = logits.softmax(-1)
            if args.min_p:
                probs[probs < probs.max() * args.min_p] = 0
            if args.top_p < 1.0:
                sorted_probs, sorted_ids = probs.sort(descending=True)
                remove = sorted_probs.cumsum(0) - sorted_probs > args.top_p
                probs[sorted_ids[remove]] = 0
            probs /= probs.sum()
            token = torch.multinomial(probs, 1).item()
            if token == end:
                break
            ids.append(token)
    print(tokenizer.decode(ids, skip_special_tokens=False))


if __name__ == "__main__":
    main()
