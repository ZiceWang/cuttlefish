"""Identical-seed generation comparison for base and continued checkpoints."""
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
from pathlib import Path

import torch
from tokenizers import Tokenizer

from train_ultrafineweb_continue import make_model




def load(path, device):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    model = make_model(512)
    model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in state.items()})
    return model.to(device).eval()


def generate(model, tokenizer, prompt, length, seed, device, temperature):
    torch.manual_seed(seed)
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    generated = []
    with torch.inference_mode():
        for _ in range(length):
            x = torch.tensor([ids[-512:]], device=device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16,
                                enabled=device == "cuda"):
                logits = model(x)[0, -1].float() / temperature
            seen = torch.tensor(list(set(ids[-256:])), device=device)
            selected = logits[seen]
            logits[seen] = torch.where(selected < 0, selected * 1.1, selected / 1.1)
            cutoff = torch.topk(logits, 50).values[-1]
            logits[logits < cutoff] = -torch.inf
            token = torch.multinomial(torch.softmax(logits, -1), 1).item()
            ids.append(token)
            generated.append(token)
    return tokenizer.decode(generated)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--continued", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--length", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.8)
    args = ap.parse_args()
    tok = Tokenizer.from_file(str(HERE / "bpe4096.json"))
    prompts = [
        "The sky appears blue during the day because",
        "Question: Why is the sky blue?\nAnswer:",
        "Question: What is photosynthesis and why is it important?\nAnswer:",
        "Question: What is the capital of France?\nAnswer:",
        "Artificial intelligence can help scientific research by",
    ]
    for label, path in (("BASE", args.base), ("CONTINUED_10K", args.continued)):
        model = load(path, args.device)
        print(f"\n######## {label} ########", flush=True)
        for i, prompt in enumerate(prompts):
            text = generate(model, tok, prompt, args.length, 7000 + i, args.device,
                            args.temperature)
            print(f"\nPROMPT: {prompt}\n{text}", flush=True)
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
