"""从大数据集训练的 bdh_best checkpoint 采样文本（byte-level, vocab=256）。

用法: python sample_big.py --tag r3|r1 [--length 400] [--temp 0.8] [--top-k 50]
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
from pathlib import Path

import torch

import bdh_best
import rope_group_gate

VOCAB = 256


def decode_bytes(ids):
    try:
        return bytes(ids).decode("utf-8", errors="replace")
    except Exception:
        return "".join(chr(b) for b in ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="r3")
    ap.add_argument("--length", type=int, default=500)
    ap.add_argument("--seed-text", default="The quick brown fox jumps over the lazy dog. ")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--repetition-window", type=int, default=512)
    ap.add_argument("--model", choices=["bdh_best", "ropegate8noabs", "signedexpnoabs"], default="bdh_best")
    ap.add_argument("--tokenizer", choices=["byte", "bpe"], default="byte")
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--context-window", type=int, default=256)
    ap.add_argument("--hidden-size", type=int, default=450)
    ap.add_argument("--mixer-width", type=int, default=2250)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()

    torch.manual_seed(1337)
    device = (("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else args.device)
    ckpt = args.checkpoint or HERE / "runs" / f"big_{args.tag}.pt"
    tokenizer = None
    vocab = VOCAB
    if args.tokenizer == "bpe":
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(HERE / "bpe4096.json"))
        vocab = 4096
    config = bdh_best.Config(
        hidden_size=args.hidden_size if args.model in ("ropegate8noabs", "signedexpnoabs") else 232,
        heads=4, mlp_mult=16,
        layers=args.layers if args.model in ("ropegate8noabs", "signedexpnoabs") else 5,
        recurrences=1 if args.model in ("ropegate8noabs", "signedexpnoabs") else (3 if args.tag == "r3" else 1),
        vocab_size=vocab, context_length=args.context_window, dropout=0.0,
        mixer_mode="swiglu", swiglu_in_loop=False,
        mixer_width=args.mixer_width if args.model in ("ropegate8noabs", "signedexpnoabs") else 0)
    if args.model == "signedexpnoabs":
        model = rope_group_gate.RoPEGroupGateSignedExp(
            config, groups=8, learned_absolute=False).to(device)
    elif args.model == "ropegate8noabs":
        model = rope_group_gate.RoPEGroupGateNoAbs(config, groups=8).to(device)
    else:
        model = bdh_best.BDHBest(config).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    ids = tokenizer.encode(args.seed_text).ids if tokenizer else list(args.seed_text.encode("utf-8"))
    with torch.no_grad():
        for _ in range(args.length):
            ctx = torch.tensor([ids[-args.context_window:]], dtype=torch.long, device=device)
            logits = model(ctx)[0, -1] / args.temp
            if args.repetition_penalty != 1.0:
                seen = torch.tensor(list(set(ids[-args.repetition_window:])), device=device)
                selected = logits[seen]
                logits[seen] = torch.where(
                    selected < 0,
                    selected * args.repetition_penalty,
                    selected / args.repetition_penalty,
                )
            if args.top_k:
                v, _ = torch.topk(logits, args.top_k)
                logits[logits < v[-1]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            ids.append(torch.multinomial(probs, 1).item())

    print("=" * 78)
    print(f"[{args.model} {args.tag}] (big-data)")
    print("-" * 78)
    print(tokenizer.decode(ids) if tokenizer else decode_bytes(ids))
    print(flush=True)


if __name__ == "__main__":
    main()
