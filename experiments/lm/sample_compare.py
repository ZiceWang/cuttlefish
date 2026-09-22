"""从 4 个已训 checkpoint 滚动采样文本，对比生成效果。

用法: python sample_compare.py [--model all|bdh_best|transformer|liv|hyena]
      [--length 600] [--seed-text "..."] [--temp 0.8] [--top-k 50]
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

from train_compare import MODELS, load_vocab, DATA



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all")
    ap.add_argument("--length", type=int, default=600)
    ap.add_argument("--seed-text", default="ROMEO:")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--tag", default="", help="checkpoint 文件名后缀（默认=model名）")
    args = ap.parse_args()

    torch.manual_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data, vocab_size = load_vocab(DATA)
    chars = sorted(set(Path(DATA).read_text()))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}

    names = [args.model] if args.model != "all" else ["transformer", "bdh_best", "hyena", "liv"]

    for name in names:
        ckpt = HERE / "runs" / f"compare_{args.tag or name}_5L.pt"
        if not ckpt.exists():
            print(f"[{name}] 无 checkpoint，跳过")
            continue
        model = MODELS[name]().to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())

        ids = [stoi[c] for c in args.seed_text]
        with torch.no_grad():
            for _ in range(args.length):
                ctx = torch.tensor([ids[-256:]], dtype=torch.long, device=device)
                logits = model(ctx)[0, -1] / args.temp
                if args.top_k:
                    v, _ = torch.topk(logits, args.top_k)
                    logits[logits < v[-1]] = float("-inf")
                probs = torch.softmax(logits, dim=-1)
                ids.append(torch.multinomial(probs, 1).item())

        text = "".join(itos[i] for i in ids)
        print("=" * 78)
        print(f"[{name}]  params={n_params:,}")
        print("-" * 78)
        print(text)
        print(flush=True)


if __name__ == "__main__":
    main()
