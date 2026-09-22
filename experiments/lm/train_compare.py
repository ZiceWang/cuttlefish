"""统一对比训练：5 层、~10M 参数、同一训练流程，对比 5 个模块的 PPL。

模块: bdh_orig / bdh_best / transformer / liv / hyena
流程: 同一数据集(tinyshakespeare) / 同 seed / 同 lr+cosine / 同 batch/ctx / 同 eval
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
import dataclasses
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import bdh_orig
import bdh_best
import liv
import transformer
import hyena

DATA = HERE.parent / "bdh_sudoku4" / "data" / "tinyshakespeare.txt"

# 5 层、~10M 参数配置（不含原始 BDH，它层间共享参数且已验证）
MODELS = {
    "bdh_best": lambda: bdh_best.BDHBest(bdh_best.Config(
        hidden_size=232, heads=4, mlp_mult=16, layers=5, recurrences=3,
        vocab_size=65, context_length=256, dropout=0.1)),
    "transformer": lambda: transformer.Transformer(transformer.Config(
        hidden_size=400, heads=4, layers=5, vocab_size=65, context_length=256)),
    "liv": lambda: liv.LIV(liv.Config(
        hidden_size=350, intermediate_mult=4, layers=5, vocab_size=65, context_length=256)),
    "hyena": lambda: hyena.Hyena(hyena.Config(
        hidden_size=400, layers=5, order=2, filter_order=64, ffn_mult=4,
        vocab_size=65, context_length=256)),
}


def load_vocab(path):
    text = Path(path).read_text()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    return data, len(chars)


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix]).to(device)
    return x, y


@torch.no_grad()
def evaluate(model, data, block_size, batch_size, device, n_batches=20):
    model.eval()
    total, count = 0.0, 0
    for _ in range(n_batches):
        x, y = get_batch(data, block_size, batch_size, device)
        logits = model(x)
        total += F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)).item()
        count += 1
    model.train()
    return total / count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS) + ["all"], default="all")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--recurrences", type=int, default=None,
                    help="覆盖 bdh_best 的递归次数")
    ap.add_argument("--tag", default="", help="结果文件名后缀")
    ap.add_argument("--params-only", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    data, vocab_size = load_vocab(DATA)
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]

    names = [args.model] if args.model != "all" else list(MODELS)
    results = {}
    tag = args.tag or args.model
    for name in names:
        model = MODELS[name]()
        if args.recurrences is not None and name == "bdh_best":
            model = bdh_best.BDHBest(bdh_best.Config(
                hidden_size=232, heads=4, mlp_mult=16, layers=5,
                recurrences=args.recurrences, vocab_size=65, context_length=256,
                dropout=0.1))
        model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[{name}] params={n_params:,}  (目标 ~10M)", flush=True)
        if args.params_only:
            continue
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)

        best = float("inf")
        t0 = time.time()
        for step in range(1, args.steps + 1):
            x, y = get_batch(train_data, args.block_size, args.batch_size, device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            if step % args.eval_every == 0 or step == 1:
                eval_loss = evaluate(model, val_data, args.block_size, args.batch_size, device)
                best = min(best, eval_loss)
                tok_s = args.batch_size * args.block_size / max(time.time() - t0, 1e-9)
                t0 = time.time()
                print(f"  [{name}] step={step}/{args.steps} train={loss.item():.4f} "
                      f"eval={eval_loss:.4f} ppl={eval_loss:.4f} best={best:.4f} "
                      f"{tok_s:,.0f} tok/s", flush=True)
        results[name] = {
            "params": n_params,
            "best_loss": best,
            "best_ppl": best,
            "steps": args.steps,
            "recurrences": args.recurrences,
        }
        torch.save(model.state_dict(), HERE / "runs" / f"compare_{tag}_5L.pt")

    if results:
        out = HERE / "runs" / f"compare_{tag}_5L_10M.json"
        out.write_text(json.dumps(results, indent=2))
        print("\n=== 结果 ===")
        for n, r in sorted(results.items(), key=lambda kv: kv[1]["best_loss"]):
            print(f"{n:14} params={r['params']:>9,}  best_loss={r['best_loss']:.4f}  "
                  f"ppl={r['best_ppl']:.4f}")


if __name__ == "__main__":
    main()
