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
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from bdh_orig import BDH_GPU

DATA = HERE.parent / "bdh_sudoku4" / "data" / "tinyshakespeare.txt"


def load_vocab(path):
    text = Path(path).read_text()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    return data, len(chars), stoi, itos


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix]).to(device)
    return x, y


@torch.no_grad()
def evaluate(model, data, block_size, batch_size, device):
    model.eval()
    total, count = 0.0, 0
    for _ in range(20):
        x, y = get_batch(data, block_size, batch_size, device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total += loss.item()
        count += 1
    model.train()
    return total / count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    data, vocab_size, stoi, itos = load_vocab(DATA)
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]
    print(f"data={len(data)} chars vocab={vocab_size} device={device}", flush=True)

    model = BDH_GPU()
    if args.compile:
        model = torch.compile(model)
    model.to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"model params={params:,}", flush=True)

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
            tok_s = args.batch_size * args.block_size / (time.time() - t0)
            t0 = time.time()
            print(
                f"step={step}/{args.steps} train={loss.item():.4f} "
                f"eval={eval_loss:.4f} ppl={eval_loss:.4f} best={best:.4f} "
                f"{tok_s:,.0f} tok/s",
                flush=True,
            )

    torch.save(model.state_dict(), HERE / "runs" / "bdh_orig.pt")
    print("done", flush=True)


if __name__ == "__main__":
    main()
