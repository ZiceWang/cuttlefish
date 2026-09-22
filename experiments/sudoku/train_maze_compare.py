"""Maze-30x30 四架构对比：同参、path 位置 mask loss、teacher-forcing + 自回归评估。

数据：data/maze-30x30-hard-1k-5pct（train 400 / val 1000），序列 900，vocab 6。
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
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import bdh_best
import transformer
import liv
import hyena

DATA = HERE.parent.parent / "data" / "maze-30x30-hard-1k-5pct"
VOCAB = 6
CTX = 900
PATH = 5  # path token

# 同参数量对齐（基准 Cuttlefish d128 L4 R2 ≈ 2.78M @ctx163 / ≈2.87M @ctx900）
DEFAULT = {
    "bdhbest": dict(hidden=128, layers=4),
    "transformer": dict(hidden=168, layers=8),
    "liv": dict(hidden=184, layers=5),
    "hyena": dict(hidden=208, layers=5),
}


def load():
    tr = np.load(DATA / "train" / "all__inputs.npy").astype(np.int64)
    ty = np.load(DATA / "train" / "all__labels.npy").astype(np.int64)
    vr = np.load(DATA / "val" / "all__inputs.npy").astype(np.int64)
    vy = np.load(DATA / "val" / "all__labels.npy").astype(np.int64)
    return tr, ty, vr, vy


def build(arch, hidden, layers, ctx=CTX):
    if arch == "bdhbest":
        c = bdh_best.Config(hidden_size=hidden, heads=4, mlp_mult=16, layers=layers,
                            recurrences=2, vocab_size=VOCAB, context_length=ctx,
                            dropout=0.0, mixer_mode="swiglu", swiglu_in_loop=False,
                            mixer_width=5 * hidden)
        return bdh_best.BDHBest(c)
    if arch == "transformer":
        return transformer.Transformer(transformer.Config(
            hidden_size=hidden, heads=4, layers=layers, vocab_size=VOCAB, context_length=ctx))
    if arch == "liv":
        return liv.LIV(liv.Config(hidden_size=hidden, intermediate_mult=4, layers=layers,
                                  vocab_size=VOCAB, context_length=ctx))
    if arch == "hyena":
        return hyena.Hyena(hyena.Config(hidden_size=hidden, layers=layers, order=2,
                                        filter_order=64, ffn_mult=4, vocab_size=VOCAB,
                                        context_length=ctx))
    raise ValueError(arch)


def bfs_connected(grid_pred, start, goal):
    """自回归生成的 30x30 grid：从 start 沿 path(5) 格 4 邻域能否到 goal（只走模型画的解）。"""
    import collections
    SIDE = 30
    grid = grid_pred.reshape(SIDE, SIDE)
    start = (start // SIDE, start % SIDE)
    goal = (goal // SIDE, goal % SIDE)
    if grid[start] not in (3, 5):
        return False
    if grid[goal] not in (4, 5):
        return False
    seen = {start}
    q = collections.deque([start])
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < SIDE and 0 <= nc < SIDE and (nr, nc) not in seen \
               and grid[nr, nc] == 5:
                seen.add((nr, nc))
                q.append((nr, nc))
    return False


@torch.no_grad()
def evaluate(model, vr, vy, rng, device, n=64, n_autoreg=20):
    model.eval()
    idx = rng.sample(range(len(vr)), n)
    x = torch.tensor(vr[idx], device=device)
    y = torch.tensor(vy[idx], device=device)
    logits = model(x[:, :-1])
    pred = logits.argmax(-1)
    pm = y[:, 1:] == PATH
    tp = ((pred == PATH) & pm).float().sum().item()
    fp = ((pred == PATH) & ~pm).float().sum().item()
    fn = pm.float().sum().item() - tp
    precision = tp / max(tp + fp, 1e-9)
    recall = tp / max(tp + fn, 1e-9)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    row_exact = ((pred == PATH).float() * pm.float()).sum(-1) == pm.float().sum(-1)
    row_exact = row_exact.float().mean().item()
    # 自回归 + BFS 连通率（20 样本）
    auto_idx = rng.sample(range(len(vr)), n_autoreg)
    ax = torch.tensor(vr[auto_idx], device=device)
    ay = torch.tensor(vy[auto_idx], device=device)
    seq = ax[:, :1].clone()
    for _ in range(CTX - 1):
        lg = model(seq)
        nxt = lg[:, -1].argmax(-1, keepdim=True)
        seq = torch.cat((seq, nxt), dim=1)
    a_pred = seq.cpu().numpy()  # [B, 900]（第0格=input，其余为生成）
    connected = 0
    for i in range(n_autoreg):
        inp = vr[auto_idx[i]]
        start = int(np.argwhere(inp == 3).flatten()[0]) if (inp == 3).any() else 0
        goal = int(np.argwhere(inp == 4).flatten()[0]) if (inp == 4).any() else 0
        if bfs_connected(a_pred[i], start, goal):
            connected += 1
    model.train()
    return f1, row_exact, connected / n_autoreg


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--arch", choices=["bdhbest", "transformer", "liv", "hyena"], required=True)
    a.add_argument("--hidden", type=int, default=128)
    a.add_argument("--layers", type=int, default=4)
    a.add_argument("--steps", type=int, default=600)
    a.add_argument("--batch-size", type=int, default=32)
    a.add_argument("--lr", type=float, default=3e-4)
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--compile", action="store_true")
    a.add_argument("--output", type=Path)
    args = a.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)
    tr, ty, vr, vy = load()
    args.hidden = DEFAULT[args.arch]["hidden"]
    args.layers = DEFAULT[args.arch]["layers"]
    model = build(args.arch, args.hidden, args.layers).to(device)
    if args.compile:
        model = torch.compile(model, mode="reduce-overhead")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    params = sum(p.numel() for p in model.parameters())
    print(f"[{args.arch}] d={args.hidden} L={args.layers} params={params:,} device={device}",
          flush=True)
    n_train = len(tr)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = rng.sample(range(n_train), args.batch_size)
        x = torch.tensor(tr[idx], device=device)
        y = torch.tensor(ty[idx], device=device)
        x, y = x[:, :-1], y[:, 1:]
        logits = model(x)
        # 全序列 loss（模型需复制 wall/space + 输出 path，无法只刷 path token）
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            f1, row_exact, conn = evaluate(model, vr, vy, rng, device)
            el = time.time() - t0
            print(f"step={step}/{args.steps} loss={loss.item():.4f} "
                  f"path_f1={f1:.4f} path_row={row_exact:.4f} auto_conn={conn:.3f} "
                  f"({el:.0f}s)", flush=True)
    result = {"arch": args.arch, "params": params, "hidden": args.hidden, "layers": args.layers,
              "steps": args.steps, "path_f1": round(f1, 4),
              "path_row_exact": round(row_exact, 4), "auto_connected": round(conn, 4)}
    out = args.output or HERE / "runs" / f"maze_{args.arch}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\n[{args.arch}] path_f1={f1:.4f} path_row={row_exact:.4f} auto_conn={conn:.3f}")


if __name__ == "__main__":
    main()
