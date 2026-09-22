"""Cuttlefish (bdh_best) 长序列检查：数值稳定性 + 显存 + 速度。

针对 Gram T² 部分在长序列下的行为：检查 logits/梯度 NaN、OOM、tok/s。
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
import time

import torch
import torch.nn.functional as F
import torch._dynamo

torch._dynamo.config.recompile_limit = 200

import bdh_best

VOCAB = 4096


def make_model(hidden, layers, recurrences, context_length, mixer_width=None, qk_mult=2):
    config = bdh_best.Config(
        hidden_size=hidden, heads=4, mlp_mult=16, layers=layers,
        recurrences=recurrences, vocab_size=VOCAB, context_length=context_length,
        dropout=0.0, mixer_mode="swiglu", swiglu_in_loop=False,
        mixer_width=mixer_width or 5 * hidden,
    )
    return bdh_best.BDHBest(config).cuda()


def step(model, optimizer, b, t):
    tok = torch.randint(0, VOCAB, (b, t + 1), device="cuda")
    inputs, targets = tok[:, :-1], tok[:, 1:]
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(inputs)
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
    nan_logits = bool(torch.isnan(logits.float()).any().item())
    max_abs = float(logits.float().abs().max().item())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item(), nan_logits, max_abs


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--hidden", type=int, default=232)
    a.add_argument("--layers", type=int, default=4)
    a.add_argument("--recurrences", type=int, default=2)
    a.add_argument("--batch", type=int, default=4)
    a.add_argument("--ts", default="512,1024,2048,4096,8192")
    a.add_argument("--compile", action="store_true")
    a.add_argument("--warmup", type=int, default=2)
    a.add_argument("--steps", type=int, default=5)
    args = a.parse_args()
    ts = [int(x) for x in args.ts.split(",")]
    max_t = max(ts)
    model = make_model(args.hidden, args.layers, args.recurrences, max_t + 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    if args.compile:
        model = torch.compile(model, mode="reduce-overhead")
    params = sum(p.numel() for p in model.parameters())
    report = {}
    for t in ts:
        try:
            for _ in range(args.warmup):
                loss, _, _ = step(model, optimizer, args.batch, t)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(args.steps):
                loss, nan_logits, max_abs = step(model, optimizer, args.batch, t)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            ms = elapsed * 1000 / args.steps
            tok = args.batch * t * args.steps / elapsed
            grad_max = max(
                (float(p.grad.abs().max().item()) if p.grad is not None else 0.0)
                for p in model.parameters()
            )
            mem = torch.cuda.max_memory_allocated() / 2**30
            report[f"t={t}"] = {
                "ms": round(ms, 2), "tok/s": int(tok), "nan_logits": nan_logits,
                "max_abs_logit": round(max_abs, 3), "max_grad": round(grad_max, 4),
                "peak_gib": round(mem, 2), "loss": round(loss, 4),
            }
            print(f"t={t}: {ms:.1f}ms {int(tok)} tok/s nan={nan_logits} "
                  f"max|logit|={max_abs:.3f} max|grad|={grad_max:.4f} "
                  f"peak={mem:.2f}GiB loss={loss:.4f}", flush=True)
            torch.cuda.reset_peak_memory_stats()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            report[f"t={t}"] = {"ms": "OOM"}
            print(f"t={t}: OOM", flush=True)
    print(json.dumps({"params_M": round(params / 1e6, 2), "config": vars(args), "report": report}, indent=2), flush=True)


if __name__ == "__main__":
    main()
