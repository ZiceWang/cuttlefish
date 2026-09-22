"""Continue the 4096-token Cuttlefish pretrain on two English Ultra-FineWeb shards."""
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
import random
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

import bdh_best
import rope_group_gate


DATA = Path("data/ultrafineweb_l3_probe/data/ultrafineweb_en_l3")
SOURCE = HERE / "runs/big_formal512_10shard_1ep.ckpt.pt"


def source_files():
    files = []
    for subset in ("multi_style", "qa"):
        matches = sorted((DATA / subset).glob("*.parquet"))
        if len(matches) != 1:
            raise RuntimeError(f"expected exactly one {subset} parquet, found {len(matches)}")
        files.extend(matches)
    return files


class PackedParquet(torch.utils.data.IterableDataset):
    def __init__(self, paths, tokenizer_path, context, seed):
        self.paths = list(paths)
        self.tokenizer_path = str(tokenizer_path)
        self.context = context
        self.seed = seed

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        wid, workers = (worker.id, worker.num_workers) if worker else (0, 1)
        tokenizer = Tokenizer.from_file(self.tokenizer_path)
        separator = tokenizer.encode("\n", add_special_tokens=False).ids
        tasks = [(path, rg) for path in self.paths
                 for rg in range(pq.ParquetFile(path).num_row_groups)]
        random.Random(self.seed).shuffle(tasks)
        tasks = tasks[wid::workers]
        buffer = []
        for task_index, (path, rg) in enumerate(tasks):
            texts = pq.ParquetFile(path).read_row_group(rg, columns=["content"])["content"].to_pylist()
            random.Random(self.seed + wid * 100003 + task_index).shuffle(texts)
            for start in range(0, len(texts), 128):
                batch = [text or "" for text in texts[start:start + 128]]
                for encoded in tokenizer.encode_batch(batch, add_special_tokens=False):
                    buffer.extend(encoded.ids)
                    buffer.extend(separator)
                    while len(buffer) >= self.context + 1:
                        seq = torch.tensor(buffer[:self.context + 1], dtype=torch.long)
                        yield seq[:-1], seq[1:]
                        del buffer[:self.context]


def make_model(context):
    config = bdh_best.Config(
        hidden_size=512, heads=4, mlp_mult=16, layers=12, recurrences=1,
        vocab_size=4096, context_length=context, dropout=0.1,
        mixer_mode="swiglu", swiglu_in_loop=False, mixer_width=2560)
    return rope_group_gate.RoPEGroupGateSignedExp(
        config, groups=8, learned_absolute=False)


def lr_factor(step, warmup, total, floor):
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = min(1.0, (step - warmup) / max(total - warmup, 1))
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--warmup-steps", type=int, default=1000)
    ap.add_argument("--total-tokens", type=int, required=True)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=10000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--tag", default="ultrafineweb_en_warm_continue")
    ap.add_argument("--seed", type=int, default=20260912)
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.device == "cuda":
        torch.set_float32_matmul_precision("high")
    model = make_model(args.context)
    source = torch.load(args.source, map_location="cpu", weights_only=False)
    state = source["model"] if "model" in source else source
    model.load_state_dict(state, strict=True)
    model = model.to(args.device)
    params = sum(p.numel() for p in model.parameters())
    if args.compile:
        model = torch.compile(model, mode="reduce-overhead")

    # Full batches only: the iterable cannot provide a partial optimizer update at EOF.
    steps = args.total_tokens // (args.batch_size * args.context)
    if args.max_steps:
        steps = min(steps, args.max_steps)
    warmup = min(args.warmup_steps, max(1, steps // 10))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_factor(step, warmup, steps, args.min_lr_ratio))
    paths = source_files()
    dataset = PackedParquet(paths, HERE / "bpe4096.json", args.context, args.seed)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.workers,
        pin_memory=args.device == "cuda", persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None, drop_last=True)
    run_dir = HERE / "runs" / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"source": str(args.source), "source_step": source.get("step"),
                      "files": [str(x) for x in paths], "parameters": params,
                      "total_tokens": args.total_tokens, "planned_steps": steps,
                      "warmup_steps": warmup, "peak_lr": args.lr}), flush=True)

    model.train()
    started = time.perf_counter()
    tokens = 0
    final_step = 0
    for step, (x, y) in enumerate(loader, 1):
        if step > steps:
            break
        x, y = x.to(args.device, non_blocking=True), y.to(args.device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=args.device, dtype=torch.bfloat16,
                            enabled=args.device == "cuda"):
            logits = model(x)
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        final_step, tokens = step, step * args.batch_size * args.context
        if step == 1 or step % args.log_every == 0:
            elapsed = time.perf_counter() - started
            print(f"step={step}/{steps} loss={loss.item():.4f} ppl={math.exp(min(loss.item(), 20)):.3f} "
                  f"lr={scheduler.get_last_lr()[0]:.3e} grad={float(grad_norm):.3f} "
                  f"tokens={tokens:,} tok/s={tokens / elapsed:,.0f}", flush=True)
        if args.save_every and step % args.save_every == 0:
            sd = {k.removeprefix("_orig_mod."): v for k, v in model.state_dict().items()}
            torch.save({"model": sd, "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "step": step,
                        "stage_tokens": tokens, "source": str(args.source)},
                       run_dir / f"step_{step:07d}.ckpt.pt")

    elapsed = time.perf_counter() - started
    sd = {k.removeprefix("_orig_mod."): v for k, v in model.state_dict().items()}
    torch.save(sd, run_dir / "final.pt")
    result = {"steps": final_step, "tokens": tokens, "seconds": elapsed,
              "tokens_per_second": tokens / max(elapsed, 1e-9), "complete_epoch": final_step >= steps}
    (run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
