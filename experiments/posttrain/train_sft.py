"""ChatML SFT for the 136M SignedExp-noabs model.

The JSONL stream is packed into fixed-length blocks. Loss is computed only on
assistant content and the assistant <|im_end|> token; no examples are dropped.
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
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

import bdh_best
from rope_group_gate import RoPEGroupGateSignedExp

DATA_DIR = HERE / "data" / "sft_cherry_v1"
IGNORE_INDEX = -100


def encode_messages(messages, tokenizer):
    start = tokenizer.token_to_id("<|im_start|>")
    end = tokenizer.token_to_id("<|im_end|>")
    newline = tokenizer.encode("\n", add_special_tokens=False).ids
    ids, supervise = [], []
    for message in messages:
        role = message["role"]
        prefix = [start] + tokenizer.encode(role + "\n", add_special_tokens=False).ids
        content = tokenizer.encode(message["content"], add_special_tokens=False).ids
        suffix = [end] + newline
        ids.extend(prefix + content + suffix)
        if role == "assistant":
            supervise.extend([False] * len(prefix) + [True] * (len(content) + 1)
                             + [False] * len(newline))
        else:
            supervise.extend([False] * (len(prefix) + len(content) + len(suffix)))
    return ids, supervise


class PackedChatDataset(torch.utils.data.IterableDataset):
    def __init__(self, path, tokenizer_path, context_length, repeat=False,
                 shuffle=False, seed=20260912):
        self.paths = [Path(item) for item in (path if isinstance(path, (list, tuple)) else [path])]
        self.tokenizer_path = str(tokenizer_path)
        self.context_length = context_length
        self.repeat = repeat
        self.shuffle = shuffle
        self.seed = seed

    def _records(self, epoch):
        if not self.shuffle or len(self.paths) == 1:
            for path in self.paths:
                with path.open() as stream:
                    yield from stream
            return
        streams = [path.open() for path in self.paths]
        remaining = []
        try:
            for path in self.paths:
                manifest = json.loads((path.parent / "manifest.json").read_text())
                key = "train_examples" if path.name.startswith("train") else "validation_examples"
                remaining.append(int(manifest[key]))
            rng = random.Random(self.seed + epoch)
            while sum(remaining):
                pick = rng.randrange(sum(remaining))
                cumulative = 0
                for index, count in enumerate(remaining):
                    cumulative += count
                    if pick < cumulative:
                        line = next(streams[index])
                        remaining[index] -= 1
                        yield line
                        break
        finally:
            for stream in streams:
                stream.close()

    def __iter__(self):
        tokenizer = Tokenizer.from_file(self.tokenizer_path)
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker else 0
        workers = worker.num_workers if worker else 1
        token_buffer, mask_buffer = [], []
        epoch = 0
        while True:
            for line_no, line in enumerate(self._records(epoch)):
                if line_no % workers != worker_id:
                    continue
                record = json.loads(line)
                ids, masks = encode_messages(record["messages"], tokenizer)
                token_buffer.extend(ids)
                mask_buffer.extend(masks)
                required = self.context_length + 1
                while len(token_buffer) >= required:
                    window = token_buffer[:required]
                    supervised = mask_buffer[1:required]
                    labels = [token if keep else IGNORE_INDEX
                              for token, keep in zip(window[1:], supervised)]
                    if any(supervised):
                        yield (torch.tensor(window[:-1], dtype=torch.long),
                               torch.tensor(labels, dtype=torch.long))
                    del token_buffer[:self.context_length]
                    del mask_buffer[:self.context_length]
            if not self.repeat:
                # Preserve the final assistant span by padding the stream tail.
                if len(token_buffer) >= 2 and any(mask_buffer[1:]):
                    pad = tokenizer.token_to_id("<|pad|>")
                    required = self.context_length + 1
                    missing = required - len(token_buffer)
                    token_buffer.extend([pad] * missing)
                    mask_buffer.extend([False] * missing)
                    supervised = mask_buffer[1:required]
                    labels = [token if keep else IGNORE_INDEX
                              for token, keep in zip(token_buffer[1:required], supervised)]
                    yield (torch.tensor(token_buffer[:self.context_length], dtype=torch.long),
                           torch.tensor(labels, dtype=torch.long))
                break
            epoch += 1


def make_model(vocab_size=4099, context_length=2048):
    config = bdh_best.Config(
        hidden_size=512, heads=4, mlp_mult=16, layers=12, recurrences=1,
        vocab_size=vocab_size, context_length=context_length, dropout=0.1,
        mixer_mode="swiglu", swiglu_in_loop=False, mixer_width=2560,
    )
    return RoPEGroupGateSignedExp(config, groups=8, learned_absolute=False)


def load_pretrained_resized(model, checkpoint):
    old = torch.load(checkpoint, map_location="cpu", weights_only=True)
    current = model.state_dict()
    for key, value in old.items():
        if key not in current:
            raise KeyError(f"unexpected pretrained key: {key}")
        if current[key].shape == value.shape:
            current[key] = value
        elif key in ("token_embedding.weight", "output.weight"):
            if value.shape[1:] != current[key].shape[1:] or value.shape[0] >= current[key].shape[0]:
                raise ValueError(f"cannot resize {key}: {value.shape} -> {current[key].shape}")
            current[key][:value.shape[0]] = value
            # Initialize control tokens near the pretrained vocabulary centroid.
            mean = value.mean(dim=0, keepdim=True)
            noise = torch.randn_like(current[key][value.shape[0]:]) * 0.002
            current[key][value.shape[0]:] = mean + noise
        else:
            raise ValueError(f"shape mismatch for {key}: {value.shape} != {current[key].shape}")
    model.load_state_dict(current)


@torch.inference_mode()
def evaluate(model, loader, device, batches):
    model.eval()
    loss_sum = token_count = 0
    for step, (x, labels) in enumerate(loader):
        if step >= batches:
            break
        x, labels = x.to(device), labels.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            logits = model(x)
            loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(),
                                   ignore_index=IGNORE_INDEX, reduction="sum")
        loss_sum += loss.float().item()
        token_count += (labels != IGNORE_INDEX).sum().item()
    model.train()
    mean = loss_sum / max(token_count, 1)
    return {"loss": mean, "perplexity": math.exp(mean), "assistant_tokens": token_count}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-data", type=Path, nargs="+", default=[DATA_DIR / "train.jsonl"])
    ap.add_argument("--val-data", type=Path, nargs="+", default=[DATA_DIR / "validation.jsonl"])
    ap.add_argument("--tokenizer", type=Path, default=DATA_DIR / "bpe4099_chatml.json")
    ap.add_argument("--pretrained", type=Path,
                    default=HERE / "runs" / "big_formal512_10shard_1ep.best.pt")
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--context-length", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--schedule-steps", type=int, default=0,
                    help="Optimizer updates from start to the cosine endpoint; required for training")
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--shuffle-seed", type=int, default=20260912)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-batches", type=int, default=32)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--tag", default="chatml_2048")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if not args.smoke:
        if args.schedule_steps <= args.warmup_steps:
            ap.error("--schedule-steps must be greater than --warmup-steps")
        if args.max_steps and args.max_steps > args.schedule_steps:
            ap.error("--max-steps cannot exceed the cosine --schedule-steps")

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    assert tokenizer.get_vocab_size() == 4099
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = make_model(tokenizer.get_vocab_size(), args.context_length)
    resume_state = None
    if args.resume:
        resume_state = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(resume_state["model"])
    else:
        load_pretrained_resized(model, args.pretrained)
    model.to(device)
    if args.compile:
        model = torch.compile(model, mode="reduce-overhead")
    print(f"model_params={sum(p.numel() for p in model.parameters()):,} device={device} "
          f"context={args.context_length} vocab={tokenizer.get_vocab_size()}", flush=True)

    train_data = PackedChatDataset(args.train_data, args.tokenizer, args.context_length,
                                   shuffle=True, seed=args.shuffle_seed)
    val_data = PackedChatDataset(args.val_data, args.tokenizer, args.context_length)
    loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                         num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size,
                                             num_workers=args.workers)
    if args.smoke:
        x, labels = next(iter(loader))
        supervised = (labels != IGNORE_INDEX).sum().item()
        assert x.shape == labels.shape == (args.batch_size, args.context_length)
        assert supervised > 0
        with torch.inference_mode():
            logits = model(x.to(device))
        print(f"smoke_ok shape={tuple(logits.shape)} supervised_tokens={supervised}", flush=True)
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    def lr_factor(step):
        # LambdaLR calls this once at construction with step=0, then after each
        # optimizer update. The final scheduled update therefore lands exactly
        # on min_lr_ratio rather than silently continuing a stale warmup LR.
        if step < args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        progress = min(1.0, (step - args.warmup_steps)
                       / max(args.schedule_steps - args.warmup_steps, 1))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    start_update = 0
    start_epoch = 0
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        start_update = resume_state["update"]
        start_epoch = resume_state["epoch"]
        print(f"resumed update={start_update} epoch={start_epoch + 1}", flush=True)
    optimizer.zero_grad(set_to_none=True)
    output = HERE / "runs" / f"sft_{args.tag}.ckpt.pt"
    best_output = HERE / "runs" / f"sft_{args.tag}.best.pt"
    update = start_update
    micro = 0
    skip_micro = start_update * args.grad_accum
    supervised_tokens = 0
    best_loss = (resume_state.get("best_loss", float("inf"))
                 if resume_state is not None else float("inf"))
    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs):
        for x, labels in loader:
            if micro < skip_micro:
                micro += 1
                continue
            x, labels = x.to(device), labels.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                logits = model(x)
                loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(),
                                       ignore_index=IGNORE_INDEX) / args.grad_accum
            loss.backward()
            supervised_tokens += (labels != IGNORE_INDEX).sum().item()
            micro += 1
            if micro % args.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            update += 1
            if update == 1 or update % 50 == 0:
                elapsed = time.perf_counter() - started
                print(f"update={update} epoch={epoch + 1} loss={loss.item() * args.grad_accum:.4f} "
                      f"lr={optimizer.param_groups[0]['lr']:.3e} supervised_tok/s="
                      f"{supervised_tokens / max(elapsed, 1e-9):,.0f}", flush=True)
            do_eval = args.eval_every and update % args.eval_every == 0
            do_save = args.save_every and update % args.save_every == 0
            metrics = evaluate(model, val_loader, device, args.eval_batches) if do_eval else None
            if do_eval:
                print(f"eval update={update} {metrics}", flush=True)
            if do_save or do_eval:
                is_best = do_eval and metrics["loss"] < best_loss
                if is_best:
                    best_loss = metrics["loss"]
                state = {k.removeprefix("_orig_mod."): v.cpu() for k, v in model.state_dict().items()}
                torch.save({"model": state, "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(), "update": update,
                            "epoch": epoch, "metrics": metrics, "best_loss": best_loss,
                            "args": vars(args)}, output)
                if is_best:
                    torch.save(state, best_output)
            if args.max_steps and update >= args.max_steps:
                break
        if args.max_steps and update >= args.max_steps:
            break
    state = {k.removeprefix("_orig_mod."): v.cpu() for k, v in model.state_dict().items()}
    torch.save({"model": state, "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "update": update,
                "epoch": max(args.epochs - 1, 0), "metrics": None,
                "best_loss": best_loss, "args": vars(args)}, output)
    torch.save(state, HERE / "runs" / f"sft_{args.tag}.pt")
    print(f"complete updates={update} supervised_tokens={supervised_tokens:,}", flush=True)


if __name__ == "__main__":
    main()
