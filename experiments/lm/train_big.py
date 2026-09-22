"""大数据集训练：fineweb/dclm 混合语料（byte-level, vocab=256），~10M 参数。

与 bdh_sudoku4/train_mixed_corpus_10m.py 相同的加载/训练流程（bfloat16 autocast,
AdamW 1e-3 cosine, ctx256 bs16），便于直接对照 mixed_bdh_* 的 PPL。

用法:
  python train_big.py --model bdh_best --recurrences 3 --tag r3
  python train_big.py --model bdh_best --recurrences 1 --tag r1
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
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import pyarrow.parquet as pq
from datasets import Dataset

import bdh_best
import gram_norm
import rope_group_gate
import liv
import transformer
import mix_model
import hyena

# Hugging Face cache root; override with HF_HOME (defaults to ~/.cache/huggingface).
HF_HOME = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
DATASET_DIR = (
    HF_HOME / "datasets" / "HuggingFaceFW___finepdfs_edu_50_bt-"
    "dclm_30_bt-fineweb_edu_20_bt-shuffled" / "default-fdc261053abb4d32" / "0.0.0"
    / "8904a95879538b9e7db6cf8636b2cd16b5e86a76"
)
ARROW_PREFIX = "finepdfs_edu_50_bt-dclm_30_bt-fineweb_edu_20_bt-shuffled-train"
PARQUET_DIR = (
    HF_HOME / "hub"
    / "datasets--HuggingFaceFW--finepdfs_edu_50BT-dclm_30BT-fineweb_edu_20BT-shuffled"
    / "snapshots" / "8904a95879538b9e7db6cf8636b2cd16b5e86a76" / "data"
)

VOCAB = 256
MODELS = {
    "bdh_norm": lambda rec: gram_norm.NormDeep(bdh_best.Config(
        hidden_size=232, heads=4, mlp_mult=16, layers=5, recurrences=rec,
        vocab_size=256, context_length=256, dropout=0.1)),
    "ropegate8noabs": lambda rec: rope_group_gate.RoPEGroupGateNoAbs(bdh_best.Config(
        hidden_size=450, heads=4, mlp_mult=16, layers=12, recurrences=rec,
        vocab_size=4096, context_length=256, dropout=0.1,
        mixer_mode="swiglu", swiglu_in_loop=False, mixer_width=2250), groups=8),
    "ropegate8": lambda rec: rope_group_gate.RoPEGroupGate(bdh_best.Config(
        hidden_size=450, heads=4, mlp_mult=16, layers=12, recurrences=rec,
        vocab_size=4096, context_length=256, dropout=0.1,
        mixer_mode="swiglu", swiglu_in_loop=False, mixer_width=2250), groups=8),
    "ropegate8signedexpnoabs": lambda rec: rope_group_gate.RoPEGroupGateSignedExp(
        bdh_best.Config(hidden_size=450, heads=4, mlp_mult=16, layers=12,
        recurrences=rec, vocab_size=4096, context_length=256, dropout=0.1,
        mixer_mode="swiglu", swiglu_in_loop=False, mixer_width=2250),
        groups=8, learned_absolute=False),
    "bdh_best": lambda rec: bdh_best.BDHBest(bdh_best.Config(
        hidden_size=232, heads=4, mlp_mult=16, layers=5, recurrences=rec,
        vocab_size=VOCAB, context_length=256, dropout=0.1)),
    "transformer": lambda rec: transformer.Transformer(transformer.Config(
        hidden_size=400, heads=4, layers=5, vocab_size=VOCAB, context_length=256)),
    "liv": lambda rec: liv.LIV(liv.Config(
        hidden_size=350, intermediate_mult=4, layers=5, vocab_size=VOCAB,
        context_length=256)),
    "hyena": lambda rec: hyena.Hyena(hyena.Config(
        hidden_size=400, layers=5, order=2, filter_order=64, ffn_mult=4,
        vocab_size=VOCAB, context_length=256)),
    "mix": lambda rec: mix_model.MixModel(mix_model.MixConfig(
        hidden_size=232, layers=6, pattern=("T", "T", "B", "T", "T", "B"),
        recurrences=rec, vocab_size=VOCAB, context_length=256)),
}


def load_shards(indices):
    return [Dataset.from_file(str(DATASET_DIR / f"{ARROW_PREFIX}-{i:05d}-of-00008.arrow"))
            for i in indices]


def sample_batch(shards, batch_size, context_length, rng, device, tok=None):
    windows = []
    required = context_length + 1
    while len(windows) < batch_size:
        shard = shards[rng.randrange(len(shards))]
        text_raw = shard[rng.randrange(len(shard))]["text"]
        if tok is None:
            text = text_raw.encode("utf-8")
            seq = list(text)
        else:
            seq = tok.encode(text_raw).ids
        if len(seq) < required:
            continue
        start = rng.randrange(len(seq) - required + 1)
        windows.append(torch.tensor(seq[start:start + required], dtype=torch.long))
    batch = torch.stack(windows).to(device)
    return batch[:, :-1], batch[:, 1:]


class TrainDataset(torch.utils.data.IterableDataset):
    """顺序遍历 Parquet，将文档连续打包为不重复的因果训练窗口。"""

    def __init__(self, parquet_paths, context_length, tok=None):
        self.parquet_paths = list(parquet_paths)
        self.ctx = context_length
        self.tok = tok

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        paths = self.parquet_paths if worker is None else self.parquet_paths[worker.id::worker.num_workers]
        buffer = []
        for path in paths:
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=64, columns=["text"]):
                texts = batch.column(0).to_pylist()
                if self.tok is None:
                    encoded = [list(text.encode("utf-8")) for text in texts]
                    separator = [10]
                else:
                    encoded = [encoding.ids for encoding in self.tok.encode_batch(texts)]
                    separator = self.tok.encode("\n").ids
                for seq in encoded:
                    buffer.extend(seq)
                    buffer.extend(separator)
                    while len(buffer) >= self.ctx + 1:
                        window = torch.tensor(buffer[:self.ctx + 1], dtype=torch.long)
                        yield window[:-1], window[1:]
                        del buffer[:self.ctx]


@torch.inference_mode()
def evaluate(model, shards, batch_size, context_length, rng, device, n_batches=32, tok=None):
    model.eval()
    loss_sum, correct, count = 0.0, 0, 0
    for _ in range(n_batches):
        x, y = sample_batch(shards, batch_size, context_length, rng, device, tok=tok)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="sum")
        loss_sum += loss.float().item()
        correct += (logits.argmax(dim=-1) == y).sum().item()
        count += y.numel()
    model.train()
    mean_loss = loss_sum / count
    return {"loss": mean_loss, "perplexity": math.exp(mean_loss),
            "accuracy": correct / count, "tokens": count}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), default="bdh_best")
    ap.add_argument("--recurrences", type=int, default=3)
    ap.add_argument("--steps", type=int, default=0,
                    help="最多训练步数；0 表示完整遍历一轮")
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=10000)
    ap.add_argument("--eval-batches", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--tag", default="")
    ap.add_argument("--swiglu-out", action="store_true",
                    help="bdh_best: dense SwiGLU 移出递归循环（每层一次）")
    ap.add_argument("--mixer-mode", choices=["swiglu", "linear", "add", "had", "merged", "gate"],
                    default="swiglu", help="bdh_best 读出融合方式")
    ap.add_argument("--swiglu-mlp", action="store_true",
                    help="bdh_best: 块级后加 h->2h->h SwiGLU MLP + 残差流")
    ap.add_argument("--ffn-mult", type=int, default=2,
                    help="bdh_best: 块级 FFN 内部宽度倍数 (internal=ffn_mult*d)")
    ap.add_argument("--mixer-width", type=int, default=0,
                    help="bdh_best: GLU 内部宽度 n_mixer (0=用 mlp_mult 逻辑)")
    ap.add_argument("--hidden-size", type=int, default=512,
                    help="隐藏宽度；正式 RoPEGroupGate 配置对齐 H100 为 512")
    ap.add_argument("--tf-hidden", type=int, default=400,
                    help="transformer 隐藏宽度")
    ap.add_argument("--tokenizer", choices=["byte", "bpe"], default="byte",
                    help="分词器：byte(256) 或 bpe(4096, bpe4096.json)")
    ap.add_argument("--layers", type=int, default=12,
                    help="自定义构造的层数（bdh_best / transformer）")
    ap.add_argument("--workers", type=int, default=16,
                    help="DataLoader worker 数（多进程并行 tokenize）")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile 模型（~2× 提速）")
    ap.add_argument("--resume", action="store_true",
                    help="从 runs/big_{tag}.ckpt.pt 恢复")
    ap.add_argument("--params-only", action="store_true")
    ap.add_argument("--source-shards", type=int, default=10,
                    help="顺序使用 train-00000 起的 Parquet shard 数")
    args = ap.parse_args()

    global VOCAB
    tok = None
    if args.tokenizer == "bpe":
        from tokenizers import Tokenizer as _T
        tok = _T.from_file(str(HERE / "bpe4096.json"))
        VOCAB = 4096

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    train_shards = load_shards(range(7))
    val_shards = load_shards([7])
    rng = random.Random(args.seed)

    model = MODELS[args.model](args.recurrences).to(device)
    if args.tf_hidden != 400 and args.model == "transformer":
        model = transformer.Transformer(transformer.Config(
            hidden_size=args.tf_hidden, heads=4, layers=args.layers, vocab_size=VOCAB,
            context_length=256)).to(device)
    if args.swiglu_out and args.model == "bdh_best":
        model = bdh_best.BDHBest(bdh_best.Config(
            hidden_size=232, heads=4, mlp_mult=16, layers=args.layers, recurrences=args.recurrences,
            vocab_size=VOCAB, context_length=256, dropout=0.1, swiglu_in_loop=False)).to(device)
    if args.model in ("bdh_best", "bdh_norm") and (args.hidden_size != 232 or args.mixer_mode != "swiglu"
                                     or args.swiglu_mlp or args.mixer_width > 0):
        cls = gram_norm.NormDeep if args.model == "bdh_norm" else bdh_best.BDHBest
        model = cls(bdh_best.Config(
            hidden_size=args.hidden_size, heads=4, mlp_mult=16, layers=args.layers, recurrences=args.recurrences,
            vocab_size=VOCAB, context_length=256, dropout=0.1,
            mixer_mode=args.mixer_mode, swiglu_mlp=args.swiglu_mlp, ffn_mult=args.ffn_mult,
            mixer_width=args.mixer_width)).to(device)
    if args.model in ("ropegate8", "ropegate8noabs", "ropegate8signedexpnoabs"):
        config = bdh_best.Config(
            hidden_size=args.hidden_size, heads=4, mlp_mult=16, layers=args.layers,
            recurrences=args.recurrences, vocab_size=VOCAB, context_length=args.block_size,
            dropout=0.1, mixer_mode="swiglu", swiglu_in_loop=False,
            mixer_width=5 * args.hidden_size)
        if args.model == "ropegate8signedexpnoabs":
            model = rope_group_gate.RoPEGroupGateSignedExp(
                config, groups=8, learned_absolute=False).to(device)
        else:
            cls = (rope_group_gate.RoPEGroupGateNoAbs
                   if args.model == "ropegate8noabs" else rope_group_gate.RoPEGroupGate)
            model = cls(config, groups=8).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if args.compile:
        model = torch.compile(model, mode="reduce-overhead")
    print(f"[{args.model}] recurrences={args.recurrences} params={n_params:,} "
          f"device={torch.cuda.get_device_name() if device=='cuda' else 'cpu'}", flush=True)
    if args.params_only:
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    # 完整十 shard 标称约 10B token；steps=0 时据此设置一轮 cosine 长度。
    schedule_steps = args.steps or math.ceil(10_000_000_000 / (args.batch_size * args.block_size))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=schedule_steps)

    best = None
    start_step = 0
    ckpt_path = HERE / "runs" / f"big_{args.tag or (args.model + '_r' + str(args.recurrences))}.ckpt.pt"
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ck["model"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        optimizer.load_state_dict(ck["optimizer"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=schedule_steps)
        scheduler.load_state_dict(ck["scheduler"])
        start_step = ck["step"]
        best = ck["best"]
        print(f"恢复 checkpoint: step={start_step} best={best['loss'] if best else None}", flush=True)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=schedule_steps)

    t0 = time.time()
    parquet_paths = [PARQUET_DIR / f"train-{i:05d}-of-00100.parquet"
                     for i in range(args.source_shards)]
    missing = [str(path) for path in parquet_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"缺少 Parquet shard: {missing}")
    loader = torch.utils.data.DataLoader(
        TrainDataset(parquet_paths, args.block_size, tok=tok),
        batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=4, drop_last=True)
    train_started = time.perf_counter()
    completed_steps = start_step
    for step, (x, y) in enumerate(loader, 1):
        if step <= start_step:
            continue
        if args.steps and step > args.steps:
            break
        x = x.to(device)
        y = y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        completed_steps = step

        if args.eval_every > 0 and (step % args.eval_every == 0 or step == 1):
            m = evaluate(model, val_shards, args.batch_size, args.block_size, rng,
                         device, n_batches=args.eval_batches, tok=tok)
            is_best = best is None or m["loss"] < best["loss"]
            if is_best:
                best = m
            sd = model.state_dict()
            sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
            torch.save({
                "model": sd,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
                "best": best,
            }, ckpt_path)
            if is_best:
                torch.save(sd, HERE / "runs" / f"big_{args.tag or (args.model + '_r' + str(args.recurrences))}.best.pt")
            tok_s = args.batch_size * args.block_size / max(time.time() - t0, 1e-9)
            t0 = time.time()
            print(f"step={step}/{args.steps} train={loss.item():.4f} "
                  f"eval={m['loss']:.4f} ppl={m['perplexity']:.4f} "
                  f"acc={m['accuracy']:.4f} best={best['loss']:.4f} "
                  f"{tok_s * args.eval_every:,.0f} tok/s", flush=True)

    if device == "cuda":
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - train_started
    trained_tokens = max(0, completed_steps - start_step) * args.batch_size * args.block_size
    result = {
        "model": args.model,
        "recurrences": args.recurrences,
        "parameters": n_params,
        "steps": completed_steps,
        "training_tokens": trained_tokens,
        "training_seconds": training_seconds,
        "tokens_per_second": trained_tokens / max(training_seconds, 1e-9),
        "peak_cuda_memory_gib": (torch.cuda.max_memory_allocated() / 2**30
                                 if device == "cuda" else None),
        "best_metrics": ({k: best[k] for k in ("loss", "perplexity", "accuracy", "tokens")}
                         if best is not None else None),
        "elapsed_minutes": training_seconds / 60,
    }
    tag = args.tag or f"{args.model}_r{args.recurrences}"
    out = HERE / "runs" / f"big_{tag}.json"
    out.write_text(json.dumps(result, indent=2))
    sd = model.state_dict()
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    torch.save(sd, HERE / "runs" / f"big_{tag}.pt")
    if best is not None:
        print(f"\n[{args.model} r{args.recurrences}] best_loss={best['loss']:.4f} "
              f"ppl={best['perplexity']:.4f} acc={best['accuracy']:.4f}", flush=True)
    print(f"trained_steps={completed_steps - start_step} tokens={trained_tokens:,} "
          f"seconds={training_seconds:.3f} tok/s={result['tokens_per_second']:,.0f} "
          f"peak_cuda_gib={result['peak_cuda_memory_gib']}", flush=True)


if __name__ == "__main__":
    main()
