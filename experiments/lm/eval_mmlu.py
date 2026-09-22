"""MMLU 四选项 log-likelihood 零样本评测。

score_i = Σ log P(option_i | 题干) / len(option_i tokens)   （长度归一化）
pred = argmax_i score_i。与 gold 比较得 acc。随机基线 25%。
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
import glob
import os
import json
from pathlib import Path

import pandas as pd
import torch

# Hugging Face cache root; override with HF_HOME (defaults to ~/.cache/huggingface).
HF_HOME = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
SNAP = glob.glob(str(HF_HOME / "hub" / "datasets--cais--mmlu" / "snapshots" / "*"))[0]
LETTERS = ["A", "B", "C", "D"]


def build_ctx(row):
    q = row["question"].strip()
    choices = row["choices"]
    return (f"The following are multiple choice questions about {row['subject'].replace('_',' ')}.\n\n"
            f"{q}\n"
            + "\n".join(f"{L}. {c}" for L, c in zip(LETTERS, choices))
            + "\nAnswer:")


@torch.no_grad()
def score_options(model, tok, ctx, options, device):
    all_ids = tok.encode(ctx).ids
    out = []
    for opt in options:
        opt_ids = tok.encode(" " + opt.strip()).ids
        cap = max(256 - len(opt_ids) - 1, 16)
        ctx_ids = all_ids[-cap:]
        ids = torch.tensor([ctx_ids + opt_ids], device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ids[:, :-1]).float()
        logprobs = torch.log_softmax(logits, -1)
        tgt = ids[0, len(ctx_ids):].clamp(0, 4095)
        lp = logprobs[0, len(ctx_ids) - 1:, :].gather(1, tgt.unsqueeze(1)).sum().item()
        out.append(lp / max(len(opt_ids), 1))
    return out


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--n", type=int, default=500)
    a.add_argument("--arch", default="both")
    args = a.parse_args()
    device = "cuda"
    import bdh_best, gram_norm, transformer
    from tokenizers import Tokenizer as _T
    tok = _T.from_file(str(HERE / "bpe4096.json"))

    rows = []
    for f in sorted(glob.glob(f"{SNAP}/*/test-00000-of-00001.parquet")):
        if "/global/" in f or "global" in f.split("/")[-2]:
            continue
        df = pd.read_parquet(f)
        rows.append(df.head(15))
    df = pd.concat(rows).sample(n=min(args.n, 20000), random_state=0).reset_index(drop=True)
    print(f"MMLU 题数: {len(df)}")

    models = {}
    if args.arch in ("both", "cut"):
        sd = torch.load("runs/big_night2_norm.best.pt", map_location=device)
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
        m = gram_norm.NormDeep(bdh_best.Config(hidden_size=450, heads=4, mlp_mult=16, layers=12,
            recurrences=1, vocab_size=4096, context_length=256, dropout=0.0,
            mixer_mode="swiglu", mixer_width=2250)).to(device)
        m.load_state_dict(sd); m.eval()
        models["cuttlefish"] = m
    if args.arch in ("both", "tf"):
        import glob as g
        p = sorted(g.glob("runs/big_night_tf2.best.pt"), key=os.path.getmtime)[-1]
        sd = torch.load(p, map_location=device)
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
        m = transformer.Transformer(transformer.Config(hidden_size=832, heads=4, layers=12,
            vocab_size=4096, context_length=256)).to(device)
        m.load_state_dict(sd); m.eval()
        models["transformer"] = m

    results = {k: {"correct": 0, "pred_dist": [0, 0, 0, 0]} for k in models}
    skipped = 0
    for _, row in df.iterrows():
        ctx = build_ctx(row)
        if len(tok.encode(ctx)) < 20 or any(len(str(c).strip()) == 0 for c in row["choices"]):
            skipped += 1
            continue
        gold = int(row["answer"]) if not isinstance(row["answer"], str) else LETTERS.index(row["answer"])
        for name, m in models.items():
            scores = score_options(m, tok, ctx, row["choices"], device)
            if any(x != x for x in scores):
                skipped += 1
                continue
            pred = max(range(4), key=lambda i: scores[i])
            results[name]["correct"] += int(pred == gold)
            results[name]["pred_dist"][pred] += 1
    out = {}
    for name, r in results.items():
        acc = r["correct"] / max(len(df) - skipped, 1)
        out[name] = {"acc": round(acc, 4), "n": len(df),
                     "pred_dist": [d / len(df) for d in r["pred_dist"]]}
        print(f"[skipped {skipped}] {name}: acc={acc:.4f} (随机基线 0.25)  选项分布 A/B/C/D = "
              f"{[round(x,2) for x in out[name]['pred_dist']]}")
    Path("runs/mmlu_probe.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
