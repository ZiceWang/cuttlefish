"""用 sudoku-extreme 真实题（min_difficulty 过滤）评估四架构。

teacher-forcing token_acc + 自回归 exact_acc，跨难度档位。
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
import csv
import json
import os
import random
from pathlib import Path

import torch

import bdh_best
import transformer
import liv
import hyena
import hyena_fish
import train_sudoku9_bdhbest as TC
import train_sudoku9_compare as TS

# Hugging Face cache root; override with HF_HOME (defaults to ~/.cache/huggingface).
HF_HOME = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
EXTREME = str(
    HF_HOME / "hub" / "datasets--sapientinc--sudoku-extreme"
    / "snapshots" / "58942f96baeb572ca3127e2a9e9c70f330783d6b" / "train.csv"
)


def build(arch):
    if arch == "hyenafish":
        return hyena_fish.HyenaFish(hyena_fish.Config(
            hidden_size=148, layers=4, recurrences=2, vocab_size=TC.VOCAB_SIZE,
            context_length=TC.CTX))
    if arch == "bdhbest":
        c = bdh_best.Config(hidden_size=128, heads=4, mlp_mult=16, layers=4,
                            recurrences=2, vocab_size=TC.VOCAB_SIZE, context_length=TC.CTX,
                            dropout=0.0, mixer_mode="swiglu", swiglu_in_loop=False,
                            mixer_width=640)
        return bdh_best.BDHBest(c)
    d = TS.DEFAULT[arch]["hidden"]
    L = TS.DEFAULT[arch]["layers"]
    return TS.build_model(arch, d, L)


def load_extreme(min_difficulty, n, seed):
    random.seed(seed)
    rows = []
    with open(EXTREME) as f:
        r = csv.DictReader(f)
        for row in r:
            if int(row["rating"]) >= min_difficulty:
                q = row["question"].replace(".", "0")
                if len(q) == 81 and len(row["answer"]) == 81:
                    rows.append((q, row["answer"], int(row["rating"])))
                if len(rows) >= n * 4:
                    break
    random.shuffle(rows)
    return rows[:n]


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--archs", default="bdhbest,transformer,liv,hyena")
    a.add_argument("--diffs", default="0,30,100")
    a.add_argument("--n", type=int, default=1000)
    a.add_argument("--n-autoreg", type=int, default=400)
    a.add_argument("--seed", type=int, default=2026)
    args = a.parse_args()
    device = "cuda"
    models = {}
    for arch in args.archs.split(","):
        m = build(arch).to(device)
        pt = HERE / "runs" / f"sudoku9_{arch}.best.pt"
        sd = torch.load(pt, map_location=device)
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        m.load_state_dict(sd)
        m.eval()
        models[arch] = m
    report = {}
    for dmin in [int(x) for x in args.diffs.split(",")]:
        rows = load_extreme(dmin, args.n, args.seed)
        q = torch.tensor([[int(c) for c in r[0]] for r in rows], device=device)
        ans = torch.tensor([[int(c) for c in r[1]] for r in rows], device=device)
        # teacher-forcing token acc（solution 部分）
        seq = torch.cat([q, torch.full((len(rows), 1), TC.SEP, device=device), ans], dim=1)
        tf_acc = {}
        exact = {}
        for arch, m in models.items():
            with torch.no_grad():
                lg = m(seq[:, :-1])
            pred = lg.argmax(-1)
            sol = seq[:, TC.PUZZLE_LEN + 1:]
            pred_sol = pred[:, TC.PUZZLE_LEN:]
            tf = (pred_sol == sol).float().mean().item()
            tf_acc[arch] = round(tf, 4)
            # 自回归 exact（子集）
            if len(rows) <= args.n_autoreg:
                idxs = range(len(rows))
            else:
                rng = random.Random(args.seed + dmin)
                idxs = rng.sample(range(len(rows)), args.n_autoreg)
            prompts = torch.cat([q[idxs], torch.full((len(idxs), 1), TC.SEP, device=device)], dim=1)
            sols = ans[idxs]
            preds = TC.generate_solution(m, prompts)
            exact[arch] = round((preds == sols).all(dim=1).float().mean().item(), 4)
        report[f"difficulty>={dmin} (n={len(rows)})"] = {"teacher_token": tf_acc, "autoreg_exact": exact}
        print(f"difficulty>={dmin} n={len(rows)}:")
        print(f"  teacher_token: {tf_acc}")
        print(f"  autoreg_exact: {exact}", flush=True)
    out = HERE / "runs" / "sudoku9_extreme.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"已保存 {out}")


if __name__ == "__main__":
    main()
