"""Sudoku9 四架构 4096 样本正式评估（用各自 best.pt）。"""
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
from pathlib import Path

import torch

import bdh_best
import transformer
import liv
import hyena
import hyena_fish
import gated_fish
import hyena_rope
import hyena_dilated
import gram_norm
import bdh_best
import transformer_rope
import transformer_cosine
import cut_noemb
import shared_gram
import shared_gram
import train_sudoku9_bdhbest as TC
import train_sudoku9_compare as TS



def build(arch):
    if arch == "tfcos":
        return transformer_cosine.TransformerCosine(transformer.Config(
            hidden_size=168, heads=4, layers=8, vocab_size=TC.VOCAB_SIZE,
            context_length=TC.CTX))
    if arch == "tfrope":
        return transformer_rope.TransformerRoPE(transformer.Config(
            hidden_size=168, heads=4, layers=8, vocab_size=TC.VOCAB_SIZE,
            context_length=TC.CTX))
    if arch == "cutnoemb":
        return cut_noemb.CutNoEmb(bdh_best.Config(
            hidden_size=92, heads=4, mlp_mult=16, layers=8, recurrences=1,
            vocab_size=TC.VOCAB_SIZE, context_length=TC.CTX, dropout=0.0,
            mixer_mode="swiglu", swiglu_in_loop=False, mixer_width=460))
    if arch == "pat112norm":
        return shared_gram.SharedGramModel(shared_gram.Config(
            hidden_size=120, layers=6, group_pattern=(0, 0, 1, 0, 0, 1),
            norm_attn=True, posify=False, vocab_size=TC.VOCAB_SIZE,
            context_length=TC.CTX))
    if arch == "deepnorm":
        return gram_norm.NormDeep(bdh_best.Config(
            hidden_size=92, heads=4, mlp_mult=16, layers=8, recurrences=1,
            vocab_size=TC.VOCAB_SIZE, context_length=TC.CTX, dropout=0.0,
            mixer_mode="swiglu", swiglu_in_loop=False, mixer_width=460), posify=False)
    if arch == "deepnormpos":
        return gram_norm.NormDeep(bdh_best.Config(
            hidden_size=92, heads=4, mlp_mult=16, layers=8, recurrences=1,
            vocab_size=TC.VOCAB_SIZE, context_length=TC.CTX, dropout=0.0,
            mixer_mode="swiglu", swiglu_in_loop=False, mixer_width=460), posify=True)
    if arch == "cutpat112":
        return shared_gram.SharedGramModel(shared_gram.Config(
            hidden_size=120, layers=6, group_pattern=(0, 0, 1, 0, 0, 1),
            vocab_size=TC.VOCAB_SIZE, context_length=TC.CTX))
    if arch == "cutshared4":
        return shared_gram.SharedGramModel(shared_gram.Config(
            hidden_size=108, layers=8, n_groups=4, vocab_size=TC.VOCAB_SIZE,
            context_length=TC.CTX))
    if arch == "cutshared2":
        return shared_gram.SharedGramModel(shared_gram.Config(
            hidden_size=108, layers=8, n_groups=2, vocab_size=TC.VOCAB_SIZE,
            context_length=TC.CTX))
    if arch == "cutshared":
        return shared_gram.SharedGramModel(shared_gram.Config(
            hidden_size=112, layers=8, vocab_size=TC.VOCAB_SIZE,
            context_length=TC.CTX))
    if arch == "cutdeep":
        return bdh_best.BDHBest(bdh_best.Config(
            hidden_size=92, heads=4, mlp_mult=16, layers=8, recurrences=1,
            vocab_size=TC.VOCAB_SIZE, context_length=TC.CTX, dropout=0.0,
            mixer_mode="swiglu", swiglu_in_loop=False, mixer_width=460))
    if arch == "cuttr1":
        return bdh_best.BDHBest(bdh_best.Config(
            hidden_size=128, heads=4, mlp_mult=16, layers=4, recurrences=1,
            vocab_size=TC.VOCAB_SIZE, context_length=TC.CTX, dropout=0.0,
            mixer_mode="swiglu", swiglu_in_loop=False, mixer_width=640))
    if arch == "hyenadil":
        return hyena_dilated.HyenaDilated(hyena.Config(
            hidden_size=208, layers=5, order=2, filter_order=64,
            ffn_mult=4, vocab_size=TC.VOCAB_SIZE, context_length=TC.CTX))
    if arch == "hyenarope":
        return hyena_rope.HyenaRoPE(hyena.Config(
            hidden_size=208, layers=5, order=2, filter_order=64,
            ffn_mult=4, vocab_size=TC.VOCAB_SIZE, context_length=TC.CTX))
    if arch == "gatedfish":
        return gated_fish.GatedFish(gated_fish.Config(
            hidden_size=118, layers=4, recurrences=2, vocab_size=TC.VOCAB_SIZE,
            context_length=TC.CTX))
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


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--archs", default="bdhbest,transformer,liv,hyena")
    a.add_argument("--examples", type=int, default=4096)
    a.add_argument("--seed", type=int, default=999)
    a.add_argument("--min-clues", type=int, default=None, help="更难题：限制最少线索数")
    a.add_argument("--max-clues", type=int, default=None, help="更难题：限制最多线索数")
    args = a.parse_args()
    device = "cuda"
    rng = random.Random(args.seed)
    report = {}
    for arch in args.archs.split(","):
        model = build(arch).to(device)
        pt = HERE / "runs" / f"sudoku9_{arch}.best.pt"
        if not pt.exists():
            print(f"{arch}: 无 {pt}", flush=True)
            continue
        sd = torch.load(pt, map_location=device)
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd)
        model.eval()
        kw = {}
        if args.min_clues is not None:
            kw["min_clues"] = args.min_clues
        if args.max_clues is not None:
            kw["max_clues"] = args.max_clues
        pred_all, sol_all = [], []
        for i in range(0, args.examples, 128):
            batch = [TC.make_example(rng, **kw) for _ in range(128)]
            prompts = torch.tensor([p + [TC.SEP] for p, _, _ in batch],
                                   dtype=torch.long, device=device)
            solutions = torch.tensor([s for _, s, _ in batch], dtype=torch.long, device=device)
            preds = TC.generate_solution(model, prompts)
            pred_all.append(preds)
            sol_all.append(solutions)
        pred = torch.cat(pred_all)
        sol = torch.cat(sol_all)
        tok = (pred == sol).float().mean().item()
        exact = (pred == sol).all(dim=1).float().mean().item()
        report[arch] = {"token_acc": round(tok, 4), "exact_acc": round(exact, 4),
                        "examples": args.examples}
        print(f"{arch}: token_acc={tok:.4f} exact_acc={exact:.4f} ({args.examples} 样本)",
              flush=True)
    out = HERE / "runs" / "sudoku9_official.json"
    existing = json.loads(out.read_text()) if out.exists() else {}
    key = f"clues_{args.min_clues or 'any'}_{args.max_clues or 'any'}"
    existing[key] = report
    out.write_text(json.dumps(existing, indent=2))
    print(f"已保存 {out}")


if __name__ == "__main__":
    main()
