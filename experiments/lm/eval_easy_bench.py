"""简单 bench 四件套：PIQA(2) / HellaSwag(4) / Winogrande(2) / ARC-Easy(多)。

统一 LF 评分：score_i = Σ log P(option_i tokens | 上下文) / len(option_i)。
位置嵌入上限 256：ctx 动态截断。基线：PIQA/Winogrande 50%，HellaSwag/ARC 25%。
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
import glob
import json
import argparse
import os
from pathlib import Path

import pandas as pd
import torch

# Hugging Face cache root; override with HF_HOME (defaults to ~/.cache/huggingface).
HF_HOME = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
ROOT = str(HF_HOME / "hub")
CTX_CAP = 256


def lf_batch(models, tok, ctx, options, device):
    all_ids = tok.encode(ctx).ids
    scores_all = {k: [] for k in models}
    for opt in options:
        opt_ids = tok.encode(" " + str(opt).strip()).ids
        cap = max(CTX_CAP - len(opt_ids) - 1, 16)
        ctx_ids = all_ids[-cap:]
        ids = torch.tensor([ctx_ids + opt_ids], device=device)
        for name, m in models.items():
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits = m(ids[:, :-1]).float()
            logprobs = torch.log_softmax(logits, -1)
            tgt = ids[0, len(ctx_ids):].clamp(0, 4095)
            lp = logprobs[0, len(ctx_ids) - 1:, :].gather(1, tgt.unsqueeze(1)).sum().item()
            scores_all[name].append(lp / max(len(opt_ids), 1))
    return {k: max(range(len(v)), key=lambda i: v[i]) for k, v in scores_all.items()}


def load_piqa(n):
    f = glob.glob(f"{ROOT}/datasets--baber--piqa/snapshots/*/piqa_validation.parquet")[0]
    df = pd.read_parquet(f).sample(n=min(n, 1838), random_state=0)
    items = []
    for _, r in df.iterrows():
        items.append({"ctx": f"Question: {r['goal']}\nAnswer:",
                      "options": [r["sol1"], r["sol2"]], "gold": int(r["label"])})
    return items


def load_hellaswag(n):
    f = glob.glob(f"{ROOT}/datasets--Rowan--hellaswag/snapshots/*/data/validation-*.parquet")[0]
    df = pd.read_parquet(f).sample(n=min(n, 10042), random_state=0)
    items = []
    for _, r in df.iterrows():
        items.append({"ctx": r["ctx"].strip(),
                      "options": list(r["endings"]), "gold": int(r["label"])})
    return items


def load_winogrande(n):
    f = glob.glob(f"{ROOT}/datasets--allenai--winogrande/snapshots/*/winogrande_xl/validation-*.parquet")[0]
    df = pd.read_parquet(f).sample(n=min(n, 1267), random_state=0)
    items = []
    for _, r in df.iterrows():
        s = r["sentence"]
        pre, _, post = s.partition("_")
        items.append({"ctx": pre.strip(),
                      "options": [f"{r['option1']} {post}", f"{r['option2']} {post}"],
                      "gold": int(r["answer"]) - 1})
    return items


def load_arc(n):
    files = glob.glob(f"{ROOT}/datasets--allenai--ai2_arc/snapshots/*/ARC-Easy/test-*.parquet")
    df = pd.read_parquet(files[0])
    df = df.sample(n=min(n, len(df)), random_state=0)
    items = []
    for _, r in df.iterrows():
        labels = list(r["choices"]["label"])
        texts = list(r["choices"]["text"])
        if r["answerKey"] not in labels or len(labels) != 4:
            continue
        items.append({"ctx": f"Question: {r['question']}\nAnswer:",
                      "options": texts, "gold": labels.index(r["answerKey"])})
    return items


def main():
    import os
    from tokenizers import Tokenizer as _T
    import bdh_best, gram_norm, transformer, rope_group_gate
    ap = argparse.ArgumentParser()
    ap.add_argument("--ropegate8noabs", action="store_true")
    ap.add_argument("--signedexpnoabs", action="store_true")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--hidden-size", type=int, default=450)
    ap.add_argument("--mixer-width", type=int, default=2250)
    ap.add_argument("--layers", type=int, default=12)
    args = ap.parse_args()
    device = "cuda"
    here = HERE
    tok = _T.from_file(str(here / "bpe4096.json"))
    models = {}
    checkpoint = (args.checkpoint if args.checkpoint is not None else
                  here / "runs" / "big_signedexpnoabs_r1_full.best.pt"
                  if args.signedexpnoabs else
                  here / "runs" / "big_ropegate8noabs_r1_full.best.pt"
                  if args.ropegate8noabs else here / "runs" / "big_night2_norm.best.pt")
    sd = torch.load(checkpoint, map_location=device)
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    config = bdh_best.Config(hidden_size=args.hidden_size, heads=4, mlp_mult=16,
        layers=args.layers,
        recurrences=1, vocab_size=4096, context_length=256, dropout=0.0,
        mixer_mode="swiglu", swiglu_in_loop=False, mixer_width=args.mixer_width)
    m = (rope_group_gate.RoPEGroupGateSignedExp(
            config, groups=8, learned_absolute=False)
         if args.signedexpnoabs else
         rope_group_gate.RoPEGroupGateNoAbs(config, groups=8)
         if args.ropegate8noabs else gram_norm.NormDeep(config)).to(device)
    m.load_state_dict(sd); m.eval()
    model_name = ("signedexpnoabs" if args.signedexpnoabs else
                  "ropegate8noabs" if args.ropegate8noabs else "cuttlefish")
    models[model_name] = m
    p = sorted(glob.glob(str(here / "runs" / "big_night_tf2.best.pt")), key=os.path.getmtime)[-1]
    sd = torch.load(p, map_location=device)
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    m = transformer.Transformer(transformer.Config(hidden_size=832, heads=4, layers=12,
        vocab_size=4096, context_length=256)).to(device)
    m.load_state_dict(sd); m.eval(); models["transformer"] = m

    N = 300
    out = {}
    for name, loader, base in [("PIQA", load_piqa, 0.5), ("HellaSwag", load_hellaswag, 0.25),
                               ("Winogrande", load_winogrande, 0.5), ("ARC-Easy", load_arc, 0.25)]:
        items = loader(N)
        items = [it for it in items
                 if all(0 < len(tok.encode(" " + str(o).strip()).ids) <= 200
                        for o in it["options"])]
        correct = {k: 0 for k in models}
        for it in items:
            preds = lf_batch(models, tok, it["ctx"], it["options"], device)
            for k, pr in preds.items():
                correct[k] += int(pr == it["gold"])
        line = {k: round(correct[k] / len(items), 4) for k in models}
        line["baseline"] = base
        line["n"] = len(items)
        out[name] = line
        print(f"{name:12s} n={len(items)}  {model_name}={line[model_name]:.3f}  "
              f"transformer={line['transformer']:.3f}  (基线 {base})", flush=True)
    output = args.output or Path(here / "runs" / "easy_bench.json")
    output.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
