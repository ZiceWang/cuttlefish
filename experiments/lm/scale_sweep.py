"""宽深兼顾 sweep：同参下变 L/d，bf16+compile 测训练吞吐。"""
# --- repo path bootstrap: flat sibling imports + runs/data/tokenizer paths ---
import sys as _sys
from pathlib import Path as _Path
_EXP = _Path(__file__).resolve().parents[1]
for _d in [_EXP, *sorted(_p for _p in _EXP.iterdir() if _p.is_dir() and _p.name != "__pycache__")]:
    if str(_d) not in _sys.path:
        _sys.path.insert(0, str(_d))
HERE = _EXP.parent
# ---------------------------------------------------------------------------
import time, math
import torch
from torch.nn import functional as F
import bdh_best
import transformer

BS, CTX, VOCAB = 16, 256, 256
HOURS = 10.0


def build_bdh(target_m, L):
    d = int(math.sqrt((target_m * 1e6) / (L * 42)))
    c = bdh_best.Config(hidden_size=d, heads=4, mlp_mult=16, layers=L, recurrences=2,
                        vocab_size=VOCAB, context_length=CTX, mixer_mode="swiglu",
                        swiglu_in_loop=False, mixer_width=5 * d)
    m = bdh_best.BDHBest(c)
    p = sum(v.numel() for v in m.parameters())
    return m, p, d


def build_tf(target_m, L):
    d = int(math.sqrt((target_m * 1e6) / (L * 12)))
    d = (d // 4) * 4
    t = transformer.Transformer(transformer.Config(hidden_size=d, heads=4, layers=L,
                                                   vocab_size=VOCAB, context_length=CTX))
    p = sum(v.numel() for v in t.parameters())
    return t, p, d


def bench(name, build_fn, tm, L, n=12):
    torch.manual_seed(0)
    model, p, d = build_fn(tm, L)
    m = model.bfloat16().cuda()
    x = torch.randint(0, 256, (BS, CTX)).cuda()
    y = x.clone()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = m(x)
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        opt.zero_grad(); loss.backward(); opt.step()

    mc = torch.compile(m, mode="reduce-overhead")
    for _ in range(4): step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n): step()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / n
    tok = BS * CTX / dt
    night = tok * HOURS * 3600
    print(f"{name:22} L={L:>2} d={d:>4}  {p/1e6:7.2f}M  {tok/1000:7.1f}k tok/s  一晚 {night/1e9:5.2f}B")
    del m, mc, opt, model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    for L in [5, 8, 12, 16]:
        bench("BDH 100M", build_bdh, 100, L)
    print()
    for L in [8, 12, 16, 24]:
        bench("TF  100M", build_tf, 100, L)
