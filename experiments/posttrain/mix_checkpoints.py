"""Linearly merge two architecture-identical model state dictionaries."""
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
from pathlib import Path

import torch


def state(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return obj["model"] if isinstance(obj, dict) and "model" in obj else obj


ap = argparse.ArgumentParser()
ap.add_argument("--a", type=Path, required=True)
ap.add_argument("--b", type=Path, required=True)
ap.add_argument("--b-weight", type=float, default=0.5)
ap.add_argument("--base", type=Path, help="common base for task-vector merge: a + b - base")
ap.add_argument("--output", type=Path, required=True)
args = ap.parse_args()
if not 0 <= args.b_weight <= 1:
    raise ValueError("--b-weight must be in [0, 1]")
a, b = state(args.a), state(args.b)
if a.keys() != b.keys():
    raise ValueError(f"state keys differ: {len(a)} versus {len(b)}")
mixed = {}
base = state(args.base) if args.base else None
if base is not None and base.keys() != a.keys():
    raise ValueError("base state keys differ")
for key in a:
    if a[key].shape != b[key].shape or a[key].dtype != b[key].dtype:
        raise ValueError(f"incompatible tensor {key}")
    if a[key].is_floating_point():
        if base is not None:
            if base[key].shape != a[key].shape or base[key].dtype != a[key].dtype:
                raise ValueError(f"incompatible base tensor {key}")
            mixed[key] = a[key] + b[key] - base[key]
        else:
            mixed[key] = torch.lerp(a[key], b[key], args.b_weight)
    else:
        if not torch.equal(a[key], b[key]):
            raise ValueError(f"non-floating tensor differs: {key}")
        mixed[key] = a[key]
args.output.parent.mkdir(parents=True, exist_ok=True)
torch.save(mixed, args.output)
print(f"saved {args.output} tensors={len(mixed)} mode={'task-vector' if base is not None else 'lerp'}")
