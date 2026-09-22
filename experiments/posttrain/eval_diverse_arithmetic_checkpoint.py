# --- repo path bootstrap: flat sibling imports + runs/data/tokenizer paths ---
import sys as _sys
from pathlib import Path as _Path
_EXP = _Path(__file__).resolve().parents[1]
for _d in [_EXP, *sorted(_p for _p in _EXP.iterdir() if _p.is_dir() and _p.name != "__pycache__")]:
    if str(_d) not in _sys.path:
        _sys.path.insert(0, str(_d))
HERE = _EXP.parent
# ---------------------------------------------------------------------------
import argparse,json,sys
from pathlib import Path
from tokenizers import Tokenizer

from compare_continue_samples import load
from train_diverse_arithmetic_qa import KINDS,evaluate

ap=argparse.ArgumentParser();ap.add_argument("--checkpoint",type=Path,required=True);ap.add_argument("--per-kind",type=int,default=20);ap.add_argument("--output",type=Path);args=ap.parse_args()
tok=Tokenizer.from_file(str(HERE/"bpe4096.json"));model=load(args.checkpoint,"cuda")
scores,examples=evaluate(model,tok,args.per_kind)
result={"checkpoint":str(args.checkpoint),"scores":scores,"seen_macro":sum(v for k,v in scores.items() if k.startswith('seen_'))/len(KINDS),"unseen_macro":sum(v for k,v in scores.items() if k.startswith('unseen_'))/len(KINDS),"examples":examples}
print(json.dumps(result,indent=2));
if args.output:args.output.write_text(json.dumps(result,indent=2)+"\n")
