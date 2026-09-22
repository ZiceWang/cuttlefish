# --- repo path bootstrap: flat sibling imports + runs/data/tokenizer paths ---
import sys as _sys
from pathlib import Path as _Path
_EXP = _Path(__file__).resolve().parents[1]
for _d in [_EXP, *sorted(_p for _p in _EXP.iterdir() if _p.is_dir() and _p.name != "__pycache__")]:
    if str(_d) not in _sys.path:
        _sys.path.insert(0, str(_d))
HERE = _EXP.parent
# ---------------------------------------------------------------------------
import json,random,re,sys
from pathlib import Path
from tokenizers import Tokenizer
from compare_continue_samples import load
from diverse_arithmetic_generator import generate
from train_diverse_arithmetic_qa import answer

model=load(HERE/'runs/arithmetic_expr_poly_20k/step_04000.pt','cuda');tok=Tokenizer.from_file(str(HERE/'bpe4096.json'));rng=random.Random(48121);rows=[]
for _ in range(20):
 ex=generate(rng,'val','poly3',False);out=answer(model,tok,ex.prompt);nums=re.findall(r'-?\d+',out);pred=int(nums[-1]) if nums else None
 rows.append({'prompt':ex.prompt,'gold_answer':ex.answer,'model_output':out,'gold':ex.gold,'pred':pred,'correct':pred==ex.gold})
print(json.dumps(rows,indent=2))
