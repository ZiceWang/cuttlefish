# --- repo path bootstrap: flat sibling imports + runs/data/tokenizer paths ---
import sys as _sys
from pathlib import Path as _Path
_EXP = _Path(__file__).resolve().parents[1]
for _d in [_EXP, *sorted(_p for _p in _EXP.iterdir() if _p.is_dir() and _p.name != "__pycache__")]:
    if str(_d) not in _sys.path:
        _sys.path.insert(0, str(_d))
HERE = _EXP.parent
# ---------------------------------------------------------------------------
import sys
from pathlib import Path
import torch
from tokenizers import Tokenizer
from compare_continue_samples import load

raw_tests=[
 ('json','{"operation":"add","left":2,"right":3,"result":'),
 ('rpn','RPN evaluation: 2 3 + =>'),
 ('arrow','2 | ADD | 3  →'),
 ('table','left=2; operator=ADD; right=3; output='),
 ('fill_blank','2 plus 3 makes [____]'),
 ('words','two added to three equals'),
 ('roman','II + III ='),
 ('function','add(2, 3) ->'),
]
tests=[]
for name, raw in raw_tests:
    tests.append((name, f"Solve the following arithmetic problem. Interpret the notation and calculate the result.\nExpression: {raw}\nAnswer:"))
tok=Tokenizer.from_file(str(HERE/'bpe4096.json'));model=load(HERE/'runs/arithmetic_expr_poly_20k/step_10000.pt','cuda')
for name,prompt in tests:
 ids=tok.encode(prompt,add_special_tokens=False).ids;new=[]
 with torch.inference_mode():
  for _ in range(32):
   x=torch.tensor([ids[-512:]],device='cuda')
   with torch.autocast('cuda',dtype=torch.bfloat16):n=model(x)[0,-1].argmax().item()
   ids.append(n);new.append(n)
   if '\n\n' in tok.decode(new):break
 print(f'{name}\nPROMPT: {prompt}\nOUTPUT: {tok.decode(new)!r}\n',flush=True)
