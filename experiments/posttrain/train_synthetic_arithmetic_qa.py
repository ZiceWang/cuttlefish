"""Short continued pretraining on deterministic synthetic arithmetic QA."""
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
import hashlib
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from compare_continue_samples import load


SOURCE = HERE / "runs/ultrafineweb_en_1ep/final.pt"
KINDS = ("add", "sub", "mul", "div", "mul_add", "add_mul", "paren_mul")


def split_of(key):
    return "val" if int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) % 20 == 0 else "train"


def make_example(rng, split, forced_kind=None):
    while True:
        kind = forced_kind or rng.choice(KINDS)
        if kind == "add":
            a, b = rng.randint(0, 999), rng.randint(0, 999); q=f"What is {a} + {b}?"; ans=f"{a} + {b} = {a+b}"
        elif kind == "sub":
            a, b = rng.randint(0, 999), rng.randint(0, 999); a,b=max(a,b),min(a,b); q=f"What is {a} - {b}?"; ans=f"{a} - {b} = {a-b}"
        elif kind == "mul":
            a, b = rng.randint(0, 99), rng.randint(0, 99); q=f"What is {a} x {b}?"; ans=f"{a} x {b} = {a*b}"
        elif kind == "div":
            b, c = rng.randint(1, 99), rng.randint(0, 99); a=b*c; q=f"What is {a} / {b}?"; ans=f"{a} / {b} = {c}"
        elif kind == "mul_add":
            a,b,c=rng.randint(0,30),rng.randint(0,30),rng.randint(0,99); mid=a*b; q=f"What is {a} x {b} + {c}?"; ans=f"{a} x {b} + {c} = {mid} + {c} = {mid+c}"
        elif kind == "add_mul":
            a,b,c=rng.randint(0,99),rng.randint(0,30),rng.randint(0,30); mid=b*c; q=f"What is {a} + {b} x {c}?"; ans=f"{a} + {b} x {c} = {a} + {mid} = {a+mid}"
        else:
            a,b,c=rng.randint(0,30),rng.randint(0,30),rng.randint(0,20); mid=a+b; q=f"What is ({a} + {b}) x {c}?"; ans=f"({a} + {b}) x {c} = {mid} x {c} = {mid*c}"
        key=f"{kind}|{q}"
        if split_of(key) == split:
            return kind, f"Question: {q}\nAnswer: {ans}\n\n", str(eval_answer(kind,a,b,locals().get('c')))


def eval_answer(kind, a, b, c=None):
    if kind == "add": return a+b
    if kind == "sub": return a-b
    if kind == "mul": return a*b
    if kind == "div": return c
    if kind == "mul_add": return a*b+c
    if kind == "add_mul": return a+b*c
    return (a+b)*c


def batch(rng, tok, batch_size, context, split="train"):
    rows=[]
    for _ in range(batch_size):
        ids=[]
        while len(ids) < context+1:
            ids.extend(tok.encode(make_example(rng,split)[1], add_special_tokens=False).ids)
        rows.append(torch.tensor(ids[:context+1]))
    z=torch.stack(rows)
    return z[:,:-1],z[:,1:]


def exact_eval(model, tok, device, per_kind=30):
    model.eval(); out={}; rng=random.Random(88031)
    with torch.inference_mode():
        for kind in KINDS:
            ok=0
            for _ in range(per_kind):
                _, text, gold=make_example(rng,"val",kind)
                prompt=text.split("Answer:")[0]+"Answer:"
                ids=tok.encode(prompt,add_special_tokens=False).ids; new=[]
                for _ in range(32):
                    x=torch.tensor([ids[-512:]],device=device)
                    with torch.autocast("cuda",dtype=torch.bfloat16): nxt=model(x)[0,-1].argmax().item()
                    ids.append(nxt); new.append(nxt)
                    decoded=tok.decode(new)
                    if "\n" in decoded: break
                nums=re.findall(r"-?\d+",decoded)
                ok += bool(nums and nums[-1] == gold)
            out[kind]=ok/per_kind
    model.train(); return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--steps",type=int,default=5000); ap.add_argument("--batch-size",type=int,default=16)
    ap.add_argument("--context",type=int,default=512); ap.add_argument("--lr",type=float,default=3e-5)
    ap.add_argument("--warmup",type=int,default=100); ap.add_argument("--compile",action="store_true")
    ap.add_argument("--source",type=Path,default=SOURCE); ap.add_argument("--tag",default="arithmetic_qa_5k")
    ap.add_argument("--smoke",action="store_true"); args=ap.parse_args()
    device="cuda"; tok=Tokenizer.from_file(str(HERE/"bpe4096.json")); model=load(args.source,device)
    if args.compile: model=torch.compile(model,mode="reduce-overhead")
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    def factor(s):
        if s<args.warmup:return (s+1)/max(args.warmup,1)
        p=(s-args.warmup)/max(args.steps-args.warmup,1);return .1+.9*.5*(1+math.cos(math.pi*min(p,1)))
    sched=torch.optim.lr_scheduler.LambdaLR(opt,factor); rng=random.Random(20260912)
    run=HERE/"runs"/args.tag;run.mkdir(parents=True,exist_ok=True)
    if args.smoke:
        for k in KINDS: print(k,make_example(rng,"train",k)[1].strip())
        x,y=batch(rng,tok,2,args.context);print(x.shape,y.shape,int(x.max()));return
    started=time.perf_counter()
    for step in range(1,args.steps+1):
        x,y=batch(rng,tok,args.batch_size,args.context);x,y=x.to(device),y.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16): loss=F.cross_entropy(model(x).flatten(0,1),y.flatten())
        loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();sched.step()
        if step==1 or step%100==0:
            print(f"step={step}/{args.steps} loss={loss.item():.4f} ppl={math.exp(loss.item()):.3f} lr={sched.get_last_lr()[0]:.3e} grad={float(gn):.3f}",flush=True)
        if step%1000==0:
            sd={k.removeprefix('_orig_mod.'):v for k,v in model.state_dict().items()};torch.save(sd,run/f"step_{step:05d}.pt")
    scores=exact_eval(model,tok,device)
    sd={k.removeprefix('_orig_mod.'):v for k,v in model.state_dict().items()};torch.save(sd,run/"final.pt")
    result={"steps":args.steps,"tokens":args.steps*args.batch_size*args.context,"seconds":time.perf_counter()-started,"exact":scores,"macro":sum(scores.values())/len(scores)}
    (run/"result.json").write_text(json.dumps(result,indent=2)+"\n");print("RESULT "+json.dumps(result),flush=True)


if __name__=="__main__":main()
