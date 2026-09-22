"""Answer-masked training on diverse symbolic and natural-language arithmetic QA."""
# --- repo path bootstrap: flat sibling imports + runs/data/tokenizer paths ---
import sys as _sys
from pathlib import Path as _Path
_EXP = _Path(__file__).resolve().parents[1]
for _d in [_EXP, *sorted(_p for _p in _EXP.iterdir() if _p.is_dir() and _p.name != "__pycache__")]:
    if str(_d) not in _sys.path:
        _sys.path.insert(0, str(_d))
HERE = _EXP.parent
# ---------------------------------------------------------------------------
import argparse,json,math,random,re,sys,time
from pathlib import Path
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from compare_continue_samples import load
from diverse_arithmetic_generator import KINDS,generate

def make_batch(rng,tok,n,device,gsm_rows=None,gsm_ratio=0.0):
    rows=[]
    for _ in range(n):
        if gsm_rows and rng.random()<gsm_ratio:
            row=gsm_rows[rng.randrange(len(gsm_rows))];prompt,answer=row["prompt"],row["answer"]
        else:
            ex=generate(rng,"train");prompt,answer=ex.prompt,ex.answer
        p=tok.encode(prompt,add_special_tokens=False).ids;a=tok.encode(answer,add_special_tokens=False).ids
        ids=(p+a)[:256];labels=([-100]*len(p)+a)[:256];rows.append((ids,labels))
    # Fixed width avoids recompilation; 256 leaves room for four-digit column CoT.
    width=256;x=torch.zeros((n,width),dtype=torch.long);y=torch.full((n,width),-100,dtype=torch.long)
    for i,(ids,labels) in enumerate(rows):x[i,:len(ids)]=torch.tensor(ids);y[i,:len(labels)]=torch.tensor(labels)
    return x[:,:-1].to(device),y[:,1:].to(device)

def answer(model,tok,prompt,max_new_tokens=192):
    ids=tok.encode(prompt,add_special_tokens=False).ids;new=[]
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            x=torch.tensor([ids[-512:]],device="cuda")
            with torch.autocast("cuda",dtype=torch.bfloat16):n=model(x)[0,-1].argmax().item()
            ids.append(n);new.append(n)
            if "\n\n" in tok.decode(new):break
    return tok.decode(new)

def evaluate(model,tok,per_kind=40):
    model.eval();rng=random.Random(741993);scores={};examples=[]
    for mode,heldout in (("seen_template",False),("unseen_template",True)):
        for kind in KINDS:
            ok=0
            for _ in range(per_kind):
                ex=generate(rng,"val",kind,heldout);text=answer(model,tok,ex.prompt);nums=re.findall(r"-?\d+",text);good=bool(nums and int(nums[-1])==ex.gold);ok+=good
                if len(examples)<12:examples.append({"mode":mode,"kind":kind,"prompt":ex.prompt,"output":text,"gold":ex.gold,"ok":good})
            scores[f"{mode}/{kind}"]=ok/per_kind
    model.train();return scores,examples

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--steps",type=int,default=10000);ap.add_argument("--batch-size",type=int,default=64);ap.add_argument("--lr",type=float,default=2e-5);ap.add_argument("--warmup",type=int,default=200);ap.add_argument("--compile",action="store_true");ap.add_argument("--source",type=Path,default=HERE/"runs/arithmetic_qa_5k/final.pt");ap.add_argument("--tag",default="arithmetic_diverse_10k");ap.add_argument("--smoke",action="store_true");ap.add_argument("--gsm8k",type=Path);ap.add_argument("--gsm8k-ratio",type=float,default=0.25);ap.add_argument("--save-every",type=int,default=2000);args=ap.parse_args()
    tok=Tokenizer.from_file(str(HERE/"bpe4096.json"));rng=random.Random(20260912)
    gsm_rows=[json.loads(line) for line in args.gsm8k.open()] if args.gsm8k else []
    if not 0<=args.gsm8k_ratio<=1:raise ValueError("--gsm8k-ratio must be in [0, 1]")
    if args.gsm8k and not gsm_rows:raise ValueError("GSM8K file is empty")
    print(f"gsm8k_rows={len(gsm_rows)} gsm8k_ratio={args.gsm8k_ratio if gsm_rows else 0}",flush=True)
    if args.smoke:
        for held in (False,True):
            for k in KINDS:print(generate(rng,"val",k,held))
        x,y=make_batch(rng,tok,8,"cpu",gsm_rows,args.gsm8k_ratio);print(x.shape,y.shape,(y!=-100).sum().item());return
    model=load(args.source,"cuda");model=torch.compile(model,mode="reduce-overhead") if args.compile else model
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    def factor(s):
        if s<args.warmup:return (s+1)/args.warmup
        p=min(1,(s-args.warmup)/max(1,args.steps-args.warmup));return .1+.9*.5*(1+math.cos(math.pi*p))
    sch=torch.optim.lr_scheduler.LambdaLR(opt,factor);run=HERE/"runs"/args.tag;run.mkdir(parents=True,exist_ok=True);start=time.perf_counter();supervised=0
    for step in range(1,args.steps+1):
        x,y=make_batch(rng,tok,args.batch_size,"cuda",gsm_rows,args.gsm8k_ratio);supervised+=(y!=-100).sum().item();opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):loss=F.cross_entropy(model(x).flatten(0,1),y.flatten(),ignore_index=-100)
        loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();sch.step()
        if step==1 or step%100==0:print(f"step={step}/{args.steps} loss={loss.item():.4f} ppl={math.exp(loss.item()):.3f} lr={sch.get_last_lr()[0]:.3e} grad={float(gn):.3f} supervised_tokens={supervised:,}",flush=True)
        if step%args.save_every==0:
            sd={k.removeprefix('_orig_mod.'):v for k,v in model.state_dict().items()};torch.save(sd,run/f"step_{step:05d}.pt")
    scores,examples=evaluate(model,tok);sd={k.removeprefix('_orig_mod.'):v for k,v in model.state_dict().items()};torch.save(sd,run/"final.pt")
    result={"scores":scores,"seen_macro":sum(v for k,v in scores.items() if k.startswith('seen_'))/len(KINDS),"unseen_macro":sum(v for k,v in scores.items() if k.startswith('unseen_'))/len(KINDS),"examples":examples,"supervised_tokens":supervised,"seconds":time.perf_counter()-start};(run/"result.json").write_text(json.dumps(result,indent=2)+"\n");print("RESULT "+json.dumps({k:v for k,v in result.items() if k!='examples'}),flush=True)
if __name__=="__main__":main()
