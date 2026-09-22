"""Short verifiable-policy-gradient probe on two-digit addition candidates."""
# --- repo path bootstrap: flat sibling imports + runs/data/tokenizer paths ---
import sys as _sys
from pathlib import Path as _Path
_EXP = _Path(__file__).resolve().parents[1]
for _d in [_EXP, *sorted(_p for _p in _EXP.iterdir() if _p.is_dir() and _p.name != "__pycache__")]:
    if str(_d) not in _sys.path:
        _sys.path.insert(0, str(_d))
HERE = _EXP.parent
# ---------------------------------------------------------------------------
import argparse, copy, json, math, random, time
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from train_sft import HERE, make_model


def problem(rng):
    a, b = rng.randint(10, 89), rng.randint(10, 89)
    answer = a + b
    pool = {answer}
    while len(pool) < 10:
        pool.add(max(0, answer + rng.randint(-18, 18)))
    choices = list(pool)
    rng.shuffle(choices)
    return f"{a} + {b} =", choices, choices.index(answer)


def candidate_logits(model, tokenizer, batch, device):
    sequences, spans = [], []
    for prompt, choices, _ in batch:
        prefix = tokenizer.encode(prompt, add_special_tokens=False).ids
        for choice in choices:
            suffix = tokenizer.encode(" " + str(choice), add_special_tokens=False).ids
            sequences.append(prefix + suffix)
            spans.append((len(prefix), len(suffix)))
    width = max(map(len, sequences))
    padded = torch.zeros(len(sequences), width, dtype=torch.long, device=device)
    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = torch.tensor(seq, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(padded)
    token_logp = F.log_softmax(logits.float(), -1)
    scores = []
    for row, (start, length) in enumerate(spans):
        ids = padded[row, start:start + length]
        positions = torch.arange(start - 1, start + length - 1, device=device)
        scores.append(token_logp[row, positions, ids].mean())
    return torch.stack(scores).view(len(batch), 10)


@torch.inference_mode()
def evaluate(model, tok, seed, count, batch_size, device):
    rng = random.Random(seed)
    correct = probability = 0.0
    for offset in range(0, count, batch_size):
        batch = [problem(rng) for _ in range(min(batch_size, count - offset))]
        scores = candidate_logits(model, tok, batch, device)
        probs = scores.softmax(-1)
        targets = torch.tensor([x[2] for x in batch], device=device)
        correct += (scores.argmax(-1) == targets).sum().item()
        probability += probs.gather(1, targets[:, None]).sum().item()
    return {"accuracy": correct / count, "correct_probability": probability / count}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=HERE / "runs/big_formal512_10shard_1ep.best.pt")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--group-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--kl-beta", type=float, default=0.03)
    ap.add_argument("--target-kl", type=float, default=0.0,
                    help="enable adaptive KL coefficient when positive")
    ap.add_argument("--kl-adapt-rate", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20260912)
    ap.add_argument("--tag", default="pretrain_add_grpo_probe")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    tok = Tokenizer.from_file(str(HERE / "bpe4096.json"))
    policy = make_model(4096, 64).to(device)
    policy.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    reference = copy.deepcopy(policy).eval()
    for p in reference.parameters(): p.requires_grad_(False)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0)
    before = evaluate(policy.eval(), tok, args.seed + 1, 512, 64, device)
    print("before", before, flush=True)
    policy.train(); rng = random.Random(args.seed)
    history=[]; started=time.perf_counter();kl_beta=args.kl_beta;kl_ema=0.0
    for step in range(1, args.steps + 1):
        batch=[problem(rng) for _ in range(args.batch_size)]
        scores=candidate_logits(policy,tok,batch,device)
        log_probs=scores.log_softmax(-1); probs=log_probs.exp()
        with torch.no_grad():
            ref_log_probs=candidate_logits(reference,tok,batch,device).log_softmax(-1)
            sampled=torch.multinomial(probs, args.group_size, replacement=True)
            targets=torch.tensor([x[2] for x in batch],device=device)[:,None]
            rewards=(sampled==targets).float()
            advantages=rewards-rewards.mean(1,keepdim=True)
        selected=log_probs.gather(1,sampled)
        pg=-(advantages*selected).mean()
        kl=(probs*(log_probs-ref_log_probs)).sum(-1).mean()
        loss=pg+kl_beta*kl
        optimizer.zero_grad(set_to_none=True); loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(policy.parameters(),1.0)
        optimizer.step()
        current_kl=kl.item()
        kl_ema=current_kl if step==1 else .9*kl_ema+.1*current_kl
        if args.target_kl>0:
            error=max(-1.0,min(1.0,kl_ema/args.target_kl-1.0))
            kl_beta=max(1e-3,min(10.0,kl_beta*math.exp(args.kl_adapt_rate*error)))
        row={"step":step,"reward":rewards.mean().item(),
             "group_hit":(rewards.sum(1)>0).float().mean().item(),
             "pg":pg.item(),"kl":current_kl,"kl_ema":kl_ema,
             "kl_beta":kl_beta,"grad_norm":float(grad)}
        history.append(row)
        if step==1 or step%5==0: print(row,flush=True)
    after=evaluate(policy.eval(),tok,args.seed+1,512,64,device)
    result={"args":vars(args),"before":before,"after":after,"history":history,
            "elapsed_seconds":time.perf_counter()-started}
    out=HERE/f"runs/{args.tag}.json"
    out.write_text(json.dumps(result,indent=2,default=str))
    torch.save(policy.state_dict(),HERE/f"runs/{args.tag}.pt")
    print("after",after,"saved",out,flush=True)

if __name__ == "__main__": main()
