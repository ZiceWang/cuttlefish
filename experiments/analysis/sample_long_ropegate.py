"""Generate past the training context and print boundary-aligned text segments."""
# --- repo path bootstrap: flat sibling imports + runs/data/tokenizer paths ---
import sys as _sys
from pathlib import Path as _Path
_EXP = _Path(__file__).resolve().parents[1]
for _d in [_EXP, *sorted(_p for _p in _EXP.iterdir() if _p.is_dir() and _p.name != "__pycache__")]:
    if str(_d) not in _sys.path:
        _sys.path.insert(0, str(_d))
HERE = _EXP.parent
# ---------------------------------------------------------------------------
from pathlib import Path

import torch
from tokenizers import Tokenizer

import bdh_best
import rope_group_gate




def main():
    torch.manual_seed(1337)
    device = "cuda"
    tokenizer = Tokenizer.from_file(str(HERE / "bpe4096.json"))
    config = bdh_best.Config(
        hidden_size=450, heads=4, mlp_mult=16, layers=12, recurrences=1,
        vocab_size=4096, context_length=512, dropout=0.0,
        mixer_mode="swiglu", swiglu_in_loop=False, mixer_width=2250)
    model = rope_group_gate.RoPEGroupGateNoAbs(config, groups=8).to(device)
    state = torch.load(HERE / "runs" / "big_ropegate8noabs_r1_full.best.pt",
                       map_location=device)
    model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in state.items()})
    model.eval()

    prompt = "The development of scientific knowledge depends on"
    prompt_ids = tokenizer.encode(prompt).ids
    ids = list(prompt_ids)
    generated = []
    with torch.inference_mode():
        for _ in range(320):
            context = torch.tensor([ids[-512:]], dtype=torch.long, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(context)[0, -1].float() / 0.8
            values, _ = torch.topk(logits, 50)
            logits[logits < values[-1]] = float("-inf")
            token = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()
            ids.append(token)
            generated.append(token)

    print("PROMPT")
    print(prompt)
    for start, end in ((0, 192), (192, 256), (256, 320)):
        print(f"\nGENERATED TOKENS {start + 1}-{end}")
        print(tokenizer.decode(generated[start:end]))
    print("\nFULL TEXT")
    print(tokenizer.decode(ids))


if __name__ == "__main__":
    main()
