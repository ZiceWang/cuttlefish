"""Download and normalize the complete GSM8K train split (non-streaming)."""
# --- repo path bootstrap: flat sibling imports + runs/data/tokenizer paths ---
import sys as _sys
from pathlib import Path as _Path
_EXP = _Path(__file__).resolve().parents[1]
for _d in [_EXP, *sorted(_p for _p in _EXP.iterdir() if _p.is_dir() and _p.name != "__pycache__")]:
    if str(_d) not in _sys.path:
        _sys.path.insert(0, str(_d))
HERE = _EXP.parent
# ---------------------------------------------------------------------------
import json
import re
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer

OUT = HERE / "data/gsm8k_train.jsonl"
tok = Tokenizer.from_file(str(HERE / "bpe4096.json"))
dataset = load_dataset("openai/gsm8k", "main", split="train", streaming=False)

kept = []
dropped = 0
lengths = []
for row in dataset:
    # Remove calculator-only annotations while retaining the human rationale.
    raw = re.sub(r"<<[^<>]*>>", "", row["answer"]).strip()
    if "####" not in raw:
        dropped += 1
        continue
    reasoning, final = raw.rsplit("####", 1)
    final = final.strip().replace(",", "")
    answer = reasoning.strip() + f" Therefore, the answer is {final}.\n\n"
    prompt = f"Question: {row['question'].strip()}\nAnswer:"
    length = len(tok.encode(prompt + answer, add_special_tokens=False).ids)
    if length > 256:
        dropped += 1
        continue
    kept.append({"prompt": prompt, "answer": " " + answer, "gold": final, "tokens": length})
    lengths.append(length)

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w") as f:
    for row in kept:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
lengths.sort()
print(json.dumps({
    "source_rows": len(dataset), "kept": len(kept), "dropped": dropped,
    "tokens_p50": lengths[len(lengths)//2],
    "tokens_p95": lengths[int(len(lengths)*.95)], "tokens_max": max(lengths),
    "output": str(OUT),
}, indent=2))
for row in kept[:3]:
    print(json.dumps(row, ensure_ascii=False))
