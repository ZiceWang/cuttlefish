"""Build a balanced 150k-example SmolTalk increment in native ChatML form."""
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
import random
from pathlib import Path

from datasets import load_dataset

SPECS = {
    "everyday-conversations": 35_000,
    "smol-constraints": 30_000,
    "smol-rewrite": 20_000,
    "smol-summarize": 20_000,
    "openhermes-100k": 45_000,
}


def clean(text):
    return str(text or "").replace("\x00", " ").strip()


def convert(row):
    messages = []
    for message in row.get("messages", []):
        role, content = message.get("role"), clean(message.get("content"))
        if role in ("system", "user", "assistant") and content:
            messages.append({"role": role, "content": content})
    if len(messages) < 2 or not any(m["role"] == "assistant" for m in messages):
        return None
    return messages


def render(messages):
    return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
                   for m in messages)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path,
                    default=HERE / "data" / "sft_cherry_v2_additional")
    ap.add_argument("--seed", type=int, default=20260912)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    records, seen, stats = [], set(), {}
    for config, target in SPECS.items():
        accepted = scanned = 0
        dataset = load_dataset("HuggingFaceTB/smoltalk", config, split="train",
                               streaming=True)
        dataset = dataset.shuffle(seed=args.seed, buffer_size=10_000)
        for row in dataset:
            scanned += 1
            messages = convert(row)
            if messages is None:
                continue
            text = render(messages)
            digest = hashlib.sha256(text.lower().encode()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            records.append({"messages": messages, "text": text,
                            "source": "HuggingFaceTB/smoltalk", "subset": config,
                            "character_count": len(text), "sha256": digest})
            accepted += 1
            if accepted >= target:
                break
        stats[config] = {"target": target, "accepted": accepted, "scanned": scanned}
        print(config, stats[config], flush=True)
    rng.shuffle(records)
    n_val = max(1, round(len(records) * 0.01))
    for filename, rows in (("validation.jsonl", records[:n_val]),
                           ("train.jsonl", records[n_val:])):
        with (args.output_dir / filename).open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "name": "sft_cherry_v2_additional", "seed": args.seed,
        "format": "ChatML", "train_examples": len(records) - n_val,
        "validation_examples": n_val, "total_examples": len(records),
        "max_tokens": None, "sources": stats,
        "usage": "Incremental continuation after sft_cherry_v1; assistant-span loss only.",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
