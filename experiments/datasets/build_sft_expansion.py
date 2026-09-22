"""Build a cross-deduplicated expansion taking the full SFT pool to ~5x v1."""
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
    "smol-magpie-ultra": 60_000,
    "apigen-80k": 40_000,
    "explore-instruct-rewriting": 25_000,
    "longalign": 15_000,
    "metamathqa-50k": 45_000,
    "numina-cot-100k": 70_000,
    "self-oss-instruct": 30_000,
    "systemchats-30k": 18_000,
}


def clean(value):
    return str(value or "").replace("\x00", " ").strip()


def render(messages):
    return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
                   for m in messages)


def digest_messages(messages):
    return hashlib.sha256(render(messages).lower().encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path,
                    default=HERE / "data" / "sft_cherry_v3_expansion")
    ap.add_argument("--seed", type=int, default=20260912)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Cross-batch deduplication: never count an old example as new coverage.
    seen = set()
    for dirname in ("sft_cherry_v1", "sft_cherry_v2_additional"):
        for split in ("train.jsonl", "validation.jsonl"):
            path = HERE / "data" / dirname / split
            with path.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    seen.add(row.get("sha256") or digest_messages(row["messages"]))
    print(f"existing_unique={len(seen)}", flush=True)

    records, stats = [], {}
    for config, target in SPECS.items():
        accepted = scanned = 0
        dataset = load_dataset("HuggingFaceTB/smoltalk", config, split="train",
                               streaming=True)
        dataset = dataset.shuffle(seed=args.seed, buffer_size=10_000)
        for row in dataset:
            scanned += 1
            messages = []
            for message in row.get("messages", []):
                role = message.get("role")
                content = clean(message.get("content"))
                if role in ("system", "user", "assistant") and content:
                    messages.append({"role": role, "content": content})
            if len(messages) < 2 or not any(m["role"] == "assistant" for m in messages):
                continue
            key = digest_messages(messages)
            if key in seen:
                continue
            seen.add(key)
            text = render(messages)
            records.append({"messages": messages, "text": text,
                            "source": "HuggingFaceTB/smoltalk", "subset": config,
                            "character_count": len(text), "sha256": key})
            accepted += 1
            if accepted >= target:
                break
        stats[config] = {"target": target, "accepted": accepted, "scanned": scanned}
        print(config, stats[config], flush=True)

    random.Random(args.seed).shuffle(records)
    n_val = max(1, round(len(records) * 0.01))
    for filename, rows in (("validation.jsonl", records[:n_val]),
                           ("train.jsonl", records[n_val:])):
        with (args.output_dir / filename).open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "name": "sft_cherry_v3_expansion", "seed": args.seed,
        "format": "ChatML", "train_examples": len(records) - n_val,
        "validation_examples": n_val, "total_examples": len(records),
        "target_total_examples": sum(SPECS.values()), "sources": stats,
        "deduplication": "SHA-256 over normalized ChatML; excludes v1 and v2",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
