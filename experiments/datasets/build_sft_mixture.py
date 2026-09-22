"""Build an English-first ChatML SFT mixture without a context-length cutoff."""
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
from tokenizers import Tokenizer

SPECS = {
    "ultrachat": ("HuggingFaceH4/ultrachat_200k", None, "train_sft", 60_000, "mit"),
    "dolly": ("databricks/databricks-dolly-15k", None, "train", 12_000, "cc-by-sa-3.0"),
    "sciq": ("allenai/sciq", None, "train", 10_000, "cc-by-nc-3.0"),
    "squad": ("rajpurkar/squad", None, "train", 10_000, "cc-by-sa-4.0"),
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy", "train", 2_251, "cc-by-sa-4.0"),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "train", 1_119, "cc-by-sa-4.0"),
    "commonsenseqa": ("tau/commonsense_qa", None, "train", 9_741, "mit"),
}


def clean(text):
    return " ".join(str(text or "").replace("\x00", " ").split()).strip()


def mc_prompt(question, choices):
    return clean(question) + "\n" + "\n".join(
        f"{label}. {clean(text)}" for label, text in zip(choices["label"], choices["text"])
    )


def convert(name, row):
    if name == "ultrachat":
        messages = [{"role": x["role"], "content": clean(x["content"])}
                    for x in row["messages"]
                    if x["role"] in ("system", "user", "assistant")]
    elif name == "dolly":
        prompt = clean(row["instruction"])
        if clean(row["context"]):
            prompt += "\n\nContext:\n" + clean(row["context"])
        messages = [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": clean(row["response"])}]
    elif name == "sciq":
        prompt = clean(row["question"])
        answer = clean(row["correct_answer"])
        support = clean(row["support"])
        response = answer + (f". {support}" if support and support.lower() != answer.lower() else "")
        messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
    elif name == "squad":
        answers = row["answers"]["text"]
        if not answers:
            return None
        prompt = f"Context:\n{clean(row['context'])}\n\nQuestion:\n{clean(row['question'])}"
        messages = [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": clean(answers[0])}]
    elif name.startswith("arc_") or name == "commonsenseqa":
        labels, texts = row["choices"]["label"], row["choices"]["text"]
        try:
            answer = clean(texts[labels.index(row["answerKey"])])
        except ValueError:
            return None
        messages = [{"role": "user", "content": mc_prompt(row["question"], row["choices"])},
                    {"role": "assistant", "content": f"{row['answerKey']}. {answer}"}]
    else:
        raise KeyError(name)
    if len(messages) < 2 or messages[-1]["role"] != "assistant":
        return None
    if any(not m["content"] for m in messages):
        return None
    return messages


def render(messages):
    return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=HERE / "data" / "sft_cherry_v1")
    ap.add_argument("--seed", type=int, default=20260911)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    records, seen, stats = [], set(), {}
    for name, (repo, config, split, target, license_name) in SPECS.items():
        accepted = scanned = 0
        dataset = load_dataset(repo, config, split=split, streaming=True)
        dataset = dataset.shuffle(seed=args.seed, buffer_size=10_000)
        for row in dataset:
            scanned += 1
            messages = convert(name, row)
            if messages is None:
                continue
            text = render(messages)
            if len(text) < 8:
                continue
            key = hashlib.sha256(text.lower().encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            records.append({"messages": messages, "text": text, "source": repo,
                            "subset": config, "license": license_name,
                            "character_count": len(text),
                            "sha256": key})
            accepted += 1
            if accepted >= target:
                break
        stats[name] = {"target": target, "accepted": accepted, "scanned": scanned,
                       "repo": repo, "config": config, "license": license_name}
        print(name, stats[name], flush=True)
    rng.shuffle(records)
    n_val = max(1, round(len(records) * 0.01))
    val, train = records[:n_val], records[n_val:]
    for filename, rows in (("train.jsonl", train), ("validation.jsonl", val)):
        with (args.output_dir / filename).open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "name": "sft_cherry_v1", "seed": args.seed, "format": "ChatML",
        "max_tokens": None, "train_examples": len(train),
        "validation_examples": len(val), "total_examples": len(records),
        "total_characters": sum(x["character_count"] for x in records),
        "special_tokens_required": ["<|im_start|>", "<|im_end|>", "<|pad|>"],
        "chatml_tokenizer": "bpe4099_chatml.json",
        "chatml_vocab_size": 4099,
        "loss_policy": "assistant spans only, including assistant <|im_end|>",
        "length_policy": "No build-time truncation; bucket/pack at the chosen SFT context length.",
        "usage_note": "Contains SciQ CC-BY-NC-3.0; default mixture is research/non-commercial.",
        "sources": stats,
    }
    chatml_tok = Tokenizer.from_file(str(HERE / "bpe4096.json"))
    added = chatml_tok.add_special_tokens(manifest["special_tokens_required"])
    if added != 3 or chatml_tok.get_vocab_size() != 4099:
        raise RuntimeError(f"unexpected ChatML vocabulary: added={added}, size={chatml_tok.get_vocab_size()}")
    chatml_tok.save(str(args.output_dir / manifest["chatml_tokenizer"]))
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
