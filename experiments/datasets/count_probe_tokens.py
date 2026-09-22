# --- repo path bootstrap: flat sibling imports + runs/data/tokenizer paths ---
import sys as _sys
from pathlib import Path as _Path
_EXP = _Path(__file__).resolve().parents[1]
for _d in [_EXP, *sorted(_p for _p in _EXP.iterdir() if _p.is_dir() and _p.name != "__pycache__")]:
    if str(_d) not in _sys.path:
        _sys.path.insert(0, str(_d))
HERE = _EXP.parent
# ---------------------------------------------------------------------------
#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import pyarrow.parquet as pq
from tokenizers import Tokenizer


ROOT = Path("data/ultrafineweb_l3_probe/data")
TOKENIZER = Path("experiments/bdh_orig/data/minicpm5_tokenizer/tokenizer.json")
OUT = Path("experiments/bdh_orig/runs/ultrafineweb_probe_token_counts.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER)
    args = parser.parse_args()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    files = [args.file] if args.file else sorted(ROOT.glob("**/*.parquet"))
    results = []
    for path in files:
        parquet = pq.ParquetFile(path)
        total = 0
        rows = 0
        started = time.time()
        for rg in range(parquet.num_row_groups):
            texts = parquet.read_row_group(rg, columns=["content"])["content"].to_pylist()
            # Encode in bounded batches so a large row group cannot create a huge temporary list.
            for start in range(0, len(texts), 1024):
                batch = [x or "" for x in texts[start : start + 1024]]
                total += sum(len(enc.ids) for enc in tokenizer.encode_batch(batch, add_special_tokens=False))
            rows += len(texts)
            print(f"{path.parent.name}/{path.name}: row_group={rg + 1}/{parquet.num_row_groups} rows={rows:,} tokens={total:,}", flush=True)
        results.append({
            "path": str(path), "bytes": path.stat().st_size, "rows": rows,
            "tokens": total, "tokens_per_row": total / rows,
            "seconds": time.time() - started, "shards_for_3b": 3_000_000_000 / total,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(f"RESULT={args.output}", flush=True)


if __name__ == "__main__":
    main()
