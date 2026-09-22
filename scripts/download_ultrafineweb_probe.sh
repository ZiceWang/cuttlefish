#!/usr/bin/env bash
set -euo pipefail
repo="openbmb/Ultra-FineWeb-L3"
out="${OUT:-$(cd "$(dirname "$0")/.." && pwd)/data/ultrafineweb_l3_probe}"
files=(
  "data/ultrafineweb_en_l3/multi_style/part-00000-f36f5a53-4a77-434b-b9bc-67ed69b93fe2-c000.snappy.parquet"
  "data/ultrafineweb_en_l3/qa/part-00000-37dc9f21-f87f-4f43-8dd2-134424f1537a-c000.snappy.parquet"
  "data/ultrafineweb_zh_l3/multi_style/part-00000-c13afd3b-b5fb-4acd-97dc-e045a844c126-c000.snappy.parquet"
  "data/ultrafineweb_zh_l3/qa/part-00000-100cee91-ea2e-4268-9b98-1403cd5d9d11-c000.snappy.parquet"
)
for file in "${files[@]}"; do
  echo "START $file"
  target="$out/$file"
  mkdir -p "$(dirname "$target")"
  curl -fL -C - --retry 5 --retry-delay 3 \
    "https://hf-mirror.com/datasets/$repo/resolve/main/$file" -o "$target"
  echo "DONE $file"
done
