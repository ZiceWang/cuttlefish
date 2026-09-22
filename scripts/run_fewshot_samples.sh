#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"
prompts=(
  "Write a Python function add(a, b) that returns their sum. Return only the code."
  "Write a Python function factorial(n) that correctly handles zero. Return only the code."
  "Write a Python function fibonacci(n) that returns the first n Fibonacci numbers. Return only the code."
)
for prompt in "${prompts[@]}"; do
  echo "=== PROMPT: $prompt ==="
  "$PYTHON" experiments/posttrain/sample_chatml.py \
    --checkpoint runs/sft_chatml_stage2_5x.best.pt \
    --demo-json data/code_fewshot.json --prompt "$prompt" \
    --max-new-tokens 128 --temperature 0.5 --top-k 40 --top-p 0.85 \
    --min-p 0.08 --repetition-penalty 1.08 --no-repeat-ngram 3 --device cuda
done
