#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"
prompts=(
  "Why is the sky blue? Explain briefly."
  "What is the capital of France? Answer in one sentence."
  "Tell me a short programmer joke."
  "Give exactly three concise tips for learning Python."
)
for prompt in "${prompts[@]}"; do
  echo "=== PROMPT: ${prompt} ==="
  "$PYTHON" experiments/posttrain/sample_chatml.py \
    --checkpoint runs/sft_chatml_stage2_5x.best.pt \
    --prompt "${prompt}" --max-new-tokens 128 --temperature 0.7 \
    --top-k 50 --top-p 0.9 --min-p 0.05 \
    --repetition-penalty 1.12 --no-repeat-ngram 3 --seed 2027 --device cuda
  echo
done
