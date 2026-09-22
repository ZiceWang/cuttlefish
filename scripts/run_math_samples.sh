#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"
checkpoint="${CHECKPOINT:-runs/sft_chatml_stage2_5x.best.pt}"
prompts=(
  "Calculate 37 + 58. Give only the answer."
  "Ava has 24 apples. She gave 7 to Ben and bought 5 more. How many apples does she have now? Show your reasoning briefly."
  "A train travels 60 miles per hour for 2.5 hours. It then travels 40 miles per hour for 1.5 hours. What total distance does it travel? Show your calculation."
  "Solve for x: 3x + 7 = 25. Show your reasoning briefly."
)
for prompt in "${prompts[@]}"; do
  echo "=== PROMPT: $prompt ==="
  "$PYTHON" experiments/posttrain/sample_chatml.py \
    --checkpoint "$checkpoint" --prompt "$prompt" \
    --max-new-tokens 128 --temperature 0.3 --top-k 30 --top-p 0.85 \
    --min-p 0.1 --repetition-penalty 1.08 --no-repeat-ngram 3 --device cuda
  echo
done
