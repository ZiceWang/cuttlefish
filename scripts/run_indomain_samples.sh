#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"
checkpoint="${CHECKPOINT:-runs/sft_chatml_stage2_5x.best.pt}"
prompts=(
  "What is the capital of France? Answer in one sentence."
  "Why is the sky blue? Explain briefly."
  "Tell me a short programmer joke."
  "Summarize the following passage in one short sentence: A research team installed solar panels on three university buildings in 2024. After one year, electricity purchased from the grid fell by 28 percent, and the university plans to equip two more buildings next summer."
  "Rewrite this message more concisely while preserving its meaning: I am writing to let you know that the meeting we originally planned for Tuesday afternoon will need to be moved to Thursday morning because several team members cannot attend at the original time."
  "Give exactly three bullet points, each under eight words, describing benefits of regular exercise."
  "Answer in one sentence: Water freezes at what temperature in degrees Celsius under standard atmospheric pressure?"
)
for prompt in "${prompts[@]}"; do
  echo "=== PROMPT: $prompt ==="
  "$PYTHON" experiments/posttrain/sample_chatml.py \
    --checkpoint "$checkpoint" --prompt "$prompt" \
    --max-new-tokens "${MAX_NEW_TOKENS:-128}" --temperature 0.5 --top-k 40 --top-p 0.85 \
    --min-p 0.08 --repetition-penalty 1.08 --no-repeat-ngram 3 --device cuda
  echo
done
