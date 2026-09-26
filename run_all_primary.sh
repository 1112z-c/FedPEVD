#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 ENCODER_PATH OUTPUT_ROOT [PYTHON]" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENCODER="$1"
OUTPUT_ROOT="$2"
PYTHON_BIN="${3:-python}"

for seed in 45 51 57; do
  for dataset in chrome github libav; do
    "$PYTHON_BIN" "$ROOT/run_primary.py" \
      --dataset "$dataset" --seed "$seed" \
      --encoder_path "$ENCODER" \
      --output_dir "$OUTPUT_ROOT/seed${seed}/${dataset}" \
      --python "$PYTHON_BIN"
  done
done
