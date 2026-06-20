#!/usr/bin/env bash
# Build data + run the LoRA fine-tune. From project root:  ./scripts/train.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source ../.venv/bin/activate 2>/dev/null || source .venv/bin/activate

echo "==> Building dataset"
python data/build_dataset.py

echo "==> LoRA fine-tune"
mlx_lm.lora --config config/lora_config.yaml

echo "==> Done. Adapters in ./adapters  (tip: ./scripts/watch_train.py for a live view)"
