#!/usr/bin/env bash
# Fuse LoRA adapters into the base, then make MLX artifacts:
#   dist/fused      (HF-format fp16, input for GGUF export)
#   dist/mlx-4bit   (4-bit MLX, publish this for Mac users)
# For GGUF (llama.cpp/Ollama/LM Studio), run ./scripts/export_gguf.sh afterward.
set -euo pipefail
cd "$(dirname "$0")/.."
source ../.venv/bin/activate 2>/dev/null || source .venv/bin/activate

BASE="Qwen/Qwen2.5-Coder-7B-Instruct"

echo "==> [1/2] Fuse adapters -> dist/fused (NOTE: mlx_lm.fuse has NO --hf-path)"
mlx_lm.fuse --model "$BASE" --adapter-path adapters --save-path dist/fused

echo "==> [2/2] Quantize the FUSED (fine-tuned) model to 4-bit -> dist/mlx-4bit"
mlx_lm.convert --hf-path dist/fused --mlx-path dist/mlx-4bit -q --q-bits 4 --q-group-size 64

echo "==> Done. dist/mlx-4bit ready to publish. Next: ./scripts/export_gguf.sh"
