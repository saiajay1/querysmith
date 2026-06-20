#!/usr/bin/env bash
# Publish model + dataset to the Hugging Face Hub.
# PREREQ (once): source .venv/bin/activate && hf auth login   (WRITE token)
# Run:  HF_USER=your-username ./scripts/publish.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source ../.venv/bin/activate 2>/dev/null || source .venv/bin/activate

: "${HF_USER:?Set HF_USER, e.g. HF_USER=ajay ./scripts/publish.sh}"
MODEL_REPO="$HF_USER/Qwen2.5-Coder-7B-Querysmith"     # e.g. Qwen2.5-Coder-1.5B-querysmith
DATA_REPO="$HF_USER/querysmith-spider-bird"    # e.g. querysmith-data

echo "==> Creating repos (idempotent)"
hf repos create "$MODEL_REPO" --repo-type model   --exist-ok || true
hf repos create "$DATA_REPO"  --repo-type dataset --exist-ok || true

echo "==> Model card -> the published model page"
cp card/model_card.md dist/mlx-4bit/README.md

echo "==> Upload 4-bit MLX model (+ card) and GGUF if present"
hf upload "$MODEL_REPO" dist/mlx-4bit . --repo-type model
if [ -f dist/querysmith-f16.gguf ]; then
  hf upload "$MODEL_REPO" dist/querysmith-f16.gguf querysmith-f16.gguf --repo-type model
fi

echo "==> Upload dataset splits + card"
hf upload "$DATA_REPO" data/train.jsonl train.jsonl --repo-type dataset
hf upload "$DATA_REPO" data/valid.jsonl valid.jsonl --repo-type dataset
hf upload "$DATA_REPO" card/dataset_card.md README.md --repo-type dataset

echo "==> Done."
echo "    Model:   https://huggingface.co/$MODEL_REPO"
echo "    Dataset: https://huggingface.co/datasets/$DATA_REPO"
echo "    Next: create a Gradio Space and upload the space/ folder."
