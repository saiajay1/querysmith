#!/usr/bin/env bash
# Export the fused model to GGUF via llama.cpp (mlx-lm can't GGUF-export Qwen2).
# Handles the two gotchas: isolated converter venv (it conflicts with mlx-lm's
# transformers), and copying the BASE tokenizer (mlx re-serializes it in a format
# the converter rejects). Prereq: ./scripts/quantize.sh produced dist/fused.
set -euo pipefail
cd "$(dirname "$0")/.."
source ../.venv/bin/activate 2>/dev/null || source .venv/bin/activate

BASE="Qwen/Qwen2.5-Coder-7B-Instruct"
FUSED="dist/fused"
OUT="dist/querysmith-f16.gguf"
TOOLS=".tools"; LLAMA="$TOOLS/llama.cpp"; CONV_VENV="$TOOLS/gguf-venv"

[ -d "$FUSED" ] || { echo "ERROR: $FUSED missing — run ./scripts/quantize.sh first"; exit 1; }

echo "==> Copy upstream base tokenizer into $FUSED (converter needs the original format)"
CACHE="models--${BASE//\//--}"
BASE_SNAP=$(find ~/.cache/huggingface/hub/"$CACHE"/snapshots -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1)
if [ -n "${BASE_SNAP:-}" ]; then
  for tf in tokenizer.json tokenizer_config.json vocab.json merges.txt; do
    [ -f "$BASE_SNAP/$tf" ] && cp "$BASE_SNAP/$tf" "$FUSED/$tf"
  done
fi

echo "==> Fetch llama.cpp converter (shallow clone, once)"
mkdir -p "$TOOLS"
[ -d "$LLAMA" ] || git clone --depth 1 https://github.com/ggerganov/llama.cpp "$LLAMA"

echo "==> Isolated converter venv (keeps the mlx-lm env clean)"
[ -d "$CONV_VENV" ] || python -m venv "$CONV_VENV"
"$CONV_VENV/bin/pip" install --quiet -r "$LLAMA/requirements/requirements-convert_hf_to_gguf.txt"

echo "==> Convert $FUSED -> $OUT (f16)"
"$CONV_VENV/bin/python" "$LLAMA/convert_hf_to_gguf.py" "$FUSED" --outfile "$OUT" --outtype f16

echo "==> Done: $OUT"
echo "Optional small 4-bit GGUF (one-time llama.cpp build):"
echo "  cmake -S $LLAMA -B $LLAMA/build -DCMAKE_BUILD_TYPE=Release >/dev/null"
echo "  cmake --build $LLAMA/build --target llama-quantize -j"
echo "  $LLAMA/build/bin/llama-quantize $OUT dist/querysmith-Q4_K_M.gguf Q4_K_M"
