#!/bin/bash
# Merge a LoRA adapter into its base and export to GGUF (Q4_K_M) for
# llama.cpp / Ollama deployment. Paper evaluation uses HuggingFace fp16
# directly; this script is only needed for deployment.
#
# Usage: bash export_gguf.sh [adapter_path] [output_name]
#   adapter_path: path to LoRA adapter (default: ./lora_output/final)
#   output_name: name for output files (default: qwen3-8b-laimark)

set -e

ADAPTER_PATH="${1:-./lora_output/final}"
OUTPUT_NAME="${2:-qwen3-8b-laimark}"
MODEL_ID="Qwen/Qwen3-8B"
MERGED_DIR="./merged_model"
GGUF_DIR="./gguf_output"

echo "=== Step 1: Merge LoRA adapter with base model ==="
python -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print('Loading base model...')
model = AutoModelForCausalLM.from_pretrained(
    '${MODEL_ID}', torch_dtype=torch.bfloat16, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained('${MODEL_ID}', trust_remote_code=True)

print('Loading LoRA adapter...')
model = PeftModel.from_pretrained(model, '${ADAPTER_PATH}')

print('Merging...')
model = model.merge_and_unload()

print('Saving merged model...')
model.save_pretrained('${MERGED_DIR}')
tokenizer.save_pretrained('${MERGED_DIR}')
print('Done.')
"

echo "=== Step 2: Convert to GGUF ==="
# Install llama.cpp if not present
if [ ! -d "llama.cpp" ]; then
    echo "Cloning llama.cpp..."
    git clone --depth 1 https://github.com/ggerganov/llama.cpp.git
    pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
fi

mkdir -p "${GGUF_DIR}"

echo "Converting to GGUF (f16 first)..."
python llama.cpp/convert_hf_to_gguf.py "${MERGED_DIR}" \
    --outfile "${GGUF_DIR}/${OUTPUT_NAME}-f16.gguf" \
    --outtype f16

echo "=== Step 3: Quantize to Q4_K_M ==="
# Build llama-quantize if not present
if [ ! -f "llama.cpp/build/bin/llama-quantize" ]; then
    echo "Building llama.cpp quantize tool..."
    cd llama.cpp && mkdir -p build && cd build && cmake .. && make llama-quantize -j$(nproc) && cd ../..
fi

llama.cpp/build/bin/llama-quantize \
    "${GGUF_DIR}/${OUTPUT_NAME}-f16.gguf" \
    "${GGUF_DIR}/${OUTPUT_NAME}-Q4_K_M.gguf" \
    Q4_K_M

echo "=== Done ==="
echo "GGUF file: ${GGUF_DIR}/${OUTPUT_NAME}-Q4_K_M.gguf"
ls -lh "${GGUF_DIR}/"
echo ""
echo "Download with: scp user@<host>:${GGUF_DIR}/${OUTPUT_NAME}-Q4_K_M.gguf ."
