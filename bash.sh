#!/usr/bin/env bash
set -euo pipefail

# Override these variables when running on a different machine.
MODEL_ROOT="${MODEL_ROOT:-models/llama3.1-8b}"
FP16_CHECKPOINT="${FP16_CHECKPOINT:-${MODEL_ROOT}/llama3.1-8b-instruct}"
QUANTIZED_CHECKPOINT="${QUANTIZED_CHECKPOINT:-${MODEL_ROOT}/llama3.1-8b-gptq-int4}"
DATASET_TYPE="${DATASET_TYPE:-klar}"
QUANT_TYPE="${QUANT_TYPE:-gptq-int4}"
MODEL_TYPE="${MODEL_TYPE:-llama3.1}"
LANGUAGES="${LANGUAGES:-ar,ca,el,en,es,fr,he,hu,ja,ko,nl,tr,zh}"
OUTPUT_DIR="${OUTPUT_DIR:-result/recovery}"

python recovery_experiment.py \
  --dataset_type "${DATASET_TYPE}" \
  --checkpoint "${QUANTIZED_CHECKPOINT}" \
  --quant_type "${QUANT_TYPE}" \
  --model_type "${MODEL_TYPE}" \
  --languages "${LANGUAGES}" \
  --batch_size 32 \
  --max_new_tokens 8 \
  --stages 1,2 \
  --save_path "${OUTPUT_DIR}" \
  --resume true \
  --calculate_ppl false \
  --seed 42 \
  --fp16_checkpoint "${FP16_CHECKPOINT}"
