#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="Llama-3.2-1B-Instruct"
SAVE_DIR="Llama-3.2-1B-Instruct-NVFP4"

TASKS="ifeval"
RESULT_DIR="result"

mkdir -p "${RESULT_DIR}"

COMMON_ARGS=(
  --model vllm
  --tasks "${TASKS}"
  --batch_size auto
  --apply_chat_template
  --log_samples
  --seed 42
)

echo "[1/2] Running ${MODEL_ID} on GPU 0"

CUDA_VISIBLE_DEVICES=0 lm_eval \
  "${COMMON_ARGS[@]}" \
  --model_args "pretrained=${MODEL_ID},tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.85,max_model_len=8192" \
  --output_path "${RESULT_DIR}/${MODEL_ID}" \
  > "${RESULT_DIR}/${MODEL_ID}.log" 2>&1  &

echo "[2/2] Running ${SAVE_DIR} on GPU 1 with linear_backend=emulation"

CUDA_VISIBLE_DEVICES=1 lm_eval \
  "${COMMON_ARGS[@]}" \
  --model_args "pretrained=${SAVE_DIR},tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.85,max_model_len=8192,linear_backend=emulation" \
  --output_path "${RESULT_DIR}/${SAVE_DIR}" \
  > "${RESULT_DIR}/${SAVE_DIR}.log" 2>&1  &

