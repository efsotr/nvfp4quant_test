#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/print_ifeval_result.py" "Llama-3.2-1B-Instruct"
echo
python "${SCRIPT_DIR}/print_ifeval_result.py" "Llama-3.2-1B-Instruct-NVFP4"
echo
python "${SCRIPT_DIR}/print_ifeval_result.py" "Llama-3.2-1B-Instruct-NVFP4_scalesweep_mse"
