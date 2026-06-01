#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p result

sm="$(python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is required")
major, minor = torch.cuda.get_device_capability()
print(major * 10 + minor)
PY
)"

if (( sm >= 100 )); then
  echo "current device is sm_${sm}; use scripts/run_sm_ge100.sh for non-simulate kernels" >&2
  exit 1
fi

rm -f result/bench_*_results.json result/gemm_*_results.json result/benchmark_report.md

python bench.py \
  AbsMax_simulate_fp4_swizzled \
  ScaleSweep_MSE_simulate_fp4_swizzled \
  ScaleSweep_simulate_fp4_swizzled \
  --output-dir result

python gemm_simulate_nvfp4_perf.py --output-dir result
python generate_markdown.py --output-dir result
