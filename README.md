# Preliminary ScaleSweep NVFP4 Quant-Operator/GEMM Speed and Quantization Error Evaluation

For details of ScaleSweep, see `ScaleSweep_paper/`. 

`kernel_ScaleSweep*.py` provides preliminary implementations of ScaleSweep and still requires further refinement.

NVFP4 quantization and GEMM benchmark scripts. Benchmark outputs are written as JSON under `result/`, and `generate_markdown.py` can convert those JSON files into `result/benchmark_report.md`.

## Run

For `sm >= 100` hardware, use the native NVFP4/CUTLASS path:

```bash
./scripts/run_sm_ge100.sh
```

For `sm < 100` hardware, use the simulate swizzled path:

```bash
./scripts/run_sm_lt100.sh
```

The scripts run the matching `bench.py` kernels, run the matching GEMM script, and write `result/benchmark_report.md`.

## Outputs

- `bench.py` writes `result/bench_<kernel>_results.json`.
- `gemm_simulate_nvfp4_perf.py` writes `result/gemm_simulate_nvfp4_perf_results.json`.
- `gemm_nvfp4_perf.py` writes `result/gemm_nvfp4_perf_results.json`.
- `generate_markdown.py` writes `result/benchmark_report.md`.

Both GEMM scripts use the same JSON shape and default output directory. GEMM input and weight tensors use `Laplace(loc=0, scale=1)`. Input-channel square norm uses `mean(x^2)` instead of `sum(x^2)`.

## Current Results

The tables below are built from the current files under `result/`.

### Environment

| torch | vllm | cuda | gpu_driver | gpu | sm | sm_count | total_memory_bytes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2.11.0+cu130 | 0.22.0 | 13.0 | 580.95.05 | NVIDIA RTX PRO 6000 Blackwell Server Edition | 120 | 188 | 101974081536 |

### Quantization Speed

Latency in milliseconds. Columns are `bsz`, with `dim=8192` for all rows.
The `vllm` quantization baseline uses `vllm._custom_ops.scaled_fp4_quant`.

| kernel | 1 | 8 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vllm | 0.00584533 | 0.00656058 | 0.00802169 | 0.00940061 | 0.0122513 | 0.0162843 | 0.0253864 | 0.0406222 | 0.0760823 | 0.134747 |
| ScaleSweep_MSE_round_swizzled | 0.00633512 | 0.00779489 | 0.00929372 | 0.0101168 | 0.0126149 | 0.0168762 | 0.0256528 | 0.0414329 | 0.0722104 | 0.135609 |
| ScaleSweep_MSE_swizzled | 0.0064172 | 0.00809576 | 0.00927953 | 0.0100796 | 0.0125797 | 0.0163772 | 0.0256723 | 0.0410426 | 0.0726136 | 0.135918 |
| ScaleSweep_swizzled | 0.00852955 | 0.0087639 | 0.010818 | 0.0129502 | 0.0164173 | 0.0222497 | 0.0327516 | 0.0573654 | 0.100374 | 0.187389 |

### Quantization Error

Error metrics are shown only for `bsz=8192`.

| kernel | bsz | mse | weighted_mse | max_abs_error |
| --- | --- | --- | --- | --- |
| vllm | 8192 | 0.0171616 | - | 1.80059 |
| ScaleSweep_MSE_round_swizzled | 8192 | 0.0135951 | - | 1.34375 |
| ScaleSweep_MSE_swizzled | 8192 | 0.0135951 | - | 1.34375 |
| ScaleSweep_swizzled | 8192 | 0.0135951 | 0.0135951 | 1.34375 |


### GEMM Result

The `vllm` GEMM baseline uses `vllm._custom_ops.cutlass_scaled_fp4_mm`.

| kernel | status | mse | max_abs_error |
| --- | --- | --- | --- |
| vllm | ok | 562.049 | 132 |
| ScaleSweep_MSE_swizzled | ok | 444.243 | 117.812 |
| ScaleSweep_MSE_round_swizzled | ok | 444.243 | 117.812 |
| ScaleSweep_swizzled | ok | 444.177 | 115.625 |
