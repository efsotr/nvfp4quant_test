# nvfp4quant_test

NVFP4 quantization and GEMM benchmark scripts. Results are written as JSON under `result/` by default, and `generate_markdown.py` converts the generated JSON files into a markdown report.

## Run

Current hardware is `sm_89`, so use the simulate swizzled path:

```bash
./scripts/run_sm_lt100.sh
```

For `sm >= 100` hardware, use the native NVFP4/CUTLASS path:

```bash
./scripts/run_sm_ge100.sh
```

The scripts run only the swizzled matching `bench.py` kernels, run the matching GEMM script, and then write `result/benchmark_report.md`.

## Outputs

- `bench.py` writes `result/bench_<kernel>_results.json`.
- `gemm_simulate_nvfp4_perf.py` writes `result/gemm_simulate_nvfp4_perf_results.json`.
- `gemm_nvfp4_perf.py` writes `result/gemm_nvfp4_perf_results.json`.
- `generate_markdown.py` writes `result/benchmark_report.md`.

Both GEMM scripts use the same JSON shape and default output directory. GEMM input and weight tensors use `Laplace(loc=0, scale=1)`. Input-channel square norm uses `mean(x^2)` instead of `sum(x^2)`.

## Current Generated Results

Generated on `sm_89`, `sm_count=142`, `dim=8192`. Because `sm < 100`, only simulate kernels were executed.

### Environment

| torch | vllm | cuda | gpu_driver | gpu | sm | sm_count | total_memory_bytes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2.8.0+cu126 | 0.11.0 | 12.6 | 525.105.17 | NVIDIA L40 | 89 | 142 | 47620882432 |

### GEMM

| mode | kernel | status | latency_ms | mse | max_abs_error |
| --- | --- | --- | --- | --- | --- |
| simulate | AbsMax_simulate_fp4_swizzled | ok | - | 563.097 | 145.75 |
| simulate | ScaleSweep_MSE_simulate_fp4_swizzled | ok | - | 444.354 | 119 |
| simulate | ScaleSweep_simulate_fp4_swizzled | ok | - | 444.293 | 120.75 |

### Quantization Bench Summary

The table below shows the `bsz=8192` rows from the generated bench results. The full table is in `result/benchmark_report.md`.

| kernel | layout | bsz | dim | latency_ms | mse | weighted_mse | max_abs_error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AbsMax_simulate_fp4_swizzled | swizzled | 8192 | 8192 | 0.297366 | 0.0171627 | - | 1.80059 |
| ScaleSweep_MSE_simulate_fp4_swizzled | swizzled | 8192 | 8192 | 0.565407 | 0.0135958 | - | 1.34375 |
| ScaleSweep_simulate_fp4_swizzled | swizzled | 8192 | 8192 | 0.810476 | 0.0135958 | 0.0135958 | 1.34375 |
