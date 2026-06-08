# ScaleSweep MSE NVFP4 Quantization

This directory contains a standalone ScaleSweep MSE NVFP4 quantization path, its simulate-FP4 variant for pre-Blackwell GPUs, PyTorch reference helpers, tests, and benchmarks. Test and benchmark result files are written under `../result/` by default.

Keep this README in sync with the code. If any file is added, removed, renamed, or its public behavior or internal organization changes, update the corresponding section in this README in the same change.

## Files

### `scalesweep_mse_nvfp4_quant.py`

Native Triton implementation of ScaleSweep MSE NVFP4 quantization. This path uses native FP4 PTX instructions and requires hardware/toolchain support for `e2m1` conversion.

Internal organization:
- Constants: `BLOCK_SIZE`, candidate scale sweep bounds, FP4/FP8 max values, and max positive FP8 scale raw value.
- Allocation helpers: `round_up`, `create_fp4_scale_tensor`, and `create_fp4_output_tensors`.
- Scale layout helper: `swizzled_scale_offsets` computes the 128x4 swizzled scale-factor layout.
- FP4 helpers: `_fp32x16_to_e2m1_u32x2` packs 16 FP32 values into two `uint32` words using native `cvt.rn.satfinite.e2m1x2.f32`; `_fp32x2_e2m1_quant_squared_error` and `_fp32x2_e2m1_quant_squared_error_acc` compute native FP4 roundtrip squared error with inline PTX.
- Block helpers: `_max_abs_16` and `_load_normalized_16_cols` operate on one 16-value quantization block.
- Kernel: `_scalesweep_mse_nvfp4_quant_kernel` tries candidate FP8 scales around the absmax-derived base scale, selects the scale with minimum FP4 reconstruction error, writes the selected scale, then packs final FP4 codes.
- Public API: `scalesweep_mse_nvfp4_quant`; `scaled_fp4_quant` is an alias.

### `scalesweep_mse_nvfp4_quant_simulate.py`

Triton simulate-FP4 implementation of the same ScaleSweep MSE algorithm. This file mirrors `scalesweep_mse_nvfp4_quant.py` except where native FP4 conversion is not supported.

Internal organization:
- Same constants, allocation helpers, swizzled layout helper, block helpers, autotune configs, kernel structure, and public API shape as the native file.
- Simulated FP4 conversion helpers: `fp32_round_to_fp4_code` and `fp32_round_to_fp4_value` replace unsupported native FP4 conversion with Triton arithmetic and `libdevice.round`.
- Inline asm is still used for non-FP4-specific packing and squared-error arithmetic. `_fp32x16_to_e2m1_u32x2` still uses inline asm to assemble bytes into two `uint32` words; `_fp32x2_e2m1_quant_squared_error` and `_fp32x2_e2m1_quant_squared_error_acc` still use inline asm for subtraction, multiply, and FMA after simulated FP4 values are produced.
- Public API: `scalesweep_mse_nvfp4_quant_simulate`; `scalesweep_mse_nvfp4_quant`, `scaled_fp4_quant`, and `scaled_fp4_quant_simulate` are aliases inside this module for drop-in use.

### `absmax_nvfp4_quant_simulate.py`

Pure PyTorch AbsMax simulate-FP4 quantizer used as a baseline in simulate benchmarks.

Internal organization:
- Imports common constants and allocation helpers from `scalesweep_mse_nvfp4_quant_simulate.py`.
- `_torch_round_to_fp4_code` implements simulated E2M1 FP4 code generation in PyTorch.
- `_swizzled_scale_indices` computes the same scale swizzle indices on tensors.
- `absmax_nvfp4_quant_simulate` normalizes by the global inverse scale, computes per-block absmax scale, quantizes the scale to FP8, packs two FP4 codes per byte, and optionally writes scales in swizzled layout.
- `scaled_fp4_quant_simulate` is an alias for the AbsMax simulate function.

### `test_scalesweep_mse_nvfp4_quant.py`

Pytest coverage for ScaleSweep MSE NVFP4. The same test file covers native FP4 by default and simulate-FP4 when pytest is run with `--simulate`.

Internal organization:
- Skips when CUDA is unavailable.
- Skips native runs on GPUs below compute capability 10.0 because native FP4 PTX is required.
- Defines PyTorch reference helpers for FP4 unpacking, native/simulate FP4 value rounding, global scale computation, swizzled-scale recovery, and full ScaleSweep MSE reference quantization.
- The reference selects candidate scales with the same absmax-derived raw FP8 base scale and `LOWER_BOUND..UPPER_BOUND` search window used by the Triton kernels, rather than enumerating every positive FP8 scale.
- Tests both `float16` and `bfloat16`, several shapes, and both swizzled and non-swizzled scale layouts.
- Includes the lower-bound and upper-bound examples from `../fp4_bound_example.py`.

### `conftest.py`

Pytest CLI and result-report helpers.

Internal organization:
- Adds `--simulate` to select the simulate-FP4 path.
- Adds `--result-path` to override the JSON result file.
- Writes JSON test results to `../result/test_scalesweep_mse_nvfp4_quant_native_results.json` or `../result/test_scalesweep_mse_nvfp4_quant_simulate_results.json` by default.

### `benchmark_scalesweep_mse_nvfp4_quant.py`

Benchmark for native and simulate ScaleSweep MSE NVFP4 quantization. Native is the default; pass `--simulate` for the simulate-FP4 benchmark.

Internal organization:
- Native mode requires device capability 10.0 or higher through `vllm.platforms.current_platform`.
- Collects environment information and driver/package versions.
- Provides helpers for global scale computation, swizzled-scale recovery, FP4 unpacking, dequantization, and error metrics.
- Native `quantize` dispatches between `vllm` and `scalesweep_mse`; `benchmark` is a Triton perf-report benchmark over model weight shapes from `weight_shapes.py`.
- Simulate mode compares `absmax_torch_simulate` against `scalesweep_mse_triton_simulate`.
- CLI arguments include native options (`--models`, `--tp-sizes`, `--error-batches`, `--skip-speed`, `--skip-error`) and simulate options (`--k`, `--batches`, `--bench-iters`).
- Default native output path: `../result/bench_scalesweep_mse_nvfp4_quant_results`.
- Default simulate output path: `../result/bench_scalesweep_mse_simulate_fp4_results.json`.

### `weight_shapes.py`

Model weight-shape catalog used by the native benchmark CLI.

Internal organization:
- Defines `WEIGHT_SHAPES`, keyed by model name.
- Each entry contains `[K, N]` plus the tensor-parallel dimension index to divide by the selected TP size.
- Used by `prepare_shapes` in `benchmark_scalesweep_mse_nvfp4_quant.py`.

### `__init__.py`

Package exports for the quantization helpers.

Internal organization:
- Exports common allocation helpers from the native module.
- Exports native APIs: `scalesweep_mse_nvfp4_quant` and `scaled_fp4_quant`.
- Exports simulate APIs: `scalesweep_mse_nvfp4_quant_simulate`, `scaled_fp4_quant_simulate`, and `absmax_nvfp4_quant_simulate`.
- Maintains `__all__` for explicit package-level imports.

## Running Tests

Run from this directory, because tests insert the directory itself on `sys.path` and import sibling modules directly:

```bash
cd scalesweep_mse
python -m pytest -q test_scalesweep_mse_nvfp4_quant.py
python -m pytest -q test_scalesweep_mse_nvfp4_quant.py --simulate
```

The native test requires compute capability 10.0 or newer. On older CUDA GPUs, it should skip.
Both commands write JSON results to `../result/` unless `--result-path` is provided.

## Running Benchmarks

Simulate-FP4 benchmark:

```bash
cd scalesweep_mse
python benchmark_scalesweep_mse_nvfp4_quant.py --simulate --bench-iters 50
```

Native benchmark:

```bash
cd scalesweep_mse
python benchmark_scalesweep_mse_nvfp4_quant.py
```

The native benchmark requires Blackwell-class FP4 support and vLLM native FP4 ops. The simulate benchmark is the path to use on pre-Blackwell CUDA GPUs.

## Maintenance Rule

When changing any file in this directory, update this README if the change affects:
- File names or added/removed files.
- Public functions, aliases, CLI arguments, or output files.
- Test assumptions, skip conditions, benchmark behavior, or required hardware.
- Internal organization described above, especially native-vs-simulate FP4 behavior.
