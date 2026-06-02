# NVFP4 Benchmark Report

## Environment

| torch | vllm | cuda | gpu_driver | gpu | sm | sm_count | total_memory_bytes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2.11.0+cu130 | 0.22.0 | 13.0 | 580.95.05 | NVIDIA RTX PRO 6000 Blackwell Server Edition | 120 | 188 | 101974081536 |

## Result Files

| file | name | mode | sm | sm_count | dim | weight_distribution | input_distribution | channel_square_norm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bench_ScaleSweep_MSE_round_swizzled_results.json | triton.ScaleSweep_MSE_round_swizzled | - | 120 | 188 | 8192 | Laplace(loc=0, scale=1) | - | - |
| bench_ScaleSweep_MSE_swizzled_results.json | triton.ScaleSweep_MSE_swizzled | - | 120 | 188 | 8192 | Laplace(loc=0, scale=1) | - | - |
| bench_ScaleSweep_swizzled_results.json | triton.ScaleSweep_swizzled | - | 120 | 188 | 8192 | Laplace(loc=0, scale=1) | - | - |
| bench_vllm_results.json | triton.vllm | - | 120 | 188 | 8192 | Laplace(loc=0, scale=1) | - | - |
| gemm_nvfp4_perf_results.json | gemm_nvfp4_perf | native | 120 | 188 | 8192 | Laplace(loc=0, scale=1) | Laplace(loc=0, scale=1) | mean |

## GEMM Results

| mode | kernel | status | latency_ms | mse | max_abs_error | note |
| --- | --- | --- | --- | --- | --- | --- |
| native | vllm | ok | 0.720463 | 562.049 | 132 | - |
| native | ScaleSweep_MSE_swizzled | ok | 0.702919 | 444.243 | 117.812 | - |
| native | ScaleSweep_MSE_round_swizzled | ok | 0.702615 | 444.243 | 117.812 | - |
| native | ScaleSweep_swizzled | ok | 0.703141 | 444.177 | 115.625 | - |

## Quantization Bench Results

| kernel | layout | bsz | dim | latency_ms | mse | weighted_mse | max_abs_error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ScaleSweep_MSE_round_swizzled | swizzled | 1 | 8192 | 0.00633512 | 0.0134908 | - | 0.5625 |
| ScaleSweep_MSE_round_swizzled | swizzled | 8 | 8192 | 0.00779489 | 0.0137242 | - | 0.738281 |
| ScaleSweep_MSE_round_swizzled | swizzled | 64 | 8192 | 0.00929372 | 0.0136257 | - | 1 |
| ScaleSweep_MSE_round_swizzled | swizzled | 128 | 8192 | 0.0101168 | 0.0136095 | - | 1 |
| ScaleSweep_MSE_round_swizzled | swizzled | 256 | 8192 | 0.0126149 | 0.0135946 | - | 0.96875 |
| ScaleSweep_MSE_round_swizzled | swizzled | 512 | 8192 | 0.0168762 | 0.0135962 | - | 1.34375 |
| ScaleSweep_MSE_round_swizzled | swizzled | 1024 | 8192 | 0.0256528 | 0.0136004 | - | 1.34375 |
| ScaleSweep_MSE_round_swizzled | swizzled | 2048 | 8192 | 0.0414329 | 0.0136047 | - | 1.34375 |
| ScaleSweep_MSE_round_swizzled | swizzled | 4096 | 8192 | 0.0722104 | 0.0136001 | - | 1.34375 |
| ScaleSweep_MSE_round_swizzled | swizzled | 8192 | 8192 | 0.135609 | 0.0135951 | - | 1.34375 |
| ScaleSweep_MSE_swizzled | swizzled | 1 | 8192 | 0.0064172 | 0.0134908 | - | 0.5625 |
| ScaleSweep_MSE_swizzled | swizzled | 8 | 8192 | 0.00809576 | 0.0137242 | - | 0.738281 |
| ScaleSweep_MSE_swizzled | swizzled | 64 | 8192 | 0.00927953 | 0.0136257 | - | 1 |
| ScaleSweep_MSE_swizzled | swizzled | 128 | 8192 | 0.0100796 | 0.0136095 | - | 1 |
| ScaleSweep_MSE_swizzled | swizzled | 256 | 8192 | 0.0125797 | 0.0135946 | - | 0.96875 |
| ScaleSweep_MSE_swizzled | swizzled | 512 | 8192 | 0.0163772 | 0.0135962 | - | 1.34375 |
| ScaleSweep_MSE_swizzled | swizzled | 1024 | 8192 | 0.0256723 | 0.0136004 | - | 1.34375 |
| ScaleSweep_MSE_swizzled | swizzled | 2048 | 8192 | 0.0410426 | 0.0136047 | - | 1.34375 |
| ScaleSweep_MSE_swizzled | swizzled | 4096 | 8192 | 0.0726136 | 0.0136001 | - | 1.34375 |
| ScaleSweep_MSE_swizzled | swizzled | 8192 | 8192 | 0.135918 | 0.0135951 | - | 1.34375 |
| ScaleSweep_swizzled | swizzled | 1 | 8192 | 0.00852955 | 0.0134908 | 0.0134908 | 0.5625 |
| ScaleSweep_swizzled | swizzled | 8 | 8192 | 0.0087639 | 0.0137242 | 0.0137242 | 0.738281 |
| ScaleSweep_swizzled | swizzled | 64 | 8192 | 0.010818 | 0.0136257 | 0.0136257 | 1 |
| ScaleSweep_swizzled | swizzled | 128 | 8192 | 0.0129502 | 0.0136095 | 0.0136095 | 1 |
| ScaleSweep_swizzled | swizzled | 256 | 8192 | 0.0164173 | 0.0135946 | 0.0135946 | 0.96875 |
| ScaleSweep_swizzled | swizzled | 512 | 8192 | 0.0222497 | 0.0135962 | 0.0135962 | 1.34375 |
| ScaleSweep_swizzled | swizzled | 1024 | 8192 | 0.0327516 | 0.0136004 | 0.0136004 | 1.34375 |
| ScaleSweep_swizzled | swizzled | 2048 | 8192 | 0.0573654 | 0.0136047 | 0.0136047 | 1.34375 |
| ScaleSweep_swizzled | swizzled | 4096 | 8192 | 0.100374 | 0.0136001 | 0.0136001 | 1.34375 |
| ScaleSweep_swizzled | swizzled | 8192 | 8192 | 0.187389 | 0.0135951 | 0.0135951 | 1.34375 |
| vllm | swizzled | 1 | 8192 | 0.00584533 | 0.0170494 | - | 0.790179 |
| vllm | swizzled | 8 | 8192 | 0.00656058 | 0.0172774 | - | 0.964286 |
| vllm | swizzled | 64 | 8192 | 0.00802169 | 0.0172161 | - | 1.05804 |
| vllm | swizzled | 128 | 8192 | 0.00940061 | 0.0171915 | - | 1.25893 |
| vllm | swizzled | 256 | 8192 | 0.0122513 | 0.0171765 | - | 1.44048 |
| vllm | swizzled | 512 | 8192 | 0.0162843 | 0.0171593 | - | 1.80059 |
| vllm | swizzled | 1024 | 8192 | 0.0253864 | 0.0171738 | - | 1.80059 |
| vllm | swizzled | 2048 | 8192 | 0.0406222 | 0.0171768 | - | 1.80059 |
| vllm | swizzled | 4096 | 8192 | 0.0760823 | 0.0171713 | - | 1.80059 |
| vllm | swizzled | 8192 | 8192 | 0.134747 | 0.0171616 | - | 1.80059 |
