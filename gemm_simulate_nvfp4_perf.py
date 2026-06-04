import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from bench import KERNEL_CASES, current_sm, should_skip_kernel
from helper import (
    DTYPE,
    SEED,
    DEVICE,
    dequantize,
    error_stats,
    get_environment_info,
    get_nvfp4_global_scale,
    make_laplace,
    make_w,
)
from kernels.kernel_ScaleSweep_MSE import BLOCK_SIZE
from kernels.kernel_vllm import unswizzle_vllm_fp4_scale


DIM = 8192
OUTPUT_DIR = Path("result")
OUTPUT_NAME = "gemm_simulate_nvfp4_perf_results.json"
SIMULATE_KERNELS = tuple(
    case for case in KERNEL_CASES if "simulate_fp4" in case.name and case.scale_layout == "swizzled"
)


def make_input(m: int, k: int) -> torch.Tensor:
    return make_laplace((m, k), seed=SEED + 1)


def make_channel_square_norm(x: torch.Tensor) -> torch.Tensor:
    return torch.mean(x.float() * x.float(), dim=0, keepdim=True).contiguous()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=DIM)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def write_results(results, args):
    output = args.output or args.output_dir / OUTPUT_NAME
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(results, indent=2) + "\n"
    output.write_text(text)
    print(text, end="")


def quantize_case(case, tensor, imp, global_scale_inv):
    if case.kind == "weighted":
        return case.quantize(
            tensor,
            imp,
            global_scale_inv,
            BLOCK_SIZE,
            case.lower_bound,
            case.upper_bound,
            is_swizzle=case.scale_layout == "swizzled",
        )
    if case.kind == "mse":
        return case.quantize(
            tensor,
            global_scale_inv,
            BLOCK_SIZE,
            case.lower_bound,
            case.upper_bound,
            is_swizzle=case.scale_layout == "swizzled",
        )
    if case.kind == "absmax":
        return case.quantize(
            tensor,
            global_scale_inv,
            BLOCK_SIZE,
            is_swizzle=case.scale_layout == "swizzled",
        )
    if case.kind == "vllm":
        from vllm._custom_ops import scaled_fp4_quant

        code, scale = scaled_fp4_quant(tensor, global_scale_inv)
        return scale, code
    raise ValueError(f"unknown kernel kind for {case.name}: {case.kind}")


def quantize_dequantize(case, tensor, imp):
    global_scale, global_scale_inv = get_nvfp4_global_scale(tensor, FP8_MAX=case.fp8_max)
    scale, code = quantize_case(case, tensor, imp, global_scale_inv)
    if case.scale_layout == "swizzled":
        scale = unswizzle_vllm_fp4_scale(
            scale,
            m=tensor.shape[0],
            n=tensor.shape[1],
            block_size=BLOCK_SIZE,
        )
    return dequantize("base", code, scale, global_scale).to(DTYPE)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run gemm_simulate_nvfp4_perf.py")

    sm = current_sm()
    sm_count = torch.cuda.get_device_properties("cuda").multi_processor_count

    x = make_input(args.dim, args.dim)
    weight = make_w(args.dim, args.dim)
    weight_imp = make_channel_square_norm(x)
    x_imp = make_channel_square_norm(weight)

    ref = F.linear(x, weight)

    results = {
        "name": "gemm_simulate_nvfp4_perf",
        "sm_count": sm_count,
        "sm": sm,
        "dim": args.dim,
        "mode": "simulate",
        "input_distribution": "Laplace(loc=0, scale=1)",
        "weight_distribution": "Laplace(loc=0, scale=1)",
        "channel_square_norm": "mean",
        "environment": get_environment_info(),
        "results": [],
    }

    for case in SIMULATE_KERNELS:
        skip_reason = should_skip_kernel(case, sm)
        if skip_reason is not None:
            results["results"].append(
                {
                    "kernel": case.name,
                    "status": "skip",
                    "latency_ms": None,
                    "mse": None,
                    "max_abs_error": None,
                    "reason": skip_reason,
                }
            )
            continue

        try:
            reconstructed_x = quantize_dequantize(case, x, x_imp)
            reconstructed_weight = quantize_dequantize(case, weight, weight_imp)
            pred = F.linear(reconstructed_x, reconstructed_weight)
            mse, max_abs_err = error_stats(ref, pred)

            results["results"].append(
                {
                    "kernel": case.name,
                    "status": "ok",
                    "latency_ms": None,
                    "mse": mse,
                    "max_abs_error": max_abs_err,
                    "reason": None,
                }
            )

            del reconstructed_x, reconstructed_weight, pred
            torch.cuda.empty_cache()
        except Exception as exc:
            results["results"].append(
                {
                    "kernel": case.name,
                    "status": "error",
                    "latency_ms": None,
                    "mse": None,
                    "max_abs_error": None,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    write_results(results, args)


if __name__ == "__main__":
    main()
