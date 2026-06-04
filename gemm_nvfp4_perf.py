import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import triton.testing as tts

from bench import KERNEL_CASES, current_sm, should_skip_kernel
from helper import DTYPE, SEED, DEVICE, error_stats, get_environment_info, get_nvfp4_global_scale, make_laplace, make_w
from kernels.kernel_ScaleSweep_MSE import BLOCK_SIZE


DIM = 8192
OUTPUT_DIR = Path("result")
OUTPUT_NAME = "gemm_nvfp4_perf_results.json"
WARMUP = 10
REP = 100
NATIVE_KERNELS = tuple(
    case for case in KERNEL_CASES if "simulate_fp4" not in case.name and case.scale_layout == "swizzled"
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
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--rep", type=int, default=REP)
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
            is_swizzle=True,
        )
    if case.kind == "mse":
        return case.quantize(
            tensor,
            global_scale_inv,
            BLOCK_SIZE,
            case.lower_bound,
            case.upper_bound,
            is_swizzle=True,
        )
    if case.kind == "vllm":
        from vllm._custom_ops import scaled_fp4_quant

        code, scale = scaled_fp4_quant(tensor, global_scale_inv)
        return scale, code
    raise ValueError(f"unknown kernel kind for {case.name}: {case.kind}")


def quantize_for_cutlass(case, tensor, imp):
    _, global_scale_inv = get_nvfp4_global_scale(tensor, FP8_MAX=case.fp8_max)
    scale, code = quantize_case(case, tensor, imp, global_scale_inv)
    return code, scale, global_scale_inv


def run_cutlass_gemm(
    x_code,
    weight_code,
    x_scale,
    weight_scale,
    x_global_scale_inv,
    weight_global_scale_inv,
):
    from vllm._custom_ops import cutlass_scaled_fp4_mm

    alpha = torch.reciprocal(x_global_scale_inv * weight_global_scale_inv)
    return cutlass_scaled_fp4_mm(
        x_code,
        weight_code,
        x_scale,
        weight_scale,
        alpha,
        DTYPE,
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run gemm_nvfp4_perf.py")

    sm = current_sm()
    sm_count = torch.cuda.get_device_properties("cuda").multi_processor_count
    results = {
        "name": "gemm_nvfp4_perf",
        "sm_count": sm_count,
        "sm": sm,
        "dim": args.dim,
        "mode": "native",
        "input_distribution": "Laplace(loc=0, scale=1)",
        "weight_distribution": "Laplace(loc=0, scale=1)",
        "channel_square_norm": "mean",
        "environment": get_environment_info(),
        "warmup": args.warmup,
        "rep": args.rep,
        "results": [],
    }
    if sm < 100:
        skip_reason = f"cutlass_scaled_fp4_mm requires sm_100, current device is sm_{sm}"
        for case in NATIVE_KERNELS:
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
        write_results(results, args)
        return

    x = make_input(args.dim, args.dim)
    weight = make_w(args.dim, args.dim)
    weight_imp = make_channel_square_norm(x)
    x_imp = make_channel_square_norm(weight)

    ref = F.linear(x, weight)

    for case in NATIVE_KERNELS:
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
            x_code, x_scale, x_global_scale_inv = quantize_for_cutlass(case, x, x_imp)
            weight_code, weight_scale, weight_global_scale_inv = quantize_for_cutlass(
                case,
                weight,
                weight_imp,
            )

            def gemm():
                return run_cutlass_gemm(
                    x_code,
                    weight_code,
                    x_scale,
                    weight_scale,
                    x_global_scale_inv,
                    weight_global_scale_inv,
                )

            latency_ms = tts.do_bench(gemm, warmup=args.warmup, rep=args.rep)
            pred = gemm()
            mse, max_abs_err = error_stats(ref, pred)

            results["results"].append(
                {
                    "kernel": case.name,
                    "status": "ok",
                    "latency_ms": latency_ms,
                    "mse": mse,
                    "max_abs_error": max_abs_err,
                    "reason": None,
                }
            )

            del x_code, x_scale, weight_code, weight_scale, pred
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
