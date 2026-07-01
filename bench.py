import argparse
import json
from pathlib import Path

import torch
import triton.testing as tts

from helper import (
    dequantize,
    error_stats,
    get_environment_info,
    get_nvfp4_global_scale,
    make_imp,
    make_w,
    weighted_error_stats,
)
from kernels.kernel_ScaleSweep import (
    LOWER_BOUND as WEIGHTED_LOWER_BOUND,
    UPPER_BOUND as WEIGHTED_UPPER_BOUND,
    scalesweep_weighted_mse_nvfp4_quant_impl,
)
from kernels.kernel_ScaleSweep_MSE import (
    BLOCK_SIZE,
    LOWER_BOUND as MSE_LOWER_BOUND,
    UPPER_BOUND as MSE_UPPER_BOUND,
    scalesweep_mse_nvfp4_quant_impl,
)
from kernels.kernel_vllm import unswizzle_vllm_fp4_scale


BSZ_LIST = [1, 8, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
BENCHMARKS = ["vllm", "ScaleSweep_MSE", "ScaleSweep"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmarks", nargs="*", choices=BENCHMARKS)
    parser.add_argument("--dim", type=int, default=8192)
    parser.add_argument("--imp", choices=["ones", "ramp", "random"], default="ones")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("result"))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=100)
    args = parser.parse_args()
    if not args.benchmarks:
        args.benchmarks = BENCHMARKS
    if args.output is not None and len(args.benchmarks) != 1:
        parser.error("--output can only be used when running one benchmark")
    return args


def current_sm():
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def make_case(bsz, dim, fp8_max, imp_kind=None):
    weight = make_w(bsz, dim)
    global_scale, global_scale_inv = get_nvfp4_global_scale(weight, FP8_MAX=fp8_max)
    case = {
        "weight": weight,
        "global_scale": global_scale,
        "global_scale_inv": global_scale_inv,
    }
    if imp_kind is not None:
        case["imp"] = make_imp(imp_kind, dim, weight.device)
    return case


def unswizzle_scale(scale, weight):
    return unswizzle_vllm_fp4_scale(
        scale,
        m=weight.shape[0],
        n=weight.shape[1],
        block_size=BLOCK_SIZE,
    )


def bench_once(fn, args):
    ms = tts.do_bench(fn, warmup=args.warmup, rep=args.rep)
    return ms, fn()


def make_results(name, args, sm_count, **extra):
    result = {
        "name": f"triton.{name}",
        "sm_count": sm_count,
        "dim": args.dim,
        "scale_layout": "swizzled",
        "weight_distribution": "Laplace(loc=0, scale=1)",
        "environment": get_environment_info(),
        "results": [],
    }
    result.update(extra)
    return result


def add_error_row(results, case, ms, code, scale, *, weighted=False):
    weight = case["weight"]
    scale = unswizzle_scale(scale, weight)
    reconstructed = dequantize("base", code, scale, case["global_scale"])
    mse, max_abs_error = error_stats(weight, reconstructed)

    row = {
        "bsz": weight.shape[0],
        "dim": weight.shape[1],
        "latency_ms": ms,
        "mse": mse,
        "max_abs_error": max_abs_error,
    }
    if weighted:
        weighted_mse, _ = weighted_error_stats(weight, reconstructed, case["imp"])
        row["weighted_mse"] = weighted_mse
    results["results"].append(row)


def run_vllm(args, sm_count):
    from vllm._custom_ops import scaled_fp4_quant

    results = make_results("vllm", args, sm_count)
    for bsz in BSZ_LIST:
        case = make_case(bsz, args.dim, fp8_max=448.0)
        ms, (code, scale) = bench_once(
            lambda: scaled_fp4_quant(case["weight"], case["global_scale_inv"]),
            args,
        )
        add_error_row(results, case, ms, code, scale)
    return results


def run_mse(args, sm_count):
    results = make_results(
        "ScaleSweep_MSE",
        args,
        sm_count,
        lower_bound=MSE_LOWER_BOUND,
        upper_bound=MSE_UPPER_BOUND,
    )
    for bsz in BSZ_LIST:
        case = make_case(bsz, args.dim, fp8_max=256.0)
        ms, (code, scale) = bench_once(
            lambda: scalesweep_mse_nvfp4_quant_impl(
                case["weight"],
                case["global_scale_inv"],
                is_sf_swizzled_layout=True,
            ),
            args,
        )
        add_error_row(results, case, ms, code, scale)
    return results


def run_weighted(args, sm_count):
    results = make_results(
        "ScaleSweep",
        args,
        sm_count,
        lower_bound=WEIGHTED_LOWER_BOUND,
        upper_bound=WEIGHTED_UPPER_BOUND,
        imp=args.imp,
    )
    for bsz in BSZ_LIST:
        case = make_case(bsz, args.dim, fp8_max=256.0, imp_kind=args.imp)
        ms, (code, scale) = bench_once(
            lambda: scalesweep_weighted_mse_nvfp4_quant_impl(
                case["weight"],
                case["imp"],
                case["global_scale_inv"],
                is_sf_swizzled_layout=True,
            ),
            args,
        )
        add_error_row(results, case, ms, code, scale, weighted=True)
    return results


def write_results(results, args):
    name = results["name"].removeprefix("triton.")
    output = args.output or args.output_dir / f"bench_{name}_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"saved results to {output}")


def main():
    args = parse_args()
    sm = current_sm()
    sm_count = torch.cuda.get_device_properties("cuda").multi_processor_count

    if sm < 100:
        raise RuntimeError(f"these NVFP4 kernels require sm_100+, got sm_{sm}")

    runners = {
        "vllm": run_vllm,
        "ScaleSweep_MSE": run_mse,
        "ScaleSweep": run_weighted,
    }
    for name in args.benchmarks:
        write_results(runners[name](args, sm_count), args)


if __name__ == "__main__":
    main()
