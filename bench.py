import argparse
import json
from pathlib import Path

import torch
import triton.testing as tts

from helper import (
    check_sm100,
    dequantize,
    error_stats,
    get_nvfp4_global_scale,
    make_imp,
    make_w,
    weighted_error_stats,
)
from kernel_AbsMax_no_convert import absmax_quantize_no_convert
from kernel_ScaleSweep import (
    LOWER_BOUND as SCALESWEEP_LOWER_BOUND,
    UPPER_BOUND as SCALESWEEP_UPPER_BOUND,
    scalesweep_quantize,
)
from kernel_ScaleSweep_MSE import (
    BLOCK_SIZE,
    LOWER_BOUND as SCALESWEEP_MSE_LOWER_BOUND,
    UPPER_BOUND as SCALESWEEP_MSE_UPPER_BOUND,
    scalesweep_quantize as mse_scalesweep_quantize,
)
from kernel_ScaleSweep_MSE_no_convert import (
    LOWER_BOUND as SCALESWEEP_MSE_NO_CONVERT_LOWER_BOUND,
    UPPER_BOUND as SCALESWEEP_MSE_NO_CONVERT_UPPER_BOUND,
    scalesweep_quantize as mse_scalesweep_no_convert_quantize,
)
from kernel_ScaleSweep_no_convert import (
    LOWER_BOUND as SCALESWEEP_NO_CONVERT_LOWER_BOUND,
    UPPER_BOUND as SCALESWEEP_NO_CONVERT_UPPER_BOUND,
    scalesweep_quantize as scalesweep_no_convert_quantize,
)
from kernel_vllm import unswizzle_vllm_fp4_scale


BSZ_LIST = [1, 8, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
WEIGHTED_BSZ_LIST = [1, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
BENCHMARKS = [
    "ScaleSweep",
    "ScaleSweep_MSE",
    "ScaleSweep_no_convert",
    "ScaleSweep_MSE_no_convert",
    "AbsMax_no_convert",
    "vllm",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmarks", nargs="*", choices=BENCHMARKS, default=BENCHMARKS)
    parser.add_argument("--dim", type=int, default=8192)
    parser.add_argument("--imp", type=str, choices=["ones", "ramp", "random"], default="ones")
    parser.add_argument("--fp8-max", type=float, default=448.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("result"))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=100)
    return parser.parse_args()


def write_results(results, args):
    output = args.output or args.output_dir / f"bench_{results['name'].removeprefix('triton.')}_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"saved results to {output}")


def benchmark_call(fn, args):
    ms = tts.do_bench(fn, warmup=args.warmup, rep=args.rep)
    return ms, fn()


def run_mse_benchmark(args, sm_count, *, no_convert):
    if no_convert:
        name = "triton.ScaleSweep_MSE_no_convert"
        quantize = mse_scalesweep_no_convert_quantize
        lower_bound = SCALESWEEP_MSE_NO_CONVERT_LOWER_BOUND
        upper_bound = SCALESWEEP_MSE_NO_CONVERT_UPPER_BOUND
    else:
        check_sm100()
        name = "triton.ScaleSweep_MSE"
        quantize = mse_scalesweep_quantize
        lower_bound = SCALESWEEP_MSE_LOWER_BOUND
        upper_bound = SCALESWEEP_MSE_UPPER_BOUND

    results = {
        "name": name,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "sm_count": sm_count,
        "dim": args.dim,
        "results": [],
    }

    for bsz in BSZ_LIST:
        weight = make_w(bsz, args.dim)
        global_scale, global_scale_inv = get_nvfp4_global_scale(weight, FP8_MAX=256)
        fn = lambda: quantize(weight, global_scale_inv, BLOCK_SIZE, lower_bound, upper_bound)
        ms, (scale, code) = benchmark_call(fn, args)

        reconstructed = dequantize("base", code, scale, global_scale)
        mse, max_abs_error = error_stats(weight, reconstructed)
        results["results"].append(
            {
                "bsz": bsz,
                "dim": weight.shape[1],
                "latency_ms": ms,
                "mse": mse,
                "max_abs_error": max_abs_error,
            }
        )

    return results


def run_weighted_benchmark(args, sm_count, *, no_convert):
    if no_convert:
        name = "triton.ScaleSweep_no_convert"
        quantize = scalesweep_no_convert_quantize
        fallback_quantize = mse_scalesweep_no_convert_quantize
        lower_bound = SCALESWEEP_NO_CONVERT_LOWER_BOUND
        upper_bound = SCALESWEEP_NO_CONVERT_UPPER_BOUND
    else:
        check_sm100()
        name = "triton.ScaleSweep"
        quantize = scalesweep_quantize
        fallback_quantize = None
        lower_bound = SCALESWEEP_LOWER_BOUND
        upper_bound = SCALESWEEP_UPPER_BOUND

    results = {
        "name": name,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "sm_count": sm_count,
        "dim": args.dim,
        "imp": args.imp,
        "results": [],
    }

    for bsz in WEIGHTED_BSZ_LIST:
        weight = make_w(bsz, args.dim)
        imp = make_imp(args.imp, weight.shape[1], weight.device)
        global_scale, global_scale_inv = get_nvfp4_global_scale(weight, FP8_MAX=256)

        if no_convert and args.imp == "ones":
            fn = lambda: fallback_quantize(
                weight,
                global_scale_inv,
                BLOCK_SIZE,
                lower_bound,
                upper_bound,
            )
        else:
            fn = lambda: quantize(
                weight,
                imp,
                global_scale_inv,
                BLOCK_SIZE,
                lower_bound,
                upper_bound,
            )
        ms, (scale, code) = benchmark_call(fn, args)

        reconstructed = dequantize("base", code, scale, global_scale)
        mse, max_abs_error = error_stats(weight, reconstructed)
        weighted_mse, _ = weighted_error_stats(weight, reconstructed, imp)
        results["results"].append(
            {
                "bsz": bsz,
                "dim": weight.shape[1],
                "latency_ms": ms,
                "mse": mse,
                "weighted_mse": weighted_mse,
                "max_abs_error": max_abs_error,
            }
        )

    return results


def run_absmax_benchmark(args, sm_count):
    results = {
        "name": "triton.AbsMax_no_convert",
        "block_size": BLOCK_SIZE,
        "fp8_max": args.fp8_max,
        "sm_count": sm_count,
        "dim": args.dim,
        "results": [],
    }

    for bsz in BSZ_LIST:
        weight = make_w(bsz, args.dim)
        global_scale, global_scale_inv = get_nvfp4_global_scale(weight, FP8_MAX=args.fp8_max)
        fn = lambda: absmax_quantize_no_convert(weight, global_scale_inv, BLOCK_SIZE)
        ms, (scale, code) = benchmark_call(fn, args)

        reconstructed = dequantize("base", code, scale, global_scale)
        mse, max_abs_error = error_stats(weight, reconstructed)
        results["results"].append(
            {
                "bsz": bsz,
                "dim": weight.shape[1],
                "latency_ms": ms,
                "mse": mse,
                "max_abs_error": max_abs_error,
            }
        )

    return results


def run_vllm_benchmark(args, sm_count):
    from vllm._custom_ops import scaled_fp4_quant

    check_sm100()
    results = {
        "name": "triton.vllm",
        "sm_count": sm_count,
        "dim": args.dim,
        "results": [],
    }

    for bsz in BSZ_LIST:
        weight = make_w(bsz, args.dim)
        global_scale, global_scale_inv = get_nvfp4_global_scale(weight)
        fn = lambda: scaled_fp4_quant(weight, global_scale_inv)
        ms, (code, scale) = benchmark_call(fn, args)
        scale = unswizzle_vllm_fp4_scale(scale, m=weight.shape[0], n=weight.shape[1], block_size=BLOCK_SIZE)

        reconstructed = dequantize("base", code, scale, global_scale)
        mse, max_abs_error = error_stats(weight, reconstructed)
        results["results"].append(
            {
                "bsz": bsz,
                "dim": weight.shape[1],
                "latency_ms": ms,
                "mse": mse,
                "max_abs_error": max_abs_error,
            }
        )

    return results


def run_benchmark(name, args, sm_count):
    if name == "ScaleSweep":
        return run_weighted_benchmark(args, sm_count, no_convert=False)
    if name == "ScaleSweep_MSE":
        return run_mse_benchmark(args, sm_count, no_convert=False)
    if name == "ScaleSweep_no_convert":
        return run_weighted_benchmark(args, sm_count, no_convert=True)
    if name == "ScaleSweep_MSE_no_convert":
        return run_mse_benchmark(args, sm_count, no_convert=True)
    if name == "AbsMax_no_convert":
        return run_absmax_benchmark(args, sm_count)
    if name == "vllm":
        return run_vllm_benchmark(args, sm_count)
    raise ValueError(f"unknown benchmark: {name}")


def main():
    args = parse_args()
    if args.output is not None and len(args.benchmarks) != 1:
        raise ValueError("--output can only be used when running one benchmark")

    sm_count = torch.cuda.get_device_properties("cuda").multi_processor_count

    for name in args.benchmarks:
        results = run_benchmark(name, args, sm_count)
        write_results(results, args)


if __name__ == "__main__":
    main()
