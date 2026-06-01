import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import triton.testing as tts

from helper import (
    dequantize,
    error_stats,
    get_nvfp4_global_scale,
    make_imp,
    make_w,
    weighted_error_stats,
)
from kernel_AbsMax_simulate_fp4 import absmax_quantize_simulate_fp4
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
from kernel_ScaleSweep_MSE_simulate_fp4 import (
    LOWER_BOUND as SCALESWEEP_MSE_SIMULATE_FP4_LOWER_BOUND,
    UPPER_BOUND as SCALESWEEP_MSE_SIMULATE_FP4_UPPER_BOUND,
    scalesweep_quantize as mse_scalesweep_simulate_fp4_quantize,
)
from kernel_ScaleSweep_simulate_fp4 import (
    LOWER_BOUND as SCALESWEEP_SIMULATE_FP4_LOWER_BOUND,
    UPPER_BOUND as SCALESWEEP_SIMULATE_FP4_UPPER_BOUND,
    scalesweep_quantize as scalesweep_simulate_fp4_quantize,
)
from kernel_vllm import unswizzle_vllm_fp4_scale


BSZ_LIST = [1, 8, 64, 128, 256, 512, 1024, 2048, 4096, 8192]


@dataclass(frozen=True)
class KernelCase:
    name: str
    result_name: str
    kind: str
    quantize: Callable | None = None
    lower_bound: int | None = None
    upper_bound: int | None = None
    fp8_max: float = 256.0
    min_sm: int | None = None
    scale_layout: str = "linear"


KERNEL_CASES = (
    KernelCase(
        name="vllm",
        result_name="triton.vllm",
        kind="vllm",
        fp8_max=448.0,
        min_sm=100,
        scale_layout="swizzled",
    ),
    KernelCase(
        name="ScaleSweep_MSE",
        result_name="triton.ScaleSweep_MSE",
        kind="mse",
        quantize=mse_scalesweep_quantize,
        lower_bound=SCALESWEEP_MSE_LOWER_BOUND,
        upper_bound=SCALESWEEP_MSE_UPPER_BOUND,
        min_sm=100,
    ),
    KernelCase(
        name="ScaleSweep_MSE_swizzled",
        result_name="triton.ScaleSweep_MSE_swizzled",
        kind="mse",
        quantize=mse_scalesweep_quantize,
        lower_bound=SCALESWEEP_MSE_LOWER_BOUND,
        upper_bound=SCALESWEEP_MSE_UPPER_BOUND,
        min_sm=100,
        scale_layout="swizzled",
    ),
    KernelCase(
        name="ScaleSweep",
        result_name="triton.ScaleSweep",
        kind="weighted",
        quantize=scalesweep_quantize,
        lower_bound=SCALESWEEP_LOWER_BOUND,
        upper_bound=SCALESWEEP_UPPER_BOUND,
        min_sm=100,
    ),
    KernelCase(
        name="ScaleSweep_swizzled",
        result_name="triton.ScaleSweep_swizzled",
        kind="weighted",
        quantize=scalesweep_quantize,
        lower_bound=SCALESWEEP_LOWER_BOUND,
        upper_bound=SCALESWEEP_UPPER_BOUND,
        min_sm=100,
        scale_layout="swizzled",
    ),
    KernelCase(
        name="AbsMax_simulate_fp4",
        result_name="triton.AbsMax_simulate_fp4",
        kind="absmax",
        quantize=absmax_quantize_simulate_fp4,
        fp8_max=448.0,
    ),
    KernelCase(
        name="ScaleSweep_MSE_simulate_fp4",
        result_name="triton.ScaleSweep_MSE_simulate_fp4",
        kind="mse",
        quantize=mse_scalesweep_simulate_fp4_quantize,
        lower_bound=SCALESWEEP_MSE_SIMULATE_FP4_LOWER_BOUND,
        upper_bound=SCALESWEEP_MSE_SIMULATE_FP4_UPPER_BOUND,
    ),
    KernelCase(
        name="ScaleSweep_MSE_simulate_fp4_swizzled",
        result_name="triton.ScaleSweep_MSE_simulate_fp4_swizzled",
        kind="mse",
        quantize=mse_scalesweep_simulate_fp4_quantize,
        lower_bound=SCALESWEEP_MSE_SIMULATE_FP4_LOWER_BOUND,
        upper_bound=SCALESWEEP_MSE_SIMULATE_FP4_UPPER_BOUND,
        scale_layout="swizzled",
    ),
    KernelCase(
        name="ScaleSweep_simulate_fp4",
        result_name="triton.ScaleSweep_simulate_fp4",
        kind="weighted",
        quantize=scalesweep_simulate_fp4_quantize,
        lower_bound=SCALESWEEP_SIMULATE_FP4_LOWER_BOUND,
        upper_bound=SCALESWEEP_SIMULATE_FP4_UPPER_BOUND,
    ),
    KernelCase(
        name="ScaleSweep_simulate_fp4_swizzled",
        result_name="triton.ScaleSweep_simulate_fp4_swizzled",
        kind="weighted",
        quantize=scalesweep_simulate_fp4_quantize,
        lower_bound=SCALESWEEP_SIMULATE_FP4_LOWER_BOUND,
        upper_bound=SCALESWEEP_SIMULATE_FP4_UPPER_BOUND,
        scale_layout="swizzled",
    ),
)
KERNEL_CASES_BY_NAME = {case.name: case for case in KERNEL_CASES}
BENCHMARKS = list(KERNEL_CASES_BY_NAME)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmarks", nargs="*")
    parser.add_argument("--dim", type=int, default=8192)
    parser.add_argument("--imp", type=str, choices=["ones", "ramp", "random"], default="ones")
    parser.add_argument("--fp8-max", type=float, default=448.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("result"))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=100)
    args = parser.parse_args()
    if not args.benchmarks:
        args.benchmarks = BENCHMARKS
    unknown_benchmarks = sorted(set(args.benchmarks) - set(BENCHMARKS))
    if unknown_benchmarks:
        parser.error(
            "argument benchmarks: invalid choice(s): "
            f"{', '.join(unknown_benchmarks)} "
            f"(choose from {', '.join(BENCHMARKS)})"
        )
    return args


def write_results(results, args):
    output = args.output or args.output_dir / f"bench_{results['name'].removeprefix('triton.')}_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"saved results to {output}")


def current_sm() -> int:
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def should_skip_kernel(kernel: KernelCase, sm: int) -> str | None:
    if kernel.min_sm is not None and sm < kernel.min_sm:
        return f"requires sm_{kernel.min_sm}, current device is sm_{sm}"
    return None


def benchmark_call(fn, args):
    ms = tts.do_bench(fn, warmup=args.warmup, rep=args.rep)
    return ms, fn()


def make_results(name, args, sm_count, **metadata):
    results = {
        "name": name,
        "sm_count": sm_count,
        "dim": args.dim,
        "results": [],
    }
    results.update(metadata)
    return results


def make_base_case(args, bsz, *, fp8_max):
    weight = make_w(bsz, args.dim)
    global_scale, global_scale_inv = get_nvfp4_global_scale(weight, FP8_MAX=fp8_max)
    return {
        "weight": weight,
        "global_scale": global_scale,
        "global_scale_inv": global_scale_inv,
    }


def scale_for_dequant(kernel, weight, scale):
    if kernel.scale_layout == "linear":
        return scale
    if kernel.scale_layout == "swizzled":
        return unswizzle_vllm_fp4_scale(
            scale,
            m=weight.shape[0],
            n=weight.shape[1],
            block_size=BLOCK_SIZE,
        )
    raise ValueError(f"unknown scale layout for {kernel.name}: {kernel.scale_layout}")


def append_error_result(results, kernel, case, ms, scale, code, extra_metrics=None):
    weight = case["weight"]
    scale = scale_for_dequant(kernel, weight, scale)
    reconstructed = dequantize("base", code, scale, case["global_scale"])
    mse, max_abs_error = error_stats(weight, reconstructed)

    row = {
        "bsz": weight.shape[0],
        "dim": weight.shape[1],
        "latency_ms": ms,
        "mse": mse,
        "max_abs_error": max_abs_error,
    }
    if extra_metrics is not None:
        row.update(extra_metrics(case, reconstructed))
    results["results"].append(row)


def run_quantize_benchmark(args, results, kernel, bsz_list, make_case, quantize, extra_metrics=None):
    for bsz in bsz_list:
        case = make_case(bsz)
        ms, (scale, code) = benchmark_call(lambda: quantize(case), args)
        append_error_result(results, kernel, case, ms, scale, code, extra_metrics)
    return results


def run_mse_benchmark(case, args, sm_count):
    results = make_results(
        case.result_name,
        args,
        sm_count,
        lower_bound=case.lower_bound,
        upper_bound=case.upper_bound,
        scale_layout=case.scale_layout,
    )

    return run_quantize_benchmark(
        args,
        results,
        case,
        BSZ_LIST,
        lambda bsz: make_base_case(args, bsz, fp8_max=case.fp8_max),
        lambda bench_case: case.quantize(
            bench_case["weight"],
            bench_case["global_scale_inv"],
            BLOCK_SIZE,
            case.lower_bound,
            case.upper_bound,
            is_swizzle=case.scale_layout == "swizzled",
        ),
    )


def make_weighted_case(args, bsz, *, fp8_max):
    case = make_base_case(args, bsz, fp8_max=fp8_max)
    weight = case["weight"]
    case["imp"] = make_imp(args.imp, weight.shape[1], weight.device)
    return case


def weighted_metrics(case, reconstructed):
    weighted_mse, _ = weighted_error_stats(case["weight"], reconstructed, case["imp"])
    return {"weighted_mse": weighted_mse}


def run_weighted_benchmark(case, args, sm_count):
    results = make_results(
        case.result_name,
        args,
        sm_count,
        lower_bound=case.lower_bound,
        upper_bound=case.upper_bound,
        imp=args.imp,
        scale_layout=case.scale_layout,
    )

    return run_quantize_benchmark(
        args,
        results,
        case,
        BSZ_LIST,
        lambda bsz: make_weighted_case(args, bsz, fp8_max=case.fp8_max),
        lambda bench_case: case.quantize(
            bench_case["weight"],
            bench_case["imp"],
            bench_case["global_scale_inv"],
            BLOCK_SIZE,
            case.lower_bound,
            case.upper_bound,
            is_swizzle=case.scale_layout == "swizzled",
        ),
        weighted_metrics,
    )


def run_absmax_benchmark(case, args, sm_count):
    results = make_results(
        case.result_name,
        args,
        sm_count,
        block_size=BLOCK_SIZE,
        fp8_max=args.fp8_max,
        scale_layout=case.scale_layout,
    )

    return run_quantize_benchmark(
        args,
        results,
        case,
        BSZ_LIST,
        lambda bsz: make_base_case(args, bsz, fp8_max=args.fp8_max),
        lambda bench_case: case.quantize(
            bench_case["weight"],
            bench_case["global_scale_inv"],
            BLOCK_SIZE,
        ),
    )


def run_vllm_benchmark(case, args, sm_count):
    from vllm._custom_ops import scaled_fp4_quant

    results = make_results(case.result_name, args, sm_count, scale_layout=case.scale_layout)

    def quantize(case):
        code, scale = scaled_fp4_quant(case["weight"], case["global_scale_inv"])
        return scale, code

    return run_quantize_benchmark(
        args,
        results,
        case,
        BSZ_LIST,
        lambda bsz: make_base_case(args, bsz, fp8_max=case.fp8_max),
        quantize,
    )


def run_benchmark(name, args, sm_count, sm):
    case = KERNEL_CASES_BY_NAME[name]
    skip_reason = should_skip_kernel(case, sm)
    if skip_reason is not None:
        print(f"skipping {case.name}: {skip_reason}")
        return None
    if case.kind == "weighted":
        return run_weighted_benchmark(case, args, sm_count)
    if case.kind == "mse":
        return run_mse_benchmark(case, args, sm_count)
    if case.kind == "absmax":
        return run_absmax_benchmark(case, args, sm_count)
    if case.kind == "vllm":
        return run_vllm_benchmark(case, args, sm_count)
    raise ValueError(f"unknown benchmark kind for {name}: {case.kind}")


def main():
    args = parse_args()
    if args.output is not None and len(args.benchmarks) != 1:
        raise ValueError("--output can only be used when running one benchmark")

    sm_count = torch.cuda.get_device_properties("cuda").multi_processor_count
    sm = current_sm()

    for name in args.benchmarks:
        results = run_benchmark(name, args, sm_count, sm)
        if results is None:
            continue
        write_results(results, args)


if __name__ == "__main__":
    main()
