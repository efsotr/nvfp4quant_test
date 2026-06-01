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


def append_error_result(results, case, ms, scale, code, extra_metrics=None):
    weight = case["weight"]
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


def run_quantize_benchmark(args, results, bsz_list, make_case, quantize, extra_metrics=None):
    for bsz in bsz_list:
        case = make_case(bsz)
        ms, (scale, code) = benchmark_call(lambda: quantize(case), args)
        append_error_result(results, case, ms, scale, code, extra_metrics)
    return results


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

    results = make_results(
        name,
        args,
        sm_count,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )

    return run_quantize_benchmark(
        args,
        results,
        BSZ_LIST,
        lambda bsz: make_base_case(args, bsz, fp8_max=256),
        lambda case: quantize(
            case["weight"],
            case["global_scale_inv"],
            BLOCK_SIZE,
            lower_bound,
            upper_bound,
        ),
    )


def make_weighted_case(args, bsz):
    case = make_base_case(args, bsz, fp8_max=256)
    weight = case["weight"]
    case["imp"] = make_imp(args.imp, weight.shape[1], weight.device)
    return case


def weighted_metrics(case, reconstructed):
    weighted_mse, _ = weighted_error_stats(case["weight"], reconstructed, case["imp"])
    return {"weighted_mse": weighted_mse}


def run_weighted_benchmark(args, sm_count, *, no_convert):
    if no_convert:
        name = "triton.ScaleSweep_no_convert"
        quantize = scalesweep_no_convert_quantize
        lower_bound = SCALESWEEP_NO_CONVERT_LOWER_BOUND
        upper_bound = SCALESWEEP_NO_CONVERT_UPPER_BOUND
    else:
        check_sm100()
        name = "triton.ScaleSweep"
        quantize = scalesweep_quantize
        lower_bound = SCALESWEEP_LOWER_BOUND
        upper_bound = SCALESWEEP_UPPER_BOUND

    results = make_results(
        name,
        args,
        sm_count,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        imp=args.imp,
    )

    return run_quantize_benchmark(
        args,
        results,
        WEIGHTED_BSZ_LIST,
        lambda bsz: make_weighted_case(args, bsz),
        lambda case: quantize(
            case["weight"],
            case["imp"],
            case["global_scale_inv"],
            BLOCK_SIZE,
            lower_bound,
            upper_bound,
        ),
        weighted_metrics,
    )


def run_absmax_benchmark(args, sm_count):
    results = make_results(
        "triton.AbsMax_no_convert",
        args,
        sm_count,
        block_size=BLOCK_SIZE,
        fp8_max=args.fp8_max,
    )

    return run_quantize_benchmark(
        args,
        results,
        BSZ_LIST,
        lambda bsz: make_base_case(args, bsz, fp8_max=args.fp8_max),
        lambda case: absmax_quantize_no_convert(
            case["weight"],
            case["global_scale_inv"],
            BLOCK_SIZE,
        ),
    )


def run_vllm_benchmark(args, sm_count):
    from vllm._custom_ops import scaled_fp4_quant

    check_sm100()
    results = make_results("triton.vllm", args, sm_count)

    def quantize(case):
        code, scale = scaled_fp4_quant(case["weight"], case["global_scale_inv"])
        scale = unswizzle_vllm_fp4_scale(
            scale,
            m=case["weight"].shape[0],
            n=case["weight"].shape[1],
            block_size=BLOCK_SIZE,
        )
        return scale, code

    return run_quantize_benchmark(
        args,
        results,
        BSZ_LIST,
        lambda bsz: make_base_case(args, bsz, fp8_max=448.0),
        quantize,
    )


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
