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
from kernel_ScaleSweep import (
    BLOCK_SIZE,
    LOWER_BOUND,
    UPPER_BOUND,
    scalesweep_quantize,
)

sm_count = torch.cuda.get_device_properties("cuda").multi_processor_count
bsz_list = [1, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=8192)
    parser.add_argument("--imp", type=str, choices=["ones", "ramp", "random"], default="ones")
    parser.add_argument("--output", type=Path, default=Path("result/bench_ScaleSweep_results.json"))
    return parser.parse_args()


def main():
    args = parse_args()

    check_sm100()
    results = {
        "name": "triton.ScaleSweep",
        "lower_bound": LOWER_BOUND,
        "upper_bound": UPPER_BOUND,
        "sm_count": sm_count,
        "dim": args.dim,
        "imp": args.imp,
        "results": [],
    }

    for bsz in bsz_list:
        weight = make_w(bsz, args.dim)
        imp = make_imp(args.imp, weight.shape[1], weight.device)
        global_scale, global_scale_inv = get_nvfp4_global_scale(weight, FP8_MAX=256)

        fn = lambda: scalesweep_quantize(
            weight,
            imp,
            global_scale_inv,
            BLOCK_SIZE,
            LOWER_BOUND,
            UPPER_BOUND,
        )
        ms = tts.do_bench(fn, warmup=10, rep=100)
        scale, code = fn()

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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"saved results to {args.output}")

if __name__ == "__main__":
    main()
