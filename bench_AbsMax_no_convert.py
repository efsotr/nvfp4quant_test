import argparse
import json
from pathlib import Path

import torch
import triton.testing as tts

from helper import (
    dequantize,
    error_stats,
    get_nvfp4_global_scale,
    make_w,
)
from kernel_AbsMax_no_convert import BLOCK_SIZE, absmax_quantize_no_convert

sm_count = torch.cuda.get_device_properties("cuda").multi_processor_count
bsz_list = [1, 8, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=8192)
    parser.add_argument("--fp8-max", type=float, default=448.0)
    parser.add_argument("--output", type=Path, default=Path("result/bench_AbsMax_no_convert_results.json"))
    return parser.parse_args()


def main():
    args = parse_args()

    results = {
        "name": "triton.AbsMax_no_convert",
        "block_size": BLOCK_SIZE,
        "fp8_max": args.fp8_max,
        "sm_count": sm_count,
        "dim": args.dim,
        "results": [],
    }

    for bsz in bsz_list:
        weight = make_w(bsz, args.dim)
        global_scale, global_scale_inv = get_nvfp4_global_scale(weight, FP8_MAX=args.fp8_max)

        fn = lambda: absmax_quantize_no_convert(
            weight,
            global_scale_inv,
            BLOCK_SIZE,
        )
        ms = tts.do_bench(fn, warmup=10, rep=100)
        scale, code = fn()

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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"saved results to {args.output}")


if __name__ == "__main__":
    main()
