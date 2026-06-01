import argparse
import json
from pathlib import Path

import torch
from vllm._custom_ops import scaled_fp4_quant
import triton.testing as tts

from helper import (
    check_sm100,
    make_w,
    get_nvfp4_global_scale,
    error_stats,
    dequantize,
)

def round_up(x, y):
    return (x + y - 1) // y * y


def unswizzle_vllm_fp4_scale(scale_swizzled, m, n, block_size=16):
    assert n % block_size == 0

    k = n // block_size
    m_pad = round_up(m, 128)
    k_pad = round_up(k, 4)

    scale_swizzled = scale_swizzled.reshape(m_pad, k_pad)

    scale = scale_swizzled.reshape(
        m_pad // 128,
        k_pad // 4,
        32,
        4,
        4,
    )
    scale = scale.permute(0, 3, 2, 1, 4).contiguous()
    scale = scale.reshape(m_pad, k_pad)

    return scale[:m, :k].contiguous()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=8192)
    parser.add_argument("--output", type=Path, default=Path("result/bench_vllm_results.json"))
    return parser.parse_args()

def main():
    args = parse_args()
    check_sm100()
    results = {
        "name": "vllm._custom_ops.scaled_fp4_quant",
        "dim": args.dim,
        "results": [],
    }

    for bsz in [1, 8, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
        W = make_w(bsz, args.dim)
        global_scale, global_scale_inv = get_nvfp4_global_scale(W)

        fn = lambda: scaled_fp4_quant(W, global_scale_inv)
        ms = tts.do_bench(fn, warmup=10, rep=100)
        q, s = fn()
        s = unswizzle_vllm_fp4_scale(s, m=W.shape[0], n=W.shape[1], block_size=16)

        W_hat = dequantize("base", q, s, global_scale)

        mse, max_abs_error = error_stats(W, W_hat)

        results["results"].append(
            {
                "bsz": bsz,
                "dim": W.shape[1],
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
