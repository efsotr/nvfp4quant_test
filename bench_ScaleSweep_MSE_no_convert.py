import argparse
import json
from pathlib import Path

from helper import (
    dequantize,
    error_stats,
    get_nvfp4_global_scale,
    make_w,
)
from bench_ScaleSweep_MSE import (
    SCALESWEEP_CONFIGS,
    _load_normalized_16_cols,
    _max_abs_16,
)

import torch
sm_count = torch.cuda.get_device_properties("cuda").multi_processor_count
bsz_list = [1, 8, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

import triton
import triton.language as tl
from triton.language.extra import libdevice

BLOCK_SIZE = 16
LOWER_BOUND = -3
UPPER_BOUND = 7


@triton.jit
def fp32_round_to_fp4_code(x):
    ax = tl.abs(x)
    le2 = ax <= 2.0
    le4 = ax <= 4.0
    exp = tl.where(le2, 0.5, tl.where(le4, 1.0, 2.0))
    r = libdevice.round(ax / exp)
    mag = tl.where(le2, r, tl.where(le4, r + 2.0, tl.minimum(r + 4.0, 7.0))).to(tl.uint8)
    sign = (x < 0.0).to(tl.uint8) << 3
    return mag | sign


@triton.jit
def fp32_round_to_fp4_value(x):
    ax = tl.abs(x)
    exp = tl.where(ax <= 2.0, 0.5, tl.where(ax <= 4.0, 1.0, 2.0))
    q = libdevice.round(x / exp) * exp
    return tl.minimum(tl.maximum(q, -6.0), 6.0)


@triton.jit
def _fp32x16_to_e2m1_u32x2(
    x0, x1, x2, x3,
    x4, x5, x6, x7,
    x8, x9, x10, x11,
    x12, x13, x14, x15,
):
    b0 = fp32_round_to_fp4_code(x0).to(tl.uint32) | (fp32_round_to_fp4_code(x1).to(tl.uint32) << 4)
    b1 = fp32_round_to_fp4_code(x2).to(tl.uint32) | (fp32_round_to_fp4_code(x3).to(tl.uint32) << 4)
    b2 = fp32_round_to_fp4_code(x4).to(tl.uint32) | (fp32_round_to_fp4_code(x5).to(tl.uint32) << 4)
    b3 = fp32_round_to_fp4_code(x6).to(tl.uint32) | (fp32_round_to_fp4_code(x7).to(tl.uint32) << 4)
    b4 = fp32_round_to_fp4_code(x8).to(tl.uint32) | (fp32_round_to_fp4_code(x9).to(tl.uint32) << 4)
    b5 = fp32_round_to_fp4_code(x10).to(tl.uint32) | (fp32_round_to_fp4_code(x11).to(tl.uint32) << 4)
    b6 = fp32_round_to_fp4_code(x12).to(tl.uint32) | (fp32_round_to_fp4_code(x13).to(tl.uint32) << 4)
    b7 = fp32_round_to_fp4_code(x14).to(tl.uint32) | (fp32_round_to_fp4_code(x15).to(tl.uint32) << 4)

    lo = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
    hi = b4 | (b5 << 8) | (b6 << 16) | (b7 << 24)
    return lo, hi


@triton.jit
def _fp32_pair_e2m1_roundtrip_squared_error(x0, x1):
    q0 = fp32_round_to_fp4_value(x0)
    q1 = fp32_round_to_fp4_value(x1)
    d0 = q0 - x0
    d1 = q1 - x1
    return d0 * d0 + d1 * d1


@triton.jit
def _fp32x16_e2m1_roundtrip_squared_error(
    x0, x1, x2, x3,
    x4, x5, x6, x7,
    x8, x9, x10, x11,
    x12, x13, x14, x15,
):
    s01 = _fp32_pair_e2m1_roundtrip_squared_error(x0, x1)
    s23 = _fp32_pair_e2m1_roundtrip_squared_error(x2, x3)
    s45 = _fp32_pair_e2m1_roundtrip_squared_error(x4, x5)
    s67 = _fp32_pair_e2m1_roundtrip_squared_error(x6, x7)
    s89 = _fp32_pair_e2m1_roundtrip_squared_error(x8, x9)
    sAB = _fp32_pair_e2m1_roundtrip_squared_error(x10, x11)
    sCD = _fp32_pair_e2m1_roundtrip_squared_error(x12, x13)
    sEF = _fp32_pair_e2m1_roundtrip_squared_error(x14, x15)

    return ((s01 + s23) + (s45 + s67)) + ((s89 + sAB) + (sCD + sEF))


@triton.jit
def _squared_error_16_cols_after_e2m1_roundtrip(
    v0, v1, v2, v3,
    v4, v5, v6, v7,
    v8, v9, v10, v11,
    v12, v13, v14, v15,
    inv_scale,
    scale,
):
    x0 = v0 * inv_scale
    x1 = v1 * inv_scale
    x2 = v2 * inv_scale
    x3 = v3 * inv_scale
    x4 = v4 * inv_scale
    x5 = v5 * inv_scale
    x6 = v6 * inv_scale
    x7 = v7 * inv_scale
    x8 = v8 * inv_scale
    x9 = v9 * inv_scale
    x10 = v10 * inv_scale
    x11 = v11 * inv_scale
    x12 = v12 * inv_scale
    x13 = v13 * inv_scale
    x14 = v14 * inv_scale
    x15 = v15 * inv_scale

    squared_error = _fp32x16_e2m1_roundtrip_squared_error(
        x0, x1, x2, x3,
        x4, x5, x6, x7,
        x8, x9, x10, x11,
        x12, x13, x14, x15,
    )
    return squared_error * (scale * scale)


@triton.jit
def _pack_final_code_16_cols(
    v0, v1, v2, v3,
    v4, v5, v6, v7,
    v8, v9, v10, v11,
    v12, v13, v14, v15,
    inv_scale,
):
    x0 = v0 * inv_scale
    x1 = v1 * inv_scale
    x2 = v2 * inv_scale
    x3 = v3 * inv_scale
    x4 = v4 * inv_scale
    x5 = v5 * inv_scale
    x6 = v6 * inv_scale
    x7 = v7 * inv_scale
    x8 = v8 * inv_scale
    x9 = v9 * inv_scale
    x10 = v10 * inv_scale
    x11 = v11 * inv_scale
    x12 = v12 * inv_scale
    x13 = v13 * inv_scale
    x14 = v14 * inv_scale
    x15 = v15 * inv_scale

    return _fp32x16_to_e2m1_u32x2(
        x0, x1, x2, x3,
        x4, x5, x6, x7,
        x8, x9, x10, x11,
        x12, x13, x14, x15,
    )


@triton.autotune(
    configs=SCALESWEEP_CONFIGS,
    key=["NUM_BLOCKS", "LOWER_BOUND", "NUM_CANDIDATES"],
)
@triton.jit
def scalesweep_quantize_kernel(
    weight_ptr,
    scale_ptr,
    code_i32_ptr,
    global_scale_inv_ptr,
    NUM_BLOCKS: tl.constexpr,
    LOWER_BOUND: tl.constexpr,
    NUM_CANDIDATES: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    global_scale_inv = tl.load(global_scale_inv_ptr)

    pid = tl.program_id(0)
    block_start = pid * BLOCKS_PER_PROGRAM

    block_offsets = block_start + tl.arange(0, BLOCKS_PER_PROGRAM)
    block_mask = block_offsets < NUM_BLOCKS

    (
        v0, v1, v2, v3,
        v4, v5, v6, v7,
        v8, v9, v10, v11,
        v12, v13, v14, v15,
    ) = _load_normalized_16_cols(
        weight_ptr,
        block_offsets,
        block_mask,
        global_scale_inv,
    )
    
    abs_max = _max_abs_16(
        v0, v1, v2, v3,
        v4, v5, v6, v7,
        v8, v9, v10, v11,
        v12, v13, v14, v15,
    )

    base_scale = abs_max * (1.0 / 6.0)
    base_fp8 = base_scale.to(tl.float8e4nv)
    base_raw = base_fp8.to(tl.uint8, bitcast=True).to(tl.int32) - (
        base_fp8.to(tl.float32) > base_scale
    ).to(tl.int32)

    best_mse = tl.full((BLOCKS_PER_PROGRAM,), float("inf"), tl.float32)
    best_scale_fp8 = tl.full((BLOCKS_PER_PROGRAM,), 0, tl.float8e4nv)

    for i in tl.range(0, NUM_CANDIDATES, loop_unroll_factor=1):
        raw_i = tl.minimum(
            tl.maximum(base_raw + LOWER_BOUND + i, 1),
            126,
        ).to(tl.uint8)

        scale_fp8 = raw_i.to(tl.float8e4nv, bitcast=True)
        scale_i = scale_fp8.to(tl.float32)
        inv_scale_i = 1.0 / scale_i

        mse_i = _squared_error_16_cols_after_e2m1_roundtrip(
            v0, v1, v2, v3,
            v4, v5, v6, v7,
            v8, v9, v10, v11,
            v12, v13, v14, v15,
            inv_scale_i,
            scale_i,
        )

        better = mse_i < best_mse
        best_mse = tl.where(better, mse_i, best_mse)
        best_scale_fp8 = tl.where(better, scale_fp8, best_scale_fp8)

    tl.store(
        scale_ptr + block_offsets,
        best_scale_fp8,
        mask=block_mask,
    )

    best_scale_inv = 1.0 / best_scale_fp8.to(tl.float32)

    lo, hi = _pack_final_code_16_cols(
        v0, v1, v2, v3,
        v4, v5, v6, v7,
        v8, v9, v10, v11,
        v12, v13, v14, v15,
        best_scale_inv,
    )

    code_i32_offsets = block_offsets * 2

    tl.store(
        code_i32_ptr + code_i32_offsets + 0,
        lo.to(tl.int32),
        mask=block_mask,
    )
    tl.store(
        code_i32_ptr + code_i32_offsets + 1,
        hi.to(tl.int32),
        mask=block_mask,
    )


def scalesweep_quantize(
    weight,
    global_scale_inv,
    block_size,
    lower_bound,
    upper_bound,
):
    if block_size != 16:
        raise ValueError("optimized kernel is specialized for block_size == 16")
    if weight.numel() % 16 != 0:
        raise ValueError("weight.numel() must be divisible by 16")

    num_blocks = weight.numel() // 16

    scale = torch.empty(
        num_blocks,
        device=weight.device,
        dtype=torch.float8_e4m3fn,
    )

    # 16 FP4 values = 8 bytes = 2 int32.
    # Store as int32 for fast vectorized writes, return uint8 view.
    code_i32 = torch.empty(
        num_blocks * 2,
        device=weight.device,
        dtype=torch.int32,
    )

    meta = lambda config: (triton.cdiv(num_blocks, config["BLOCKS_PER_PROGRAM"]), )

    scalesweep_quantize_kernel[meta](
        weight,
        scale,
        code_i32,
        global_scale_inv,
        num_blocks,
        LOWER_BOUND=lower_bound,
        NUM_CANDIDATES=upper_bound - lower_bound + 1,
    )

    code = code_i32.view(torch.uint8)

    scale_shape = (*weight.shape[:-1], weight.shape[-1] // 16)
    code_shape = (*weight.shape[:-1], weight.shape[-1] // 2)

    return scale.view(scale_shape), code.view(code_shape)

import triton.testing as tts

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=8192)
    parser.add_argument("--output", type=Path, default=Path("result/bench_ScaleSweep_MSE_no_convert_results.json"))
    return parser.parse_args()


def main():
    args = parse_args()

    results = {
        "name": "triton.ScaleSweep_MSE_no_convert",
        "lower_bound": LOWER_BOUND,
        "upper_bound": UPPER_BOUND,
        "sm_count": sm_count,
        "dim": args.dim,
        "results": [],
    }

    for bsz in bsz_list:
        weight = make_w(bsz, args.dim)
        global_scale, global_scale_inv = get_nvfp4_global_scale(weight, FP8_MAX=256)

        fn = lambda: scalesweep_quantize(
            weight,
            global_scale_inv,
            BLOCK_SIZE,
            LOWER_BOUND,
            UPPER_BOUND,
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
