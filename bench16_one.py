import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dim", type=int, default=8192)
parser.add_argument("--load", type=str, choices=["default", "sep", "trans"], default="sep")
parser.add_argument("--mse", type=str, choices=["direct", "default"], default="default")
args = parser.parse_args()
print(args)

from helper import (
    check_sm100,
    dequantize,
    error_stats,
    get_nvfp4_global_scales,
    make_w,
)
from helper16 import (
    _load_16_cols_2d_seperate,
    _max_abs_16,
    _mse_after_e2m1_roundtrip_16_cols_direct,
    _pack_final_code_16_cols,
)

LOAD_FN = _load_16_cols_2d_seperate
MSE_FN = _mse_after_e2m1_roundtrip_16_cols_direct

import torch
sm_count = torch.cuda.get_device_properties("cuda").multi_processor_count
bsz_list = [1, 8, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

import triton
import triton.language as tl

BLOCK_SIZE = 16
LOWER_BOUND = -3
UPPER_BOUND = 7

SCALESWEEP_CONFIGS = [
    triton.Config({"BLOCKS_PER_PROGRAM": 32, "NUM_STAGES": 2}, num_warps=1),
    triton.Config({"BLOCKS_PER_PROGRAM": 64, "NUM_STAGES": 2}, num_warps=2),
    triton.Config({"BLOCKS_PER_PROGRAM": 128, "NUM_STAGES": 2}, num_warps=4),
    triton.Config({"BLOCKS_PER_PROGRAM": 256, "NUM_STAGES": 2}, num_warps=8),
    triton.Config({"BLOCKS_PER_PROGRAM": 512, "NUM_STAGES": 2}, num_warps=16),
    triton.Config({"BLOCKS_PER_PROGRAM": 1024, "NUM_STAGES": 2}, num_warps=32),
]

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
    ) = LOAD_FN(
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

    base_fp8 = (abs_max * (1.0 / 6.0)).to(tl.float8e4nv)
    base_raw = base_fp8.to(tl.uint8, bitcast=True).to(tl.int32)

    for i in tl.static_range(0, NUM_CANDIDATES):
        raw_i = tl.minimum(
            tl.maximum(base_raw + LOWER_BOUND + i, 1),
            126,
        ).to(tl.uint8)

        scale_fp8 = raw_i.to(tl.float8e4nv, bitcast=True)
        scale_i = scale_fp8.to(tl.float32)
        inv_scale_i = 1.0 / scale_i

        mse_i = MSE_FN(
            v0, v1, v2, v3,
            v4, v5, v6, v7,
            v8, v9, v10, v11,
            v12, v13, v14, v15,
            inv_scale_i,
            scale_i,
        )

        if i > 0:
            better = mse_i < best_mse
            best_mse = tl.where(better, mse_i, best_mse)
            best_scale_fp8 = tl.where(better, scale_fp8, best_scale_fp8)
        else:
            best_mse = mse_i
            best_scale_fp8 = scale_fp8

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

def main():
    check_sm100()
    print(f"[triton.ScaleSweep [{LOWER_BOUND}, {UPPER_BOUND}]] [SM {sm_count}]")

    for bsz in bsz_list:
        weight = make_w(bsz, args.dim)
        global_scale, global_scale_inv = get_nvfp4_global_scales(weight, FP8_MAX=256)

        fn = lambda: scalesweep_quantize(
            weight,
            global_scale_inv,
            BLOCK_SIZE,
            LOWER_BOUND,
            UPPER_BOUND,
        )
        ms = tts.do_bench(fn, warmup=10, rep=100)
        scale, code = fn()

        reconstructed = dequantize("base", code, scale, global_scale, high_first=False)
        mse, max_abs_error = error_stats(weight, reconstructed)

        print(f"bsz = {bsz}, dim = {weight.shape[1]}")
        print(f"latency_ms    = {ms:.6f}")
        print(f"mse           = {mse:.8e}")
        print(f"max_abs_error = {max_abs_error:.8e}")
        print(flush=True)
    print("-" * 25)

if __name__ == "__main__":
    main()