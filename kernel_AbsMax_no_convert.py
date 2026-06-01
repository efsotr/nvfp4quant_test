from kernel_ScaleSweep_MSE_no_convert import (
    _load_normalized_16_cols,
    _max_abs_16,
    _pack_final_code_16_cols,
)

import torch
import triton
import triton.language as tl

BLOCK_SIZE = 16


ABSMAX_CONFIGS = [
    triton.Config({"BLOCKS_PER_PROGRAM": 32}, num_warps=1),
    triton.Config({"BLOCKS_PER_PROGRAM": 64}, num_warps=2),
    triton.Config({"BLOCKS_PER_PROGRAM": 128}, num_warps=4),
    triton.Config({"BLOCKS_PER_PROGRAM": 256}, num_warps=8),
    triton.Config({"BLOCKS_PER_PROGRAM": 512}, num_warps=16),
    triton.Config({"BLOCKS_PER_PROGRAM": 1024}, num_warps=32),
]


@triton.autotune(
    configs=ABSMAX_CONFIGS,
    key=["NUM_BLOCKS"],
)
@triton.jit
def absmax_quantize_no_convert_kernel(
    weight_ptr,
    scale_ptr,
    code_i32_ptr,
    global_scale_inv_ptr,
    NUM_BLOCKS: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
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

    scale_fp8 = (abs_max * (1.0 / 6.0)).to(tl.float8e4nv)
    tl.store(scale_ptr + block_offsets, scale_fp8, mask=block_mask)

    scale = scale_fp8.to(tl.float32)
    inv_scale = tl.where(scale == 0.0, 0.0, 1.0 / scale)

    lo, hi = _pack_final_code_16_cols(
        v0, v1, v2, v3,
        v4, v5, v6, v7,
        v8, v9, v10, v11,
        v12, v13, v14, v15,
        inv_scale,
    )

    code_i32_offsets = block_offsets * 2
    tl.store(code_i32_ptr + code_i32_offsets + 0, lo.to(tl.int32), mask=block_mask)
    tl.store(code_i32_ptr + code_i32_offsets + 1, hi.to(tl.int32), mask=block_mask)


def absmax_quantize_no_convert(weight, global_scale_inv, block_size):
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
    code_i32 = torch.empty(
        num_blocks * 2,
        device=weight.device,
        dtype=torch.int32,
    )

    meta = lambda config: (triton.cdiv(num_blocks, config["BLOCKS_PER_PROGRAM"]), )

    absmax_quantize_no_convert_kernel[meta](
        weight,
        scale,
        code_i32,
        global_scale_inv,
        num_blocks,
    )

    code = code_i32.view(torch.uint8)
    scale_shape = (*weight.shape[:-1], weight.shape[-1] // 16)
    code_shape = (*weight.shape[:-1], weight.shape[-1] // 2)

    return scale.view(scale_shape), code.view(code_shape)
