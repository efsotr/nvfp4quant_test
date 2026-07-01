import math

import torch
import triton
import triton.language as tl

from kernels.kernel_vllm import round_up
from vllm._custom_ops import create_fp4_output_tensors

BLOCK_SIZE = 16
LOWER_BOUND = -8
UPPER_BOUND = 7


@triton.jit
def _swizzled_scale_offsets(
    row,
    col,
    BLOCKS_PER_COL_OUT_PAD: tl.constexpr,
):
    major_m = row >> 7
    row_in_tile = row & 127
    tile_m = row_in_tile >> 5
    inner_m = row & 31

    major_k = col >> 2
    inner_k = col & 3

    return (
        major_m * (BLOCKS_PER_COL_OUT_PAD * 128)
        + major_k * 512
        + inner_m * 16
        + tile_m * 4
        + inner_k
    )


@triton.jit
def _fp32x16_to_e2m1_u32x2(
    x0, x1, x2, x3,
    x4, x5, x6, x7,
    x8, x9, x10, x11,
    x12, x13, x14, x15,
):
    lo, hi = tl.inline_asm_elementwise(
        asm="""
        {
          .reg .b8 b0;
          .reg .b8 b1;
          .reg .b8 b2;
          .reg .b8 b3;
          .reg .b8 b4;
          .reg .b8 b5;
          .reg .b8 b6;
          .reg .b8 b7;

          cvt.rn.satfinite.e2m1x2.f32 b0,  $3,  $2;
          cvt.rn.satfinite.e2m1x2.f32 b1,  $5,  $4;
          cvt.rn.satfinite.e2m1x2.f32 b2,  $7,  $6;
          cvt.rn.satfinite.e2m1x2.f32 b3,  $9,  $8;
          cvt.rn.satfinite.e2m1x2.f32 b4,  $11, $10;
          cvt.rn.satfinite.e2m1x2.f32 b5,  $13, $12;
          cvt.rn.satfinite.e2m1x2.f32 b6,  $15, $14;
          cvt.rn.satfinite.e2m1x2.f32 b7,  $17, $16;

          mov.b32 $0, {b0, b1, b2, b3};
          mov.b32 $1, {b4, b5, b6, b7};
        }
        """,
        constraints="=r,=r,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f",
        args=[
            x0, x1, x2, x3,
            x4, x5, x6, x7,
            x8, x9, x10, x11,
            x12, x13, x14, x15,
        ],
        dtype=(tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )
    return lo, hi


@triton.jit
def _fp32x2_e2m1_quant_weighted_squared_error(x0, x1, iw0, iw1):
    return tl.inline_asm_elementwise(
        asm=r"""
        {
          .reg .b8  b;
          .reg .b32 h;
          .reg .b16 lo;
          .reg .b16 hi;
          .reg .f32 q0;
          .reg .f32 q1;
          .reg .f32 d0;
          .reg .f32 d1;

          cvt.rn.satfinite.e2m1x2.f32 b, $2, $1;
          cvt.rn.f16x2.e2m1x2 h, b;

          mov.b32 {lo, hi}, h;
          cvt.f32.f16 q0, lo;
          cvt.f32.f16 q1, hi;

          sub.rn.f32 d0, q0, $1;
          sub.rn.f32 d1, q1, $2;
          mul.rn.f32 d0, d0, d0;
          mul.rn.f32 d1, d1, d1;
          mul.rn.f32 d0, d0, $3;
          fma.rn.f32 $0, d1, $4, d0;
        }
        """,
        constraints="=f,f,f,f,f",
        args=[x0, x1, iw0, iw1],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _scaled_fp32x16_e2m1_quant_weighted_squared_error(
    x0, x1, x2, x3,
    x4, x5, x6, x7,
    x8, x9, x10, x11,
    x12, x13, x14, x15,
    iw0, iw1, iw2, iw3,
    iw4, iw5, iw6, iw7,
    iw8, iw9, iw10, iw11,
    iw12, iw13, iw14, iw15,
):
    e01 = _fp32x2_e2m1_quant_weighted_squared_error(x0, x1, iw0, iw1)
    e23 = _fp32x2_e2m1_quant_weighted_squared_error(x2, x3, iw2, iw3)
    e45 = _fp32x2_e2m1_quant_weighted_squared_error(x4, x5, iw4, iw5)
    e67 = _fp32x2_e2m1_quant_weighted_squared_error(x6, x7, iw6, iw7)
    e89 = _fp32x2_e2m1_quant_weighted_squared_error(x8, x9, iw8, iw9)
    eAB = _fp32x2_e2m1_quant_weighted_squared_error(x10, x11, iw10, iw11)
    eCD = _fp32x2_e2m1_quant_weighted_squared_error(x12, x13, iw12, iw13)
    eEF = _fp32x2_e2m1_quant_weighted_squared_error(x14, x15, iw14, iw15)

    return ((e01 + e23) + (e45 + e67)) + ((e89 + eAB) + (eCD + eEF))


@triton.jit
def _fp32x16_e2m1_quant_weighted_squared_error(
    v0, v1, v2, v3,
    v4, v5, v6, v7,
    v8, v9, v10, v11,
    v12, v13, v14, v15,
    iw0, iw1, iw2, iw3,
    iw4, iw5, iw6, iw7,
    iw8, iw9, iw10, iw11,
    iw12, iw13, iw14, iw15,
    inv_scale,
    scale,
):
    weighted_squared_error = _scaled_fp32x16_e2m1_quant_weighted_squared_error(
        v0 * inv_scale, v1 * inv_scale,
        v2 * inv_scale, v3 * inv_scale,
        v4 * inv_scale, v5 * inv_scale,
        v6 * inv_scale, v7 * inv_scale,
        v8 * inv_scale, v9 * inv_scale,
        v10 * inv_scale, v11 * inv_scale,
        v12 * inv_scale, v13 * inv_scale,
        v14 * inv_scale, v15 * inv_scale,
        iw0, iw1, iw2, iw3,
        iw4, iw5, iw6, iw7,
        iw8, iw9, iw10, iw11,
        iw12, iw13, iw14, iw15,
    )
    return weighted_squared_error * (scale * scale)


@triton.jit
def _max_abs_16(
    v0, v1, v2, v3,
    v4, v5, v6, v7,
    v8, v9, v10, v11,
    v12, v13, v14, v15,
):
    m0 = tl.maximum(tl.abs(v0), tl.abs(v1))
    m1 = tl.maximum(tl.abs(v2), tl.abs(v3))
    m2 = tl.maximum(tl.abs(v4), tl.abs(v5))
    m3 = tl.maximum(tl.abs(v6), tl.abs(v7))
    m4 = tl.maximum(tl.abs(v8), tl.abs(v9))
    m5 = tl.maximum(tl.abs(v10), tl.abs(v11))
    m6 = tl.maximum(tl.abs(v12), tl.abs(v13))
    m7 = tl.maximum(tl.abs(v14), tl.abs(v15))

    m01 = tl.maximum(m0, m1)
    m23 = tl.maximum(m2, m3)
    m45 = tl.maximum(m4, m5)
    m67 = tl.maximum(m6, m7)
    return tl.maximum(tl.maximum(m01, m23), tl.maximum(m45, m67))


@triton.jit
def _load_normalized_16_cols(ptr, block_offsets, block_mask, global_scale_inv):
    base_elem = block_offsets * 16

    v0 = tl.load(ptr + base_elem + 0, mask=block_mask, other=0.0).to(tl.float32)
    v1 = tl.load(ptr + base_elem + 1, mask=block_mask, other=0.0).to(tl.float32)
    v2 = tl.load(ptr + base_elem + 2, mask=block_mask, other=0.0).to(tl.float32)
    v3 = tl.load(ptr + base_elem + 3, mask=block_mask, other=0.0).to(tl.float32)
    v4 = tl.load(ptr + base_elem + 4, mask=block_mask, other=0.0).to(tl.float32)
    v5 = tl.load(ptr + base_elem + 5, mask=block_mask, other=0.0).to(tl.float32)
    v6 = tl.load(ptr + base_elem + 6, mask=block_mask, other=0.0).to(tl.float32)
    v7 = tl.load(ptr + base_elem + 7, mask=block_mask, other=0.0).to(tl.float32)
    v8 = tl.load(ptr + base_elem + 8, mask=block_mask, other=0.0).to(tl.float32)
    v9 = tl.load(ptr + base_elem + 9, mask=block_mask, other=0.0).to(tl.float32)
    v10 = tl.load(ptr + base_elem + 10, mask=block_mask, other=0.0).to(tl.float32)
    v11 = tl.load(ptr + base_elem + 11, mask=block_mask, other=0.0).to(tl.float32)
    v12 = tl.load(ptr + base_elem + 12, mask=block_mask, other=0.0).to(tl.float32)
    v13 = tl.load(ptr + base_elem + 13, mask=block_mask, other=0.0).to(tl.float32)
    v14 = tl.load(ptr + base_elem + 14, mask=block_mask, other=0.0).to(tl.float32)
    v15 = tl.load(ptr + base_elem + 15, mask=block_mask, other=0.0).to(tl.float32)

    return (
        v0 * global_scale_inv, v1 * global_scale_inv,
        v2 * global_scale_inv, v3 * global_scale_inv,
        v4 * global_scale_inv, v5 * global_scale_inv,
        v6 * global_scale_inv, v7 * global_scale_inv,
        v8 * global_scale_inv, v9 * global_scale_inv,
        v10 * global_scale_inv, v11 * global_scale_inv,
        v12 * global_scale_inv, v13 * global_scale_inv,
        v14 * global_scale_inv, v15 * global_scale_inv,
    )


@triton.jit
def _load_shared_importance_16_cols(
    imp_ptr,
    block_offsets,
    block_mask,
    BLOCKS_PER_COL_IN: tl.constexpr,
):
    # imp is [1, d_in], contiguous. The same 16 importance weights are shared
    # across all rows for the same input-column block.
    col = block_offsets % BLOCKS_PER_COL_IN
    base_elem = col * 16

    iw0 = tl.load(imp_ptr + base_elem + 0, mask=block_mask, other=0.0).to(tl.float32)
    iw1 = tl.load(imp_ptr + base_elem + 1, mask=block_mask, other=0.0).to(tl.float32)
    iw2 = tl.load(imp_ptr + base_elem + 2, mask=block_mask, other=0.0).to(tl.float32)
    iw3 = tl.load(imp_ptr + base_elem + 3, mask=block_mask, other=0.0).to(tl.float32)
    iw4 = tl.load(imp_ptr + base_elem + 4, mask=block_mask, other=0.0).to(tl.float32)
    iw5 = tl.load(imp_ptr + base_elem + 5, mask=block_mask, other=0.0).to(tl.float32)
    iw6 = tl.load(imp_ptr + base_elem + 6, mask=block_mask, other=0.0).to(tl.float32)
    iw7 = tl.load(imp_ptr + base_elem + 7, mask=block_mask, other=0.0).to(tl.float32)
    iw8 = tl.load(imp_ptr + base_elem + 8, mask=block_mask, other=0.0).to(tl.float32)
    iw9 = tl.load(imp_ptr + base_elem + 9, mask=block_mask, other=0.0).to(tl.float32)
    iw10 = tl.load(imp_ptr + base_elem + 10, mask=block_mask, other=0.0).to(tl.float32)
    iw11 = tl.load(imp_ptr + base_elem + 11, mask=block_mask, other=0.0).to(tl.float32)
    iw12 = tl.load(imp_ptr + base_elem + 12, mask=block_mask, other=0.0).to(tl.float32)
    iw13 = tl.load(imp_ptr + base_elem + 13, mask=block_mask, other=0.0).to(tl.float32)
    iw14 = tl.load(imp_ptr + base_elem + 14, mask=block_mask, other=0.0).to(tl.float32)
    iw15 = tl.load(imp_ptr + base_elem + 15, mask=block_mask, other=0.0).to(tl.float32)

    return (
        iw0, iw1, iw2, iw3,
        iw4, iw5, iw6, iw7,
        iw8, iw9, iw10, iw11,
        iw12, iw13, iw14, iw15,
    )


SCALESWEEP_CONFIGS = [
    triton.Config({"BLOCKS_PER_PROGRAM": 32}, num_warps=1),
    triton.Config({"BLOCKS_PER_PROGRAM": 64}, num_warps=2),
    triton.Config({"BLOCKS_PER_PROGRAM": 128}, num_warps=4),
    triton.Config({"BLOCKS_PER_PROGRAM": 256}, num_warps=8),
    triton.Config({"BLOCKS_PER_PROGRAM": 512}, num_warps=16),
    triton.Config({"BLOCKS_PER_PROGRAM": 1024}, num_warps=32),
]


@triton.heuristics({"LOG2_NUM_ROW": lambda args: int(math.log2(args["NUM_ROW"]))})
@triton.autotune(
    configs=SCALESWEEP_CONFIGS,
    key=[
        "LOG2_NUM_ROW",
        "BLOCKS_PER_COL_IN",
        "BLOCKS_PER_COL_OUT",
        "LOWER_BOUND",
        "NUM_CANDIDATES",
        "IS_SWIZZLE_SCALE",
        "BLOCKS_PER_COL_OUT_PAD",
    ],
)
@triton.jit
def _scalesweep_weighted_mse_nvfp4_quant_kernel(
    input_ptr,
    importance_ptr,
    output_scale_ptr,
    output_i32_ptr,
    global_scale_inv_ptr,
    NUM_OUTPUT_BLOCKS: tl.constexpr,
    NUM_ROW: tl.constexpr,
    BLOCKS_PER_COL_IN: tl.constexpr,
    BLOCKS_PER_COL_OUT: tl.constexpr,
    LOWER_BOUND: tl.constexpr,
    NUM_CANDIDATES: tl.constexpr,
    IS_SWIZZLE_SCALE: tl.constexpr,
    BLOCKS_PER_COL_OUT_PAD: tl.constexpr,
    LOG2_NUM_ROW: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
):
    global_scale_inv = tl.load(global_scale_inv_ptr)
    pid = tl.program_id(0)
    output_block_offsets = pid * BLOCKS_PER_PROGRAM + tl.arange(0, BLOCKS_PER_PROGRAM)
    output_block_mask = output_block_offsets < NUM_OUTPUT_BLOCKS

    row = output_block_offsets // BLOCKS_PER_COL_OUT
    col = output_block_offsets % BLOCKS_PER_COL_OUT
    if BLOCKS_PER_COL_IN == BLOCKS_PER_COL_OUT:
        input_block_offsets = output_block_offsets
        input_block_mask = output_block_mask
    else:
        input_block_offsets = row * BLOCKS_PER_COL_IN + col
        input_block_mask = output_block_mask & (col < BLOCKS_PER_COL_IN)

    (
        v0, v1, v2, v3,
        v4, v5, v6, v7,
        v8, v9, v10, v11,
        v12, v13, v14, v15,
    ) = _load_normalized_16_cols(
        input_ptr,
        input_block_offsets,
        input_block_mask,
        global_scale_inv,
    )

    (
        iw0, iw1, iw2, iw3,
        iw4, iw5, iw6, iw7,
        iw8, iw9, iw10, iw11,
        iw12, iw13, iw14, iw15,
    ) = _load_shared_importance_16_cols(
        importance_ptr,
        input_block_offsets,
        input_block_mask,
        BLOCKS_PER_COL_IN,
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

    for i in tl.static_range(0, NUM_CANDIDATES):
        raw_i = tl.minimum(
            tl.maximum(base_raw + (LOWER_BOUND + i), 1),
            126,
        ).to(tl.uint8)
        scale_fp8 = raw_i.to(tl.float8e4nv, bitcast=True)
        scale_i = scale_fp8.to(tl.float32)
        inv_scale_i = 1.0 / scale_i

        mse_i = _fp32x16_e2m1_quant_weighted_squared_error(
            v0, v1, v2, v3,
            v4, v5, v6, v7,
            v8, v9, v10, v11,
            v12, v13, v14, v15,
            iw0, iw1, iw2, iw3,
            iw4, iw5, iw6, iw7,
            iw8, iw9, iw10, iw11,
            iw12, iw13, iw14, iw15,
            inv_scale_i,
            scale_i,
        )

        better = mse_i < best_mse
        best_mse = tl.where(better, mse_i, best_mse)
        best_scale_fp8 = tl.where(better, scale_fp8, best_scale_fp8)

    if IS_SWIZZLE_SCALE:
        scale_offsets = _swizzled_scale_offsets(
            row,
            col,
            BLOCKS_PER_COL_OUT_PAD,
        )
    else:
        scale_offsets = output_block_offsets

    tl.store(
        output_scale_ptr + scale_offsets,
        best_scale_fp8,
        mask=output_block_mask,
    )

    inv_scale = 1.0 / best_scale_fp8.to(tl.float32)
    lo, hi = _fp32x16_to_e2m1_u32x2(
        v0 * inv_scale, v1 * inv_scale,
        v2 * inv_scale, v3 * inv_scale,
        v4 * inv_scale, v5 * inv_scale,
        v6 * inv_scale, v7 * inv_scale,
        v8 * inv_scale, v9 * inv_scale,
        v10 * inv_scale, v11 * inv_scale,
        v12 * inv_scale, v13 * inv_scale,
        v14 * inv_scale, v15 * inv_scale,
    )

    output_i32_offsets = output_block_offsets * 2
    tl.store(output_i32_ptr + output_i32_offsets, lo, mask=output_block_mask)
    tl.store(output_i32_ptr + output_i32_offsets + 1, hi, mask=output_block_mask)


def _check_inputs(input: torch.Tensor, importance: torch.Tensor) -> None:
    if input.ndim != 2:
        raise ValueError(f"input must be 2D [num_row, dim], got {tuple(input.shape)}")
    if input.shape[-1] % BLOCK_SIZE != 0:
        raise ValueError(f"input.shape[-1] must be divisible by {BLOCK_SIZE}")
    if importance.shape != (1, input.shape[-1]):
        raise ValueError(
            f"importance must have shape [1, dim], got {tuple(importance.shape)}, "
            f"expected {(1, input.shape[-1])}"
        )
    if input.device != importance.device:
        raise ValueError(
            f"input and importance must be on the same device: "
            f"{input.device} vs {importance.device}"
        )


def scalesweep_weighted_mse_nvfp4_quant_out(
    input: torch.Tensor,
    importance: torch.Tensor,
    input_scale: torch.Tensor,
    is_sf_swizzled_layout: bool = True,
    *,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    lower_bound: int = LOWER_BOUND,
    upper_bound: int = UPPER_BOUND,
) -> None:
    _check_inputs(input, importance)

    if not input.is_contiguous():
        input = input.contiguous()
    if not importance.is_contiguous():
        importance = importance.contiguous()

    num_row, n = input.shape
    physical_n = output.shape[-1] * 2
    blocks_per_col_in = n // BLOCK_SIZE
    blocks_per_col_out = physical_n // BLOCK_SIZE
    num_output_blocks = num_row * blocks_per_col_out

    output_i32 = output.view(torch.int32)
    output_scale_fp8 = output_scale.view(torch.float8_e4m3fn)

    grid = lambda meta: (
        triton.cdiv(num_output_blocks, meta["BLOCKS_PER_PROGRAM"]),
    )
    _scalesweep_weighted_mse_nvfp4_quant_kernel[grid](
        input,
        importance,
        output_scale_fp8,
        output_i32,
        input_scale,
        num_output_blocks,
        NUM_ROW=num_row,
        BLOCKS_PER_COL_IN=blocks_per_col_in,
        BLOCKS_PER_COL_OUT=blocks_per_col_out,
        LOWER_BOUND=lower_bound,
        NUM_CANDIDATES=upper_bound - lower_bound + 1,
        IS_SWIZZLE_SCALE=is_sf_swizzled_layout,
        BLOCKS_PER_COL_OUT_PAD=round_up(blocks_per_col_out, 4),
    )


def scalesweep_weighted_mse_nvfp4_quant_impl(
    input: torch.Tensor,
    importance: torch.Tensor,
    input_scale: torch.Tensor,
    is_sf_swizzled_layout: bool = True,
    padded_n: int | None = None,
    *,
    lower_bound: int = LOWER_BOUND,
    upper_bound: int = UPPER_BOUND,
) -> tuple[torch.Tensor, torch.Tensor]:
    _check_inputs(input, importance)

    m, n = input.shape
    output, output_scale = create_fp4_output_tensors(
        m,
        n,
        input.device,
        is_sf_swizzled_layout,
        padded_n=padded_n,
    )
    scalesweep_weighted_mse_nvfp4_quant_out(
        input,
        importance,
        input_scale,
        is_sf_swizzled_layout,
        output=output,
        output_scale=output_scale,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    return output, output_scale.view(torch.float8_e4m3fn)

