import triton
import triton.language as tl


def round_up(x, y):
    return (x + y - 1) // y * y


@triton.jit
def vllm_swizzled_scale_offsets(
    block_offsets,
    BLOCKS_PER_OUT: tl.constexpr,
    K_PAD: tl.constexpr,
):
    row = block_offsets // BLOCKS_PER_OUT
    col = block_offsets - row * BLOCKS_PER_OUT

    major_m = row >> 7
    row_in_tile = row & 127
    tile_m = row_in_tile >> 5
    inner_m = row_in_tile & 31

    major_k = col >> 2
    inner_k = col & 3

    return major_m * (K_PAD * 128) + major_k * 512 + inner_m * 16 + tile_m * 4 + inner_k


def swizzle_vllm_fp4_scale(scale, m, n, block_size=16):
    assert n % block_size == 0
    assert scale.shape == (m, n // block_size)

    k = n // block_size
    m_pad = round_up(m, 128)
    k_pad = round_up(k, 4)

    scale_padded = scale.new_zeros((m_pad, k_pad))
    scale_padded[:m, :k] = scale

    scale_swizzled = scale_padded.reshape(
        m_pad // 128,
        4,
        32,
        k_pad // 4,
        4,
    )
    scale_swizzled = scale_swizzled.permute(0, 3, 2, 1, 4).contiguous()
    return scale_swizzled.reshape(-1).contiguous()


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
