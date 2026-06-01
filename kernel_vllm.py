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
