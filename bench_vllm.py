import torch
from vllm._custom_ops import scaled_fp4_quant
import triton.testing as tts

from helper import (
    check_sm100,
    make_w,
    get_nvfp4_global_scales,
    time_cuda,
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

def main():
    check_sm100()
    print("[vllm._custom_ops.scaled_fp4_quant]")
    for bsz in [1, 8, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
        W = make_w(bsz, 8192)
        global_scale, global_scale_inv = get_nvfp4_global_scales(W)

        fn = lambda: scaled_fp4_quant(W, global_scale_inv)
        ms = tts.do_bench(fn, warmup=10, rep=100)
        q, s = fn()
        s = unswizzle_vllm_fp4_scale(s, m=W.shape[0], n=W.shape[1], block_size=16)

        W_hat = dequantize("base", q, s, global_scale, high_first=False)

        mse, max_abs_error = error_stats(W, W_hat)

        print(f"bsz = {bsz}, dim = {W.shape[1]}")
        print(f"latency_ms    = {ms:.6f}")
        print(f"mse           = {mse:.8e}")
        print(f"max_abs_error = {max_abs_error:.8e}")
        print()


if __name__ == "__main__":
    main()