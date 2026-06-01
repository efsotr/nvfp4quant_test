from pathlib import Path

import torch
import torch.nn.functional as F

from bench import KERNEL_CASES, current_sm, should_skip_kernel
from helper import DTYPE, SEED, DEVICE, dequantize, error_stats, get_nvfp4_global_scale, make_w
from kernel_ScaleSweep_MSE import BLOCK_SIZE
from kernel_vllm import unswizzle_vllm_fp4_scale


DIM = 8192
OUTPUT = Path("result/gemm_perf.log")


def make_input(m: int, k: int) -> torch.Tensor:
    torch.manual_seed(SEED + 1)
    torch.cuda.manual_seed_all(SEED + 1)
    return torch.randn((m, k), device=DEVICE, dtype=DTYPE).contiguous()


def make_channel_square_norm(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(x.float() * x.float(), dim=0, keepdim=True).contiguous()


def quantize_case(case, tensor, imp, global_scale_inv):
    if case.kind == "weighted":
        return case.quantize(
            tensor,
            imp,
            global_scale_inv,
            BLOCK_SIZE,
            case.lower_bound,
            case.upper_bound,
            is_swizzle=case.scale_layout == "swizzled",
        )
    if case.kind == "mse":
        return case.quantize(
            tensor,
            global_scale_inv,
            BLOCK_SIZE,
            case.lower_bound,
            case.upper_bound,
            is_swizzle=case.scale_layout == "swizzled",
        )
    if case.kind == "absmax":
        return case.quantize(tensor, global_scale_inv, BLOCK_SIZE)
    if case.kind == "vllm":
        from vllm._custom_ops import scaled_fp4_quant

        code, scale = scaled_fp4_quant(tensor, global_scale_inv)
        return scale, code
    raise ValueError(f"unknown kernel kind for {case.name}: {case.kind}")


def quantize_dequantize(case, tensor, imp):
    global_scale, global_scale_inv = get_nvfp4_global_scale(tensor, FP8_MAX=case.fp8_max)
    scale, code = quantize_case(case, tensor, imp, global_scale_inv)
    if case.scale_layout == "swizzled":
        scale = unswizzle_vllm_fp4_scale(
            scale,
            m=tensor.shape[0],
            n=tensor.shape[1],
            block_size=BLOCK_SIZE,
        )
    return dequantize("base", code, scale, global_scale).to(DTYPE)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run gemm_perf.py")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    sm = current_sm()

    x = make_input(DIM, DIM)
    weight = make_w(DIM, DIM)
    weight_imp = make_channel_square_norm(x)
    x_imp = make_channel_square_norm(weight)

    ref = F.linear(x, weight)

    lines = ["kernel,mse,max_abs_err"]

    for case in KERNEL_CASES:
        skip_reason = should_skip_kernel(case, sm)
        if skip_reason is not None:
            lines.append(f"{case.name},SKIP,{skip_reason}")
            continue

        try:
            reconstructed_x = quantize_dequantize(case, x, x_imp)
            reconstructed_weight = quantize_dequantize(case, weight, weight_imp)
            pred = F.linear(reconstructed_x, reconstructed_weight)
            mse, max_abs_err = error_stats(ref, pred)

            lines.append(f"{case.name},{mse:.4f},{max_abs_err:.4f}")

            del reconstructed_x, reconstructed_weight, pred
            torch.cuda.empty_cache()
        except Exception as exc:
            lines.append(f"{case.name},ERROR,{type(exc).__name__}: {exc}")

    text = "\n".join(lines) + "\n"
    OUTPUT.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
