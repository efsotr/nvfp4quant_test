from pathlib import Path

import torch
import torch.nn.functional as F
import triton.testing as tts

from bench import KERNEL_CASES, current_sm, should_skip_kernel
from helper import DTYPE, SEED, DEVICE, error_stats, get_nvfp4_global_scale, make_w
from kernel_ScaleSweep_MSE import BLOCK_SIZE
from kernel_vllm import round_up, swizzle_vllm_fp4_scale


DIM = 8192
OUTPUT = Path("result/gemm_nvfp4_perf.log")
WARMUP = 10
REP = 100


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
            is_swizzle=True,
        )
    if case.kind == "mse":
        return case.quantize(
            tensor,
            global_scale_inv,
            BLOCK_SIZE,
            case.lower_bound,
            case.upper_bound,
            is_swizzle=True,
        )
    if case.kind == "vllm":
        from vllm._custom_ops import scaled_fp4_quant

        code, scale = scaled_fp4_quant(tensor, global_scale_inv)
        return scale, code
    raise ValueError(f"unknown kernel kind for {case.name}: {case.kind}")


def quantize_for_cutlass(case, tensor, imp):
    _, global_scale_inv = get_nvfp4_global_scale(tensor, FP8_MAX=case.fp8_max)
    scale, code = quantize_case(case, tensor, imp, global_scale_inv)
    return code, scale, global_scale_inv


def run_cutlass_gemm(
    x_code,
    weight_code,
    x_scale,
    weight_scale,
    x_global_scale_inv,
    weight_global_scale_inv,
):
    from vllm._custom_ops import cutlass_scaled_fp4_mm

    alpha = torch.reciprocal(x_global_scale_inv * weight_global_scale_inv)
    return cutlass_scaled_fp4_mm(
        x_code,
        weight_code,
        x_scale,
        weight_scale,
        alpha,
        DTYPE,
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run gemm_nvfp4_perf.py")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    sm = current_sm()
    lines = ["kernel,latency_ms,mse,max_abs_err"]
    if sm < 100:
        skip_reason = f"cutlass_scaled_fp4_mm requires sm_100, current device is sm_{sm}"
        for case in KERNEL_CASES:
            lines.append(f"{case.name},SKIP,{skip_reason}")
        text = "\n".join(lines) + "\n"
        OUTPUT.write_text(text)
        print(text, end="")
        return

    x = make_input(DIM, DIM)
    weight = make_w(DIM, DIM)
    weight_imp = make_channel_square_norm(x)
    x_imp = make_channel_square_norm(weight)

    ref = F.linear(x, weight)

    for case in KERNEL_CASES:
        skip_reason = should_skip_kernel(case, sm)
        if skip_reason is not None:
            lines.append(f"{case.name},SKIP,{skip_reason}")
            continue

        try:
            x_code, x_scale, x_global_scale_inv = quantize_for_cutlass(case, x, x_imp)
            weight_code, weight_scale, weight_global_scale_inv = quantize_for_cutlass(
                case,
                weight,
                weight_imp,
            )

            def gemm():
                return run_cutlass_gemm(
                    x_code,
                    weight_code,
                    x_scale,
                    weight_scale,
                    x_global_scale_inv,
                    weight_global_scale_inv,
                )

            latency_ms = tts.do_bench(gemm, warmup=WARMUP, rep=REP)
            pred = gemm()
            mse, max_abs_err = error_stats(ref, pred)

            lines.append(f"{case.name},{latency_ms:.4f},{mse:.4f},{max_abs_err:.4f}")

            del x_code, x_scale, weight_code, weight_scale, pred
            torch.cuda.empty_cache()
        except Exception as exc:
            lines.append(f"{case.name},ERROR,{type(exc).__name__}: {exc}")

    text = "\n".join(lines) + "\n"
    OUTPUT.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
