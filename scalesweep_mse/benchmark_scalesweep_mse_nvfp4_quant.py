import argparse
import copy
import importlib.metadata
import itertools
import json
import subprocess
from pathlib import Path

import torch
import triton
from absmax_nvfp4_quant_simulate import absmax_nvfp4_quant_simulate
from scalesweep_mse_nvfp4_quant import (
    BLOCK_SIZE,
    FP4_E2M1_MAX,
    FP8_E4M3_MAX,
    round_up,
    scalesweep_mse_nvfp4_quant,
)
from scalesweep_mse_nvfp4_quant_simulate import scalesweep_mse_nvfp4_quant_simulate
from weight_shapes import WEIGHT_SHAPES

PROVIDER_CFGS = {
    "vllm": dict(backend="vllm", enabled=True),
    "scalesweep_mse": dict(backend="scalesweep_mse", enabled=True),
}
SIMULATE_PROVIDERS = {
    "absmax_torch_simulate": absmax_nvfp4_quant_simulate,
    "scalesweep_mse_triton_simulate": scalesweep_mse_nvfp4_quant_simulate,
}

_enabled = [k for k, v in PROVIDER_CFGS.items() if v["enabled"]]

E2M1_TO_FLOAT32 = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def driver_version() -> str | None:
    get_driver_version = getattr(torch._C, "_cuda_getDriverVersion", None)
    if get_driver_version is not None:
        version = get_driver_version()
        return f"{version // 1000}.{(version % 1000) // 10}"
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    versions = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return versions[0] if versions else None


def get_environment_info() -> dict:
    props = torch.cuda.get_device_properties("cuda")
    major, minor = torch.cuda.get_device_capability()
    return {
        "torch": torch.__version__,
        "vllm": package_version("vllm"),
        "cuda": torch.version.cuda,
        "gpu_driver": driver_version(),
        "gpu": {
            "name": props.name,
            "sm": major * 10 + minor,
            "capability": f"{major}.{minor}",
            "sm_count": props.multi_processor_count,
            "total_memory_bytes": props.total_memory,
        },
    }


def compute_global_scale_inv(tensor: torch.Tensor) -> torch.Tensor:
    amax = torch.abs(tensor).max().to(torch.float32)
    return FP8_E4M3_MAX * FP4_E2M1_MAX / amax


def recover_swizzled_scales(scale: torch.Tensor, m: int, n: int) -> torch.Tensor:
    scale_n = n // BLOCK_SIZE
    rounded_m = round_up(m, 128)
    rounded_n = round_up(scale_n, 4)
    tmp = torch.reshape(scale, (1, rounded_m // 128, rounded_n // 4, 32, 4, 4))
    tmp = torch.permute(tmp, (0, 1, 4, 3, 2, 5))
    result = torch.reshape(tmp, (rounded_m, rounded_n)).to(torch.float32)
    return result[:m, :scale_n]


def cast_from_fp4(x: torch.Tensor, m: int, n: int) -> torch.Tensor:
    v_2nd = x & 0xF
    v_1st = (x >> 4) & 0xF
    c = torch.stack((v_2nd, v_1st), dim=-1)
    lut = torch.tensor(E2M1_TO_FLOAT32, device=x.device, dtype=torch.float32)
    return lut[c.long()].reshape(m, n)


def dequantize_output(
    output: torch.Tensor,
    output_scale: torch.Tensor,
    global_scale_inv: torch.Tensor,
) -> torch.Tensor:
    m, n = output.shape[0], output.shape[1] * 2
    scale = recover_swizzled_scales(output_scale, m, n)
    values = cast_from_fp4(output, m, n)
    scales = scale.to(torch.float32).repeat_interleave(BLOCK_SIZE, dim=1)
    return values * scales * global_scale_inv.reciprocal()


def error_stats(ref: torch.Tensor, pred: torch.Tensor) -> tuple[float, float]:
    diff = pred.float() - ref.float()
    mse = torch.mean(diff * diff).item()
    max_abs_error = torch.max(torch.abs(diff)).item()
    return mse, max_abs_error


def quantize(provider: str, x: torch.Tensor, global_scale_inv: torch.Tensor):
    if provider == "vllm":
        from vllm import _custom_ops as ops

        return ops.scaled_fp4_quant(x, global_scale_inv, is_sf_swizzled_layout=True)
    if provider == "scalesweep_mse":
        return scalesweep_mse_nvfp4_quant(x, global_scale_inv, is_sf_swizzled_layout=True)
    raise ValueError(f"unknown provider: {provider}")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=[1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        x_log=False,
        line_arg="provider",
        line_vals=_enabled,
        line_names=_enabled,
        ylabel="us (lower is better)",
        plot_name="scalesweep_mse NVFP4 Quantization Latency (us)",
        args={},
    )
)
def benchmark(batch_size, provider, N, K):
    del N
    x = torch.randn((batch_size, K), device="cuda", dtype=torch.bfloat16)
    global_scale_inv = compute_global_scale_inv(x)
    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
        lambda: quantize(provider, x, global_scale_inv),
        quantiles=[0.5, 0.2, 0.8],
    )
    return ms * 1000, max_ms * 1000, min_ms * 1000


def prepare_shapes(args):
    out = []
    for model, tp_size in itertools.product(args.models, args.tp_sizes):
        for KN, tp_dim in copy.deepcopy(WEIGHT_SHAPES[model]):
            KN[tp_dim] //= tp_size
            KN.append(model)
            out.append(KN)
    return out


@torch.inference_mode()
def error_once(M: int, K: int, dtype: torch.dtype) -> dict[str, float]:
    x = torch.randn((M, K), device="cuda", dtype=dtype)
    global_scale_inv = compute_global_scale_inv(x)

    vllm_out, vllm_scale = quantize("vllm", x, global_scale_inv)
    scalesweep_mse_out, scalesweep_mse_scale = quantize("scalesweep_mse", x, global_scale_inv)

    vllm_reconstructed = dequantize_output(vllm_out, vllm_scale, global_scale_inv)
    scalesweep_mse_reconstructed = dequantize_output(
        scalesweep_mse_out,
        scalesweep_mse_scale,
        global_scale_inv,
    )

    vllm_mse, vllm_max_abs_error = error_stats(x, vllm_reconstructed)
    scalesweep_mse_mse, scalesweep_mse_max_abs_error = error_stats(
        x,
        scalesweep_mse_reconstructed,
    )
    return {
        "vllm_mse": vllm_mse,
        "vllm_max_abs_error": vllm_max_abs_error,
        "scalesweep_mse_mse": scalesweep_mse_mse,
        "scalesweep_mse_max_abs_error": scalesweep_mse_max_abs_error,
    }


def print_error_table(K: int, batches: list[int]):
    print("\nError vs vLLM backend")
    print("environment:", get_environment_info())
    print("| batch_size | K | vllm_mse | scalesweep_mse_mse | vllm_max_abs_error | scalesweep_mse_max_abs_error |")
    print("| --- | --- | --- | --- | --- | --- |")
    for M in batches:
        stats = error_once(M, K, torch.bfloat16)
        print(
            f"| {M} | {K} | {stats['vllm_mse']:.6g} | "
            f"{stats['scalesweep_mse_mse']:.6g} | "
            f"{stats['vllm_max_abs_error']:.6g} | "
            f"{stats['scalesweep_mse_max_abs_error']:.6g} |"
        )


@torch.inference_mode()
def benchmark_once(
    provider: str,
    batch_size: int,
    k: int,
    dtype: torch.dtype,
    bench_iters: int,
) -> dict:
    x = torch.randn((batch_size, k), device="cuda", dtype=dtype)
    global_scale_inv = compute_global_scale_inv(x)
    fn = SIMULATE_PROVIDERS[provider]

    fn(x, global_scale_inv, is_sf_swizzled_layout=True)
    torch.cuda.synchronize()

    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: fn(x, global_scale_inv, is_sf_swizzled_layout=True),
        warmup=10,
        rep=bench_iters,
        quantiles=[0.5, 0.2, 0.8],
    )
    out, scale = fn(x, global_scale_inv, is_sf_swizzled_layout=True)
    reconstructed = dequantize_output(out, scale, global_scale_inv)
    mse, max_abs_error = error_stats(x, reconstructed)
    return {
        "provider": provider,
        "batch_size": batch_size,
        "k": k,
        "dtype": str(dtype).replace("torch.", ""),
        "latency_us": ms * 1000,
        "latency_min_us": min_ms * 1000,
        "latency_max_us": max_ms * 1000,
        "mse": mse,
        "max_abs_error": max_abs_error,
    }


def run_simulate_benchmark(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("simulate fp4 benchmark requires CUDA")

    results = {
        "environment": get_environment_info(),
        "benchmark": {
            "simulate": True,
            "k": args.k,
            "batches": args.batches,
            "bench_iters": args.bench_iters,
            "dtype": "bfloat16",
        },
        "rows": [],
    }

    print("environment:", results["environment"])
    print("| provider | batch_size | K | latency_us | min_us | max_us | mse | max_abs_error |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for batch_size in args.batches:
        for provider in SIMULATE_PROVIDERS:
            row = benchmark_once(provider, batch_size, args.k, torch.bfloat16, args.bench_iters)
            results["rows"].append(row)
            print(
                f"| {provider} | {row['batch_size']} | {row['k']} | "
                f"{row['latency_us']:.6g} | {row['latency_min_us']:.6g} | "
                f"{row['latency_max_us']:.6g} | {row['mse']:.6g} | "
                f"{row['max_abs_error']:.6g} |"
            )

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nsaved: {save_path}")


def run_native_benchmark(args) -> None:
    from vllm.platforms import current_platform

    if not current_platform.has_device_capability(100):
        raise RuntimeError("NVFP4 requires compute capability of 10.0 (Blackwell)")

    for K, N, model in prepare_shapes(args):
        print(f"\n{model}, N={N} K={K}")
        if not args.skip_speed:
            benchmark.run(print_data=True, save_path=args.save_path, N=N, K=K)
        if not args.skip_error:
            print_error_table(K, args.error_batches)

    print("\nBenchmark finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark scalesweep_mse NVFP4 quantization against vLLM."
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run the simulate-FP4 benchmark instead of the native FP4/vLLM benchmark.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=["meta-llama/Llama-3.3-70B-Instruct"],
        choices=list(WEIGHT_SHAPES.keys()),
    )
    parser.add_argument("--tp-sizes", nargs="+", type=int, default=[1])
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--error-batches", nargs="+", type=int, default=[1, 1024, 8192])
    parser.add_argument("--skip-speed", action="store_true")
    parser.add_argument("--skip-error", action="store_true")
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 16, 64, 256, 1024])
    parser.add_argument("--bench-iters", type=int, default=100)
    args = parser.parse_args()

    if args.save_path is None:
        if args.simulate:
            args.save_path = "../result/bench_scalesweep_mse_simulate_fp4_results.json"
        else:
            args.save_path = "../result/bench_scalesweep_mse_nvfp4_quant_results"

    if args.simulate:
        run_simulate_benchmark(args)
    else:
        run_native_benchmark(args)
