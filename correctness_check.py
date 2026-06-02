from dataclasses import dataclass
from typing import Callable, Iterable

import torch

from fp4_bound_example import (
    LOWER_EXAMPLES,
    UPPER_EXAMPLES,
    ScaleExample,
    verify_scales_from_x,
)
from kernel_ScaleSweep import (
    LOWER_BOUND as SCALESWEEP_LOWER_BOUND,
    UPPER_BOUND as SCALESWEEP_UPPER_BOUND,
    scalesweep_quantize,
)
from kernel_ScaleSweep_MSE import (
    BLOCK_SIZE,
    LOWER_BOUND as SCALESWEEP_MSE_LOWER_BOUND,
    UPPER_BOUND as SCALESWEEP_MSE_UPPER_BOUND,
    scalesweep_quantize as mse_scalesweep_quantize,
)
from kernel_ScaleSweep_MSE_round import (
    LOWER_BOUND as SCALESWEEP_MSE_ROUND_LOWER_BOUND,
    UPPER_BOUND as SCALESWEEP_MSE_ROUND_UPPER_BOUND,
    scalesweep_quantize as mse_round_scalesweep_quantize,
)
from kernel_ScaleSweep_MSE_simulate_fp4 import (
    LOWER_BOUND as SCALESWEEP_MSE_SIMULATE_FP4_LOWER_BOUND,
    UPPER_BOUND as SCALESWEEP_MSE_SIMULATE_FP4_UPPER_BOUND,
    scalesweep_quantize as mse_scalesweep_simulate_fp4_quantize,
)
from kernel_ScaleSweep_MSE_simulate_fp4_round import (
    LOWER_BOUND as SCALESWEEP_MSE_SIMULATE_FP4_ROUND_LOWER_BOUND,
    UPPER_BOUND as SCALESWEEP_MSE_SIMULATE_FP4_ROUND_UPPER_BOUND,
    scalesweep_quantize as mse_round_scalesweep_simulate_fp4_quantize,
)
from kernel_ScaleSweep_simulate_fp4 import (
    LOWER_BOUND as SCALESWEEP_SIMULATE_FP4_LOWER_BOUND,
    UPPER_BOUND as SCALESWEEP_SIMULATE_FP4_UPPER_BOUND,
    scalesweep_quantize as scalesweep_simulate_fp4_quantize,
)


DEVICE = "cuda"


@dataclass(frozen=True)
class KernelCase:
    name: str
    lower_bound: int
    upper_bound: int
    quantize: Callable
    weighted: bool
    min_sm: int | None = None


@dataclass(frozen=True)
class Failure:
    kernel: str
    group: str
    index: int
    b: int
    reason: str


KERNELS = (
    KernelCase(
        name="ScaleSweep",
        lower_bound=SCALESWEEP_LOWER_BOUND,
        upper_bound=SCALESWEEP_UPPER_BOUND,
        quantize=scalesweep_quantize,
        weighted=True,
        min_sm=100,
    ),
    KernelCase(
        name="ScaleSweep_MSE",
        lower_bound=SCALESWEEP_MSE_LOWER_BOUND,
        upper_bound=SCALESWEEP_MSE_UPPER_BOUND,
        quantize=mse_scalesweep_quantize,
        weighted=False,
        min_sm=100,
    ),
    KernelCase(
        name="ScaleSweep_MSE_round",
        lower_bound=SCALESWEEP_MSE_ROUND_LOWER_BOUND,
        upper_bound=SCALESWEEP_MSE_ROUND_UPPER_BOUND,
        quantize=mse_round_scalesweep_quantize,
        weighted=False,
        min_sm=100,
    ),
    KernelCase(
        name="ScaleSweep_simulate_fp4",
        lower_bound=SCALESWEEP_SIMULATE_FP4_LOWER_BOUND,
        upper_bound=SCALESWEEP_SIMULATE_FP4_UPPER_BOUND,
        quantize=scalesweep_simulate_fp4_quantize,
        weighted=True,
    ),
    KernelCase(
        name="ScaleSweep_MSE_simulate_fp4",
        lower_bound=SCALESWEEP_MSE_SIMULATE_FP4_LOWER_BOUND,
        upper_bound=SCALESWEEP_MSE_SIMULATE_FP4_UPPER_BOUND,
        quantize=mse_scalesweep_simulate_fp4_quantize,
        weighted=False,
    ),
    KernelCase(
        name="ScaleSweep_MSE_simulate_fp4_round",
        lower_bound=SCALESWEEP_MSE_SIMULATE_FP4_ROUND_LOWER_BOUND,
        upper_bound=SCALESWEEP_MSE_SIMULATE_FP4_ROUND_UPPER_BOUND,
        quantize=mse_round_scalesweep_simulate_fp4_quantize,
        weighted=False,
    ),
)

EXAMPLE_GROUPS = (
    ("lower", LOWER_EXAMPLES),
    ("upper", UPPER_EXAMPLES),
)


def check_cuda_available() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run the Triton correctness checks")


def current_sm() -> int:
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def should_skip_kernel(kernel: KernelCase, sm: int) -> str | None:
    if kernel.min_sm is not None and sm < kernel.min_sm:
        return f"requires sm_{kernel.min_sm}, current device is sm_{sm}"
    return None


def check_examples_are_valid() -> list[Failure]:
    failures = []
    for group, examples in EXAMPLE_GROUPS:
        for index, ex in enumerate(examples):
            verified = verify_scales_from_x(ex)
            if not verified["all_ok"]:
                failures.append(
                    Failure(
                        kernel="fp4_bound_example",
                        group=group,
                        index=index,
                        b=ex.b,
                        reason=f"example does not infer expected scales: {verified}",
                    )
                )
    return failures


def quantize_examples(
    kernel: KernelCase,
    examples: Iterable[ScaleExample],
) -> tuple[list[float], list[int]]:
    weight = torch.tensor(
        [ex.x for ex in examples],
        device=DEVICE,
        dtype=torch.float32,
    ).contiguous()
    global_scale_inv = torch.tensor(1.0, device=DEVICE, dtype=torch.float32)

    if kernel.weighted:
        imp = torch.ones((1, weight.shape[-1]), device=DEVICE, dtype=torch.float32)
        scale, _ = kernel.quantize(
            weight,
            imp,
            global_scale_inv,
            BLOCK_SIZE,
            kernel.lower_bound,
            kernel.upper_bound,
        )
    else:
        scale, _ = kernel.quantize(
            weight,
            global_scale_inv,
            BLOCK_SIZE,
            kernel.lower_bound,
            kernel.upper_bound,
        )

    torch.cuda.synchronize()
    return (
        scale.reshape(-1).float().cpu().tolist(),
        scale.reshape(-1).view(torch.uint8).cpu().tolist(),
    )


def check_kernel(kernel: KernelCase, group: str, examples: list[ScaleExample]) -> list[Failure]:
    failures = []
    got_scales, got_bits = quantize_examples(kernel, examples)

    for index, (ex, got_scale, got_bit) in enumerate(zip(examples, got_scales, got_bits)):
        if got_bit != ex.target_bit:
            failures.append(
                Failure(
                    kernel=kernel.name,
                    group=group,
                    index=index,
                    b=ex.b,
                    reason=(
                        f"got scale {got_scale} (bit 0x{got_bit:02X}), expected "
                        f"{ex.target_scale} (bit 0x{ex.target_bit:02X}); "
                        f"base bit 0x{ex.base_bit:02X}, bit diff {ex.bit_diff}"
                    ),
                )
            )

    return failures


def main() -> None:
    check_cuda_available()

    sm = current_sm()
    failures = check_examples_are_valid()
    skipped = []
    checked = 0

    for kernel in KERNELS:
        print(
            f"checking {kernel.name} with bounds "
            f"[{kernel.lower_bound}, {kernel.upper_bound}]"
        )

        skip_reason = should_skip_kernel(kernel, sm)
        if skip_reason is not None:
            print(f"  SKIP: {skip_reason}")
            skipped.append((kernel.name, skip_reason))
            continue

        for group, examples in EXAMPLE_GROUPS:
            group_failures = check_kernel(kernel, group, examples)
            failures.extend(group_failures)
            checked += len(examples)
            if group_failures:
                print(f"  {group}: FAIL ({len(group_failures)} failures)")
            else:
                print(f"  {group}: OK ({len(examples)} examples)")

    if failures:
        print("\nFailures:")
        for failure in failures:
            print(
                f"- {failure.kernel} {failure.group}[{failure.index}] "
                f"b={failure.b}: {failure.reason}"
            )
        raise SystemExit(1)

    print(f"\nAll runnable ScaleSweep lower/upper checks passed ({checked} examples).")
    if skipped:
        print("Skipped kernels:")
        for name, reason in skipped:
            print(f"- {name}: {reason}")


if __name__ == "__main__":
    main()
