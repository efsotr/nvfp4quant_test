import sys
from pathlib import Path

import pytest
import torch

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(ROOT_DIR))

from fp4_bound_example import LOWER_EXAMPLES, UPPER_EXAMPLES
from scalesweep_mse_nvfp4_quant import (
    BLOCK_SIZE,
    FP4_E2M1_MAX,
    FP8_E4M3_MAX,
    LOWER_BOUND,
    REF_MAX_SCALE_RAW,
    UPPER_BOUND,
    create_fp4_output_tensors,
    round_up,
    scalesweep_mse_nvfp4_quant,
)
from scalesweep_mse_nvfp4_quant_simulate import (
    scalesweep_mse_nvfp4_quant_simulate,
)

if not torch.cuda.is_available():
    pytest.skip(reason="NVFP4 quantization tests require CUDA.", allow_module_level=True)

DTYPES = [torch.float16, torch.bfloat16]
SHAPES = [(1, 16), (3, 64), (32, 128), (128, 64), (150, 80)]
CUDA_DEVICES = ["cuda:0"]

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


def cast_from_fp4(x: torch.Tensor, m: int, n: int) -> torch.Tensor:
    v_2nd = x & 0xF
    v_1st = (x >> 4) & 0xF
    c = torch.stack((v_2nd, v_1st), dim=-1)
    lut = torch.tensor(E2M1_TO_FLOAT32, device=x.device, dtype=torch.float32)
    return lut[c.long()].reshape(m, n)


def cast_to_fp4(x: torch.Tensor) -> torch.Tensor:
    sign = torch.sign(x)
    x = torch.abs(x)
    out = torch.empty_like(x)
    out[(x >= 0.0) & (x <= 0.25)] = 0.0
    out[(x > 0.25) & (x < 0.75)] = 0.5
    out[(x >= 0.75) & (x <= 1.25)] = 1.0
    out[(x > 1.25) & (x < 1.75)] = 1.5
    out[(x >= 1.75) & (x <= 2.5)] = 2.0
    out[(x > 2.5) & (x < 3.5)] = 3.0
    out[(x >= 3.5) & (x <= 5.0)] = 4.0
    out[x > 5.0] = 6.0
    return out * sign


def cast_to_fp4_simulate(x: torch.Tensor) -> torch.Tensor:
    sign = torch.sign(x)
    ax = torch.abs(x)
    exp = torch.where(
        ax <= 2.0,
        0.5,
        torch.where(ax <= 4.0, 1.0, 2.0),
    )
    q = torch.floor(ax / exp + 0.5) * exp
    return torch.clamp(q, max=6.0) * sign


def compute_global_scale_inv(x: torch.Tensor) -> torch.Tensor:
    tensor_amax = torch.abs(x).max().to(torch.float32)
    return FP8_E4M3_MAX * FP4_E2M1_MAX / tensor_amax


def recover_swizzled_scales(scale: torch.Tensor, m: int, n: int) -> torch.Tensor:
    scale_n = n // BLOCK_SIZE
    rounded_m = round_up(m, 128)
    rounded_n = round_up(scale_n, 4)
    tmp = torch.reshape(scale, (1, rounded_m // 128, rounded_n // 4, 32, 4, 4))
    tmp = torch.permute(tmp, (0, 1, 4, 3, 2, 5))
    result = torch.reshape(tmp, (rounded_m, rounded_n)).to(torch.float32)
    return result[:m, :scale_n]


def ref_scalesweep_mse_nvfp4_quant(
    x: torch.Tensor,
    global_scale_inv: torch.Tensor,
    simulate: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert global_scale_inv.dtype == torch.float32
    assert x.ndim == 2
    m, n = x.shape
    blocks = x.reshape(m, n // BLOCK_SIZE, BLOCK_SIZE).to(torch.float32)
    blocks = blocks * global_scale_inv

    abs_max = torch.abs(blocks).amax(dim=-1)
    base_scale = abs_max * (1.0 / FP4_E2M1_MAX)
    base_raw = base_scale.to(torch.float8_e4m3fn).view(torch.uint8).to(torch.int32)
    offsets = torch.arange(LOWER_BOUND, UPPER_BOUND + 1, device=x.device, dtype=torch.int32)
    scale_raw = torch.clamp(base_raw.unsqueeze(-1) + offsets, 1, REF_MAX_SCALE_RAW).to(torch.uint8)
    scales = scale_raw.view(torch.float8_e4m3fn).to(torch.float32)

    scaled = blocks.unsqueeze(2) / scales.unsqueeze(-1)
    fp4_round = cast_to_fp4_simulate if simulate else cast_to_fp4
    quantized = fp4_round(scaled)
    reconstructed = quantized * scales.unsqueeze(-1)
    squared_error = torch.sum((reconstructed - blocks.unsqueeze(2)) ** 2, dim=-1)
    best_index = torch.argmin(squared_error, dim=-1)

    best_scale_raw = torch.gather(scale_raw, dim=2, index=best_index.unsqueeze(-1)).squeeze(-1)
    best_scale = best_scale_raw.view(torch.float8_e4m3fn)
    best_quantized = torch.gather(
        quantized,
        dim=2,
        index=best_index[:, :, None, None].expand(-1, -1, 1, BLOCK_SIZE),
    ).squeeze(2)

    return best_quantized.reshape(m, n), best_scale


def selected_quantizer(request):
    simulate = request.config.getoption("--simulate")
    if simulate:
        return scalesweep_mse_nvfp4_quant_simulate, True

    major, minor = torch.cuda.get_device_capability()
    if major < 10:
        pytest.skip(
            reason=f"NVFP4 requires compute capability of 10 or above, got {major}.{minor}.",
        )
    return scalesweep_mse_nvfp4_quant, False


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("is_sf_swizzled_layout", [True, False])
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_scalesweep_mse_nvfp4_quant(
    request,
    dtype: torch.dtype,
    shape: tuple[int, int],
    is_sf_swizzled_layout: bool,
    device: str,
) -> None:
    quantizer, simulate = selected_quantizer(request)
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    x = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    global_scale_inv = compute_global_scale_inv(x)

    out_ref, scale_ref = ref_scalesweep_mse_nvfp4_quant(x, global_scale_inv, simulate)
    out, out_scale = quantizer(
        x,
        global_scale_inv,
        is_sf_swizzled_layout=is_sf_swizzled_layout,
    )
    expected_out, expected_scale = create_fp4_output_tensors(
        shape[0],
        shape[1],
        torch.device(device),
        is_sf_swizzled_layout,
    )

    assert out.shape == expected_out.shape
    assert out.dtype == expected_out.dtype
    assert out_scale.shape == expected_scale.shape
    assert out_scale.dtype == torch.float8_e4m3fn

    out_ans = cast_from_fp4(out, *shape)
    if is_sf_swizzled_layout:
        scale_ans = recover_swizzled_scales(out_scale, *shape)
    else:
        scale_ans = out_scale.to(torch.float32)

    torch.testing.assert_close(out_ans, out_ref)
    torch.testing.assert_close(scale_ans, scale_ref.to(torch.float32))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("is_sf_swizzled_layout", [True, False])
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_scalesweep_mse_nvfp4_quant_bound_examples(
    request,
    dtype: torch.dtype,
    is_sf_swizzled_layout: bool,
    device: str,
) -> None:
    quantizer, simulate = selected_quantizer(request)
    examples = LOWER_EXAMPLES + UPPER_EXAMPLES
    x = torch.tensor([ex.x for ex in examples], device=device, dtype=dtype)
    global_scale_inv = torch.tensor(1.0, device=device, dtype=torch.float32)

    out_ref, scale_ref = ref_scalesweep_mse_nvfp4_quant(x, global_scale_inv, simulate)
    out, out_scale = quantizer(
        x,
        global_scale_inv,
        is_sf_swizzled_layout=is_sf_swizzled_layout,
    )

    out_ans = cast_from_fp4(out, *x.shape)
    if is_sf_swizzled_layout:
        scale_ans = recover_swizzled_scales(out_scale, *x.shape)
    else:
        scale_ans = out_scale.to(torch.float32)

    expected_raw = torch.tensor(
        [ex.target_bit for ex in examples],
        device=device,
        dtype=torch.uint8,
    ).view(torch.float8_e4m3fn).to(torch.float32)
    torch.testing.assert_close(out_ans, out_ref)
    torch.testing.assert_close(scale_ans, scale_ref.to(torch.float32))
    torch.testing.assert_close(scale_ans[:, 0], expected_raw)
