import torch

SEED = 0
DTYPE = torch.bfloat16
DEVICE = "cuda"

FP4_E2M1_MAX = 6.0
FP8_E4M3_MAX_NVFP4 = 448.0
FP8_E4M3_MAX_4OVER6 = 256.0


def check_sm100():
    assert torch.cuda.is_available()
    major, minor = torch.cuda.get_device_capability()
    assert major >= 10, f"need SM >= 100, got sm_{major}{minor}"


def make_w(M=4096, K=4096):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    return torch.randn((M, K), device=DEVICE, dtype=DTYPE).contiguous()


def get_nvfp4_global_scale(x, FP8_MAX=FP8_E4M3_MAX_NVFP4):
    amax = x.abs().max().float()
    global_scale = amax / (FP4_E2M1_MAX * FP8_MAX)
    global_scale_inv = global_scale.reciprocal()
    return global_scale, global_scale_inv


def error_stats(ref, pred):
    diff = pred.float() - ref.float()
    mse = torch.mean(diff * diff).item()
    max_abs_error = torch.max(torch.abs(diff)).item()
    return mse, max_abs_error


_FP4_E2M1_LUT = None


def fp4_e2m1_lut(device):
    global _FP4_E2M1_LUT
    if _FP4_E2M1_LUT is None or _FP4_E2M1_LUT.device != device:
        _FP4_E2M1_LUT = torch.tensor(
            [
                0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
            ],
            device=device,
            dtype=torch.float32,
        )
    return _FP4_E2M1_LUT


def unpack_fp4_e2m1(packed):
    assert packed.dtype == torch.uint8

    lut = fp4_e2m1_lut(packed.device)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    out = torch.stack((lut[low.long()], lut[high.long()]), dim=-1)

    return out.reshape(packed.shape[0], packed.shape[1] * 2)

def dequantize_base(q, scale_fp8, global_scale):
    values = unpack_fp4_e2m1(q)
    scales = scale_fp8.to(torch.float32).repeat_interleave(16, dim=1)
    return values * scales * global_scale.float()

def dequantize(kind, *args, **kwargs):
    if kind == "base":
        return dequantize_base(*args, **kwargs)

    raise ValueError(f"unknown dequant kind: {kind}")
