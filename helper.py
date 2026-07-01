import importlib.metadata
import subprocess

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


def _driver_version_from_torch():
    get_driver_version = getattr(torch._C, "_cuda_getDriverVersion", None)
    if get_driver_version is None:
        return None
    version = get_driver_version()
    major = version // 1000
    minor = (version % 1000) // 10
    return f"{major}.{minor}"


def _driver_version_from_nvidia_smi():
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


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def get_environment_info():
    info = {
        "torch": torch.__version__,
        "vllm": _package_version("vllm"),
        "cuda": torch.version.cuda,
        "gpu_driver": _driver_version_from_torch() or _driver_version_from_nvidia_smi(),
        "gpu": None,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties("cuda")
        major, minor = torch.cuda.get_device_capability()
        info["gpu"] = {
            "name": props.name,
            "sm": major * 10 + minor,
            "capability": f"{major}.{minor}",
            "sm_count": props.multi_processor_count,
            "total_memory_bytes": props.total_memory,
        }
    return info


def make_gaussian(shape, *, seed=SEED, device=DEVICE, dtype=DTYPE, loc=0.0, scale=1.0):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    x = torch.normal(loc=loc, scale=scale, size=shape, generator=generator, device=device, dtype=dtype)
    return x


def make_w(M=4096, K=4096):
    return make_gaussian((M, K), seed=SEED)


def make_imp(kind, dim, device):
    if kind == "ones":
        return torch.ones((1, dim), device=device, dtype=torch.float32)
    if kind == "ramp":
        return torch.linspace(0.25, 2.0, dim, device=device, dtype=torch.float32).view(1, dim)
    if kind == "random":
        g = torch.Generator(device=device)
        g.manual_seed(123)
        return 0.25 + 1.75 * torch.rand((1, dim), device=device, dtype=torch.float32, generator=g)
    raise NotImplementedError(f"unsupported --imp {kind}")


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


def weighted_error_stats(weight, reconstructed, imp):
    err = reconstructed.float() - weight.float()
    weighted_mse = torch.mean(err * err * imp.float()).item()
    max_abs_error = torch.max(torch.abs(err)).item()
    return weighted_mse, max_abs_error


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
