import torch

try:
    from torch.library import register_fake
except ImportError:
    from torch.library import impl_abstract as register_fake

def round_up(x: int, y: int) -> int:
    """Round up x to the nearest multiple of y."""
    return ((x + y - 1) // y) * y

def create_fp4_scale_tensor(
    m: int,
    n: int,
    device: torch.device,
    is_sf_swizzled_layout: bool,
) -> torch.Tensor:
    """
    Allocate the output scale tensor for scaled_fp4_quant.

    When is_sf_swizzled_layout=True, we use rounded values to store the
    swizzled scales. Due to the requirement of the Tensor Core, the minimum
    tile is 128x4 for the scales. So, we first pad the scales to multiples
    of 128 (rows) and 4 (cols). Then, the scales (in float8_e4m3fn) are
    packed into an int32 for every 4 values. More:
    https://docs.nvidia.com/cuda/parallel-thread-execution/
    #tcgen05-mma-scale-factor-b-layout-4x
    """

    block_size = 16
    if is_sf_swizzled_layout:
        rounded_m = round_up(m, 128)
        scale_n = n // block_size
        rounded_n = round_up(scale_n, 4)
        return torch.empty(
            (rounded_m, rounded_n // 4), device=device, dtype=torch.int32
        )
    else:
        return torch.empty((m, n // block_size), device=device, dtype=torch.uint8)

def create_fp4_output_tensors(
    m: int,
    n: int,
    device: torch.device,
    is_sf_swizzled_layout: bool,
    padded_n: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Allocate both output tensors for scaled_fp4_quant:
    (quantized_output, output_scale).

    Must match the C++ scaled_fp4_quant_func allocation exactly when
    ``padded_n`` is ``None``. When ``padded_n`` is provided, allocate a larger
    packed-FP4 output/scale buffer so the quantization kernel can write
    CUTLASS-compatible K padding directly
    """
    physical_n = padded_n if padded_n is not None else n
    output = torch.empty((m, physical_n // 2), device=device, dtype=torch.uint8)
    output_scale = create_fp4_scale_tensor(m, physical_n, device, is_sf_swizzled_layout)
    return output, output_scale

if hasattr(torch.ops, "_C") and hasattr(torch.ops._C, "scaled_fp4_quant"):

    @register_fake("_C::scaled_fp4_quant")
    def _scaled_fp4_quant_fake(
        input: torch.Tensor,
        input_scale: torch.Tensor,
        is_sf_swizzled_layout: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = input.shape[-1]
        m = input.numel() // n
        return create_fp4_output_tensors(m, n, input.device, is_sf_swizzled_layout)

    @register_fake("_C::scaled_fp4_quant.out")
    def _scaled_fp4_quant_out_fake(
        input: torch.Tensor,
        input_scale: torch.Tensor,
        is_sf_swizzled_layout: bool,
        *,
        output: torch.Tensor,
        output_scale: torch.Tensor,
    ) -> None:
        return None

# fp4
def scaled_fp4_quant(
    input: torch.Tensor,
    input_global_scale: torch.Tensor,
    is_sf_swizzled_layout: bool = True,
    backend: str = "none",
    padded_n: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to FP4 and return quantized tensor and scale.

    This function quantizes the last dimension of the given tensor `input`. For
    every 16 consecutive elements, a single dynamically computed scaling factor
    is shared. This scaling factor is quantized using the `input_global_scale`
    and is stored in a swizzled layout (see
    https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-b-layout-4x).

    Args:
        input: The input tensor to be quantized to FP4
        input_global_scale: A scalar scaling factor for the entire tensor.
        use_8x4_sf_layout: Whether to use the 8x4 or 128x4 layout for the scaling
        padded_n: Optional padded K dimension. When provided, the quantized
            output and scale tensors are allocated for ``padded_n``

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The output tensor in FP4 but every
            two values are packed into a uint8 and float8_e4m3 scaling factors
            in the sizzled layout.
    """
    # assert not current_platform.is_rocm()
    assert input.ndim >= 1, f"input.ndim needs to be >= 1, but got {input.ndim}."
    other_dims = 1 if input.ndim == 1 else -1
    input = input.reshape(other_dims, input.shape[-1])
    m, n = input.shape
    block_size = 16

    assert n % block_size == 0, f"last dim has to be multiple of 16, but got {n}."
    assert input.dtype in (torch.float16, torch.bfloat16), (
        f"input.dtype needs to be fp16 or bf16 but got {input.dtype}."
    )
    if padded_n is not None:
        assert padded_n >= n, f"padded_n must be >= n, got padded_n={padded_n}, n={n}."
        assert padded_n % block_size == 0, (
            f"padded_n has to be a multiple of {block_size}, but got {padded_n}."
        )

    use_8x4_sf_layout = True if "trtllm" in backend and m <= 32 else False  # noqa: SIM210
    if use_8x4_sf_layout and padded_n is not None and padded_n != n:
        # TODO: support this case
        raise ValueError("padded_n is not supported with TRTLLM 8x4 scale layout.")
    if use_8x4_sf_layout:
        output, output_scale = flashinfer_quant_nvfp4_8x4_sf_layout(
            input, input_global_scale
        )
    else:
        # Pre-allocate and call .out variant (same behavior as old in-place API)
        output, output_scale = create_fp4_output_tensors(
            m,
            n,
            input.device,
            is_sf_swizzled_layout,
            padded_n=padded_n,
        )
        torch.ops._C.scaled_fp4_quant.out(
            input,
            input_global_scale,
            is_sf_swizzled_layout,
            output=output,
            output_scale=output_scale,
        )

    output_scale = output_scale.view(torch.float8_e4m3fn)
    return output, output_scale