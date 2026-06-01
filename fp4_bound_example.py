from dataclasses import dataclass
from fractions import Fraction
from typing import Tuple, List


FP4_CODEBOOK = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


@dataclass(frozen=True)
class ScaleExample:
    b: int

    # base FP8 floor scale q_b = floor_fp8(max(x) / 6)
    base_scale: float
    base_bit: int

    # ratio is the only exact Fraction field
    target_r: Fraction
    target_r_float: float

    # target scale in original x-space
    target_scale: float
    target_bit: int
    bit_diff: int

    # raw vector, not normalized y
    x: Tuple[float, ...]

    # floor-bin information
    max_x: float
    unfloored_scale: float
    bin_next_scale: float
    floor_bin_ok: bool


def fp8_e4m3_value(bit: int) -> float:
    exp = ((bit >> 3) & 0xF) - 7
    mant = bit & 0x7
    return (1.0 + mant / 8.0) * (2.0 ** exp)


def fp8_positive_normal_scales() -> List[tuple[float, int]]:
    return [
        (fp8_e4m3_value((e << 3) | m), (e << 3) | m)
        for e in range(1, 15)
        for m in range(8)
    ]


FP8_SCALES = sorted(fp8_positive_normal_scales(), key=lambda t: t[0])


def fmt_bit(bit: int) -> str:
    return f"0x{bit:02X}"


def bit_for_scale(scale: float, tol: float = 1e-12) -> int:
    hits = [bit for s, bit in FP8_SCALES if abs(s - scale) <= tol * max(1.0, abs(scale))]
    if len(hits) != 1:
        raise ValueError(f"scale={scale} is not a unique positive normal FP8 E4M3 scale; hits={hits}")
    return hits[0]


def floor_fp8_scale(value: float) -> tuple[float, int]:
    candidates = [(s, bit) for s, bit in FP8_SCALES if s <= value]
    if not candidates:
        raise ValueError(f"value={value} is below minimum positive normal FP8 scale")
    return candidates[-1]


def loss_x(scale: float, x: Tuple[float, ...]) -> float:
    return sum(min((v - scale * c) ** 2 for c in FP4_CODEBOOK) for v in x)


def best_fp8_scale_from_x(x: Tuple[float, ...]) -> tuple[float, int]:
    return min(
        ((loss_x(scale, x), scale, bit) for scale, bit in FP8_SCALES),
        key=lambda t: (t[0], t[1]),
    )[1:]


def make_example(b: int, target_r: Fraction, x: Tuple[float, ...]) -> ScaleExample:
    x = tuple(float(v) for v in x)

    max_x = max(x)
    unfloored_scale = max_x / 6.0

    base_scale, base_bit = floor_fp8_scale(unfloored_scale)
    if (base_bit & 0x7) != b:
        raise ValueError(
            f"b mismatch: given b={b}, inferred mantissa={base_bit & 0x7}, "
            f"base_scale={base_scale}, base_bit={fmt_bit(base_bit)}"
        )

    target_r_float = float(target_r)
    target_scale = base_scale * target_r_float
    target_bit = bit_for_scale(target_scale)

    bin_next_scale = fp8_e4m3_value(base_bit + 1)
    floor_bin_ok = base_scale <= unfloored_scale < bin_next_scale

    return ScaleExample(
        b=b,
        base_scale=base_scale,
        base_bit=base_bit,
        target_r=target_r,
        target_r_float=target_r_float,
        target_scale=target_scale,
        target_bit=target_bit,
        bit_diff=target_bit - base_bit,
        x=x,
        max_x=max_x,
        unfloored_scale=unfloored_scale,
        bin_next_scale=bin_next_scale,
        floor_bin_ok=floor_bin_ok,
    )


def verify_scales_from_x(ex: ScaleExample, tol: float = 1e-12) -> dict:
    inferred_base_scale, inferred_base_bit = floor_fp8_scale(max(ex.x) / 6.0)
    inferred_target_scale, inferred_target_bit = best_fp8_scale_from_x(ex.x)

    base_ok = (
        abs(ex.base_scale - inferred_base_scale) <= tol
        and ex.base_bit == inferred_base_bit
    )
    target_ok = (
        abs(ex.target_scale - inferred_target_scale) <= tol
        and ex.target_bit == inferred_target_bit
    )

    return {
        "b": ex.b,
        "base_scale": ex.base_scale,
        "base_bit": fmt_bit(ex.base_bit),
        "inferred_base_scale": inferred_base_scale,
        "inferred_base_bit": fmt_bit(inferred_base_bit),
        "base_ok": base_ok,

        "target_r": ex.target_r,
        "target_r_float": ex.target_r_float,
        "target_scale": ex.target_scale,
        "target_bit": fmt_bit(ex.target_bit),
        "inferred_target_scale": inferred_target_scale,
        "inferred_target_bit": fmt_bit(inferred_target_bit),
        "target_ok": target_ok,

        "bit_diff": ex.bit_diff,
        "max_x": ex.max_x,
        "unfloored_scale": ex.unfloored_scale,
        "bin_next_scale": ex.bin_next_scale,
        "floor_bin_ok": ex.floor_bin_ok,
        "all_ok": base_ok and target_ok and ex.floor_bin_ok,
    }


LOWER_EXAMPLES = [
    make_example(0, Fraction(13, 16), (
        6, 0.405, 0.405, 0.405, 0.405, 0.405, 0.405, 0.405,
        0.405, 4.435, 4.94, 4.94, 4.94, 4.94, 4.94, 4.94,
    )),
    make_example(1, Fraction(5, 6), (
        6.75, 0.43875, 0.43875, 0.43875, 0.43875, 0.43875, 0.43875, 0.43875,
        0.43875, 5.068125, 5.248125, 5.563125, 5.563125, 5.563125, 6.06375, 6.06375,
    )),
    make_example(2, Fraction(4, 5), (
        7.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        0.5, 0.5, 0.5, 5.80625, 5.80625, 5.80625, 6.2375, 6.50625,
    )),
    make_example(3, Fraction(9, 11), (
        8.25, 0.56375, 0.56375, 0.56375, 0.56375, 0.56375, 0.56375, 0.56375,
        0.56375, 0.56375, 4.310625, 6.373125, 6.373125, 6.620625, 7.1225, 7.1225,
    )),
    make_example(4, Fraction(5, 6), (
        9, 0.6225, 0.6225, 0.6225, 0.6225, 0.6225, 0.6225, 0.6225,
        0.6225, 4.8075, 4.8075, 7.2525, 7.2525, 7.2525, 7.8675, 7.8675,
    )),
    make_example(5, Fraction(11, 13), (
        9.75, 0.6825, 0.6825, 0.6825, 0.6825, 0.6825, 0.6825, 0.6825,
        0.6825, 5.24875, 5.24875, 7.62125, 7.873125, 8.636875, 8.636875, 8.636875,
    )),
    make_example(6, Fraction(6, 7), (
        10.5, 0.69125, 0.69125, 0.69125, 0.69125, 0.69125, 0.69125, 0.69125,
        5.6875, 5.6875, 8.25125, 8.61875, 8.61875, 9.37125, 9.37125, 9.37125,
    )),
    make_example(7, Fraction(4, 5), (
        11.25, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75,
        2.25, 2.25, 2.25, 8.634375, 8.634375, 8.634375, 9.50625, 9.50625,
    )),
]


UPPER_EXAMPLES = [
    make_example(0, Fraction(15, 8), (
        5.64, 3.721, 1.887, 5.641, 5.652, 0.945, 3.743, 6.749,
        5.608, 5.656, 2.797, 1.872, 3.616, 5.665, 3.76, 1.864,
    )),
    make_example(1, Fraction(16, 9), (
        5.99175, 6.022125, 5.99625, 0.0585, 6.364125, 4.0185, 6.50025, 0.979875,
        1.50975, 2.964375, 7.486875, 6.001875, 0, 4.026375, 6.0075, 2.92275,
    )),
    make_example(2, Fraction(9, 5), (
        1.70375, 6.765, 0.065, 8.21875, 3.40375, 4.49, 6.78125, 6.7675,
        1.1125, 0.13125, 6.79125, 4.46375, 3.4475, 4.47875, 4.65875, 8.16125,
    )),
    make_example(3, Fraction(20, 11), (
        7.4855, 7.418125, 0.048125, 0.034375, 8.160625, 7.48275, 2.520375, 2.569875,
        8.7835, 3.76475, 7.43325, 4.81525, 3.724875, 7.469, 7.46075, 8.98975,
    )),
    make_example(4, Fraction(11, 6), (
        2.7855, 8.295, 8.361, 9.744, 8.2695, 1.4205, 8.136, 2.64,
        1.818, 9.6825, 5.502, 8.2605, 8.298, 1.3695, 1.2195, 2.871,
    )),
    make_example(5, Fraction(22, 13), (
        2.770625, 8.827, 8.28425, 5.5185, 8.155875, 8.185125, 1.389375, 1.3975,
        1.378, 8.250125, 1.3975, 2.734875, 2.702375, 10.4715, 5.53475, 8.289125,
    )),
    make_example(6, Fraction(12, 7), (
        9.07725, 8.99675, 1.56975, 6.16875, 5.908, 1.54175, 11.193, 4.4835,
        3.01, 9.156, 1.407, 9.1385, 9.1, 4.59375, 0, 6.118,
    )),
    make_example(7, Fraction(26, 15), (
        9.795, 9.5175, 9.695625, 7.565625, 11.95875, 6.55125, 0, 1.674375,
        9.70875, 9.8325, 3.1725, 0.03, 7.306875, 4.87875, 9.73125, 6.525,
    )),
]


def check_examples(examples: List[ScaleExample]) -> None:
    for ex in examples:
        v = verify_scales_from_x(ex)
        assert v["all_ok"], v


def print_table(name: str, examples: List[ScaleExample]) -> None:
    print(f"\n{name}")
    for ex in examples:
        v = verify_scales_from_x(ex)
        print(v)


if __name__ == "__main__":
    check_examples(LOWER_EXAMPLES)
    check_examples(UPPER_EXAMPLES)

    print_table("LOWER_EXAMPLES", LOWER_EXAMPLES)
    print_table("UPPER_EXAMPLES", UPPER_EXAMPLES)
