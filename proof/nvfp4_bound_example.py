from fractions import Fraction
from typing import List, Tuple

import numpy as np
from helper import *

BoundKey = Tuple[str, str, int, int]
Example = Tuple[BoundKey, List[float]]

# Format:
#   ((upper/lower, normal/subnormal/saturation, b, signed_offset), y_values)
#
# q0_bits is the positive FP8 bit-pattern, not the numeric FP8 value.
# It is reconstructed from part and b:
#   normal:     q0_bits = (7 << 3) | b, b = 0,...,7
#   subnormal:  q0_bits = b,          b = 1,...,7
#   saturation: q0_bits = 120 + b,    b = 0,...,6
#
# The verifier forms
#   q0_value = value(q0_bits), x_i = q0_value * y_i,
# and checks:
#   1. len(y_values) == 16,
#   2. round_FP8(max_i |x_i| / 6) returns the FP8 value whose bit-pattern
#      is q0_bits,
#   3. the unique best positive finite FP8 scale for x has bit-pattern
#      q0_bits + signed_offset.

EXAMPLES: List[Example] = [
    (("upper", "normal", 0, 6), [
        2.625, 2.625, 3.4375, 5.062813,
        5.062813, 5.37625, 5.37625, 5.37625,
        5.37625, 5.37625, 5.37625, 5.37625,
        5.37625, 5.37625, 6.374994, 6.374994,
    ]),
    (("upper", "normal", 1, 7), [
        2.666667, 3.611111, 3.611111, 3.611111,
        5.166944, 5.555556, 5.555556, 5.555556,
        5.555556, 5.555556, 5.555556, 5.555556,
        5.555556, 6.333327, 6.333327, 6.333327,
    ]),
    (("upper", "normal", 2, 6), [
        2.4, 2.4, 2.4, 2.4,
        5.00325, 5.00325, 5.20275, 5.20275,
        5.20275, 5.20275, 5.20275, 5.20275,
        6.299994, 6.299994, 6.299994, 6.299994,
    ]),
    (("upper", "normal", 3, 6), [
        2.454545, 2.454545, 2.454545, 2.454545,
        2.454545, 5.1175, 5.1175, 5.1175,
        5.1175, 5.1175, 5.180227, 6.272721,
        6.272721, 6.272721, 6.272721, 6.272721,
    ]),
    (("upper", "normal", 4, 6), [
        2.5, 2.5, 2.5, 2.5,
        4.833333, 5.166667, 5.166667, 5.166667,
        5.166667, 5.166667, 5.166667, 5.166667,
        6.249994, 6.249994, 6.249994, 6.249994,
    ]),
    (("upper", "normal", 5, 6), [
        2.538462, 2.538462, 2.538462, 4.846154,
        4.846154, 4.846154, 4.846154, 5.306538,
        5.306538, 5.306538, 5.306538, 5.306538,
        5.306538, 5.306538, 6.230763, 6.230763,
    ]),
    (("upper", "normal", 6, 6), [
        2.571429, 2.571429, 4.8575, 4.8575,
        4.8575, 4.992143, 5.357143, 5.357143,
        5.357143, 5.357143, 5.357143, 5.357143,
        5.357143, 6.214279, 6.214279, 6.214279,
    ]),
    (("upper", "normal", 7, 6), [
        2.6, 2.6, 3.333333, 5.001333,
        5.001333, 5.001333, 5.001333, 5.001333,
        5.001333, 5.5335, 5.5335, 5.5335,
        5.5335, 5.5335, 6.199994, 6.199994,
    ]),

    (("upper", "subnormal", 1, 1), [
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 3.75, 7.5,
        7.5, 7.5, 7.5, 8.999991,
    ]),
    (("upper", "subnormal", 2, 2), [
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 6.15625,
        6.15625, 6.24375, 6.24375, 6.24375,
        7.499993, 7.499993, 7.499993, 7.499993,
    ]),
    (("upper", "subnormal", 3, 2), [
        0.75, 0.75, 0.75, 0.75,
        0.75, 0.75, 0.75, 5.168333,
        5.168333, 5.168333, 5.168333, 5.833333,
        5.833333, 6.585833, 6.585833, 6.999993,
    ]),
    (("upper", "subnormal", 4, 3), [
        0.875, 0.875, 0.875, 2.625,
        2.625, 5.439375, 5.439375, 5.439375,
        5.619375, 5.619375, 6.749993, 6.749993,
        6.749993, 6.749993, 6.749993, 6.749993,
    ]),
    (("upper", "subnormal", 5, 4), [
        2.7, 2.7, 2.7, 2.7,
        3.5, 3.5, 3.5, 3.5,
        5.302, 5.5165, 5.5165, 5.5165,
        5.5165, 5.5165, 6.599993, 6.599993,
    ]),
    (("upper", "subnormal", 6, 5), [
        2.75, 2.75, 2.75, 3.5,
        3.5, 3.5, 3.5, 5.416667,
        5.416667, 5.416667, 5.416667, 5.7525,
        5.7525, 5.7525, 5.7525, 6.499994,
    ]),
    (("upper", "subnormal", 7, 5), [
        2.571429, 2.571429, 2.571429, 5.008929,
        5.357143, 5.357143, 5.357143, 5.357143,
        5.357143, 5.357143, 5.357143, 5.357143,
        6.428565, 6.428565, 6.428565, 6.428565,
    ]),

    (("lower", "normal", 0, -3), [
        0.375, 0.375, 0.375, 0.375,
        0.375, 0.378516, 0.378516, 0.378516,
        3.125, 4.434609, 4.434609, 4.936641,
        4.936641, 4.936641, 4.936641, 5.812506,
    ]),
    (("lower", "normal", 1, -3), [
        0.361111, 0.361111, 0.361111, 0.361111,
        0.361111, 0.361111, 0.361111, 0.361111,
        0.361111, 4.166667, 4.333333, 4.333333,
        4.777708, 4.777708, 4.999375, 5.666672,
    ]),
    (("lower", "normal", 2, -2), [
        0.375, 0.375, 0.375, 0.375,
        0.375, 0.375, 0.375, 0.375,
        0.375, 3.098813, 3.15, 3.15,
        4.650188, 4.650188, 5.099062, 5.700006,
    ]),
    (("lower", "normal", 3, -2), [
        0.409091, 0.409091, 0.409091, 0.409091,
        0.409091, 0.409091, 0.409091, 2.999148,
        3.136364, 3.136364, 3.136364, 4.818239,
        4.818239, 4.818239, 5.0025, 5.727278,
    ]),
    (("lower", "normal", 4, -2), [
        0.416667, 0.416667, 0.416667, 0.416667,
        0.416667, 0.416667, 0.416667, 3.208333,
        3.208333, 3.208333, 3.208333, 4.666667,
        4.666667, 5.246094, 5.246094, 5.750006,
    ]),
    (("lower", "normal", 5, -3), [
        0.384615, 0.384615, 4.846154, 0.384615,
        0.384615, 4.846154, 4.307692, 0.384615,
        4.846154, 4.846154, 0.384615, 4.307692,
        0.384615, 0.384615, 0.384615, 5.769237,
    ]),
    (("lower", "normal", 6, -3), [
        0.392857, 0.392857, 0.392857, 0.392857,
        0.392857, 0.392857, 0.392857, 0.392857,
        0.392857, 3.0, 4.357768, 4.501473,
        4.928705, 4.928705, 5.142857, 5.78572,
    ]),
    (("lower", "normal", 7, -3), [
        0.366667, 0.366667, 0.366667, 0.366667,
        0.366667, 0.391375, 0.391375, 0.391375,
        0.391375, 3.033333, 4.398125, 4.599625,
        4.599625, 5.06075, 5.06075, 5.800006,
    ]),

    (("lower", "subnormal", 1, 0), [
        0.5, 0.5, 0.5, 0.5,
        0.5, 0.5, 0.5, 0.5,
        0.5, 0.5, 0.5, 0.5,
        0.5, 0.5, 0.5, 6.0,
    ]),
    (("lower", "subnormal", 2, 0), [
        0.5, 0.5, 0.5, 0.5,
        0.5, 0.5, 0.5, 0.5,
        0.5, 0.5, 0.5, 3.75,
        3.75, 3.75, 3.75, 6.0,
    ]),
    (("lower", "subnormal", 3, -1), [
        0.333333, 0.333333, 0.333333, 0.333333,
        0.333333, 0.333333, 0.333333, 0.333333,
        0.333333, 0.333333, 0.333333, 0.333333,
        0.333333, 0.333333, 2.5, 5.000005,
    ]),
    (("lower", "subnormal", 4, -1), [
        0.375, 0.375, 0.375, 0.375,
        0.375, 0.375, 0.375, 0.375,
        0.375, 0.375, 4.375, 4.375,
        4.375, 4.687031, 4.687031, 5.250005,
    ]),
    (("lower", "subnormal", 5, -1), [
        0.4, 0.4, 0.4, 0.4,
        0.4, 0.4, 0.4, 0.4,
        0.4, 3.0, 3.0, 3.0,
        3.0, 4.9, 4.9, 5.400005,
    ]),
    (("lower", "subnormal", 6, -1), [
        0.416667, 0.416667, 0.416667, 0.416667,
        0.416667, 0.416667, 0.416667, 0.416667,
        0.416667, 3.333333, 3.333333, 3.333333,
        5.1675, 5.1675, 5.500625, 5.500006,
    ]),
    (("lower", "subnormal", 7, -1), [
        0.428571, 0.428571, 0.428571, 0.428571,
        0.428571, 0.428571, 0.428571, 0.428571,
        0.428571, 3.214286, 3.214286, 3.214286,
        5.0, 5.0, 5.357143, 5.571434,
    ]),
    (("lower", "saturation", 0, -3), [
        0.375, 0.375, 0.375, 0.375,
        0.375, 0.378516, 0.378516, 0.378516,
        3.125, 4.434609, 4.434609, 4.936641,
        4.936641, 4.936641, 4.936641, 5.812506,
    ]),
    (("lower", "saturation", 1, -3), [
        0.361111, 0.361111, 0.361111, 0.361111,
        0.361111, 0.361111, 0.361111, 0.361111,
        0.361111, 4.166667, 4.333333, 4.333333,
        4.777708, 4.777708, 4.999375, 5.666672,
    ]),
    (("lower", "saturation", 2, -3), [
        3.0, 3.0, 3.0, 3.0,
        3.0, 3.0, 4.199995, 4.199995,
        4.200005, 4.55, 4.666667, 4.666667,
        4.899995, 4.899995, 4.899995, 5.700005,
    ]),
    (("lower", "saturation", 3, -3), [
        2.863636, 2.863636, 2.863636, 2.863636,
        2.863636, 2.95455, 2.95455, 2.95455,
        2.95455, 2.95455, 4.45454, 4.45454,
        4.666667, 4.846154, 4.846154, 5.727278,
    ]),
    (("lower", "saturation", 4, -3), [
        2.499995, 2.499995, 2.5, 2.916672,
        2.916672, 3.0, 3.0, 3.0,
        3.0, 3.0, 3.0, 3.0,
        3.0, 4.583338, 4.9, 5.750005,
    ]),
    (("lower", "saturation", 5, -3), [
        0.384615, 0.384615, 4.846154, 0.384615,
        0.384615, 4.846154, 4.307692, 0.384615,
        4.846154, 4.846154, 0.384615, 4.307692,
        0.384615, 0.384615, 0.384615, 5.769237,
    ]),
    (("lower", "saturation", 6, -4), [
        2.750005, 2.750005, 3.928571, 3.928571,
        4.166667, 4.285709, 4.285709, 4.285709,
        4.285714, 4.333333, 4.375, 4.666667,
        5.0, 5.0, 5.0, 5.785719,
    ]),
]


def q0_bits_from_part_b(part: str, b: int) -> int:
    if part == "normal":
        assert 0 <= b <= 7
        return (7 << 3) | b
    if part == "subnormal":
        assert 1 <= b <= 7
        return b
    if part == "saturation":
        assert 0 <= b <= 6
        return 120 + b
    raise ValueError(f"unknown part: {part}")


def round_fp8_bits(x: float) -> int:
    """Round positive scalar x to nearest positive finite FP8 E4M3 bits."""
    assert x > 0
    vals = [(abs(float(v) - x), bits) for bits, v in FP8]
    vals.sort()
    return vals[0][1]


def loss_for_bits(x_values: np.ndarray, scale_bits: int) -> float:
    scale = float(fp8_value_from_bits(scale_bits))
    err2 = (x_values[:, None] - scale * FP4_FLOAT[None, :]) ** 2
    return float(np.sum(np.min(err2, axis=1)))


def best_scale_bits(x_values: np.ndarray) -> Tuple[int, float, int, float]:
    vals = sorted((loss_for_bits(x_values, bits), bits) for bits, _ in FP8)
    best_loss, best_bits = vals[0]
    second_loss, second_bits = vals[1]
    return best_bits, best_loss, second_bits, second_loss


def verify_example(example: Example) -> None:
    (kind, part, b, signed_offset), y_values = example

    assert kind in {"upper", "lower"}
    assert part in {"normal", "subnormal", "saturation"}
    assert len(y_values) == 16

    q0_bits = q0_bits_from_part_b(part, b)
    q0_value = float(fp8_value_from_bits(q0_bits))
    x_values = q0_value * np.array(y_values, dtype=float)

    actual_q0_bits = round_fp8_bits(float(np.max(np.abs(x_values))) / 6.0)
    assert actual_q0_bits == q0_bits, (
        kind,
        part,
        b,
        signed_offset,
        "actual_q0_bits",
        actual_q0_bits,
        "expected_q0_bits",
        q0_bits,
    )

    target_bits = q0_bits + signed_offset
    assert 1 <= target_bits <= 127

    best_bits, best_loss, second_bits, second_loss = best_scale_bits(x_values)
    assert best_bits == target_bits, (
        kind,
        part,
        b,
        signed_offset,
        "best_bits",
        best_bits,
        "target_bits",
        target_bits,
        "second_bits",
        second_bits,
    )
    assert second_loss > best_loss, (kind, part, b, signed_offset, best_loss, second_loss)

    print(
        f"{kind:5s} {part:9s} b={b}, "
        f"offset={signed_offset:3d}, "
        f"q0_bits={q0_bits:2d}, "
        f"target_bits={target_bits:2d}, "
        f"gap={second_loss - best_loss:.6g}"
    )


def verify_examples() -> None:
    seen = set()
    for example in EXAMPLES:
        key = example[0]
        assert key not in seen, f"duplicate example key: {key}"
        seen.add(key)
        verify_example(example)

    print()
    print("All examples verified.")


if __name__ == "__main__":
    verify_examples()
