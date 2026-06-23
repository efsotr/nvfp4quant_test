from fractions import Fraction
import numpy as np

FP4 = [
    Fraction(0), Fraction(1, 2), Fraction(1), Fraction(3, 2),
    Fraction(2), Fraction(3), Fraction(4), Fraction(6),
]
FP4_FLOAT = np.array([float(c) for c in FP4])
CMAX = Fraction(6)

def pow2_frac(exp: int) -> Fraction:
    return Fraction(2 ** exp, 1) if exp >= 0 else Fraction(1, 2 ** (-exp))

def fp8_value_from_bits(bits: int) -> Fraction:
    """
    Positive finite FP8 E4M3 under the EeMm definition:
      normal:    (1+b/8) * 2^(a-7), a != 0
      subnormal: b * 2^-9,           a == 0

    bits is the positive sign-0 FP8 bit-pattern.
    """
    assert 1 <= bits <= 127
    a = bits >> 3
    b = bits & 7
    if a == 0:
        return Fraction(b, 512)
    return (Fraction(1) + Fraction(b, 8)) * pow2_frac(a - 7)

FP8 = [(bits, fp8_value_from_bits(bits)) for bits in range(1, 128)]

def fmt_decimal(x: Fraction, ndigits: int = 6) -> str:
    return f"{float(x):.{ndigits}g}"

def fmt_bin_int(x: int) -> str:
    return f"0b{x:b}"

def fmt_bin_frac(x: Fraction) -> str:
    assert x >= 0
    n = x.numerator
    d = x.denominator
    assert d & (d - 1) == 0

    shift = d.bit_length() - 1
    int_part = n >> shift
    frac_part = n & (d - 1)

    if frac_part == 0:
        return f"0b{int_part:b}"

    frac_bits = f"{frac_part:0{shift}b}".rstrip("0")
    return f"0b{int_part:b}.{frac_bits}"

def fmt_row(name: str, values: list, width: int = 14) -> None:
    print(f"{name:<18} " + " ".join(f"{str(v):>{width}s}" for v in values))
