from fractions import Fraction
from helper import *

ALPHA = Fraction(12, 7)

def upper_bits(bits: int, alpha: Fraction = ALPHA) -> int:
    """
    Let

        q0 = round_FP8(max |x_i| / 6)

    and suppose q0 has FP8 bit-pattern `bits`.

    For round-to-nearest, the largest s_base that still rounds to q0 is
    strictly below the midpoint between value(bits) and value(bits + 1).

    The largest upper sweep candidate is therefore the largest FP8 bit-pattern
    `upper` such that

        value(upper) < alpha * midpoint(value(bits), value(bits + 1)),

    except at saturation.
    """
    if bits >= 127:
        return 127

    v = fp8_value_from_bits(bits)
    next_v = fp8_value_from_bits(bits + 1)
    right_round_boundary = (v + next_v) / 2

    return max(upper for upper, u in FP8 if u < alpha * right_round_boundary)

def fmt_row(name: str, values: list, width: int = 10, decimal: bool = False) -> None:
    if decimal:
        values = [fmt_decimal(v) for v in values]
    print(f"{name:<14} " + " ".join(f"{str(v):>{width}s}" for v in values))

def main():
    print("NVFP4 upper-bound bit offsets")
    print("=" * 112)
    print("Upper scale factor alpha = 12/7 = 6/3.5.")
    print("Base scale q0 = round_FP8(max |x_i| / 6).")
    print()

    print("Normal FP8 E4M3, mantissa field b=0..7")
    print("-" * 112)

    bs = list(range(8))
    q0_bits = [(7 << 3) + b for b in bs]
    q0_value = [fp8_value_from_bits(bits) for bits in q0_bits]
    upper = [upper_bits(bits) for bits in q0_bits]
    upper_value = [fp8_value_from_bits(bits) for bits in upper]
    ratios = [upper_value[i] / q0_value[i] for i in range(len(bs))]
    offsets = [upper[i] - q0_bits[i] for i in range(len(bs))]
    m = [b + 8 for b in bs]

    fmt_row("b", bs)
    fmt_row("m", m)
    fmt_row("q0_bits", q0_bits)
    fmt_row("q0_value", q0_value, decimal=True)
    fmt_row("upper", upper)
    fmt_row("upper_value", upper_value, decimal=True)
    fmt_row("ratio", ratios, decimal=True)
    fmt_row("offset", offsets)

    print()
    print("Subnormal FP8 E4M3, mantissa field b=1..7")
    print("-" * 112)

    bs = list(range(1, 8))
    q0_bits = bs
    q0_value = [fp8_value_from_bits(bits) for bits in q0_bits]
    upper = [upper_bits(bits) for bits in q0_bits]
    upper_value = [fp8_value_from_bits(bits) for bits in upper]
    ratios = [upper_value[i] / q0_value[i] for i in range(len(bs))]
    offsets = [upper[i] - q0_bits[i] for i in range(len(bs))]

    fmt_row("b", bs)
    fmt_row("q0_bits", q0_bits)
    fmt_row("q0_value", q0_value, decimal=True)
    fmt_row("upper", upper)
    fmt_row("upper_value", upper_value, decimal=True)
    fmt_row("ratio", ratios, decimal=True)
    fmt_row("offset", offsets)

if __name__ == "__main__":
    main()