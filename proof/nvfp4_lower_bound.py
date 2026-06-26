from fractions import Fraction
from pathlib import Path
import json

import numpy as np
from scipy.optimize import linprog


N = 16
MAX_BITS = 126
MAX_VALUE = Fraction(448)
DOUBLING_VALUE_LIMIT = Fraction(240)  # candidate_value must be < 240 so that 2*candidate_value <= 448.
CERT_PATH = Path(__file__).with_name("nvfp4_lower_bound_with_saturation.json")

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
    bits=126 is the largest positive finite scale, value 448.
    bits=127 is not treated as a positive finite scale here.
    """
    assert 1 <= bits <= 127
    a = bits >> 3
    b = bits & 7
    if a == 0:
        return Fraction(b, 512)
    return (Fraction(1) + Fraction(b, 8)) * pow2_frac(a - 7)


FP8 = [(bits, fp8_value_from_bits(bits)) for bits in range(1, MAX_BITS + 1)]
assert fp8_value_from_bits(MAX_BITS) == MAX_VALUE


def fmt_decimal(x: Fraction, ndigits: int = 8) -> str:
    return f"{float(x):.{ndigits}g}"


def fmt_bin_int(x: int) -> str:
    return f"0b{x:b}"


def fmt_bin_frac(x: Fraction) -> str:
    """
    Format a nonnegative Fraction whose denominator is a power of two
    as a binary fixed-point string.
    """
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


def relative_m_from_bits(q0_bits: int, candidate_bits: int) -> Fraction:
    """
    For normal q0_bits, write

        q0 significand bit-pattern: m = 0b1bbb.

    candidate_m is candidate_bits interpreted relative to the q0 binade.
    This is a bit-pattern-side representation.
    """
    q0_a = q0_bits >> 3
    candidate_a = candidate_bits >> 3
    candidate_b = candidate_bits & 7

    local_m = Fraction(8 + candidate_b)
    exp_shift = candidate_a - q0_a
    return local_m * pow2_frac(exp_shift)


def round_alpha_range(q0_bits: int):
    """
    q0 = round_FP8(max |x_i| / 6).

    Let s_base = max |x_i| / 6 and alpha = s_base / value(q0_bits).
    This returns the round-to-nearest interval for alpha.

    For q0_bits=126, the right rounding cell is saturated, so alpha_hi=None.
    """
    q0 = fp8_value_from_bits(q0_bits)

    q_prev = Fraction(0) if q0_bits == 1 else fp8_value_from_bits(q0_bits - 1)
    alpha_lo = (q_prev + q0) / (2 * q0)

    if q0_bits == MAX_BITS:
        alpha_hi = None
    else:
        q_next = fp8_value_from_bits(q0_bits + 1)
        alpha_hi = (q0 + q_next) / (2 * q0)

    return alpha_lo, alpha_hi


def adjusted_doubling_threshold(alpha_lo: Fraction) -> Fraction:
    """
    Adjusted doubling excludes

        r < (96/127) * alpha_lo,

    but only when 2s remains an available positive finite FP8 scale.
    Since max FP8 is 448, this script applies the doubling shortcut only
    for candidate_value < 240, i.e. up to candidate_value=224.
    """
    return Fraction(96, 127) * alpha_lo


def doubling_excludes(q0_bits: int, candidate_bits: int, r: Fraction) -> bool:
    alpha_lo, _ = round_alpha_range(q0_bits)
    threshold = adjusted_doubling_threshold(alpha_lo)
    candidate_value = fp8_value_from_bits(candidate_bits)
    return candidate_value < DOUBLING_VALUE_LIMIT and r < threshold


def lower_offset_candidates(q0_bits: int):
    """
    Enumerate lower-side candidates that are not already covered by the
    adjusted-doubling proof.

    Near saturation, the adjusted-doubling proof is only used when the doubled
    candidate scale is still available. Since the largest finite FP8 E4M3 value
    is 448, candidate_value >= 240 is not skipped by doubling even if its ratio
    is below the analytic threshold.
    """
    q0 = fp8_value_from_bits(q0_bits)
    out = []

    for candidate_bits in range(1, q0_bits):
        candidate_value = fp8_value_from_bits(candidate_bits)
        r = candidate_value / q0

        if doubling_excludes(q0_bits, candidate_bits, r):
            continue

        offset = candidate_bits - q0_bits
        out.append((offset, candidate_bits, r))

    return sorted(out, key=lambda item: item[2])


def legal_ratios(q0_bits: int, hi: Fraction = Fraction(2, 1)):
    q0 = fp8_value_from_bits(q0_bits)
    return sorted({v / q0 for _, v in FP8 if Fraction(0) < v / q0 <= hi})


def d_frac(t: Fraction, y: Fraction) -> Fraction:
    return min((y - t * c) ** 2 for c in FP4)


def breakpoints_frac(scales, lo: Fraction, hi):
    pts = {lo}
    if hi is not None:
        pts.add(hi)

    for t in scales:
        for a, b in zip(FP4[:-1], FP4[1:]):
            v = t * (a + b) / 2
            if v >= lo and (hi is None or v <= hi):
                pts.add(v)

    return sorted(pts)


def exact_verify(alpha_lo: Fraction, alpha_hi, r: Fraction, U, W):
    """
    Gamma_r(y) = d_r(y) - sum_i lambda_i d_{u_i}(y).

    Finite cell:
      A = inf Gamma_r(y), y in [6 alpha_lo, 6 alpha_hi]
      B = inf Gamma_r(y), y in [0, 6 alpha_hi]

    Saturated max cell, alpha_hi=None:
      A = inf Gamma_r(y), y in [6 alpha_lo, infinity)
      B = inf Gamma_r(y), y in [0, infinity)

    Since every u_i > r and sum lambda_i = 1, Gamma_r(y) has positive
    asymptotic slope in the final FP4 cell, so finite breakpoints suffice.
    """
    assert sum(W) == 1
    assert all(w >= 0 for w in W)
    assert all(u > r for u in U)

    def Gamma(y):
        return d_frac(r, y) - sum(w * d_frac(u, y) for u, w in zip(U, W))

    hi = None if alpha_hi is None else CMAX * alpha_hi
    pts_all = breakpoints_frac([r] + list(U), Fraction(0), hi)
    pts_max = breakpoints_frac([r] + list(U), CMAX * alpha_lo, hi)

    B = min(Gamma(y) for y in pts_all)
    A = min(Gamma(y) for y in pts_max)
    margin = A + (N - 1) * B
    return A, B, margin


def d_float(t: float, y: float) -> float:
    return float(np.min((y - t * FP4_FLOAT) ** 2))


def breakpoints_float(scales, lo: float, hi):
    pts = {lo}
    if hi is not None:
        pts.add(hi)

    for t in scales:
        for a, b in zip(FP4_FLOAT[:-1], FP4_FLOAT[1:]):
            v = t * (a + b) / 2.0
            if v >= lo - 1e-12 and (hi is None or v <= hi + 1e-12):
                pts.add(round(float(v), 15))

    return sorted(pts)


def solve_lp(alpha_lo: Fraction, alpha_hi, r: Fraction, U):
    """
    Variables: lambda_0,...,lambda_{k-1}, A, B.

    Maximize A + 15B subject to:
      sum lambda_i = 1, lambda_i >= 0
      B <= Gamma_r(y) on [0, 6 alpha_hi], or [0, infinity) for saturation
      A <= Gamma_r(y) on [6 alpha_lo, 6 alpha_hi], or [6 alpha_lo, infinity)
    """
    k = len(U)
    if k == 0:
        return None

    r_f = float(r)
    U_f = [float(u) for u in U]

    hi = None if alpha_hi is None else float(CMAX * alpha_hi)
    lo_max = float(CMAX * alpha_lo)

    pts_all = breakpoints_float([r_f] + U_f, 0.0, hi)
    pts_max = breakpoints_float([r_f] + U_f, lo_max, hi)

    c = np.zeros(k + 2)
    c[k] = -1.0
    c[k + 1] = -(N - 1)

    A_ub = []
    b_ub = []

    for y in pts_all:
        row = [d_float(float(u), y) for u in U] + [0.0, 1.0]
        A_ub.append(row)
        b_ub.append(d_float(r_f, y))

    for y in pts_max:
        row = [d_float(float(u), y) for u in U] + [1.0, 0.0]
        A_ub.append(row)
        b_ub.append(d_float(r_f, y))

    A_eq = [[1.0] * k + [0.0, 0.0]]
    b_eq = [1.0]
    bounds = [(0.0, None)] * k + [(None, None), (None, None)]

    res = linprog(
        c,
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        A_eq=np.array(A_eq),
        b_eq=np.array(b_eq),
        bounds=bounds,
        method="highs",
    )

    if not res.success:
        return None

    return res.x[:k]


def rationalize_active_lambdas(vals, tol: float = 1e-10):
    """
    Keep the active support returned by the LP, rationalize it, then normalize
    exactly so that sum(lambda_i) == 1.
    """
    active = [(i, float(x)) for i, x in enumerate(vals) if float(x) > tol]
    if not active:
        return None

    active_indices = tuple(i for i, _ in active)
    active_vals = [x for _, x in active]

    for max_den in (64, 128, 512, 2048, 10000, 50000, 200000, 1000000):
        W0 = [Fraction(x).limit_denominator(max_den) for x in active_vals]
        s = sum(W0)

        if s <= 0:
            continue

        W = tuple(w / s for w in W0)
        if sum(W) == 1 and all(w >= 0 for w in W):
            return active_indices, W

    return None


def find_certificate(alpha_lo: Fraction, alpha_hi, r: Fraction, candidates):
    """
    Find an exact rational convex certificate for excluding r.

    This version solves one LP over all actually available legal candidates
    u > r. This is important near saturation, where a certificate from a lower
    binade may require a comparison scale above 448.
    """
    candidates = tuple(sorted({u for u in candidates if u > r}))
    if not candidates:
        return None

    lambdas = solve_lp(alpha_lo, alpha_hi, r, candidates)
    if lambdas is None:
        return None

    active = rationalize_active_lambdas(lambdas)
    if active is None:
        return None

    active_indices, W = active
    U = tuple(candidates[i] for i in active_indices)

    A, B, margin = exact_verify(alpha_lo, alpha_hi, r, U, W)

    return U, W, A, B, margin

def make_key(part: str, b: int, q0_bits: int, offset: int, candidate_bits: int, r: Fraction) -> str:
    return f"{part}:b={b}:q0_bits={q0_bits}:offset={offset}:candidate_bits={candidate_bits}:r={r}"


def load_cache():
    if not CERT_PATH.exists():
        return {}

    records = json.loads(CERT_PATH.read_text())
    cache = {}
    for rec in records:
        cache[rec["key"]] = rec
    return cache


def dump_cache(cache):
    records = sorted(
        cache.values(),
        key=lambda rec: (
            rec["part"],
            rec["b"],
            rec["q0_bits"],
            rec["offset"],
            rec["candidate_bits"],
        ),
    )
    CERT_PATH.write_text(json.dumps(records, indent=2))


def verify_cached_exact(rec, alpha_lo: Fraction, alpha_hi, r: Fraction):
    U = tuple(Fraction(x) for x in rec["U"])
    W = tuple(Fraction(x) for x in rec["lambda"])
    A, B, margin = exact_verify(alpha_lo, alpha_hi, r, U, W)

    if A != Fraction(rec["A"]):
        return None
    if B != Fraction(rec["B"]):
        return None
    if margin != Fraction(rec["margin"]):
        return None

    return A, B, margin


def is_excluded(part: str, b: int, q0_bits: int, offset: int, candidate_bits: int, r: Fraction, cache):
    """
    Return True iff candidate ratio r is excluded.

    The adjusted-doubling proof is applied only when the doubled scale is still
    positive finite FP8. Exact certificates are cached.
    """
    if doubling_excludes(q0_bits, candidate_bits, r):
        return True

    alpha_lo, alpha_hi = round_alpha_range(q0_bits)
    threshold = adjusted_doubling_threshold(alpha_lo)
    key = make_key(part, b, q0_bits, offset, candidate_bits, r)

    if key in cache and cache[key]["method"] == "exact_certificate":
        verified = verify_cached_exact(cache[key], alpha_lo, alpha_hi, r)
        if verified is not None:
            return verified[2] >= 0 # margin >= 0

    candidates = legal_ratios(q0_bits)
    cert = find_certificate(alpha_lo, alpha_hi, r, candidates)
    if cert is None:
        return False

    U, W, A, B, margin = cert
    cache[key] = {
        "key": key,
        "format": "nvfp4",
        "block_size": N,
        "max_bits": MAX_BITS,
        "max_value": str(MAX_VALUE),
        "doubling_value_limit": str(DOUBLING_VALUE_LIMIT),
        "method": "exact_certificate",
        "part": part,
        "b": b,
        "q0_bits": q0_bits,
        "q0_value": str(fp8_value_from_bits(q0_bits)),
        "offset": offset,
        "candidate_bits": candidate_bits,
        "candidate_value": str(fp8_value_from_bits(candidate_bits)),
        "r": str(r),
        "alpha_lo": str(alpha_lo),
        "alpha_hi": None if alpha_hi is None else str(alpha_hi),
        "doubling_threshold": str(threshold),
        "U": [str(u) for u in U],
        "lambda": [str(w) for w in W],
        "A": str(A),
        "B": str(B),
        "margin": str(margin),
    }
    return margin >= 0


def find_lower_bound_for_case(part: str, b: int, q0_bits: int, cache):
    """
    Scan legal lower-side candidates in increasing ratio order. Stop at the
    first candidate that cannot be excluded.
    """
    for offset, candidate_bits, r in lower_offset_candidates(q0_bits):
        if not is_excluded(part, b, q0_bits, offset, candidate_bits, r, cache):
            alpha_lo, alpha_hi = round_alpha_range(q0_bits)
            return {
                "part": part,
                "b": b,
                "q0_bits": q0_bits,
                "q0_value": fp8_value_from_bits(q0_bits),
                "lower_bits": candidate_bits,
                "lower_value": fp8_value_from_bits(candidate_bits),
                "lower_ratio": r,
                "bit_offset": offset,
                "alpha_lo": alpha_lo,
                "alpha_hi": alpha_hi,
                "doubling_threshold": adjusted_doubling_threshold(alpha_lo),
            }

    # If every lower-side candidate is excluded, q0 itself is the first
    # remaining feasible scale.
    alpha_lo, alpha_hi = round_alpha_range(q0_bits)
    return {
        "part": part,
        "b": b,
        "q0_bits": q0_bits,
        "q0_value": fp8_value_from_bits(q0_bits),
        "lower_bits": q0_bits,
        "lower_value": fp8_value_from_bits(q0_bits),
        "lower_ratio": Fraction(1),
        "bit_offset": 0,
        "alpha_lo": alpha_lo,
        "alpha_hi": alpha_hi,
        "doubling_threshold": adjusted_doubling_threshold(alpha_lo),
    }


def representative_q0_cases():
    for b in range(8):
        yield "normal", b, (7 << 3) | b

    for b in range(1, 8):
        yield "subnormal", b, b


def saturation_q0_cases():
    # Top finite normal binade. bits=127 is not finite, so only b=0..6 exist.
    for q0_bits in range(120, MAX_BITS + 1):
        yield "saturation", q0_bits & 7, q0_bits


def fmt_row(name: str, values: list, width: int = 14) -> None:
    print(f"{name:<18} " + " ".join(f"{str(v):>{width}s}" for v in values))


def print_normal_lower_bounds(results, title: str):
    rows = []
    for res in results:
        q0_bits = res["q0_bits"]
        lower_bits = res["lower_bits"]

        lower_m = None
        if lower_bits is not None:
            lower_m = fmt_bin_frac(relative_m_from_bits(q0_bits, lower_bits))

        rows.append({
            "b": res["b"],
            "m": fmt_bin_int(8 + res["b"]),
            "q0_bits": q0_bits,
            "q0_value": fmt_decimal(res["q0_value"]),
            "lower_bits": lower_bits,
            "lower_m": lower_m,
            "lower_value": fmt_decimal(res["lower_value"]) if res["lower_value"] is not None else None,
            "lower_ratio": fmt_decimal(res["lower_ratio"]) if res["lower_ratio"] is not None else None,
            "bit_offset": res["bit_offset"],
            "alpha_lo": fmt_decimal(res["alpha_lo"]),
            "alpha_hi": "inf" if res["alpha_hi"] is None else fmt_decimal(res["alpha_hi"]),
        })

    print(title)
    print("-" * 120)
    fmt_row("b", [r["b"] for r in rows])
    fmt_row("m", [r["m"] for r in rows])
    fmt_row("q0_bits", [r["q0_bits"] for r in rows])
    fmt_row("q0_value", [r["q0_value"] for r in rows])
    fmt_row("lower_bits", [r["lower_bits"] for r in rows])
    fmt_row("lower_m", [r["lower_m"] for r in rows])
    fmt_row("lower_value", [r["lower_value"] for r in rows])
    fmt_row("lower_ratio", [r["lower_ratio"] for r in rows])
    fmt_row("bit_offset", [r["bit_offset"] for r in rows])
    fmt_row("alpha_lo", [r["alpha_lo"] for r in rows])
    fmt_row("alpha_hi", [r["alpha_hi"] for r in rows])
    print()


def print_subnormal_lower_bounds(results):
    rows = []
    for res in results:
        rows.append({
            "b": res["b"],
            "q0_bits": res["q0_bits"],
            "q0_value": fmt_decimal(res["q0_value"]),
            "lower_bits": res["lower_bits"],
            "lower_value": fmt_decimal(res["lower_value"]) if res["lower_value"] is not None else None,
            "lower_ratio": fmt_decimal(res["lower_ratio"]) if res["lower_ratio"] is not None else None,
            "bit_offset": res["bit_offset"],
            "alpha_lo": fmt_decimal(res["alpha_lo"]),
            "alpha_hi": "inf" if res["alpha_hi"] is None else fmt_decimal(res["alpha_hi"]),
        })

    print("Subnormal FP8 E4M3 lower bounds")
    print("-" * 120)
    fmt_row("b", [r["b"] for r in rows])
    fmt_row("q0_bits", [r["q0_bits"] for r in rows])
    fmt_row("q0_value", [r["q0_value"] for r in rows])
    fmt_row("lower_bits", [r["lower_bits"] for r in rows])
    fmt_row("lower_value", [r["lower_value"] for r in rows])
    fmt_row("lower_ratio", [r["lower_ratio"] for r in rows])
    fmt_row("bit_offset", [r["bit_offset"] for r in rows])
    fmt_row("alpha_lo", [r["alpha_lo"] for r in rows])
    fmt_row("alpha_hi", [r["alpha_hi"] for r in rows])
    print()


def main():
    print("NVFP4 lower bounds for q0 = round_FP8(max |x_i| / 6)")
    print("=" * 120)
    print("Positive finite FP8 E4M3 range: bits 1..126, max value 448.")
    print("Adjusted doubling is used only when candidate_value < 240; otherwise 2*candidate_value exceeds 448.")
    print("The script scans upward and stops at the first candidate not excluded by an exact certificate.")
    print(f"Certificate cache: {CERT_PATH}")
    print()

    cache = load_cache()

    normal_results = []
    subnormal_results = []
    for part, b, q0_bits in representative_q0_cases():
        res = find_lower_bound_for_case(part, b, q0_bits, cache)
        if part == "normal":
            normal_results.append(res)
        else:
            subnormal_results.append(res)

    saturation_results = [
        find_lower_bound_for_case(part, b, q0_bits, cache)
        for part, b, q0_bits in saturation_q0_cases()
    ]

    dump_cache(cache)

    print_normal_lower_bounds(normal_results, "Representative normal FP8 E4M3 lower bounds, exponent field a=7")
    print_subnormal_lower_bounds(subnormal_results)
    print_normal_lower_bounds(saturation_results, "Top finite normal FP8 E4M3 lower bounds, saturation-aware")

    # print("lower_ratio list, representative normal:")
    # print([str(r["lower_ratio"]) for r in normal_results])
    # print("bit_offset list, representative normal:")
    # print([r["bit_offset"] for r in normal_results])
    # print()

    # print("lower_ratio list, subnormal:")
    # print([str(r["lower_ratio"]) for r in subnormal_results])
    # print("bit_offset list, subnormal:")
    # print([r["bit_offset"] for r in subnormal_results])
    # print()

    # print("lower_ratio list, top finite normal:")
    # print([str(r["lower_ratio"]) for r in saturation_results])
    # print("bit_offset list, top finite normal:")
    # print([r["bit_offset"] for r in saturation_results])
    # print()

    # print("Certificates written:", CERT_PATH)
    # print("Certificate count:", len(cache))


if __name__ == "__main__":
    main()
