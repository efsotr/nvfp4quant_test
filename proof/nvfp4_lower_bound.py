from fractions import Fraction
from pathlib import Path
import json

import numpy as np
from scipy.optimize import linprog
from helper import *

N = 16
CERT_PATH = Path("nvfp4_lower_bound.json")


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
    """
    q0 = fp8_value_from_bits(q0_bits)

    if q0_bits == 1:
        q_prev = Fraction(0)
    else:
        q_prev = fp8_value_from_bits(q0_bits - 1)

    assert q0_bits < 127
    q_next = fp8_value_from_bits(q0_bits + 1)

    alpha_lo = (q_prev + q0) / (2 * q0)
    alpha_hi = (q0 + q_next) / (2 * q0)
    return alpha_lo, alpha_hi


def adjusted_doubling_threshold(alpha_lo: Fraction) -> Fraction:
    """
    Adjusted doubling excludes

        r < (96/127) * alpha_lo.

    This follows from

        L(2r) - L(r) <= -r(24 alpha - 127r/4).
    """
    return Fraction(96, 127) * alpha_lo


def lower_offset_candidates(q0_bits: int):
    """
    Enumerate candidate_bits below q0_bits starting from the first legal ratio
    strictly above the adjusted-doubling threshold.
    """
    alpha_lo, _ = round_alpha_range(q0_bits)
    threshold = adjusted_doubling_threshold(alpha_lo)

    q0 = fp8_value_from_bits(q0_bits)
    out = []

    for candidate_bits in range(1, q0_bits):
        candidate_value = fp8_value_from_bits(candidate_bits)
        r = candidate_value / q0

        if r > threshold:
            offset = q0_bits - candidate_bits
            out.append((offset, candidate_bits, r))

    return out


def legal_ratios(q0_bits: int, hi: Fraction = Fraction(2, 1)):
    q0 = fp8_value_from_bits(q0_bits)
    return sorted({v / q0 for _, v in FP8 if Fraction(0) < v / q0 <= hi})


def d_frac(t: Fraction, y: Fraction) -> Fraction:
    return min((y - t * c) ** 2 for c in FP4)


def breakpoints_frac(scales, lo: Fraction, hi: Fraction):
    pts = {lo, hi}
    for t in scales:
        for a, b in zip(FP4[:-1], FP4[1:]):
            v = t * (a + b) / 2
            if lo <= v <= hi:
                pts.add(v)
    return sorted(pts)


def exact_verify(alpha_lo: Fraction, alpha_hi: Fraction, r: Fraction, U, W):
    """
    Gamma_r(y) = d_r(y) - sum_i lambda_i d_{u_i}(y).

    A = inf Gamma_r(y), y in [6 alpha_lo, 6 alpha_hi]
    B = inf Gamma_r(y), y in [0, 6 alpha_hi]

    If A + 15B > 0, then r is excluded.
    """
    assert sum(W) == 1
    assert all(w >= 0 for w in W)

    def Gamma(y):
        return d_frac(r, y) - sum(w * d_frac(u, y) for u, w in zip(U, W))

    hi = CMAX * alpha_hi
    pts_all = breakpoints_frac([r] + list(U), Fraction(0), hi)
    pts_max = breakpoints_frac([r] + list(U), CMAX * alpha_lo, hi)

    B = min(Gamma(y) for y in pts_all)
    A = min(Gamma(y) for y in pts_max)
    margin = A + (N - 1) * B
    return A, B, margin


def d_float(t: float, y: float) -> float:
    return float(np.min((y - t * FP4_FLOAT) ** 2))


def breakpoints_float(scales, lo: float, hi: float):
    pts = {lo, hi}
    for t in scales:
        for a, b in zip(FP4_FLOAT[:-1], FP4_FLOAT[1:]):
            v = t * (a + b) / 2.0
            if lo - 1e-12 <= v <= hi + 1e-12:
                pts.add(round(float(v), 15))
    return sorted(pts)


def solve_lp(alpha_lo: Fraction, alpha_hi: Fraction, r: Fraction, U):
    """
    Variables: lambda_0,...,lambda_{k-1}, A, B.

    Maximize A + 15B subject to:
      sum lambda_i = 1, lambda_i >= 0
      B <= Gamma_r(y) on [0, 6 alpha_hi]
      A <= Gamma_r(y) on [6 alpha_lo, 6 alpha_hi]
    """
    k = len(U)
    r_f = float(r)
    U_f = [float(u) for u in U]

    hi = float(CMAX * alpha_hi)
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

    for max_den in (64, 128, 512, 2048, 10000, 50000, 200000):
        W0 = [Fraction(x).limit_denominator(max_den) for x in active_vals]
        s = sum(W0)

        if s <= 0:
            continue

        W = tuple(w / s for w in W0)
        if sum(W) == 1 and all(w >= 0 for w in W):
            return active_indices, W

    return None


def find_certificate(alpha_lo: Fraction, alpha_hi: Fraction, r: Fraction, candidates):
    """
    Find an exact rational convex certificate for excluding r.

    This version solves one LP over all legal candidates u > r, then exact
    verifies the active support returned by the LP.
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


def verify_cached_exact(rec, alpha_lo: Fraction, alpha_hi: Fraction, r: Fraction):
    U = tuple(Fraction(x) for x in rec["U"])
    W = tuple(Fraction(x) for x in rec["lambda"])
    A, B, margin = exact_verify(alpha_lo, alpha_hi, r, U, W)

    if A != Fraction(rec["A"]):
        return None
    if B != Fraction(rec["B"]):
        return None
    if margin != Fraction(rec["margin"]):
        return None
    if margin <= 0:
        return None

    return A, B, margin


def exclude_candidate(part: str, b: int, q0_bits: int, offset: int, candidate_bits: int, r: Fraction, cache):
    alpha_lo, alpha_hi = round_alpha_range(q0_bits)
    threshold = adjusted_doubling_threshold(alpha_lo)

    assert r > threshold

    key = make_key(part, b, q0_bits, offset, candidate_bits, r)

    if key in cache and cache[key]["status"] == "excluded" and cache[key]["method"] == "exact_certificate":
        verified = verify_cached_exact(cache[key], alpha_lo, alpha_hi, r)
        if verified is not None:
            return cache[key]

    candidates = legal_ratios(q0_bits)
    cert = find_certificate(alpha_lo, alpha_hi, r, candidates)

    if cert is None:
        rec = {
            "key": key,
            "format": "nvfp4",
            "block_size": N,
            "status": "open",
            "method": None,
            "part": part,
            "b": b,
            "q0_bits": q0_bits,
            "q0_value": str(fp8_value_from_bits(q0_bits)),
            "offset": offset,
            "candidate_bits": candidate_bits,
            "candidate_value": str(fp8_value_from_bits(candidate_bits)),
            "r": str(r),
            "alpha_lo": str(alpha_lo),
            "alpha_hi": str(alpha_hi),
            "doubling_threshold": str(threshold),
        }
        cache[key] = rec
        return rec

    U, W, A, B, margin = cert
    rec = {
        "key": key,
        "format": "nvfp4",
        "block_size": N,
        "status": "excluded" if margin >= 0 else "open",
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
        "alpha_hi": str(alpha_hi),
        "doubling_threshold": str(threshold),
        "U": [str(u) for u in U],
        "lambda": [str(w) for w in W],
        "A": str(A),
        "B": str(B),
        "margin": str(margin),
    }
    cache[key] = rec
    return rec


def q0_cases():
    for b in range(8):
        yield "normal", b, (7 << 3) | b

    for b in range(1, 8):
        yield "subnormal", b, b


def summarize_normal(results_by_case):
    rows = []
    for b in range(8):
        key = ("normal", b)
        records = results_by_case[key]
        q0_bits = (7 << 3) | b
        q0_value = fp8_value_from_bits(q0_bits)

        excluded = [r for r in records if r["status"] == "excluded"]
        open_recs = [r for r in records if r["status"] == "open"]

        if open_recs:
            first_open = min(open_recs, key=lambda r: Fraction(r["r"]))
            first_open_offset = first_open["offset"]
            first_open_bits = first_open["candidate_bits"]
            first_open_ratio = Fraction(first_open["r"])
            first_open_m = fmt_bin_frac(relative_m_from_bits(q0_bits, first_open_bits))
        else:
            first_open_offset = None
            first_open_bits = None
            first_open_ratio = None
            first_open_m = None

        rows.append({
            "b": b,
            "m": fmt_bin_int(8 + b),
            "q0_bits": q0_bits,
            "q0_value": fmt_decimal(q0_value),
            "excluded_offsets": [r["offset"] for r in excluded],
            "first_open_offset": first_open_offset,
            "first_open_bits": first_open_bits,
            "first_open_m": first_open_m,
            "first_open_ratio": fmt_decimal(first_open_ratio) if first_open_ratio is not None else None,
        })

    print("Normal FP8 E4M3 lower-side offset exclusion")
    print("-" * 120)
    fmt_row("b", [r["b"] for r in rows])
    fmt_row("m", [r["m"] for r in rows])
    fmt_row("q0_bits", [r["q0_bits"] for r in rows])
    fmt_row("q0_value", [r["q0_value"] for r in rows])
    fmt_row("excluded_offsets", [r["excluded_offsets"] for r in rows])
    fmt_row("first_open_offset", [r["first_open_offset"] for r in rows])
    fmt_row("first_open_bits", [r["first_open_bits"] for r in rows])
    fmt_row("first_open_m", [r["first_open_m"] for r in rows])
    fmt_row("first_open_ratio", [r["first_open_ratio"] for r in rows])
    print()


def summarize_subnormal(results_by_case):
    rows = []
    for b in range(1, 8):
        key = ("subnormal", b)
        records = results_by_case[key]
        q0_bits = b
        q0_value = fp8_value_from_bits(q0_bits)

        excluded = [r for r in records if r["status"] == "excluded"]
        open_recs = [r for r in records if r["status"] == "open"]

        if open_recs:
            first_open = min(open_recs, key=lambda r: Fraction(r["r"]))
            first_open_offset = first_open["offset"]
            first_open_bits = first_open["candidate_bits"]
            first_open_ratio = Fraction(first_open["r"])
        else:
            first_open_offset = None
            first_open_bits = None
            first_open_ratio = None

        rows.append({
            "b": b,
            "q0_bits": q0_bits,
            "q0_value": fmt_decimal(q0_value),
            "excluded_offsets": [r["offset"] for r in excluded],
            "first_open_offset": first_open_offset,
            "first_open_bits": first_open_bits,
            "first_open_ratio": fmt_decimal(first_open_ratio) if first_open_ratio is not None else None,
        })

    print("Subnormal FP8 E4M3 lower-side offset exclusion")
    print("-" * 120)
    fmt_row("b", [r["b"] for r in rows])
    fmt_row("q0_bits", [r["q0_bits"] for r in rows])
    fmt_row("q0_value", [r["q0_value"] for r in rows])
    fmt_row("excluded_offsets", [r["excluded_offsets"] for r in rows])
    fmt_row("first_open_offset", [r["first_open_offset"] for r in rows])
    fmt_row("first_open_bits", [r["first_open_bits"] for r in rows])
    fmt_row("first_open_ratio", [r["first_open_ratio"] for r in rows])
    print()


def main():
    print("NVFP4 lower-side offset exclusion for q0 = round_FP8(max |x_i| / 6)")
    print("=" * 120)
    print("Only candidates strictly above the adjusted-doubling threshold are checked.")
    print(f"Certificate cache: {CERT_PATH}")
    print()

    cache = load_cache()
    results_by_case = {}

    for part, b, q0_bits in q0_cases():
        records = []
        for offset, candidate_bits, r in lower_offset_candidates(q0_bits):
            rec = exclude_candidate(part, b, q0_bits, offset, candidate_bits, r, cache)
            records.append(rec)
            if rec["status"] == "open":
                break
        results_by_case[(part, b)] = records

    dump_cache(cache)

    summarize_normal(results_by_case)
    summarize_subnormal(results_by_case)

    excluded = [
        r
        for records in results_by_case.values()
        for r in records
        if r["status"] == "excluded"
    ]

    print("Summary")
    print("-" * 120)
    print("Excluded candidates:", len(excluded))
    print("  by exact certificate:", sum(r["method"] == "exact_certificate" for r in excluded))
    print("Wrote", CERT_PATH)

if __name__ == "__main__":
    main()
