"""Translate DoubleML benchmark pairs to the RV diagonal scale."""

import math
import warnings


def diagonal_equivalent(cf_y, cf_d, rho=1.0):
    """Equal cf_y=cf_d strength with the same point-bound bias magnitude.

    DoubleML uses B²/(sigma² nu²) = rho² cf_y cf_d/(1-cf_d).
    The returned r solves r²/(1-r) = B²/(sigma² nu²).
    """
    values = (cf_y, cf_d, rho)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Benchmark sensitivity parameters must be finite")
    if cf_y < 0 or not 0 <= cf_d <= 1 or abs(rho) > 1:
        raise ValueError("Expected cf_y >= 0, 0 <= cf_d <= 1, |rho| <= 1")
    # At cf_d=1 the point-bias expression has denominator zero. For any
    # positive cf_y and nonzero rho its equal-strength equivalent tends to
    # the boundary r=1. When either numerator factor is zero, the bias is
    # zero even at this boundary, so the equivalent remains zero.
    if cf_d == 1:
        return 1.0 if cf_y > 0 and rho != 0 else 0.0
    strength = abs(rho) * math.sqrt(cf_y * cf_d / (1 - cf_d))
    if strength == 0:
        return 0.0
    # Stable positive root of r² + strength² r - strength² = 0.
    return 2 * strength / (strength + math.sqrt(strength * strength + 4))


def benchmark_diagonal_equivalent(cf_y, cf_d, rho=1.0, *, label="benchmark"):
    """Convert benchmark output without aborting a full sensitivity run.

    DoubleML's plug-in benchmark can be numerically inadmissible in finite
    samples (in particular, rho can exceed its theoretical correlation range
    when its denominator is small). Keep the raw benchmark fields in output,
    preserve the raw benchmark fields and continue; never silently clip inputs.
    """
    try:
        return diagonal_equivalent(float(cf_y), float(cf_d), float(rho))
    except (TypeError, ValueError, OverflowError) as exc:
        warnings.warn(
            f"{label}: cannot compute diagonal-equivalent sensitivity "
            f"(cf_y={cf_y!r}, cf_d={cf_d!r}, rho={rho!r}): {exc}. "
            "The raw benchmark is retained and the derived value is NaN.",
            RuntimeWarning,
            stacklevel=2,
        )
        return float("nan")
