"""Publication, sensitivity, and heterogeneity reporting boundaries."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Protocol
import warnings


class ReportingSpec(Protocol):
    """Minimal callbacks needed by the shared reporting entry points."""

    effect_writer: Callable[..., Any]
    sensitivity_runner: Callable[..., Any]
    gate_runner: Callable[..., Any]
    manifest_writer: Callable[..., Path]


def _fold_modes(fold_mode: str) -> tuple[str, ...]:
    choices = {
        "clustered": ("clustered",),
        "unclustered": ("unclustered",),
        "both": ("clustered", "unclustered"),
        "all": ("clustered", "unclustered"),
    }
    try:
        return choices[fold_mode]
    except KeyError as error:
        raise ValueError(f"Unknown fold mode: {fold_mode}") from error


def save_effect_outputs(
    spec: ReportingSpec,
    estimates,
    *,
    quick_sample,
    file_suffix="",
    fold_mode="both",
):
    """Write effect outputs using only the requested fold bundles."""
    return spec.effect_writer(
        estimates,
        quick_sample=quick_sample,
        file_suffix=file_suffix,
        fold_modes=_fold_modes(fold_mode),
    )


def run_sensitivity(spec: ReportingSpec, estimates, *, quick_sample, fold_mode="both"):
    """Run sensitivity reporting for the requested fold bundles."""
    return spec.sensitivity_runner(
        estimates,
        quick_sample=quick_sample,
        fold_modes=_fold_modes(fold_mode),
    )


def run_gate(spec: ReportingSpec, estimates, *, quick_sample, fold_mode="both"):
    """Run GATE reporting for the requested fold bundles."""
    return spec.gate_runner(
        estimates,
        quick_sample=quick_sample,
        fold_modes=_fold_modes(fold_mode),
    )


def write_manifest(
    spec: ReportingSpec,
    *,
    model_provenance,
    sensitivity_provenance,
    fold_mode,
):
    """Write the current analysis manifest through the configured writer."""
    return spec.manifest_writer(
        model_provenance=model_provenance,
        sensitivity_provenance=sensitivity_provenance,
        fold_mode=fold_mode,
    )


BENCHMARK_GROUPS = (
    ("wealth_index", ("windex5",)),
    ("urban", ("urban",)),
    ("water_source", ("WS1_g",)),
    ("source_ecoli", ("wq27_decile",)),
    ("under_five", ("Any_U5",)),
    ("household_children", ("Girls_less_than15", "Boys_15or_less")),
    ("toilet", ("Toilet",)),
    ("country_fixed_effects", ("country_cat",)),
    ("child_demographics", ("age", "male")),
)


def benchmark_groups(x_columns):
    """Return complete prespecified encoded covariate blocks."""
    x_columns = tuple(x_columns)
    child = "age" in x_columns or "male" in x_columns
    result = {}
    for name, variables in BENCHMARK_GROUPS:
        if name == "child_demographics" and not child:
            continue
        columns = [
            column
            for column in x_columns
            if any(
                column == variable or column.startswith(variable + "_")
                for variable in variables
            )
        ]
        for variable in variables:
            if not any(
                column == variable or column.startswith(variable + "_")
                for column in columns
            ):
                raise ValueError(
                    f"Missing benchmark control {variable!r} in {name!r}"
                )
        result[name] = columns
    unique = {column for columns in result.values() for column in columns}
    if len(unique) != sum(map(len, result.values())):
        raise ValueError("Overlapping covariate benchmark groups")
    return result


def diagonal_equivalent(cf_y, cf_d, rho=1.0):
    """Return equal confounding strength with the same point-bound bias."""
    values = (cf_y, cf_d, rho)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Benchmark sensitivity parameters must be finite")
    if cf_y < 0 or not 0 <= cf_d <= 1 or abs(rho) > 1:
        raise ValueError("Expected cf_y >= 0, 0 <= cf_d <= 1, |rho| <= 1")
    if cf_d == 1:
        return 1.0 if cf_y > 0 and rho != 0 else 0.0
    strength = abs(rho) * math.sqrt(cf_y * cf_d / (1 - cf_d))
    if strength == 0:
        return 0.0
    return 2 * strength / (strength + math.sqrt(strength * strength + 4))


def benchmark_diagonal_equivalent(cf_y, cf_d, rho=1.0, *, label="benchmark"):
    """Convert benchmark output while retaining inadmissible raw values."""
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
