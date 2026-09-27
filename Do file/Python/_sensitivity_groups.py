"""Prespecified covariate blocks for leave-one-group-out benchmarks."""

GROUPS = (
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
    """Return each full encoded block; exclude child controls for HH models.

    Fail on a missing prespecified block instead of silently benchmarking a
    fragment. The internal cluster identifier is never a benchmark control.
    """
    x_columns = tuple(x_columns)
    child = "age" in x_columns or "male" in x_columns
    result = {}
    for name, variables in GROUPS:
        if name == "child_demographics" and not child:
            continue
        columns = [column for column in x_columns if any(
            column == variable or column.startswith(variable + "_")
            for variable in variables
        )]
        for variable in variables:
            if not any(column == variable or column.startswith(variable + "_") for column in columns):
                raise ValueError(f"Missing benchmark control {variable!r} in {name!r}")
        result[name] = columns
    if len({column for columns in result.values() for column in columns}) != sum(map(len, result.values())):
        raise ValueError("Overlapping covariate benchmark groups")
    return result
