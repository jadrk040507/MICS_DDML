"""Shared MICS data selection and model-frame preparation."""

import pandas as pd


def controls_for_sample(common_controls, child_controls, child):
    """Return controls in their prespecified modeling order."""
    controls = list(common_controls)
    if child:
        controls.extend(child_controls)
    return controls


def complete_case_sample(
    data,
    outcome,
    treatment,
    controls,
    *,
    cluster=True,
    cluster_column="Cluster_var",
    country_column="country_cat",
    allowed_levels=None,
    extra_columns=(),
):
    """Select model-complete rows while retaining optional metadata."""
    required = [outcome, treatment, *controls, country_column]
    if cluster:
        required.append(cluster_column)

    frame = data[required].copy().dropna()
    if allowed_levels is not None:
        frame = frame[frame[treatment].isin(allowed_levels)].copy()
    for column in extra_columns:
        frame[column] = data.loc[frame.index, column]
    return frame.reset_index(drop=True)


def make_frame(
    data,
    outcome,
    treatment,
    controls,
    *,
    categorical_controls,
    cluster=True,
    cluster_column="Cluster_var",
    country_column="country_cat",
    allowed_levels=None,
):
    """Create the encoded DoubleML frame and ordered predictor names."""
    frame = complete_case_sample(
        data,
        outcome,
        treatment,
        controls,
        cluster=cluster,
        cluster_column=cluster_column,
        country_column=country_column,
        allowed_levels=allowed_levels,
    )

    categorical = list(categorical_controls)
    numeric = [name for name in controls if name not in categorical]
    controls_frame = frame[numeric + categorical].copy()
    x = pd.get_dummies(
        controls_frame,
        columns=categorical,
        drop_first=True,
        dtype=float,
    ).astype(float).reset_index(drop=True)
    if cluster:
        x["_cluster_model_code"] = pd.factorize(
            frame[cluster_column], sort=True
        )[0].astype(float)

    columns = [outcome, treatment]
    if cluster:
        columns.append(cluster_column)
    model_frame = pd.concat([frame[columns], x], axis=1)
    return model_frame, list(x.columns)


def load_analysis_data(
    path,
    outcome,
    controls,
    *,
    country_codes=None,
    quick_sample=False,
    sample_fraction=0.05,
    sample_seed=42,
):
    """Load only analysis columns, filter countries, then draw a sample."""
    country_column = "country_cat"
    columns = set(controls)
    columns.update({
        outcome,
        "water_treatment",
        "WQ15_g",
        country_column,
        "Cluster_var",
        "RiskSource",
    })
    data = pd.read_stata(
        path,
        columns=sorted(columns),
        convert_categoricals=False,
    )
    if country_codes is not None:
        data = data[data[country_column].isin(country_codes)].copy()
        print(f"Selected countries: {len(data):,} observations before sampling")
    if quick_sample:
        data = data.sample(
            frac=sample_fraction,
            random_state=sample_seed,
        ).reset_index(drop=True)
        print(
            f"Quick sample: {len(data):,} observations "
            f"({sample_fraction:.0%} of loaded data)"
        )
    return data
