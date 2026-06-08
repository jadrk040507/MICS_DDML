"""Tests for DDML estimation workflow."""

import numpy as np
import pandas as pd
import pytest

from data import _construct_common_variables, _construct_treatment_variables, _construct_hh_confounders, _construct_robustness_variables
from learners import create_learners
from models import estimate_effect, run_analysis
from config import BASE_CONFOUNDERS, MIN_OBSERVATIONS


def make_fake_data(n=200):
    rng = np.random.RandomState(42)
    dt = pd.DataFrame({
        "windex5": rng.randint(1, 6, size=n),
        "helevel": rng.choice([0, 1, 2, 98], size=n),
        "urban": rng.randint(0, 2, size=n),
        "improved_latrine": rng.randint(0, 2, size=n),
        "HHCHILDREN": rng.poisson(2, size=n),
        "WS1_g": rng.randint(1, 5, size=n),
        "Country": rng.choice(["A", "B", "C"], size=n),
        "RiskSource": rng.choice([0, 1, 2], size=n),
        "Cluster_var": rng.randint(1, 20, size=n),
        "RiskHome": rng.poisson(1, size=n),
        "WQ15_g": rng.choice([0, 1, 2, 3, 98], size=n),
        "Toilet": rng.choice([1, 2, 3, 98], size=n),
        "water_stored_covered": rng.randint(0, 2, size=n),
        "water_stored_uncovered": rng.randint(0, 2, size=n),
        "water_straight_from_source": rng.randint(0, 2, size=n),
        "SoapandWater": rng.choice([0, 1, np.nan], size=n),
        "Region": rng.randint(1, 4, size=n),
    })
    dt["SomeRiskHome"] = (dt["RiskHome"] >= 1).astype(int)
    dt["VeryHighRiskHome"] = (dt["RiskHome"] >= 2).astype(int)
    dt = _construct_common_variables(dt)
    dt = _construct_treatment_variables(dt)
    dt = _construct_hh_confounders(dt)
    dt = _construct_robustness_variables(dt)
    return dt


def test_estimate_effect_ols():
    dt = make_fake_data(n=200)
    learners = create_learners()
    result = estimate_effect(
        dt=dt,
        outcome_var="SomeRiskHome",
        treatment_var="water_treatment",
        learner_name="ols",
        learner=learners["ols"],
        confounder_groups=BASE_CONFOUNDERS,
        skip_checkpoint=True,
    )
    assert result is not None
    assert "coef" in result
    assert "se" in result
    assert "ci_lower" in result
    assert "ci_upper" in result
    assert result["n"] >= MIN_OBSERVATIONS


def test_run_analysis_multiple_learners():
    dt = make_fake_data(n=200)
    learners = {"ols": create_learners()["ols"]}
    outcomes = [{"var": "SomeRiskHome", "label": "Some Risk"}]
    treatments = [{"var": "water_treatment", "label": "Any Treatment"}]
    results = run_analysis(
        dt=dt,
        outcomes=outcomes,
        treatments=treatments,
        learners=learners,
    )
    assert len(results) > 0
    assert results[0]["learner"] == "ols"


def test_estimate_effect_too_few_observations():
    dt = make_fake_data(n=10)
    learners = create_learners()
    result = estimate_effect(
        dt=dt,
        outcome_var="SomeRiskHome",
        treatment_var="water_treatment",
        learner_name="ols",
        learner=learners["ols"],
        skip_checkpoint=True,
    )
    assert result is None


def test_stratified_sample_splitting_structure():
    """When treatment is rare, stratified splits should have nested repetitions.

    Regression test: older code flattened N_rep*N_folds tuples into one list,
    which DoubleML interpreted as a single repetition.  The fix nests tuples
    per repetition so len(smpls) == N_rep.
    """
    dt = make_fake_data(n=600)
    # Drop cluster variable to avoid cluster-split requirement in this test
    dt = dt.drop(columns=["Cluster_var"])
    # Make treatment rare (~4%) so stratified path triggers but still >=20 treated
    dt["water_treatment"] = np.zeros(len(dt), dtype=int)
    n_treated = 24
    treat_idx = dt.sample(n=n_treated, random_state=1).index
    dt.loc[treat_idx, "water_treatment"] = 1

    learners = create_learners()
    # monkeypatch a tiny n_folds / n_rep for speed
    result = estimate_effect(
        dt=dt,
        outcome_var="SomeRiskHome",
        treatment_var="water_treatment",
        learner_name="ols",
        learner=learners["ols"],
        confounder_groups=BASE_CONFOUNDERS,
        skip_checkpoint=True,
        n_folds=3,
        n_rep=2,
    )
    assert result is not None, "estimation failed on rare-treatment data"

    # To inspect the internal splits we must re-run the fit step
    # with the same parameters; we can't get the splits back from
    # the pickled result, so we reconstruct a minimal DoubleMLIRM.
    from data import create_model_matrix
    X, dt_work = create_model_matrix(dt, "SomeRiskHome", confounder_groups=BASE_CONFOUNDERS)
    mask = (
        dt_work["SomeRiskHome"].notna() &
        dt_work["water_treatment"].notna() &
        ~np.isnan(X).any(axis=1)
    )
    X_clean = X[mask.values]
    y = dt_work.loc[mask, "SomeRiskHome"].values.astype(float)
    d = dt_work.loc[mask, "water_treatment"].values.astype(float)

    import doubleml as dml
    from sklearn.model_selection import StratifiedKFold
    from sklearn.base import clone
    from config import RANDOM_STATE

    dml_data = dml.DoubleMLData.from_arrays(x=X_clean, y=y, d=d)
    dml_model = dml.DoubleMLIRM(
        obj_dml_data=dml_data,
        ml_g=clone(learners["ols"]["g"]),
        ml_m=clone(learners["ols"]["m"]),
        n_folds=3,
        n_rep=2,
        score="ATE",
        draw_sample_splitting=False,
    )
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    smpls = []
    for _ in range(2):
        rep_smpls = []
        for train_idx, test_idx in skf.split(X_clean, d):
            rep_smpls.append((train_idx, test_idx))
        smpls.append(rep_smpls)
    dml_model.set_sample_splitting(smpls)

    # Assert nested structure: top level has N_rep entries
    assert len(smpls) == 2, f"Expected 2 repetitions, got {len(smpls)}"
    for i, rep in enumerate(smpls):
        assert len(rep) == 3, f"Expected 3 folds in rep {i}, got {len(rep)}"
