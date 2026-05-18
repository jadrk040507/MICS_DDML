"""Tests for DDML estimation workflow."""

import numpy as np
import pandas as pd
import pytest

from final.data import _construct_common_variables, _construct_treatment_variables, _construct_hh_confounders, _construct_robustness_variables
from final.learners import create_learners
from final.models import estimate_effect, run_analysis
from final.config import BASE_CONFOUNDERS, MIN_OBSERVATIONS


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


def test_estimate_effect_no_treatment_variation():
    dt = make_fake_data(n=200)
    dt["water_treatment"] = 0
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
