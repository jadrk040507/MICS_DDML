"""Tests for data preparation."""

import pandas as pd
import numpy as np
import pytest

from final.data import create_model_matrix, _construct_common_variables, _construct_treatment_variables
from final.config import BASE_CONFOUNDERS, WQ15G_TREATMENT_MAP


def make_fake_hh_data(n=200):
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
    return dt


def test_construct_common_variables():
    dt = make_fake_hh_data()
    dt = _construct_common_variables(dt)
    assert "wealth_q1" in dt.columns
    assert "edu_none" in dt.columns
    assert "urban_bin" in dt.columns
    assert "country_B" in dt.columns or "country_C" in dt.columns
    assert "risk_source_0" in dt.columns


def test_construct_treatment_variables():
    dt = make_fake_hh_data()
    dt = _construct_common_variables(dt)
    dt = _construct_treatment_variables(dt)
    assert "water_treatment" in dt.columns
    for code, name in WQ15G_TREATMENT_MAP.items():
        assert name in dt.columns


def test_create_model_matrix_basic():
    dt = make_fake_hh_data()
    dt = _construct_common_variables(dt)
    dt = _construct_treatment_variables(dt)
    X, dt_out = create_model_matrix(dt, "SomeRiskHome", confounder_groups=BASE_CONFOUNDERS)
    assert X.shape[0] == len(dt)
    assert X.shape[1] > 0
    assert not np.isnan(X).any()
