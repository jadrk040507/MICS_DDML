"""Tests for learner creation and interfaces."""

import numpy as np
import pytest
from sklearn.base import is_regressor, is_classifier

from final.learners import create_learners
from final.config import LEARNER_NAMES


def test_all_learners_created():
    learners = create_learners()
    assert set(learners.keys()) == set(LEARNER_NAMES)


@pytest.mark.parametrize("name", LEARNER_NAMES)
def test_learner_has_g_and_m(name):
    learners = create_learners()
    learner = learners[name]
    assert "g" in learner
    assert "m" in learner
    assert is_regressor(learner["g"])
    assert is_classifier(learner["m"])


@pytest.mark.parametrize("name", LEARNER_NAMES)
def test_learner_can_fit_and_predict(name):
    learners = create_learners()
    learner = learners[name]
    n, p = 100, 5
    rng = np.random.RandomState(42)
    X = rng.randn(n, p)
    y = rng.randn(n)
    d = (rng.rand(n) > 0.5).astype(int)

    learner["g"].fit(X, y)
    y_pred = learner["g"].predict(X)
    assert len(y_pred) == n

    learner["m"].fit(X, d)
    if hasattr(learner["m"], "predict_proba"):
        d_pred = learner["m"].predict_proba(X)[:, 1]
    else:
        d_pred = learner["m"].predict(X)
    assert len(d_pred) == n
