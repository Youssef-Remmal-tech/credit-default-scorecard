"""Tests for the SQL layer, WoE binning and scorecard arithmetic.

Run with:  pytest -q
"""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.build_features import build_feature_mart
from src.evaluate import cutoff_table, psi
from src.scorecard import build_models
from src.woe import fit_feature


@pytest.fixture(scope="module")
def mart():
    return build_feature_mart()


def test_feature_mart_shape(mart):
    assert len(mart) == 30_000
    assert mart["customer_id"].is_unique
    assert set(config.MODEL_FEATURES).issubset(mart.columns)


def test_default_rate_matches_published_figure(mart):
    # UCI documents 6,636 defaults out of 30,000 (22.12%)
    assert int(mart[config.TARGET].sum()) == 6_636


def test_protected_characteristics_not_model_inputs():
    assert not set(config.MONITORING_ONLY) & set(config.MODEL_FEATURES)


def test_late_streak_bounds(mart):
    assert mart["current_late_streak"].between(0, 6).all()
    assert (mart["current_late_streak"] <= mart["months_late_6m"]).all()


def test_woe_monotonic_for_continuous_feature():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.uniform(0, 1, 5000), name="x")
    y = pd.Series((rng.uniform(0, 1, 5000) < 0.1 + 0.4 * x).astype(int))
    bins = fit_feature(x, y)
    br = bins.table["bad_rate"].to_numpy()
    assert np.all(np.diff(br) >= 0), "bad rate should rise with x"
    assert bins.iv > 0
    assert (bins.table["share"] >= 0.05).all()


def test_scorecard_points_reproduce_model(mart):
    train = mart.sample(frac=0.7, random_state=config.SEED)
    _, _, card, _, _ = build_models(train)
    # Points are rounded to integers, so score-implied PD tracks model PD closely but not exactly
    pd_points = card.pd_from_score(card.score(train))
    pd_model = card.predict_pd(train)
    assert np.corrcoef(pd_points, pd_model)[0, 1] > 0.995
    assert card.points_table.groupby("feature")["points"].nunique().min() >= 1


def test_psi_zero_for_identical_distributions():
    x = np.random.default_rng(1).normal(size=10_000)
    value, _ = psi(x, x)
    assert value == pytest.approx(0, abs=1e-9)


def test_cutoff_table_endpoints():
    y = np.array([0, 0, 1, 1, 0, 1, 0, 0, 0, 1])
    s = np.arange(10)[::-1].astype(float)
    t = cutoff_table(y, s)
    full = t.loc[t["approval_rate"] == 1.0].iloc[0]
    assert full["bad_rate_approved"] == pytest.approx(y.mean())
    assert full["defaults_declined_share"] == pytest.approx(0)
