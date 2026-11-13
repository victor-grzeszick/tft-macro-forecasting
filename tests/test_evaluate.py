"""Accuracy metrics and the Diebold-Mariano test."""
import numpy as np
import pytest
from scipy import stats

from evaluate import autocovariance, diebold_mariano, mae, rmse


def test_rmse_and_mae():
    errors = np.array([3.0, -4.0])
    assert rmse(errors) == pytest.approx(np.sqrt(12.5))
    assert mae(errors) == pytest.approx(3.5)


def test_autocovariance_lag_zero_is_variance():
    d = np.array([1.0, 2.0, 4.0, 7.0])
    assert autocovariance(d, 0) == pytest.approx(np.var(d))
    assert autocovariance(d, 1) == pytest.approx(np.sum((d[1:] - d.mean()) * (d[:-1] - d.mean())) / 4)


def test_dm_one_step_matches_hand_calculation():
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=40), rng.normal(size=40)
    d = a ** 2 - b ** 2
    n = len(d)
    expected = np.sqrt((n - 1) / n) * d.mean() / np.sqrt(np.var(d) / n)  # HLN factor for h = 1
    statistic, pvalue, kernel = diebold_mariano(a, b, 1)
    assert statistic == pytest.approx(expected)
    assert pvalue == pytest.approx(2 * stats.t.sf(abs(expected), n - 1))
    assert kernel == "rectangular"


def test_dm_is_antisymmetric_and_scale_invariant():
    rng = np.random.default_rng(4)
    a, b = rng.normal(size=60), 1.3 * rng.normal(size=60)
    forward, p_forward, _ = diebold_mariano(a, b, 3)
    backward, p_backward, _ = diebold_mariano(b, a, 3)
    scaled, _, _ = diebold_mariano(1000 * a, 1000 * b, 3)
    assert forward == pytest.approx(-backward) and p_forward == pytest.approx(p_backward)
    assert scaled == pytest.approx(forward)


def test_dm_sign_and_detection_of_a_clearly_better_model():
    rng = np.random.default_rng(5)
    good, bad = 0.1 * rng.normal(size=200), rng.normal(size=200)
    statistic, pvalue, _ = diebold_mariano(good, bad, 1)
    assert statistic < 0 and pvalue < 0.01


def test_dm_has_roughly_correct_size_under_the_null():
    """Two equally accurate models should be 'significantly different' at 5% in about 5% of samples."""
    rng = np.random.default_rng(6)
    rejections = [diebold_mariano(rng.normal(size=60), rng.normal(size=60), 3)[1] < 0.05 for _ in range(2000)]
    assert 0.02 < np.mean(rejections) < 0.09


def test_dm_falls_back_to_bartlett_when_variance_estimate_is_negative():
    d = np.array([1.0, -1.0] * 10)  # strongly negative autocorrelation -> rectangular estimate <= 0
    a, b = np.sqrt(d + 2.0), np.sqrt(np.full(20, 2.0))
    statistic, pvalue, kernel = diebold_mariano(a, b, 2)
    assert kernel == "bartlett" and np.isfinite(statistic)
