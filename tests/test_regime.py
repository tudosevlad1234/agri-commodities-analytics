import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from agri import regime as rg


def log_space_loglik(log_b, pi, a):
    log_alpha = np.log(pi) + log_b[0]
    for t in range(1, len(log_b)):
        log_alpha = logsumexp(log_alpha[:, None] + np.log(a), axis=0) + log_b[t]
    return float(logsumexp(log_alpha))


def two_state_series(seed=0, n=900):
    rng = np.random.default_rng(seed)
    state = np.zeros(n, dtype=int)
    for t in range(1, n):
        state[t] = state[t - 1] if rng.random() > 0.03 else 1 - state[t - 1]
    returns = rng.standard_normal(n) * np.where(state == 1, 0.04, 0.01)
    index = pd.date_range("2005-01-07", periods=n, freq="W-FRI")
    return pd.Series(100 * np.exp(np.cumsum(returns)), index=index), state


def test_scaled_forward_matches_log_space_recursion():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((120, 2))
    mu = np.array([[0.0, 0.0], [1.0, 1.0]])
    cov = np.array([np.eye(2), 2 * np.eye(2)])
    a = np.array([[0.9, 0.1], [0.2, 0.8]])
    pi = np.array([0.5, 0.5])
    log_b = rg.emissions(x, mu, cov)
    loglik = rg.forward_backward(log_b, pi, a)[4]
    assert abs(loglik - log_space_loglik(log_b, pi, a)) < 1e-8


def test_emission_density_matches_scipy():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((50, 3))
    cov = np.array([[1.0, 0.3, 0.0], [0.3, 2.0, 0.1], [0.0, 0.1, 0.5]])
    mean = np.array([0.1, -0.2, 0.3])
    assert np.allclose(rg.log_gaussian(x, mean, cov), multivariate_normal(mean, cov).logpdf(x))


def test_smoothed_and_filtered_probabilities_sum_to_one(corn_report):
    regime = corn_report.regime
    assert regime.ok
    assert np.allclose(regime.filtered.sum(axis=1), 1.0)
    assert np.allclose(regime.smoothed.sum(axis=1), 1.0)
    assert np.allclose(regime.transition.sum(axis=1), 1.0)


def test_em_recovers_volatility_regimes():
    prices, truth = two_state_series()
    result = rg.run_regime(prices, states=2, restarts=3)
    assert result.ok
    assert result.names == ["calm", "turbulent"]
    estimated = result.smoothed["turbulent"].to_numpy() > 0.5
    aligned = truth[-len(estimated) :] == 1
    assert (estimated == aligned).mean() > 0.85


def test_states_are_ordered_by_volatility():
    prices, _ = two_state_series(seed=4)
    result = rg.run_regime(prices, states=2, restarts=2)
    assert result.means.loc["calm", "log_vol"] < result.means.loc["turbulent", "log_vol"]
    assert all(duration > 1 for duration in result.duration_weeks.values())


def test_short_history_is_rejected():
    prices, _ = two_state_series(n=120)
    assert not rg.run_regime(prices).ok
