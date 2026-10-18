from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C

STATE_NAMES = {2: ("calm", "turbulent"), 3: ("calm", "normal", "turbulent")}
RIDGE = 1e-4


@dataclass
class RegimeResult:
    ok: bool
    note: str = ""
    names: list[str] = field(default_factory=list)
    filtered: pd.DataFrame | None = None
    smoothed: pd.DataFrame | None = None
    state: str = ""
    certainty: float = math.nan
    transition: np.ndarray | None = None
    duration_weeks: dict[str, float] = field(default_factory=dict)
    means: pd.DataFrame | None = None
    loglik: float = math.nan


def regime_features(prices: pd.Series, gap_z: pd.Series | None = None) -> pd.DataFrame:
    weekly = np.log(prices.dropna()).resample(C.WEEK_RULE).last().dropna()
    returns = weekly.diff()
    vol = returns.rolling(8, min_periods=6).std()
    frame = pd.DataFrame({"ret": returns, "log_vol": np.log(vol.where(vol > 1e-9))})
    if gap_z is not None:
        monthly = gap_z.reindex(gap_z.index.union(frame.index)).ffill().reindex(frame.index)
        frame["gap_z"] = monthly.clip(-4, 4)
    return frame.dropna()


def log_gaussian(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    chol = np.linalg.cholesky(cov)
    solved = np.linalg.solve(chol, (x - mean).T)
    log_det = 2.0 * np.log(np.diag(chol)).sum()
    return -0.5 * (x.shape[1] * math.log(2 * math.pi) + log_det + (solved**2).sum(axis=0))


def emissions(x: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    return np.column_stack([log_gaussian(x, mu[k], cov[k]) for k in range(len(mu))])


def forward_backward(log_b: np.ndarray, pi: np.ndarray, a: np.ndarray):
    t_len, k = log_b.shape
    shift = log_b.max(axis=1, keepdims=True)
    b = np.exp(log_b - shift)
    alpha = np.empty((t_len, k))
    scale = np.empty(t_len)
    current = pi * b[0]
    scale[0] = current.sum()
    alpha[0] = current / scale[0]
    for t in range(1, t_len):
        current = (alpha[t - 1] @ a) * b[t]
        scale[t] = current.sum()
        alpha[t] = current / scale[t]
    beta = np.empty((t_len, k))
    beta[-1] = 1.0
    for t in range(t_len - 2, -1, -1):
        beta[t] = (a @ (b[t + 1] * beta[t + 1])) / scale[t + 1]
    loglik = float(np.log(scale).sum() + shift.sum())
    return alpha, beta, scale, b, loglik


def initial_parameters(x: np.ndarray, k: int, rng: np.random.Generator) -> dict:
    order = np.argsort(x[:, 1] if x.shape[1] > 1 else x[:, 0])
    chunks = np.array_split(order, k)
    mu = np.array([x[c].mean(axis=0) for c in chunks]) + 0.1 * rng.standard_normal((k, x.shape[1]))
    base = np.cov(x.T) + RIDGE * np.eye(x.shape[1])
    cov = np.array([base * (0.5 + rng.random()) for _ in range(k)])
    a = np.full((k, k), 0.1 / max(k - 1, 1))
    np.fill_diagonal(a, 0.9)
    return {"pi": np.full(k, 1.0 / k), "a": a, "mu": mu, "cov": cov}


def fit_once(x: np.ndarray, k: int, seed: int, iterations: int = 80, tolerance: float = 1e-4) -> dict:
    rng = np.random.default_rng(seed)
    params = initial_parameters(x, k, rng)
    floor = RIDGE * np.eye(x.shape[1]) + 1e-3 * np.diag(np.var(x, axis=0))
    previous = -np.inf
    loglik = -np.inf
    for _ in range(iterations):
        log_b = emissions(x, params["mu"], params["cov"])
        alpha, beta, scale, b, loglik = forward_backward(log_b, params["pi"], params["a"])
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        xi = params["a"] * ((alpha[:-1] / scale[1:, None]).T @ (b[1:] * beta[1:]))
        weights = gamma.sum(axis=0)
        mu = (gamma.T @ x) / weights[:, None]
        cov = np.empty_like(params["cov"])
        for state in range(k):
            centred = x - mu[state]
            cov[state] = (gamma[:, state, None] * centred).T @ centred / weights[state] + floor
        a = xi / xi.sum(axis=1, keepdims=True)
        params = {"pi": gamma[0] / gamma[0].sum(), "a": a, "mu": mu, "cov": cov}
        if abs(loglik - previous) < tolerance * max(1.0, abs(loglik)):
            break
        previous = loglik
    params["loglik"] = loglik
    return params


def fit_hmm(x: np.ndarray, k: int, restarts: int, seed: int = 0) -> dict:
    best = None
    for attempt in range(restarts):
        try:
            candidate = fit_once(x, k, seed + attempt)
        except np.linalg.LinAlgError:
            continue
        if np.isfinite(candidate["loglik"]) and (best is None or candidate["loglik"] > best["loglik"]):
            best = candidate
    if best is None:
        raise ValueError("HMM fit failed for every restart")
    return best


def run_regime(
    prices: pd.Series,
    gap_z: pd.Series | None = None,
    states: int = C.HMM_STATES,
    restarts: int = C.HMM_RESTARTS,
    seed: int = 0,
) -> RegimeResult:
    frame = regime_features(prices, gap_z)
    if len(frame) < 200:
        return RegimeResult(False, "not enough weekly history")
    mean, deviation = frame.mean(), frame.std().replace(0, 1.0)
    x = ((frame - mean) / deviation).to_numpy()
    try:
        params = fit_hmm(x, states, restarts, seed)
    except ValueError as error:
        return RegimeResult(False, str(error))
    vol_column = list(frame.columns).index("log_vol")
    order = np.argsort(params["mu"][:, vol_column])
    names = list(STATE_NAMES.get(states, tuple(f"state{i}" for i in range(states))))
    mu, cov = params["mu"][order], params["cov"][order]
    a = params["a"][np.ix_(order, order)]
    pi = params["pi"][order]
    log_b = emissions(x, mu, cov)
    alpha, beta, _, _, loglik = forward_backward(log_b, pi, a)
    smooth = alpha * beta
    smooth /= smooth.sum(axis=1, keepdims=True)
    filtered = pd.DataFrame(alpha, index=frame.index, columns=names)
    smoothed = pd.DataFrame(smooth, index=frame.index, columns=names)
    last = filtered.iloc[-1]
    stay = np.diag(a)
    means = pd.DataFrame(mu * deviation.to_numpy() + mean.to_numpy(), index=names, columns=frame.columns)
    return RegimeResult(
        ok=True,
        names=names,
        filtered=filtered,
        smoothed=smoothed,
        state=str(last.idxmax()),
        certainty=float(last.max()),
        transition=a,
        duration_weeks={name: float(1.0 / max(1.0 - stay[i], 1e-9)) for i, name in enumerate(names)},
        means=means,
        loglik=float(loglik),
    )
