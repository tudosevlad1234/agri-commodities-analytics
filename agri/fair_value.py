from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.stattools import coint

from . import config as C
from .config import Contract

SIGN_PRIORS = {"su1": -1, "inv_su1": 1, "usd": -1, "crude": 1}
COLUMNS = {"A": ["su1", "usd", "crude"], "B": ["inv_su1", "usd", "crude"], "C": ["su1", "usd"]}
WEIGHT_WINDOW = 60
WEIGHT_MIN_OBS = 12


@dataclass
class FVResult:
    ok: bool
    note: str = ""
    price_now: float = math.nan
    fv_now: float = math.nan
    gap_pct: float = math.nan
    gap_z: float = math.nan
    sigma: float = math.nan
    r2_in_sample: float = math.nan
    r2_out_of_sample: float = math.nan
    coint_p: float = math.nan
    phi: float = math.nan
    half_life_months: float = math.nan
    su_elasticity: float = math.nan
    su_now: float = math.nan
    weights: dict[str, float] = field(default_factory=dict)
    coefficients: dict[str, float] = field(default_factory=dict)
    history: pd.DataFrame | None = None


def next_year_weight(contract: Contract, stamp: pd.Timestamp) -> float:
    months_to_start = (contract.marketing_year_start_month - stamp.month) % 12
    return (5 - months_to_start) / 5.0 if 1 <= months_to_start <= 4 else 0.0


def marketing_year(contract: Contract, stamp: pd.Timestamp) -> int:
    return stamp.year if stamp.month >= contract.marketing_year_start_month else stamp.year - 1


def su_monthly(su: pd.Series, contract: Contract, index: pd.DatetimeIndex) -> pd.Series:
    values = []
    for stamp in index:
        current = marketing_year(contract, stamp)
        weight = next_year_weight(contract, stamp)
        first = su.get(current, np.nan)
        second = su.get(current + 1, np.nan)
        if weight > 0 and np.isfinite(first) and np.isfinite(second):
            values.append((1 - weight) * first + weight * second)
        else:
            values.append(first)
    return pd.Series(values, index=index, dtype=float)


def month_end_prices(prices: pd.Series) -> pd.Series:
    clean = prices.dropna().sort_index()
    return clean.groupby(clean.index.to_period("M")).tail(1)


def by_month(series: pd.Series, periods: pd.PeriodIndex) -> pd.Series:
    clean = series.dropna()
    monthly = clean.groupby(clean.index.to_period("M")).last()
    return monthly.reindex(monthly.index.union(periods)).ffill().reindex(periods)


def monthly_frame(contract: Contract, prices: pd.Series, macro: pd.DataFrame, su: pd.Series) -> pd.DataFrame:
    px = month_end_prices(prices)
    index = px.index
    periods = index.to_period("M")
    if "cpi" in macro and macro["cpi"].notna().any():
        cpi = by_month(macro["cpi"], periods).to_numpy(dtype=float)
        ratio = pd.Series(cpi / cpi[np.isfinite(cpi)][-1], index=index).bfill()
    else:
        ratio = pd.Series(1.0, index=index)
    frame = pd.DataFrame({"price": px.to_numpy(), "cpi_ratio": ratio.to_numpy()}, index=index)
    frame["y"] = np.log(frame["price"] / frame["cpi_ratio"])
    current = su_monthly(su, contract, index)
    frame["su1"] = np.log(current)
    frame["inv_su1"] = 1.0 / current
    if "usd" in macro:
        frame["usd"] = np.log(by_month(macro["usd"], periods).to_numpy(dtype=float))
    if "crude" in macro:
        frame["crude"] = np.log(by_month(macro["crude"], periods).to_numpy(dtype=float) / frame["cpi_ratio"])
    return frame


def constrained_ridge(
    x: pd.DataFrame,
    y: pd.Series,
    alpha: float | None = None,
    fast: bool = False,
    signs: dict[str, int] | None = None,
):
    signs = SIGN_PRIORS if signs is None else signs
    mean, deviation = x.mean(), x.std().replace(0, 1.0).fillna(1.0)
    z = ((x - mean) / deviation).to_numpy()
    target = y.to_numpy()
    if alpha is None:
        grid = np.logspace(-3, 2.5, 30)
        if fast:
            alpha = float(RidgeCV(alphas=grid).fit(z, target).alpha_)
        else:
            splits = 5 if len(target) >= 100 else 3
            alpha = float(RidgeCV(alphas=grid, cv=TimeSeriesSplit(splits)).fit(z, target).alpha_)
    k = z.shape[1]
    design = np.vstack(
        [np.hstack([z, np.ones((len(target), 1))]), np.hstack([math.sqrt(alpha) * np.eye(k), np.zeros((k, 1))])]
    )
    response = np.concatenate([target, np.zeros(k)])
    lower, upper = [], []
    for column in x.columns:
        sign = signs.get(column, 0)
        lower.append(-np.inf if sign <= 0 else 0.0)
        upper.append(np.inf if sign >= 0 else 0.0)
    lower.append(-np.inf)
    upper.append(np.inf)
    solution = lsq_linear(design, response, bounds=(lower, upper), method="bvls")
    return {"coef": solution.x[:-1], "intercept": solution.x[-1], "mean": mean, "std": deviation, "alpha": alpha}


def predict(model: dict, x: pd.DataFrame) -> np.ndarray:
    return ((x - model["mean"]) / model["std"]).to_numpy() @ model["coef"] + model["intercept"]


def kalman_one_step(x: pd.DataFrame, y: pd.Series, warmup: int = 60, state_noise: float = 1e-3) -> pd.Series:
    mean, deviation = x.iloc[:warmup].mean(), x.iloc[:warmup].std().replace(0, 1.0)
    z = (x - mean) / deviation
    design = np.column_stack([np.ones(len(z)), z.to_numpy()])
    target = y.to_numpy()
    beta, *_ = np.linalg.lstsq(design[:warmup], target[:warmup], rcond=None)
    residual = target[:warmup] - design[:warmup] @ beta
    observation = float(residual.var(ddof=design.shape[1])) or 1e-4
    covariance = np.eye(design.shape[1]) * observation
    q = np.eye(design.shape[1]) * state_noise * observation
    predictions = np.full(len(target), np.nan)
    for t in range(warmup, len(target)):
        covariance = covariance + q
        row = design[t]
        predictions[t] = float(row @ beta)
        innovation_variance = float(row @ covariance @ row) + observation
        gain = covariance @ row / innovation_variance
        beta = beta + gain * (target[t] - predictions[t])
        covariance = covariance - np.outer(gain, row) @ covariance
    return pd.Series(predictions, index=y.index)


def walk_forward(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    out = pd.Series(np.nan, index=frame.index)
    for year in sorted(set(frame.index.year)):
        train = frame[frame.index < pd.Timestamp(year, 1, 1)].iloc[-C.FV_WINDOW_MONTHS :]
        if len(train) < C.FV_MIN_MONTHS:
            continue
        model = constrained_ridge(train[columns], train["y"], fast=True)
        rows = frame.index[frame.index.year == year]
        inputs = frame.loc[rows, columns]
        valid = inputs.notna().all(axis=1)
        if valid.any():
            out.loc[rows[valid.to_numpy()]] = predict(model, inputs[valid])
    return out


def inverse_mse_weights(errors: dict[str, pd.Series], window: int = WEIGHT_WINDOW) -> dict[str, float]:
    mse = {}
    for key, series in errors.items():
        tail = series.dropna().iloc[-window:]
        if len(tail) >= WEIGHT_MIN_OBS:
            mse[key] = float((tail**2).mean())
    if not mse:
        return {}
    inverse = {k: 1.0 / max(v, 1e-12) for k, v in mse.items()}
    total = sum(inverse.values())
    return {k: v / total for k, v in inverse.items()}


def rolling_weights(errors: pd.DataFrame, window: int = WEIGHT_WINDOW) -> pd.DataFrame:
    mse = (errors**2).rolling(window, min_periods=WEIGHT_MIN_OBS).mean().shift(1)
    inverse = 1.0 / mse.clip(lower=1e-12)
    return inverse.div(inverse.sum(axis=1), axis=0)


def combine(predictions: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    available = predictions.notna()
    active = weights.where(available)
    unweighted = active.sum(axis=1, min_count=1).isna()
    active.loc[unweighted] = available.loc[unweighted].astype(float)
    active = active.where(available, 0.0).fillna(0.0)
    total = active.sum(axis=1)
    return (predictions.fillna(0.0) * active).sum(axis=1) / total.where(total > 0)


def half_life_months(phi: float) -> float:
    return math.log(0.5) / math.log(phi) if 0 < phi < 1 else math.inf


def run_fair_value(contract: Contract, prices: pd.Series, macro: pd.DataFrame, su: pd.Series) -> FVResult:
    missing = [name for name in ("usd", "crude") if name not in macro]
    if missing:
        return FVResult(False, f"macro data missing: {', '.join(missing)}")
    frame = monthly_frame(contract, prices, macro, su)
    frame = frame.dropna(subset=["y", "su1", "inv_su1", "usd", "crude"])
    if len(frame) < C.FV_MIN_MONTHS + 24:
        return FVResult(False, "not enough monthly history")
    predictions = pd.DataFrame(
        {
            "A": walk_forward(frame, COLUMNS["A"]),
            "B": walk_forward(frame, COLUMNS["B"]),
            "C": kalman_one_step(frame[COLUMNS["C"]], frame["y"]),
        }
    )
    errors = predictions.rsub(frame["y"], axis=0)
    weights = inverse_mse_weights({key: errors[key] for key in errors})
    if not weights:
        return FVResult(False, "no walk-forward predictions")
    ensemble = combine(predictions, rolling_weights(errors))
    residual = frame["y"] - ensemble
    sigma = float(residual.dropna().iloc[-60:].std(ddof=1))
    window = frame.iloc[-C.FV_WINDOW_MONTHS :]
    latest = frame.iloc[[-1]]
    final = {k: constrained_ridge(window[COLUMNS[k]], window["y"]) for k in ("A", "B")}
    point = {k: float(predict(final[k], latest[COLUMNS[k]])[0]) for k in final}
    point["C"] = float(predictions["C"].iloc[-1])
    usable = {k: w for k, w in weights.items() if np.isfinite(point.get(k, math.nan))}
    total = sum(usable.values())
    fv_now = sum(w / total * point[k] for k, w in usable.items())
    price_now = float(prices.dropna().iloc[-1])
    real_now = price_now / float(frame["cpi_ratio"].iloc[-1])
    gap_log = math.log(real_now) - fv_now
    valid = residual.dropna()
    lagged = valid.shift(1).dropna()
    phi = float(np.corrcoef(valid.loc[lagged.index], lagged)[0, 1]) if len(lagged) > 24 else math.nan
    benchmark = frame["y"].expanding(min_periods=C.FV_MIN_MONTHS).mean().shift(1)
    scored = pd.concat([frame["y"], ensemble, benchmark], axis=1, keys=["y", "fv", "mean"]).dropna()
    sst = float(((scored["y"] - scored["mean"]) ** 2).sum())
    sse = float(((scored["y"] - scored["fv"]) ** 2).sum())
    model_a = final["A"]
    in_sample = window["y"] - predict(model_a, window[COLUMNS["A"]])
    try:
        p_value = float(coint(frame["y"], frame[COLUMNS["A"]], trend="c", maxlag=6, autolag=None)[1])
    except (ValueError, np.linalg.LinAlgError):
        p_value = math.nan
    history = pd.DataFrame({"y": frame["y"], "fv": ensemble, "resid": residual})
    history["gap_z"] = residual / residual.rolling(60, min_periods=24).std().shift(1)
    coefficients = {col: float(model_a["coef"][i] / model_a["std"].iloc[i]) for i, col in enumerate(COLUMNS["A"])}
    return FVResult(
        ok=True,
        price_now=price_now,
        fv_now=float(np.exp(fv_now) * frame["cpi_ratio"].iloc[-1]),
        gap_pct=float((math.exp(gap_log) - 1.0) * 100.0),
        gap_z=gap_log / sigma if sigma > 0 else math.nan,
        sigma=sigma,
        r2_in_sample=1.0 - float((in_sample**2).sum() / ((window["y"] - window["y"].mean()) ** 2).sum()),
        r2_out_of_sample=1.0 - sse / sst if sst > 0 else math.nan,
        coint_p=p_value,
        phi=phi,
        half_life_months=half_life_months(phi),
        su_elasticity=coefficients["su1"],
        su_now=float(np.exp(frame["su1"].iloc[-1])),
        weights={k: w / total for k, w in usable.items()},
        coefficients=coefficients,
        history=history,
    )
