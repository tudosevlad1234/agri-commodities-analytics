from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from . import config as C

MIN_CLASS_COUNT = 30


@dataclass
class PredictorResult:
    ok: bool
    note: str = ""
    predictions: pd.DataFrame | None = None
    auc: float = math.nan
    brier_skill: float = math.nan
    hit_rate: float = math.nan
    n_oos: int = 0
    p_up: float = math.nan
    base_rate: float = math.nan
    expected_return_pct: float = math.nan
    importances: dict[str, float] = field(default_factory=dict)


def recency_weights(index: pd.DatetimeIndex, half_life_years: float = C.ML_RECENCY_HALF_LIFE_YEARS) -> np.ndarray:
    age = (index[-1] - index).days.to_numpy() / 365.25
    return 0.5 ** (age / half_life_years)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    wanted = list(C.ML_FEATURES) + list(C.TILT_FEATURES)
    return [c for c in wanted if c in frame and frame[c].notna().sum() > 100]


def fit_models(train: pd.DataFrame, columns: list[str], seed: int) -> dict | None:
    labels = (train["target"] > 0).astype(int)
    if labels.sum() < MIN_CLASS_COUNT or (1 - labels).sum() < MIN_CLASS_COUNT:
        return None
    x = train[columns].fillna(0.0)
    scaler = StandardScaler().fit(x)
    z = scaler.transform(x)
    weights = recency_weights(train.index)
    logistic = LogisticRegression(C=0.3, max_iter=500).fit(z, labels, sample_weight=weights)
    boosted = GradientBoostingClassifier(
        n_estimators=80,
        max_depth=2,
        learning_rate=0.05,
        min_samples_leaf=40,
        subsample=0.8,
        random_state=seed,
    ).fit(x.to_numpy(), labels, sample_weight=weights)
    ridge = Ridge(alpha=50.0).fit(z, train["target"], sample_weight=weights)
    return {"scaler": scaler, "logistic": logistic, "boosted": boosted, "ridge": ridge, "columns": columns}


def predict_models(models: dict, frame: pd.DataFrame) -> pd.DataFrame:
    x = frame[models["columns"]].fillna(0.0)
    z = models["scaler"].transform(x)
    p_logistic = models["logistic"].predict_proba(z)[:, 1]
    p_boosted = models["boosted"].predict_proba(x.to_numpy())[:, 1]
    p_up = C.ML_GBM_WEIGHT * p_boosted + (1.0 - C.ML_GBM_WEIGHT) * p_logistic
    return pd.DataFrame(
        {"p_up": p_up, "expected": models["ridge"].predict(z), "p_logistic": p_logistic, "p_boosted": p_boosted},
        index=frame.index,
    )


def walk_forward_predict(frame: pd.DataFrame, horizon: int = C.HORIZON_WEEKS, seed: int = 0) -> PredictorResult:
    columns = feature_columns(frame)
    if len(columns) < 3:
        return PredictorResult(False, "not enough feature families")
    usable = frame.dropna(subset=["trend"])
    if len(usable) < C.WALK_FORWARD_MIN_WEEKS + 52:
        return PredictorResult(False, "not enough weekly history for walk-forward")
    index = usable.index
    outputs = []
    models = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for start in range(C.WALK_FORWARD_MIN_WEEKS, len(index), C.WALK_FORWARD_REFIT_WEEKS):
            stop = min(start + C.WALK_FORWARD_REFIT_WEEKS, len(index))
            cutoff = start - horizon
            train = usable.iloc[:cutoff].dropna(subset=["target"])
            fitted = fit_models(train, columns, seed)
            models = fitted or models
            if models is None:
                continue
            outputs.append(predict_models(models, usable.iloc[start:stop]))
    if not outputs:
        return PredictorResult(False, "no model could be fitted")
    predictions = pd.concat(outputs)
    scored = predictions.join(usable["target"]).dropna()
    result = PredictorResult(True, predictions=predictions, n_oos=len(scored))
    if len(scored) >= 100 and scored["target"].gt(0).nunique() == 2:
        outcome = (scored["target"] > 0).astype(float)
        result.auc = float(roc_auc_score(outcome, scored["p_up"]))
        climatology = base_rate(usable["target"], horizon).reindex(scored.index).fillna(0.5)
        reference = float(((outcome - climatology) ** 2).mean())
        brier = float(((outcome - scored["p_up"]) ** 2).mean())
        result.brier_skill = 1.0 - brier / reference if reference > 0 else math.nan
        result.hit_rate = float(((scored["p_up"] > 0.5) == (outcome > 0.5)).mean())
    climatology_now = base_rate(usable["target"], horizon).dropna()
    result.base_rate = float(climatology_now.iloc[-1]) if len(climatology_now) else math.nan
    last = predictions.iloc[-1]
    result.p_up = float(last["p_up"])
    result.expected_return_pct = float(last["expected"] * 100.0)
    result.importances = feature_importance(usable, columns, horizon, seed)
    return result


def base_rate(target: pd.Series, horizon: int) -> pd.Series:
    up = (target > 0).astype(float).where(target.notna())
    return up.expanding(min_periods=52).mean().shift(horizon)


def feature_importance(frame: pd.DataFrame, columns: list[str], horizon: int, seed: int) -> dict[str, float]:
    train = frame.iloc[:-horizon].dropna(subset=["target"])
    if len(train) < 200:
        return {}
    x = train[columns].fillna(0.0)
    z = StandardScaler().fit_transform(x)
    fit = Ridge(alpha=50.0).fit(z, train["target"])
    magnitude = np.abs(fit.coef_)
    total = magnitude.sum()
    if total <= 0:
        return {}
    return {name: float(value / total) for name, value in zip(columns, magnitude, strict=True)}
