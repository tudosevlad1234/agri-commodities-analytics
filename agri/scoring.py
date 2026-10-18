from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C
from .fair_value import FVResult
from .predictor import PredictorResult
from .regime import RegimeResult

IC_WINDOW = 260
IC_MIN_OBS = 104
IC_SHRINK = 0.7
STRONG_SCORE = 35.0
MILD_SCORE = 15.0
ACTIVE_THRESHOLD = 0.1
ML_MAX_WEIGHT = 0.5


@dataclass
class ScoreResult:
    score: float = math.nan
    composite: float = math.nan
    ml_signal: float = math.nan
    ml_weight: float = 0.0
    verdict: str = "n/a"
    p_up: float = math.nan
    trust: float = math.nan
    trust_parts: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    signals: dict[str, float] = field(default_factory=dict)
    history: pd.Series | None = None


def clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, value))) if np.isfinite(value) else 0.0


def rolling_ic(features: pd.DataFrame, horizon: int) -> pd.DataFrame:
    target = features["target"]
    out = {}
    for name in C.FAMILIES:
        if name not in features:
            continue
        correlation = features[name].rolling(IC_WINDOW, min_periods=IC_MIN_OBS).corr(target)
        out[name] = correlation.shift(horizon)
    return pd.DataFrame(out, index=features.index)


def family_weights(ic: pd.DataFrame) -> pd.DataFrame:
    prior = pd.Series({k: v for k, v in C.PRIOR_WEIGHTS.items() if k in ic.columns})
    prior = prior / prior.sum()
    positive = ic[prior.index].clip(lower=0.0)
    total = positive.sum(axis=1)
    measured = positive.div(total.where(total > 1e-9), axis=0)
    blended = IC_SHRINK * prior + (1.0 - IC_SHRINK) * measured.fillna(prior)
    return blended.div(blended.sum(axis=1), axis=0)


def composite_series(features: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    columns = list(weights.columns)
    values = features[columns]
    active = weights.where(values.notna(), 0.0)
    total = active.sum(axis=1)
    return (values.fillna(0.0) * active).sum(axis=1) / total.where(total > 1e-9)


def verdict_for(score: float) -> str:
    if not np.isfinite(score):
        return "n/a"
    side = "bullish" if score > 0 else "bearish"
    if abs(score) >= STRONG_SCORE:
        return f"{side.upper()} (strong)"
    if abs(score) >= MILD_SCORE:
        return side.capitalize()
    return "Neutral"


def has_skill(predictor: PredictorResult | None) -> bool:
    return bool(
        predictor is not None
        and predictor.ok
        and np.isfinite(predictor.auc)
        and np.isfinite(predictor.brier_skill)
        and predictor.auc > 0.5
        and predictor.brier_skill > 0
    )


def ml_weight_from_skill(predictor: PredictorResult | None) -> float:
    if not has_skill(predictor):
        return 0.0
    return ML_MAX_WEIGHT * clamp01((predictor.auc - 0.5) / 0.1)


def model_skill_part(predictor: PredictorResult | None) -> float:
    if predictor is None or not predictor.ok:
        return 0.0
    auc = clamp01((predictor.auc - 0.5) / 0.1) if np.isfinite(predictor.auc) else 0.0
    brier = clamp01(predictor.brier_skill / 0.05) if np.isfinite(predictor.brier_skill) else 0.0
    return 0.5 * (auc + brier)


def agreement_part(signals: dict[str, float], weights: dict[str, float], composite: float) -> float:
    if not np.isfinite(composite) or composite == 0:
        return 0.0
    agree, total = 0.0, 0.0
    for name, value in signals.items():
        if not np.isfinite(value) or abs(value) < ACTIVE_THRESHOLD:
            continue
        weight = weights.get(name, 0.0)
        total += weight
        agree += weight * (np.sign(value) == np.sign(composite))
    return float(agree / total) if total > 0 else 0.0


def fair_value_part(fair_value: FVResult | None) -> float:
    if fair_value is None or not fair_value.ok:
        return 0.0
    parts = []
    parts.append(clamp01(1.0 - fair_value.coint_p / 0.20) if np.isfinite(fair_value.coint_p) else 0.0)
    parts.append(clamp01(fair_value.r2_out_of_sample / 0.5) if np.isfinite(fair_value.r2_out_of_sample) else 0.0)
    half_life = fair_value.half_life_months
    parts.append(clamp01(1.0 - (half_life - 12.0) / 48.0) if np.isfinite(half_life) else 0.0)
    return float(np.mean(parts))


def regime_part(regime: RegimeResult | None) -> float:
    if regime is None or not regime.ok:
        return 0.0
    penalty = 0.7 if regime.state == "turbulent" else 1.0
    return clamp01(regime.certainty * penalty)


def coverage_part(signals: dict[str, float]) -> float:
    present = sum(1 for name in C.FAMILIES if np.isfinite(signals.get(name, math.nan)))
    return present / len(C.FAMILIES)


def stability_part(history: pd.Series) -> float:
    recent = history.dropna().iloc[-9:]
    if len(recent) < 5:
        return 0.0
    return clamp01(1.0 - float(recent.diff().abs().mean()) / 0.25)


def score_contract(
    features: pd.DataFrame,
    fair_value: FVResult | None,
    regime: RegimeResult | None,
    predictor: PredictorResult | None,
    horizon: int = C.HORIZON_WEEKS,
) -> ScoreResult:
    ic = rolling_ic(features, horizon)
    weights = family_weights(ic)
    composite = composite_series(features, weights)
    valid = composite.dropna()
    if valid.empty:
        return ScoreResult()
    latest_composite = float(valid.iloc[-1])
    latest_weights = weights.loc[valid.index[-1]].to_dict()
    signals = {name: float(features[name].iloc[-1]) for name in C.FAMILIES if name in features}
    ml_weight = ml_weight_from_skill(predictor)
    model_p = predictor.p_up if predictor is not None and predictor.ok else math.nan
    ml_signal = float(np.clip(2.0 * model_p - 1.0, -1.0, 1.0)) if np.isfinite(model_p) else math.nan
    if not np.isfinite(ml_signal):
        ml_weight = 0.0
    p_up = model_p if has_skill(predictor) else (predictor.base_rate if predictor is not None else math.nan)
    blended = (1.0 - ml_weight) * latest_composite + ml_weight * (ml_signal if ml_weight > 0 else 0.0)
    score = 100.0 * blended
    parts = {
        "model_skill": model_skill_part(predictor),
        "signal_agreement": agreement_part(signals, latest_weights, blended),
        "fair_value_quality": fair_value_part(fair_value),
        "regime_certainty": regime_part(regime),
        "data_coverage": coverage_part(signals),
        "stability": stability_part(composite),
    }
    trust = 100.0 * sum(C.TRUST_WEIGHTS[name] * value for name, value in parts.items())
    if not has_skill(predictor):
        trust = min(trust, C.TRUST_CAP_NO_SKILL)
    return ScoreResult(
        score=score,
        composite=latest_composite,
        ml_signal=ml_signal,
        ml_weight=ml_weight,
        verdict=verdict_for(score),
        p_up=p_up,
        trust=float(trust),
        trust_parts=parts,
        weights=latest_weights,
        signals=signals,
        history=composite * 100.0,
    )
