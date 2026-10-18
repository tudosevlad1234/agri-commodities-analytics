from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from . import config as C
from .crop_health import CropResult, run_crop_health
from .data import MarketBundle
from .fair_value import FVResult, run_fair_value
from .features import build_features
from .predictor import PredictorResult, walk_forward_predict
from .regime import RegimeResult, run_regime
from .scoring import ScoreResult, score_contract


@dataclass
class ContractReport:
    contract: C.Contract
    fair_value: FVResult
    crop: CropResult
    regime: RegimeResult
    predictor: PredictorResult
    score: ScoreResult
    features: pd.DataFrame | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class Analysis:
    reports: dict[str, ContractReport]
    as_of: pd.Timestamp
    horizon_weeks: int

    def summary(self) -> pd.DataFrame:
        rows = []
        for report in self.reports.values():
            fv, score, regime, crop = report.fair_value, report.score, report.regime, report.crop
            rows.append(
                {
                    "contract": report.contract.name,
                    "price": fv.price_now,
                    "fair_value": fv.fv_now,
                    "gap_pct": fv.gap_pct,
                    "gap_z": fv.gap_z,
                    "supply_shock_pct": crop.supply_shock_pct if crop.ok else math.nan,
                    "regime": regime.state if regime.ok else "n/a",
                    "score": score.score,
                    "p_up_pct": score.p_up * 100.0 if math.isfinite(score.p_up) else math.nan,
                    "verdict": score.verdict,
                    "trust": score.trust,
                }
            )
        return pd.DataFrame(rows, index=list(self.reports))


def analyse_contract(bundle: MarketBundle, key: str, horizon: int = C.HORIZON_WEEKS) -> ContractReport:
    contract = C.CONTRACTS[key]
    prices = bundle.prices[key]
    fair_value = run_fair_value(contract, prices, bundle.macro, bundle.stocks_to_use[key])
    elasticity = fair_value.su_elasticity if fair_value.ok and math.isfinite(fair_value.su_elasticity) else -0.5
    crop = run_crop_health(
        contract,
        bundle.weather,
        bundle.ndvi,
        bundle.yields,
        bundle.regions,
        su_elasticity=elasticity,
        su_now=fair_value.su_now if fair_value.ok else math.nan,
    )
    gap_z = fair_value.history["gap_z"] if fair_value.ok else None
    regime = run_regime(prices, gap_z)
    features = build_features(
        prices,
        bundle.macro,
        fair_value.history if fair_value.ok else None,
        crop.monthly if crop.ok else None,
        horizon,
    )
    predictor = walk_forward_predict(features, horizon)
    score = score_contract(features, fair_value if fair_value.ok else None, regime, predictor, horizon)
    notes = [
        f"{name}: {result.note}"
        for name, result in (
            ("fair value", fair_value),
            ("crop health", crop),
            ("regime", regime),
            ("predictor", predictor),
        )
        if not result.ok
    ]
    return ContractReport(contract, fair_value, crop, regime, predictor, score, features, notes)


def run(bundle: MarketBundle, contracts: tuple[str, ...] | None = None, horizon: int = C.HORIZON_WEEKS) -> Analysis:
    keys = [k for k in (contracts or tuple(C.CONTRACTS)) if k in bundle.prices and k in bundle.stocks_to_use]
    if not keys:
        raise ValueError("no contract has both prices and stocks-to-use data")
    reports = {key: analyse_contract(bundle, key, horizon) for key in keys}
    as_of = max(bundle.prices[k].dropna().index[-1] for k in keys)
    return Analysis(reports, as_of, horizon)
