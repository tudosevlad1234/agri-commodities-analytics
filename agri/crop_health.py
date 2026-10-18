from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import MONTH_END, Contract, Region, regions_for, season_of
from .fair_value import constrained_ridge, predict

FEATURES = ["moist", "heat", "cold", "ndvi"]
YIELD_SIGNS = {"moist": 1, "heat": -1, "cold": -1, "ndvi": 1}
MIN_DAYS_PER_MONTH = 20
MIN_CLIMATE_YEARS = 5
MIN_TREND_SEASONS = 6
MIN_TRAIN_ROWS = 12
CARRY_MONTHS = 11
YIELD_LAG_MONTHS = 3
WEATHER_STRESS_WEIGHTS = {"moist": -0.5, "heat": 0.5, "cold": 0.3}


@dataclass
class CropResult:
    ok: bool
    note: str = ""
    monthly: pd.DataFrame | None = None
    regions: pd.DataFrame | None = None
    skill: float = math.nan
    coefficients: dict[str, float] = field(default_factory=dict)
    supply_shock_pct: float = math.nan
    price_impact_pct: float = math.nan


def monthly_weather(daily: pd.DataFrame) -> pd.DataFrame:
    grouped = daily[["tmax", "tmin", "rain", "soil"]].resample(MONTH_END)
    means = grouped.mean()
    counts = grouped.count().min(axis=1)
    return means[counts >= MIN_DAYS_PER_MONTH]


def past_only_z(series: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=series.index)
    for month in range(1, 13):
        part = series[series.index.month == month]
        mean = part.expanding(min_periods=MIN_CLIMATE_YEARS).mean().shift(1)
        deviation = part.expanding(min_periods=MIN_CLIMATE_YEARS).std().shift(1)
        out.loc[part.index] = (part - mean) / deviation.where(deviation > 1e-9)
    return out.clip(-4, 4)


def monthly_anomalies(daily: pd.DataFrame, ndvi: pd.Series | None) -> pd.DataFrame:
    monthly = monthly_weather(daily)
    rain_z = past_only_z(monthly["rain"])
    soil_z = past_only_z(monthly["soil"])
    frame = pd.DataFrame(
        {
            "moist": pd.concat([rain_z, soil_z], axis=1).mean(axis=1, skipna=False),
            "heat": past_only_z(monthly["tmax"]),
            "cold": -past_only_z(monthly["tmin"]),
        }
    )
    if ndvi is not None and ndvi.notna().any():
        frame["ndvi"] = past_only_z(ndvi.resample(MONTH_END).mean().reindex(frame.index))
    else:
        frame["ndvi"] = np.nan
    return frame


def in_progress_features(region: Region, anomalies: pd.DataFrame) -> pd.DataFrame:
    frame = anomalies.copy()
    frame["cold"] = frame["cold"] * (1.0 if region.frost_sensitive else 0.0)
    inside = frame.index.month.isin(region.window)
    window = frame[inside].copy()
    window["season"] = [season_of(region, stamp) for stamp in window.index]
    running = window.groupby("season")[FEATURES].expanding(min_periods=1).mean().droplevel(0)
    window[FEATURES] = running.reindex(window.index)
    has_data = anomalies.loc[inside, ["moist", "heat"]].notna().all(axis=1).astype(int)
    window["observed"] = has_data.groupby(window["season"]).cumsum()
    carried = window.reindex(frame.index).ffill(limit=CARRY_MONTHS)
    carried["observed"] = carried["observed"].astype(float)
    return carried


def yield_deviation(log_yield: pd.Series) -> pd.Series:
    seasons = log_yield.dropna()
    out = pd.Series(np.nan, index=seasons.index, dtype=float)
    for position, season in enumerate(seasons.index):
        past = seasons.iloc[:position]
        if len(past) < MIN_TREND_SEASONS:
            continue
        slope, intercept = np.polyfit(past.index.to_numpy(dtype=float), past.to_numpy(), 1)
        out.loc[season] = seasons.iloc[position] - (intercept + slope * season)
    return out


def complete_season_rows(panel: pd.DataFrame, deviation: pd.Series, region: Region) -> pd.DataFrame:
    rows = []
    for season, group in panel.dropna(subset=["season"]).groupby("season"):
        if season not in deviation.index or not np.isfinite(deviation.loc[season]):
            continue
        if int(group["observed"].max()) < len(region.window):
            continue
        position = int(group["observed"].to_numpy().argmax())
        last = group.iloc[position]
        ended = group.index[position]
        record = last[FEATURES].astype(float).to_dict()
        record.update(
            season=int(season),
            dev=float(deviation.loc[season]),
            region=region.name,
            window_end=ended,
            known_from=ended + pd.offsets.MonthEnd(YIELD_LAG_MONTHS),
        )
        rows.append(record)
    return pd.DataFrame(rows)


class YieldModel:
    def __init__(self, rows: pd.DataFrame):
        self.rows = rows.sort_values("known_from").reset_index(drop=True)
        self.known = self.rows["known_from"].to_numpy(dtype="datetime64[ns]")
        self.cache: dict[int, dict | None] = {}

    def fit_at(self, stamp: pd.Timestamp) -> dict | None:
        count = int(
            np.searchsorted(self.known, pd.Timestamp(stamp).to_datetime64().astype("datetime64[ns]"), side="right")
        )
        if count not in self.cache:
            self.cache[count] = self._fit(self.rows.iloc[:count])
        return self.cache[count]

    @staticmethod
    def _fit(train: pd.DataFrame) -> dict | None:
        if len(train) < MIN_TRAIN_ROWS:
            return None
        features = [c for c in FEATURES if train[c].notna().mean() > 0.8]
        train = train.dropna(subset=features + ["dev"])
        if len(train) < MIN_TRAIN_ROWS or not features:
            return None
        fitted = constrained_ridge(train[features], train["dev"], signs=YIELD_SIGNS, fast=True)
        fitted["columns"] = features
        return fitted

    def predict_at(self, stamp: pd.Timestamp, row: pd.Series) -> float:
        model = self.fit_at(stamp)
        if model is None:
            return math.nan
        values = row[model["columns"]].astype(float).fillna(0.0)
        return float(predict(model, values.to_frame().T)[0])


def region_nowcast(panel: pd.DataFrame, model: YieldModel) -> pd.Series:
    predictions = pd.Series(np.nan, index=panel.index)
    for stamp, row in panel.iterrows():
        if np.isfinite(row["season"]):
            predictions.loc[stamp] = model.predict_at(stamp, row)
    return predictions


def out_of_sample_skill(model: YieldModel) -> float:
    predicted, actual = [], []
    for _, row in model.rows.iterrows():
        value = model.predict_at(row["window_end"], row)
        if np.isfinite(value):
            predicted.append(value)
            actual.append(row["dev"])
    if len(actual) < 10:
        return math.nan
    actual_values, predicted_values = np.array(actual), np.array(predicted)
    total = float((actual_values**2).sum())
    return 1.0 - float(((actual_values - predicted_values) ** 2).sum()) / total if total > 0 else math.nan


def weather_stress(panel: pd.DataFrame) -> pd.Series:
    stress = sum(weight * panel[name].fillna(0.0) for name, weight in WEATHER_STRESS_WEIGHTS.items())
    return stress.where(panel["season"].notna())


def share_weighted(values: pd.DataFrame, shares: pd.Series) -> pd.Series:
    weights = values.notna().mul(shares, axis=1)
    total = weights.sum(axis=1)
    return (values.fillna(0.0) * weights).sum(axis=1) / total.where(total > 0)


def run_crop_health(
    contract: Contract,
    weather: dict[str, pd.DataFrame],
    ndvi: dict[str, pd.Series],
    yields: dict[str, pd.Series],
    regions: tuple[Region, ...] | None = None,
    su_elasticity: float = -0.5,
    su_now: float = math.nan,
) -> CropResult:
    chosen = [r for r in (regions or ()) if r.contract == contract.key] or regions_for(contract.key)
    chosen = [r for r in chosen if r.name in weather]
    if not chosen:
        return CropResult(False, "no regions with weather data")
    panels = {r.name: in_progress_features(r, monthly_anomalies(weather[r.name], ndvi.get(r.name))) for r in chosen}
    tables = [
        complete_season_rows(panels[r.name], yield_deviation(yields[r.name]), r) for r in chosen if r.name in yields
    ]
    rows = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    if len(rows) < MIN_TRAIN_ROWS + 4:
        return CropResult(False, "not enough completed seasons with yields")
    model = YieldModel(rows)
    shares = pd.Series({r.name: r.share for r in chosen})
    predictions = pd.DataFrame({r.name: region_nowcast(panels[r.name], model) for r in chosen})
    stress = pd.DataFrame({r.name: weather_stress(panels[r.name]) for r in chosen})
    monthly = pd.DataFrame(
        {"supply_shock": share_weighted(predictions, shares), "weather_stress": share_weighted(stress, shares)}
    )
    for column, name in (("supply_shock", "shock_z"), ("weather_stress", "stress_z")):
        scale = monthly[column].expanding(min_periods=36).std().shift(1)
        monthly[name] = (monthly[column] / scale.where(scale > 1e-12)).clip(-4, 4)
    latest = monthly["supply_shock"].dropna()
    if latest.empty:
        return CropResult(False, "no nowcast available")
    final = model.fit_at(monthly.index[-1])
    coefficients = {}
    if final is not None:
        coefficients = {name: float(final["coef"][i] / final["std"].iloc[i]) for i, name in enumerate(final["columns"])}
    table = pd.DataFrame(
        {
            "share": shares,
            "predicted_yield_dev_pct": predictions.iloc[-1] * 100.0,
            "weather_stress": stress.iloc[-1],
            "season": [panels[r.name]["season"].iloc[-1] for r in chosen],
            "months_observed": [panels[r.name]["observed"].iloc[-1] for r in chosen],
        }
    )
    shock_now = float(latest.iloc[-1])
    impact = su_elasticity * shock_now / su_now * 100.0 if np.isfinite(su_now) and su_now > 0 else math.nan
    return CropResult(
        ok=True,
        monthly=monthly,
        regions=table,
        skill=out_of_sample_skill(model),
        coefficients=coefficients,
        supply_shock_pct=shock_now * 100.0,
        price_impact_pct=impact,
    )
