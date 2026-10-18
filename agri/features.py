from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

SQUASH = 1.5
MONTHLY_FILL_WEEKS = 8


def squash(values, scale: float = SQUASH):
    return np.tanh(values / scale)


def weekly_close(prices: pd.Series) -> pd.Series:
    return prices.dropna().resample(C.WEEK_RULE).last().dropna()


def to_weekly(monthly: pd.Series, index: pd.DatetimeIndex, limit: int = MONTHLY_FILL_WEEKS) -> pd.Series:
    source = monthly.dropna()
    source = source[~source.index.duplicated(keep="last")]
    filled = source.reindex(source.index.union(index)).ffill().reindex(index)
    filled[_age_weeks(source.index, index) > limit] = np.nan
    return filled


def _age_weeks(source: pd.DatetimeIndex, index: pd.DatetimeIndex) -> np.ndarray:
    positions = source.searchsorted(index, side="right") - 1
    age = np.full(len(index), np.inf)
    valid = positions >= 0
    age[valid] = (index[valid] - source[positions[valid]]).days / 7.0
    return age


def trend_signal(close: pd.Series) -> pd.Series:
    logs = np.log(close)
    weekly = logs.diff()
    sigma = weekly.rolling(104, min_periods=52).std().shift(1)
    parts = []
    for weeks in (13, 26, 52):
        move = logs - logs.shift(weeks)
        parts.append(move / (sigma * np.sqrt(weeks)))
    return pd.Series(squash(pd.concat(parts, axis=1).mean(axis=1)), index=close.index)


def seasonal_signal(close: pd.Series, horizon: int) -> pd.Series:
    logs = np.log(close)
    forward = logs.shift(-horizon) - logs
    week_of_year = pd.Series(close.index.isocalendar().week.to_numpy().clip(max=52), index=close.index)
    out = pd.Series(np.nan, index=close.index)
    for position in range(len(close)):
        known = forward.index[: max(position - horizon + 1, 0)]
        if len(known) < 52 * 6:
            continue
        distance = np.abs(week_of_year.loc[known].to_numpy() - week_of_year.iloc[position])
        near = known[np.minimum(distance, 52 - distance) <= 1]
        sample = forward.loc[near].dropna()
        if len(sample) < 12:
            continue
        spread = sample.std(ddof=1)
        if spread > 1e-12:
            out.iloc[position] = squash(sample.mean() / (spread / np.sqrt(len(sample))), 3.0)
    return out


def valuation_signal(close: pd.Series, fair_value: pd.DataFrame | None) -> pd.Series:
    if fair_value is None or fair_value.empty:
        return pd.Series(np.nan, index=close.index)
    return squash(-to_weekly(fair_value["gap_z"], close.index, limit=6))


def mean_reversion_features(close: pd.Series) -> pd.DataFrame:
    logs = np.log(close)
    long_mean = logs.rolling(260, min_periods=156).mean()
    long_std = logs.rolling(260, min_periods=156).std()
    sigma = logs.diff().rolling(104, min_periods=52).std()
    reversal = -(logs - logs.shift(130)) / (sigma * np.sqrt(130))
    return pd.DataFrame(
        {
            "value_5y": squash(-(logs - long_mean) / long_std.where(long_std > 1e-9)),
            "rev_2_5y": squash(reversal),
        }
    )


def macro_signal(macro: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    parts = []
    for column, sign in (("usd", -1.0), ("crude", 1.0)):
        if column not in macro:
            continue
        change = np.log(macro[column]).diff(3)
        deviation = change.expanding(min_periods=36).std().shift(1)
        parts.append(sign * change / deviation.where(deviation > 1e-9))
    if not parts:
        return pd.Series(np.nan, index=index)
    combined = pd.concat(parts, axis=1).mean(axis=1)
    return pd.Series(squash(to_weekly(combined, index, limit=6)), index=index)


def build_features(
    prices: pd.Series,
    macro: pd.DataFrame,
    fair_value: pd.DataFrame | None,
    crop_monthly: pd.DataFrame | None,
    horizon: int = C.HORIZON_WEEKS,
) -> pd.DataFrame:
    close = weekly_close(prices)
    index = close.index
    frame = pd.DataFrame(index=index)
    frame["valuation"] = valuation_signal(close, fair_value)
    frame["trend"] = trend_signal(close)
    frame["seasonality"] = seasonal_signal(close, horizon)
    if crop_monthly is not None and not crop_monthly.empty:
        frame["weather"] = squash(to_weekly(crop_monthly["stress_z"], index, limit=6))
        frame["crop_nowcast"] = squash(-to_weekly(crop_monthly["shock_z"], index, limit=6))
    else:
        frame["weather"] = np.nan
        frame["crop_nowcast"] = np.nan
    frame["macro"] = macro_signal(macro, index)
    frame = pd.concat([frame, mean_reversion_features(close)], axis=1)
    frame["close"] = close
    frame["target"] = np.log(close).shift(-horizon) - np.log(close)
    return frame
