from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CONTRACTS, REGIONS, Contract, Region, climate_of, regions_for, season_of
from .data import MarketBundle
from .fair_value import marketing_year, su_monthly

TRUE = {
    "moist": 0.05,
    "heat": -0.035,
    "frost": -0.03,
    "yield_noise": 0.02,
    "su_persistence": 0.5,
    "su_elasticity": -0.5,
    "usd": -0.3,
    "crude": 0.1,
}


def ar1(rng: np.random.Generator, n: int, phi: float, sigma: float) -> np.ndarray:
    out = np.zeros(n)
    shocks = rng.standard_normal(n) * sigma
    for t in range(1, n):
        out[t] = phi * out[t - 1] + shocks[t]
    return out


def season_anomalies(rng: np.random.Generator, years: range) -> dict[str, dict[str, pd.DataFrame]]:
    anomalies: dict[str, dict[str, pd.DataFrame]] = {}
    for contract in CONTRACTS:
        regions = regions_for(contract)
        common = rng.standard_normal((len(years), 3))
        table = {}
        for region in regions:
            own = rng.standard_normal((len(years), 3))
            values = 0.35 * common + np.sqrt(1 - 0.35**2) * own
            table[region.name] = pd.DataFrame(values, index=list(years), columns=["moist", "heat", "frost"])
        anomalies[contract] = table
    return anomalies


def make_weather(rng: np.random.Generator, region: Region, days: pd.DatetimeIndex, anomaly: pd.DataFrame):
    climate = climate_of(region)
    n = len(days)
    doy = days.dayofyear.to_numpy()
    base = climate["mean_temp"] + climate["amplitude"] * np.cos(2 * np.pi * (doy - climate["peak_day"]) / 365.0)
    seasons = np.array([season_of(region, d) for d in days])
    in_window = np.isin(days.month, region.window)
    moist = anomaly["moist"].reindex(seasons).to_numpy()
    heat = anomaly["heat"].reindex(seasons).to_numpy()
    frost = anomaly["frost"].reindex(seasons).to_numpy()
    noise = ar1(rng, n, 0.7, 2.0)
    tmax = base + 5.0 + 1.6 * heat * in_window + noise
    tmin = tmax - 10.0 - 3.0 * frost * in_window + 0.5 * rng.standard_normal(n)
    wet_probability = np.clip(0.35 + 0.1 * np.cos(2 * np.pi * (doy - climate["peak_day"]) / 365.0), 0.1, 0.8)
    wet = rng.random(n) < wet_probability
    amount = rng.exponential(climate["rain_mm"] / wet_probability) * np.exp(0.30 * moist * in_window)
    rain = np.where(wet, amount, 0.0)
    soil = np.zeros(n)
    level = 0.5
    for t in range(n):
        evaporation = 0.012 * max(tmax[t], 0.0) / 25.0
        level = min(1.0, max(0.0, 0.97 * level + rain[t] / 120.0 - evaporation))
        soil[t] = level
    frame = pd.DataFrame({"tmax": tmax, "tmin": tmin, "rain": rain, "soil": soil}, index=days)
    return frame


def make_ndvi(rng: np.random.Generator, region: Region, days: pd.DatetimeIndex, anomaly: pd.DataFrame) -> pd.Series:
    months = pd.date_range(days[0], days[-1], freq="ME")
    seasons = np.array([season_of(region, m) for m in months])
    in_window = np.isin(months.month, region.window)
    phase = np.cos(2 * np.pi * (months.month.to_numpy() - region.window[len(region.window) // 2]) / 12.0)
    moist = anomaly["moist"].reindex(seasons).to_numpy()
    heat = anomaly["heat"].reindex(seasons).to_numpy()
    value = 0.55 + 0.15 * phase + in_window * (0.05 * moist - 0.02 * heat) + 0.015 * rng.standard_normal(len(months))
    return pd.Series(np.clip(value, 0.05, 0.95), index=months, name=region.name)


def make_yields(rng: np.random.Generator, region: Region, anomaly: pd.DataFrame) -> pd.Series:
    years = anomaly.index.to_numpy()
    deviation = (
        TRUE["moist"] * anomaly["moist"]
        + TRUE["heat"] * anomaly["heat"]
        + (TRUE["frost"] * anomaly["frost"] if region.frost_sensitive else 0.0)
        + TRUE["yield_noise"] * rng.standard_normal(len(years))
    )
    trend = np.log(3.0) + 0.012 * (years - years[0])
    return pd.Series(trend + deviation.to_numpy(), index=years, name=region.name)


def world_shock(contract: str, deviations: dict[str, pd.Series]) -> pd.Series:
    regions = regions_for(contract)
    total = sum(region.share for region in regions)
    parts = [region.share / total * deviations[region.name] for region in regions]
    return sum(parts[1:], parts[0])


def stocks_to_use_path(rng: np.random.Generator, contract: Contract, shock: pd.Series) -> pd.Series:
    base = np.log(contract.su_base)
    values, previous = [], 0.0
    for year in shock.index:
        deviation = shock[year] / contract.su_base + TRUE["su_persistence"] * previous + 0.04 * rng.standard_normal()
        values.append(base + deviation)
        previous = deviation
    return pd.Series(np.exp(values), index=shock.index, name=contract.key)


def monthly_ar1(rng: np.random.Generator, n: int, phi_month: float, sigma_month: float) -> np.ndarray:
    phi = phi_month ** (1.0 / 21.0)
    sigma = sigma_month * np.sqrt((1.0 - phi**2) / (1.0 - phi_month**2))
    return ar1(rng, n, phi, sigma)


def make_macro(rng: np.random.Generator, business: pd.DatetimeIndex) -> pd.DataFrame:
    n = len(business)
    months_elapsed = (business - business[0]).days.to_numpy() / 30.4375
    return pd.DataFrame(
        {
            "usd": np.exp(4.6 + monthly_ar1(rng, n, 0.97, 0.018)),
            "crude": np.exp(np.log(70.0) + monthly_ar1(rng, n, 0.98, 0.06)),
            "cpi": 100.0 * np.exp(0.0017 * months_elapsed),
        },
        index=business,
    )


def month_end_macro(daily: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    monthly = daily.groupby(daily.index.to_period("M")).last()
    monthly.index = monthly.index.to_timestamp(how="end").normalize()
    return monthly[monthly.index <= end]


def make_prices(rng, contract: Contract, su: pd.Series, macro: pd.DataFrame) -> pd.Series:
    business = macro.index
    periods = business.to_period("M")
    months = periods.unique()
    su_by_month = pd.Series(su_monthly(su, contract, months.to_timestamp()).to_numpy(), index=months)
    su_daily = su_by_month.reindex(periods).to_numpy()
    fair_real = (
        np.log(contract.base_price)
        + TRUE["su_elasticity"] * (np.log(su_daily) - np.log(contract.su_base))
        + TRUE["usd"] * (np.log(macro["usd"].to_numpy()) - 4.6)
        + TRUE["crude"] * (np.log(macro["crude"].to_numpy()) - np.log(70.0))
    )
    fair_nominal = fair_real + np.log(macro["cpi"].to_numpy() / macro["cpi"].iloc[-1])
    deviation = ar1(rng, len(business), 0.992, 0.009)
    noise = 0.006 * rng.standard_normal(len(business))
    return pd.Series(np.exp(fair_nominal + deviation + noise), index=business, name=contract.key)


def window_end(region: Region, season: int) -> pd.Timestamp:
    last = region.window[-1]
    year = season + 1 if last < region.window[0] else season
    return pd.Timestamp(year, last, 1) + pd.offsets.MonthEnd(0)


def generate(seed: int = 7, start: str = "2000-01-01", end: str = "2026-10-09") -> MarketBundle:
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, end, freq="D")
    final = days[-1]
    years = range(days[0].year - 1, final.year + 2)
    anomalies = season_anomalies(rng, years)
    weather, ndvi, yields = {}, {}, {}
    for region in REGIONS:
        anomaly = anomalies[region.contract][region.name]
        weather[region.name] = make_weather(rng, region, days, anomaly)
        ndvi[region.name] = make_ndvi(rng, region, days, anomaly)
        yields[region.name] = make_yields(rng, region, anomaly)
    business = days[days.dayofweek < 5]
    daily_macro = make_macro(rng, business)
    prices, stocks = {}, {}
    for key, contract in CONTRACTS.items():
        deviations = {
            r.name: yields[r.name] - np.log(3.0) - 0.012 * (yields[r.name].index - yields[r.name].index[0])
            for r in regions_for(key)
        }
        shock = world_shock(key, deviations)
        path = stocks_to_use_path(rng, contract, shock)
        prices[key] = make_prices(rng, contract, path, daily_macro)
        stocks[key] = path[path.index <= marketing_year(contract, final) + 1]
    published = {
        region.name: yields[region.name][[window_end(region, s) <= final for s in yields[region.name].index]]
        for region in REGIONS
    }
    return MarketBundle(
        prices=prices,
        stocks_to_use=stocks,
        macro=month_end_macro(daily_macro, final),
        weather=weather,
        ndvi=ndvi,
        yields=published,
        notes={"truth": TRUE},
    )
