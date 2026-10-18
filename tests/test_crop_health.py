import numpy as np
import pandas as pd

from agri import crop_health as ch
from agri.config import CONTRACTS, REGIONS, Region, regions_for, season_of


def region_named(name: str) -> Region:
    return next(r for r in REGIONS if r.name == name)


def test_season_of_wraps_for_southern_windows():
    brazil = region_named("Brazil Mato Grosso")
    assert season_of(brazil, pd.Timestamp("2021-12-15")) == 2021
    assert season_of(brazil, pd.Timestamp("2022-01-15")) == 2021
    assert season_of(region_named("US Corn Belt"), pd.Timestamp("2021-07-15")) == 2021


def test_past_only_z_ignores_the_future():
    index = pd.date_range("2000-01-31", periods=240, freq="ME")
    series = pd.Series(np.random.default_rng(0).standard_normal(240), index=index)
    changed = series.copy()
    changed.iloc[-1] = 100.0
    a, b = ch.past_only_z(series), ch.past_only_z(changed)
    assert np.allclose(a.iloc[:-1].dropna(), b.iloc[:-1].dropna())


def test_yield_deviation_uses_only_earlier_seasons():
    seasons = pd.Series(
        np.log(3) + 0.01 * np.arange(25) + 0.02 * np.random.default_rng(1).standard_normal(25), index=range(2000, 2025)
    )
    base = ch.yield_deviation(seasons)
    changed = seasons.copy()
    changed.iloc[-1] += 5.0
    again = ch.yield_deviation(changed)
    assert np.allclose(base.iloc[:-1].dropna(), again.iloc[:-1].dropna())
    assert base.iloc[: ch.MIN_TREND_SEASONS].isna().all()


def test_in_progress_features_average_observed_window_months(bundle):
    region = region_named("US Corn Belt")
    anomalies = ch.monthly_anomalies(bundle.weather[region.name], bundle.ndvi[region.name])
    panel = ch.in_progress_features(region, anomalies)
    june = panel[(panel.index.year == 2015) & (panel.index.month == 6)].iloc[0]
    july = panel[(panel.index.year == 2015) & (panel.index.month == 7)].iloc[0]
    assert june["observed"] == 1 and july["observed"] == 2
    expected = anomalies.loc[(anomalies.index.year == 2015) & anomalies.index.month.isin([6, 7]), "heat"].mean()
    assert abs(july["heat"] - expected) < 1e-12
    december = panel[(panel.index.year == 2015) & (panel.index.month == 12)].iloc[0]
    assert december["season"] == 2015


def test_frost_feature_only_for_sensitive_regions(bundle):
    plain = region_named("US Corn Belt")
    sensitive = region_named("Russia South")
    for region, expect_zero in ((plain, True), (sensitive, False)):
        anomalies = ch.monthly_anomalies(bundle.weather[region.name], bundle.ndvi[region.name])
        panel = ch.in_progress_features(region, anomalies).dropna(subset=["season"])
        assert (panel["cold"].abs().max() == 0) == expect_zero


def test_nowcast_has_skill_and_sensible_signs(bundle):
    result = ch.run_crop_health(CONTRACTS["wheat"], bundle.weather, bundle.ndvi, bundle.yields, bundle.regions)
    assert result.ok
    assert result.skill > 0.3
    assert result.coefficients["heat"] <= 0
    assert result.coefficients["moist"] >= 0
    assert abs(result.regions["share"].sum() - sum(r.share for r in regions_for("wheat"))) < 1e-9


def test_price_impact_follows_supply_shock(bundle):
    result = ch.run_crop_health(
        CONTRACTS["corn"], bundle.weather, bundle.ndvi, bundle.yields, bundle.regions, su_elasticity=-0.5, su_now=0.2
    )
    assert abs(result.price_impact_pct - (-0.5 * result.supply_shock_pct / 0.2)) < 1e-9
    assert np.isnan(ch.run_crop_health(CONTRACTS["corn"], bundle.weather, bundle.ndvi, bundle.yields).price_impact_pct)


def test_partial_months_are_dropped():
    days = pd.date_range("2020-01-01", "2020-03-09", freq="D")
    daily = pd.DataFrame({"tmax": 20.0, "tmin": 10.0, "rain": 2.0, "soil": 0.5}, index=days)
    monthly = ch.monthly_weather(daily)
    assert list(monthly.index) == [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29")]
    assert monthly["rain"].iloc[0] == 2.0


def test_yields_are_used_only_after_publication_lag(bundle):
    region = region_named("US Corn Belt")
    panel = ch.in_progress_features(region, ch.monthly_anomalies(bundle.weather[region.name], bundle.ndvi[region.name]))
    rows = ch.complete_season_rows(panel, ch.yield_deviation(bundle.yields[region.name]), region)
    row = rows[rows["season"] == 2020].iloc[0]
    assert row["window_end"] == pd.Timestamp("2020-08-31")
    assert row["known_from"] == pd.Timestamp("2020-11-30")
    model = ch.YieldModel(rows)
    before = model.fit_at(pd.Timestamp("2020-11-29"))
    after = model.fit_at(pd.Timestamp("2020-11-30"))
    assert before is not None and after is not None
    assert len(model.cache) == 2 and before is not after


def test_missing_weather_is_reported(bundle):
    result = ch.run_crop_health(CONTRACTS["corn"], {}, {}, {}, bundle.regions)
    assert not result.ok
