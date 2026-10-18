import numpy as np
import pandas as pd

from agri import fair_value as fv
from agri.config import CONTRACTS


def test_next_year_weight_ramps_before_marketing_year():
    corn = CONTRACTS["corn"]
    assert fv.next_year_weight(corn, pd.Timestamp("2020-04-30")) == 0.0
    assert fv.next_year_weight(corn, pd.Timestamp("2020-08-31")) > fv.next_year_weight(corn, pd.Timestamp("2020-06-30"))
    assert fv.next_year_weight(corn, pd.Timestamp("2020-09-30")) == 0.0


def test_su_monthly_blends_into_next_marketing_year():
    corn = CONTRACTS["corn"]
    su = pd.Series({2019: 0.10, 2020: 0.20, 2021: 0.30})
    values = fv.su_monthly(su, corn, pd.DatetimeIndex(["2020-10-31", "2021-07-31", "2021-08-31"]))
    assert values.iloc[0] == 0.20
    assert abs(values.iloc[1] - (0.4 * 0.20 + 0.6 * 0.30)) < 1e-12
    assert abs(values.iloc[2] - (0.2 * 0.20 + 0.8 * 0.30)) < 1e-12


def test_su_monthly_falls_back_when_next_year_is_unknown():
    corn = CONTRACTS["corn"]
    su = pd.Series({2020: 0.20})
    assert fv.su_monthly(su, corn, pd.DatetimeIndex(["2021-07-31"])).iloc[0] == 0.20


def test_monthly_frame_uses_last_price_of_each_month_and_lags_missing_macro():
    corn = CONTRACTS["corn"]
    days = pd.bdate_range("2020-01-01", "2020-03-11")
    prices = pd.Series(np.arange(len(days), dtype=float) + 100.0, index=days)
    macro = pd.DataFrame(
        {"usd": [90.0, 91.0], "crude": [60.0, 61.0], "cpi": [100.0, 101.0]},
        index=pd.DatetimeIndex(["2020-01-31", "2020-02-29"]),
    )
    frame = fv.monthly_frame(corn, prices, macro, pd.Series({2019: 0.15}))
    assert list(frame.index) == [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-28"), pd.Timestamp("2020-03-11")]
    assert frame["price"].iloc[-1] == prices.iloc[-1]
    assert frame["usd"].iloc[-1] == np.log(91.0)
    assert frame["cpi_ratio"].iloc[-1] == 1.0
    assert abs(frame["cpi_ratio"].iloc[0] - 100.0 / 101.0) < 1e-12


def test_combine_uses_only_past_errors():
    index = pd.date_range("2000-01-31", periods=40, freq="ME")
    predictions = pd.DataFrame({"a": np.zeros(40), "b": np.ones(40)}, index=index)
    errors = pd.DataFrame({"a": np.full(40, 0.1), "b": np.full(40, 1.0)}, index=index)
    weights = fv.rolling_weights(errors)
    assert weights.iloc[: fv.WEIGHT_MIN_OBS].isna().all().all()
    combined = fv.combine(predictions, weights)
    assert combined.iloc[0] == 0.5
    assert combined.iloc[-1] < 0.05
    changed = errors.copy()
    changed.iloc[-1] = 50.0
    assert np.allclose(fv.rolling_weights(changed).iloc[:-1].dropna(), weights.iloc[:-1].dropna())
    assert np.allclose(fv.rolling_weights(changed).iloc[-1], weights.iloc[-1])


def test_constrained_ridge_respects_signs():
    rng = np.random.default_rng(1)
    x = pd.DataFrame({"su1": rng.standard_normal(200), "crude": rng.standard_normal(200)})
    y = 0.8 * x["su1"] - 0.5 * x["crude"] + 0.1 * rng.standard_normal(200)
    model = fv.constrained_ridge(x, y, alpha=0.1)
    assert model["coef"][0] <= 1e-9
    assert model["coef"][1] <= 1e-9 or model["coef"][1] >= 0


def test_constrained_ridge_recovers_free_coefficients():
    rng = np.random.default_rng(2)
    x = pd.DataFrame({"a": rng.standard_normal(300), "b": rng.standard_normal(300)})
    y = 2.0 * x["a"] - 1.0 * x["b"] + 0.05 * rng.standard_normal(300)
    model = fv.constrained_ridge(x, y, alpha=1e-3, signs={})
    assert np.allclose(model["coef"] / model["std"].to_numpy(), [2.0, -1.0], atol=0.05)


def test_kalman_predictions_are_one_step_ahead():
    rng = np.random.default_rng(3)
    x = pd.DataFrame({"a": rng.standard_normal(150), "b": rng.standard_normal(150)})
    y = pd.Series(x["a"] + 0.1 * rng.standard_normal(150))
    base = fv.kalman_one_step(x, y)
    altered = y.copy()
    altered.iloc[-1] += 100.0
    again = fv.kalman_one_step(x, altered)
    assert np.allclose(base.iloc[:-1].dropna(), again.iloc[:-1].dropna())


def test_inverse_mse_weights_favour_the_better_model():
    good = pd.Series(np.full(60, 0.1))
    bad = pd.Series(np.full(60, 0.5))
    weights = fv.inverse_mse_weights({"good": good, "bad": bad})
    assert abs(sum(weights.values()) - 1.0) < 1e-12
    assert weights["good"] > weights["bad"]


def test_half_life():
    assert abs(fv.half_life_months(0.5) - 1.0) < 1e-12
    assert fv.half_life_months(1.2) == float("inf")


def test_fair_value_on_synthetic_corn(bundle):
    result = fv.run_fair_value(CONTRACTS["corn"], bundle.prices["corn"], bundle.macro, bundle.stocks_to_use["corn"])
    assert result.ok
    assert result.su_elasticity < 0
    assert abs(sum(result.weights.values()) - 1) < 1e-9
    assert result.r2_out_of_sample > 0.3
    assert result.history["gap_z"].dropna().abs().max() < 10
    assert result.history["fv"].notna().sum() > 200
    assert 0 < result.su_now < 1


def test_fair_value_needs_history(bundle):
    short = bundle.prices["corn"].iloc[:400]
    result = fv.run_fair_value(CONTRACTS["corn"], short, bundle.macro, bundle.stocks_to_use["corn"])
    assert not result.ok


def test_fair_value_history_does_not_use_the_future(bundle):
    corn = CONTRACTS["corn"]
    prices = bundle.prices["corn"]
    full = fv.run_fair_value(corn, prices, bundle.macro, bundle.stocks_to_use["corn"]).history
    cut = prices[prices.index < "2020-01-01"]
    early = fv.run_fair_value(corn, cut, bundle.macro, bundle.stocks_to_use["corn"]).history
    shared = early.index[early.index < "2019-12-01"]
    assert np.allclose(full.loc[shared, "resid"], early.loc[shared, "resid"], equal_nan=True)
    assert np.allclose(full.loc[shared, "gap_z"], early.loc[shared, "gap_z"], equal_nan=True)


def test_missing_macro_is_reported(bundle):
    result = fv.run_fair_value(
        CONTRACTS["corn"], bundle.prices["corn"], bundle.macro.drop(columns="crude"), bundle.stocks_to_use["corn"]
    )
    assert not result.ok and "crude" in result.note


def test_walk_forward_does_not_use_the_future(bundle):
    corn = CONTRACTS["corn"]
    frame = fv.monthly_frame(corn, bundle.prices["corn"], bundle.macro, bundle.stocks_to_use["corn"]).dropna(
        subset=["y", "su1", "inv_su1", "usd", "crude"]
    )
    full = fv.walk_forward(frame, ["su1", "usd", "crude"])
    cut = frame.iloc[:-30]
    partial = fv.walk_forward(cut, ["su1", "usd", "crude"])
    common = partial.dropna().index[partial.dropna().index.year < cut.index[-1].year]
    assert np.allclose(full.loc[common], partial.loc[common])
