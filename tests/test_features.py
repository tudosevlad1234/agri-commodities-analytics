import numpy as np
import pandas as pd

from agri import features as ft
from agri.config import FAMILIES


def test_signals_are_bounded(corn_report):
    frame = corn_report.features
    for name in list(FAMILIES) + ["value_5y", "rev_2_5y"]:
        assert frame[name].dropna().abs().max() <= 1.0 + 1e-12


def test_every_family_is_populated(corn_report):
    latest = corn_report.features[list(FAMILIES)].iloc[-1]
    assert latest.notna().all()


def test_trend_signal_ignores_the_future(bundle):
    close = ft.weekly_close(bundle.prices["wheat"])
    full = ft.trend_signal(close)
    partial = ft.trend_signal(close.iloc[:-60])
    assert np.allclose(full.iloc[:-60].dropna(), partial.dropna())


def test_seasonal_signal_ignores_the_future(bundle):
    close = ft.weekly_close(bundle.prices["wheat"]).iloc[-700:]
    full = ft.seasonal_signal(close, 4)
    partial = ft.seasonal_signal(close.iloc[:-100], 4)
    assert np.allclose(full.iloc[:-100].dropna(), partial.dropna())


def test_to_weekly_expires_stale_values():
    monthly = pd.Series([1.0], index=pd.DatetimeIndex(["2020-01-31"]))
    weeks = pd.date_range("2020-01-31", periods=20, freq="W-FRI")
    filled = ft.to_weekly(monthly, weeks, limit=6)
    assert filled.iloc[:7].notna().all()
    assert filled.iloc[7:].isna().all()


def test_to_weekly_never_looks_ahead():
    monthly = pd.Series([5.0], index=pd.DatetimeIndex(["2020-02-29"]))
    weeks = pd.date_range("2020-01-03", periods=10, freq="W-FRI")
    filled = ft.to_weekly(monthly, weeks, limit=8)
    assert filled[weeks < "2020-02-29"].isna().all()


def test_target_is_forward_log_return(corn_report):
    frame = corn_report.features
    expected = np.log(frame["close"].iloc[10 + 4] / frame["close"].iloc[10])
    assert abs(frame["target"].iloc[10] - expected) < 1e-12
    assert frame["target"].iloc[-4:].isna().all()
