from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .config import REGIONS, Region


@dataclass
class MarketBundle:
    prices: dict[str, pd.Series]
    stocks_to_use: dict[str, pd.Series]
    macro: pd.DataFrame
    weather: dict[str, pd.DataFrame]
    ndvi: dict[str, pd.Series]
    yields: dict[str, pd.Series]
    regions: tuple[Region, ...] = REGIONS
    notes: dict = field(default_factory=dict)


def save(bundle: MarketBundle, directory: str | Path) -> None:
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    prices = pd.concat({k: v for k, v in bundle.prices.items()}, names=["contract", "date"]).rename("close")
    prices.reset_index().to_csv(folder / "prices.csv", index=False)
    rows = [
        {"contract": contract, "season": int(season), "value": float(value)}
        for contract, series in bundle.stocks_to_use.items()
        for season, value in series.items()
    ]
    pd.DataFrame(rows).to_csv(folder / "stocks_to_use.csv", index=False)
    bundle.macro.rename_axis("date").to_csv(folder / "macro.csv")
    weather = pd.concat(bundle.weather, names=["region", "date"]).reset_index()
    weather.to_csv(folder / "weather.csv", index=False)
    ndvi = pd.concat({k: v.rename("ndvi") for k, v in bundle.ndvi.items()}, names=["region", "date"])
    ndvi.reset_index().to_csv(folder / "ndvi.csv", index=False)
    yields = pd.concat({k: v.rename("log_yield") for k, v in bundle.yields.items()}, names=["region", "season"])
    yields.reset_index().to_csv(folder / "yields.csv", index=False)
    table = pd.DataFrame(
        [
            {
                "region": r.name,
                "contract": r.contract,
                "country": r.country,
                "latitude": r.latitude,
                "longitude": r.longitude,
                "share": r.share,
                "window": " ".join(map(str, r.window)),
                "frost_sensitive": r.frost_sensitive,
            }
            for r in bundle.regions
        ]
    )
    table.to_csv(folder / "regions.csv", index=False)


TRUE_WORDS = {"true", "1", "yes", "y"}


def as_bool(value) -> bool:
    return str(value).strip().lower() in TRUE_WORDS


REQUIRED = ("prices.csv", "stocks_to_use.csv", "macro.csv", "weather.csv", "ndvi.csv", "yields.csv", "regions.csv")


def load(directory: str | Path) -> MarketBundle:
    folder = Path(directory)
    missing = [name for name in REQUIRED if not (folder / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing input files in {folder}: {', '.join(missing)}")
    prices = pd.read_csv(folder / "prices.csv", parse_dates=["date"])
    stocks = pd.read_csv(folder / "stocks_to_use.csv")
    weather = pd.read_csv(folder / "weather.csv", parse_dates=["date"])
    ndvi = pd.read_csv(folder / "ndvi.csv", parse_dates=["date"])
    yields = pd.read_csv(folder / "yields.csv")
    regions = pd.read_csv(folder / "regions.csv")
    macro = pd.read_csv(folder / "macro.csv", index_col="date", parse_dates=True)
    return MarketBundle(
        prices={k: g.set_index("date")["close"].sort_index() for k, g in prices.groupby("contract")},
        stocks_to_use={k: g.set_index("season")["value"].sort_index() for k, g in stocks.groupby("contract")},
        macro=macro,
        weather={k: g.drop(columns="region").set_index("date").sort_index() for k, g in weather.groupby("region")},
        ndvi={k: g.set_index("date")["ndvi"].sort_index() for k, g in ndvi.groupby("region")},
        yields={k: g.set_index("season")["log_yield"].sort_index() for k, g in yields.groupby("region")},
        regions=tuple(
            Region(
                row.region,
                row.contract,
                row.country,
                float(row.latitude),
                float(row.longitude),
                float(row.share),
                tuple(int(m) for m in str(row.window).split()),
                as_bool(row.frost_sensitive),
            )
            for row in regions.itertuples()
        ),
    )
