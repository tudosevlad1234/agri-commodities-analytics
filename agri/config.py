from __future__ import annotations

from dataclasses import dataclass

HORIZON_WEEKS = 4
WEEK_RULE = "W-FRI"
MONTH_END = "ME"
WALK_FORWARD_MIN_WEEKS = 520
WALK_FORWARD_REFIT_WEEKS = 26
ML_RECENCY_HALF_LIFE_YEARS = 6.0
ML_GBM_WEIGHT = 0.7
FV_WINDOW_MONTHS = 120
FV_MIN_MONTHS = 60
HMM_STATES = 3
HMM_RESTARTS = 4

FAMILIES = ("valuation", "trend", "seasonality", "weather", "crop_nowcast", "macro")
PRIOR_WEIGHTS = {
    "valuation": 1.0,
    "trend": 1.0,
    "seasonality": 0.5,
    "weather": 0.8,
    "crop_nowcast": 1.0,
    "macro": 0.5,
}
ML_FEATURES = list(FAMILIES)
TILT_FEATURES = ["value_5y", "rev_2_5y"]
TRUST_WEIGHTS = {
    "model_skill": 0.30,
    "signal_agreement": 0.20,
    "fair_value_quality": 0.15,
    "regime_certainty": 0.10,
    "data_coverage": 0.15,
    "stability": 0.10,
}
TRUST_CAP_NO_SKILL = 55.0


@dataclass(frozen=True)
class Contract:
    key: str
    name: str
    unit: str
    marketing_year_start_month: int
    su_base: float
    base_price: float = 100.0


CONTRACTS: dict[str, Contract] = {
    "corn": Contract("corn", "Corn", "USc/bu", 9, 0.14, base_price=450.0),
    "soybeans": Contract("soybeans", "Soybeans", "USc/bu", 9, 0.30, base_price=1100.0),
    "wheat": Contract("wheat", "Wheat (SRW)", "USc/bu", 6, 0.35, base_price=560.0),
    "cotton": Contract("cotton", "Cotton", "USc/lb", 8, 0.45, base_price=75.0),
    "coffee": Contract("coffee", "Coffee (arabica)", "USc/lb", 10, 0.25, base_price=130.0),
    "cocoa": Contract("cocoa", "Cocoa", "USD/t", 10, 0.40, base_price=2600.0),
    "sugar": Contract("sugar", "Sugar No.11", "USc/lb", 10, 0.35, base_price=16.0),
}


@dataclass(frozen=True)
class Region:
    name: str
    contract: str
    country: str
    latitude: float
    longitude: float
    share: float
    window: tuple[int, ...]
    frost_sensitive: bool = False

    @property
    def hemisphere(self) -> str:
        return "south" if self.latitude < 0 else "north"


def window(*months: int) -> tuple[int, ...]:
    return tuple(months)


REGIONS: tuple[Region, ...] = (
    Region("US Corn Belt", "corn", "United States", 41.9, -93.6, 0.36, window(6, 7, 8)),
    Region("Brazil Parana", "corn", "Brazil", -24.5, -51.9, 0.18, window(1, 2, 3)),
    Region("Argentina Pampas", "corn", "Argentina", -33.9, -61.5, 0.10, window(1, 2, 3)),
    Region("China Northeast", "corn", "China", 45.0, 125.5, 0.22, window(7, 8)),
    Region("US Midwest Soy", "soybeans", "United States", 40.5, -89.0, 0.33, window(7, 8, 9)),
    Region("Brazil Mato Grosso", "soybeans", "Brazil", -13.0, -56.0, 0.38, window(12, 1, 2)),
    Region("Argentina Soy", "soybeans", "Argentina", -32.5, -61.0, 0.15, window(1, 2, 3)),
    Region("US Plains Wheat", "wheat", "United States", 38.5, -98.0, 0.12, window(4, 5, 6), True),
    Region("Russia South", "wheat", "Russia", 47.0, 40.0, 0.18, window(5, 6, 7), True),
    Region("EU Plain", "wheat", "France", 48.5, 2.5, 0.20, window(5, 6, 7)),
    Region("India Punjab", "wheat", "India", 30.9, 75.8, 0.14, window(2, 3), True),
    Region("US Texas Cotton", "cotton", "United States", 33.5, -101.9, 0.20, window(6, 7, 8)),
    Region("India Gujarat", "cotton", "India", 22.3, 72.0, 0.25, window(7, 8, 9)),
    Region("China Xinjiang", "cotton", "China", 41.2, 85.0, 0.25, window(6, 7, 8)),
    Region("Brazil Minas Gerais", "coffee", "Brazil", -19.0, -45.0, 0.35, window(10, 11, 12, 1), True),
    Region("Vietnam Highlands", "coffee", "Vietnam", 12.7, 108.0, 0.18, window(1, 2, 3)),
    Region("Colombia Andes", "coffee", "Colombia", 5.0, -75.5, 0.08, window(3, 4, 5)),
    Region("Ivory Coast Cocoa", "cocoa", "Ivory Coast", 6.8, -5.3, 0.38, window(5, 6, 7, 8)),
    Region("Ghana Cocoa", "cocoa", "Ghana", 6.5, -1.6, 0.18, window(5, 6, 7, 8)),
    Region("Brazil Centre-South", "sugar", "Brazil", -21.5, -48.0, 0.40, window(11, 12, 1, 2)),
    Region("India Maharashtra", "sugar", "India", 18.5, 74.5, 0.18, window(7, 8, 9)),
    Region("Thailand Central", "sugar", "Thailand", 15.0, 100.5, 0.10, window(6, 7, 8)),
)


def regions_for(contract: str) -> list[Region]:
    return [region for region in REGIONS if region.contract == contract]


def season_of(region: Region, stamp) -> int:
    first = region.window[0]
    last = region.window[-1]
    wraps = last < first
    if wraps and stamp.month <= last:
        return stamp.year - 1
    return stamp.year


def climate_of(region: Region) -> dict:
    lat = abs(region.latitude)
    tropical = lat < 15
    return {
        "mean_temp": 27.0 - 0.45 * lat if lat < 60 else 0.0,
        "amplitude": 2.0 + 0.35 * lat if not tropical else 1.5,
        "rain_mm": 3.0 if tropical else 2.4,
        "peak_day": 200 if region.latitude >= 0 else 20,
    }
