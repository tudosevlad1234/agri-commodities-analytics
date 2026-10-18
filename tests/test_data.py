import numpy as np
import pytest

from agri.data import as_bool, load, save
from agri.synthetic import TRUE, generate


def test_generate_is_deterministic():
    a = generate(seed=3, start="2010-01-01", end="2014-12-31")
    b = generate(seed=3, start="2010-01-01", end="2014-12-31")
    assert np.allclose(a.prices["corn"].to_numpy(), b.prices["corn"].to_numpy())
    assert a.notes["truth"] == TRUE


def test_generate_covers_all_contracts(bundle):
    assert set(bundle.prices) == set(bundle.stocks_to_use)
    assert len(bundle.weather) == len(bundle.regions)
    assert (bundle.prices["corn"] > 0).all()
    assert bundle.prices["corn"].notna().all()


def test_synthetic_inputs_contain_nothing_from_the_future(bundle):
    end = bundle.prices["corn"].index[-1]
    assert bundle.macro.index[-1] <= end
    assert all(series.index[-1] <= end for series in bundle.ndvi.values())
    corn_belt = bundle.yields["US Corn Belt"]
    assert corn_belt.index[-1] == end.year
    brazil = bundle.yields["Brazil Mato Grosso"]
    assert brazil.index[-1] == end.year - 1


def test_save_load_roundtrip(tmp_path):
    original = generate(seed=2, start="2012-01-01", end="2015-12-31")
    save(original, tmp_path)
    restored = load(tmp_path)
    assert set(restored.prices) == set(original.prices)
    assert np.allclose(restored.prices["wheat"].to_numpy(), original.prices["wheat"].to_numpy())
    assert restored.regions == original.regions
    assert np.allclose(restored.stocks_to_use["corn"].to_numpy(), original.stocks_to_use["corn"].to_numpy())


def test_as_bool_accepts_common_spellings():
    assert all(as_bool(v) for v in (True, "True", "yes", "1", 1))
    assert not any(as_bool(v) for v in (False, "False", "no", "0", 0))


def test_load_reports_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="prices.csv"):
        load(tmp_path)
