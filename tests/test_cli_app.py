from pathlib import Path

import pytest

from agri.__main__ import main, parse_contracts


def test_parse_contracts_validates():
    assert parse_contracts("Corn, wheat") == ("corn", "wheat")
    assert parse_contracts(None) is None
    with pytest.raises(SystemExit):
        parse_contracts("gold")


def test_demo_writes_report(tmp_path, capsys):
    code = main(["demo", "--contracts", "coffee", "--out", str(tmp_path / "out")])
    assert code == 0
    html = (tmp_path / "out" / "report.html").read_text()
    assert "Coffee" in html and "Synthetic data" in html
    assert "coffee" in capsys.readouterr().out


def test_template_then_run_on_csv(tmp_path, capsys):
    folder = tmp_path / "inputs"
    assert main(["export-template", str(folder), "--start", "2000-01-01"]) == 0
    assert (folder / "prices.csv").exists()
    assert main(["run", "--data", str(folder), "--contracts", "sugar", "--out", str(tmp_path / "o")]) == 0
    assert "Synthetic data" not in (tmp_path / "o" / "report.html").read_text()


def test_run_with_missing_folder_fails_cleanly(tmp_path, capsys):
    assert main(["run", "--data", str(tmp_path)]) == 2
    assert "missing input files" in capsys.readouterr().err


def test_streamlit_app_renders():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "agri" / "app.py"), default_timeout=180)
    app.run()
    assert not app.exception
    assert any("Score" in m.label for m in app.metric)
