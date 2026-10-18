from __future__ import annotations

import html
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import config as C
from .pipeline import Analysis, ContractReport

STATE_COLOURS = {"calm": "#2a9d8f", "normal": "#e9c46a", "turbulent": "#e76f51"}
SIGNAL_COLOURS = {True: "#2a9d8f", False: "#e76f51"}
YIELD_COLOURS = {True: "#457b9d", False: "#bc6c25"}
LAYOUT = {"template": "plotly_white", "margin": {"l": 50, "r": 20, "t": 50, "b": 40}, "height": 420}


def price_fair_value_figure(report: ContractReport, years: int = 10) -> go.Figure:
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.06)
    fv = report.fair_value
    if not fv.ok:
        figure.update_layout(title=f"{report.contract.name}: fair value unavailable ({fv.note})", **LAYOUT)
        return figure
    history = fv.history.dropna(subset=["fv"])
    history = history[history.index >= history.index[-1] - pd.DateOffset(years=years)]
    band = fv.sigma
    fair = np.exp(history["fv"])
    figure.add_trace(
        go.Scatter(x=history.index, y=fair * math.exp(band), line={"width": 0}, showlegend=False, hoverinfo="skip"),
        1,
        1,
    )
    figure.add_trace(
        go.Scatter(
            x=history.index,
            y=fair * math.exp(-band),
            fill="tonexty",
            fillcolor="rgba(120,120,120,0.18)",
            line={"width": 0},
            name="±1σ band",
            hoverinfo="skip",
        ),
        1,
        1,
    )
    figure.add_trace(go.Scatter(x=history.index, y=fair, name="Fair value", line={"color": "#264653"}), 1, 1)
    figure.add_trace(
        go.Scatter(x=history.index, y=np.exp(history["y"]), name="Month-end price", line={"color": "#e76f51"}), 1, 1
    )
    figure.add_trace(go.Bar(x=history.index, y=history["gap_z"], name="Gap z-score", marker_color="#8d99ae"), 2, 1)
    figure.add_hline(y=0, row=2, col=1, line_width=1, line_color="#555")
    figure.update_yaxes(title_text=f"{report.contract.unit}, today's money", row=1, col=1)
    figure.update_yaxes(title_text="z", row=2, col=1)
    figure.update_layout(
        title=f"{report.contract.name}: price vs fair value",
        height=520,
        template="plotly_white",
        margin=LAYOUT["margin"],
    )
    return figure


def regime_figure(report: ContractReport, years: int = 8) -> go.Figure:
    figure = go.Figure()
    regime = report.regime
    if not regime.ok:
        figure.update_layout(title=f"{report.contract.name}: regimes unavailable ({regime.note})", **LAYOUT)
        return figure
    probabilities = regime.filtered
    probabilities = probabilities[probabilities.index >= probabilities.index[-1] - pd.DateOffset(years=years)]
    for name in regime.names:
        figure.add_trace(
            go.Scatter(
                x=probabilities.index,
                y=probabilities[name],
                name=name,
                stackgroup="one",
                line={"width": 0.5, "color": STATE_COLOURS.get(name, "#999")},
            )
        )
    figure.update_yaxes(range=[0, 1], title_text="Filtered probability")
    figure.update_layout(title=f"{report.contract.name}: market regime (filtered)", **LAYOUT)
    return figure


def signals_figure(report: ContractReport) -> go.Figure:
    signals = report.score.signals
    names = [n for n in C.FAMILIES if n in signals and math.isfinite(signals[n])]
    values = [signals[n] for n in names]
    weights = [report.score.weights.get(n, 0.0) for n in names]
    figure = go.Figure(
        go.Bar(
            x=values,
            y=[f"{n} (w={w:.2f})" for n, w in zip(names, weights, strict=True)],
            orientation="h",
            marker_color=[SIGNAL_COLOURS[v >= 0] for v in values],
        )
    )
    figure.update_xaxes(range=[-1, 1], title_text="Signal (bearish ← → bullish)")
    figure.update_layout(title=f"{report.contract.name}: signal families", **LAYOUT)
    return figure


def crop_figure(report: ContractReport) -> go.Figure:
    figure = go.Figure()
    crop = report.crop
    if not crop.ok:
        figure.update_layout(title=f"{report.contract.name}: crop-health nowcast unavailable ({crop.note})", **LAYOUT)
        return figure
    table = crop.regions
    figure.add_trace(
        go.Bar(
            x=table.index,
            y=table["predicted_yield_dev_pct"],
            marker_color=[YIELD_COLOURS[v >= 0] for v in table["predicted_yield_dev_pct"].fillna(0.0)],
            customdata=table["share"],
            hovertemplate="%{x}<br>%{y:.1f}% vs trend<br>share %{customdata:.0%}<extra></extra>",
        )
    )
    figure.update_yaxes(title_text="Predicted yield vs trend (%)")
    figure.update_layout(
        title=f"{report.contract.name}: crop-health nowcast by region (above trend = more supply, bearish for price)",
        **LAYOUT,
    )
    return figure


def score_figure(report: ContractReport, years: int = 8) -> go.Figure:
    figure = go.Figure()
    history = report.score.history
    if history is None:
        figure.update_layout(title=f"{report.contract.name}: score unavailable", **LAYOUT)
        return figure
    history = history.dropna()
    history = history[history.index >= history.index[-1] - pd.DateOffset(years=years)]
    figure.add_trace(go.Scatter(x=history.index, y=history, name="Signal composite", line={"color": "#264653"}))
    figure.add_hline(y=0, line_width=1, line_color="#555")
    figure.update_yaxes(range=[-100, 100], title_text="Composite score")
    figure.update_layout(title=f"{report.contract.name}: composite signal history", **LAYOUT)
    return figure


def figures_for(report: ContractReport) -> list[go.Figure]:
    return [
        price_fair_value_figure(report),
        score_figure(report),
        signals_figure(report),
        crop_figure(report),
        regime_figure(report),
    ]


def fmt(value: float, digits: int = 1, suffix: str = "") -> str:
    if not isinstance(value, (int, float, np.floating)) or not math.isfinite(value):
        return "n/a"
    return f"{round(float(value), digits) + 0.0:.{digits}f}{suffix}"


def summary_table_html(analysis: Analysis) -> str:
    rows = []
    for key, report in analysis.reports.items():
        fv, score, regime = report.fair_value, report.score, report.regime
        rows.append(
            "<tr>"
            f"<td><a href='#{key}'>{html.escape(report.contract.name)}</a></td>"
            f"<td>{fmt(fv.price_now, 2)}</td><td>{fmt(fv.fv_now, 2)}</td>"
            f"<td>{fmt(fv.gap_pct, 1, '%')}</td><td>{fmt(fv.gap_z, 2)}</td>"
            f"<td>{html.escape(regime.state if regime.ok else 'n/a')}</td>"
            f"<td>{fmt(score.score, 0)}</td>"
            f"<td>{fmt(score.p_up * 100.0 if math.isfinite(score.p_up) else math.nan, 0, '%')}</td>"
            f"<td>{html.escape(score.verdict)}</td><td>{fmt(score.trust, 0)}</td></tr>"
        )
    head = "".join(
        f"<th>{h}</th>"
        for h in (
            "Contract",
            "Price",
            "Fair value",
            "Gap",
            "Gap z",
            "Regime",
            "Score",
            f"P(up {analysis.horizon_weeks}w)",
            "Verdict",
            "Trust",
        )
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


STYLE = """
body{font-family:system-ui,sans-serif;margin:0 auto;max-width:1100px;padding:24px;color:#222}
table{border-collapse:collapse;width:100%;margin:16px 0}
th,td{border-bottom:1px solid #ddd;padding:6px 10px;text-align:right}
th:first-child,td:first-child{text-align:left}
h2{margin-top:48px;border-top:2px solid #eee;padding-top:16px}
.note{color:#666;font-size:.9em}
"""


def build_html(analysis: Analysis, path: str | Path, synthetic: bool = False, offline: bool = True) -> Path:
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>Agricultural commodities analytics</title><style>{STYLE}</style></head><body>",
        "<h1>Agricultural commodities analytics</h1>",
        f"<p class='note'>As of {analysis.as_of:%Y-%m-%d}. Horizon {analysis.horizon_weeks} weeks.",
    ]
    if synthetic:
        parts.append(" Synthetic data: figures illustrate the method, not real markets.")
    parts.append("</p>")
    parts.append(summary_table_html(analysis))
    plotly_js = True if offline else "cdn"
    first = True
    for key, report in analysis.reports.items():
        parts.append(f"<h2 id='{key}'>{html.escape(report.contract.name)}</h2>")
        if report.notes:
            parts.append(f"<p class='note'>{html.escape('; '.join(report.notes))}</p>")
        for figure in figures_for(report):
            parts.append(figure.to_html(full_html=False, include_plotlyjs=plotly_js if first else False))
            first = False
    parts.append("</body></html>")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(parts), encoding="utf-8")
    return output


def text_summary(analysis: Analysis) -> str:
    digits = {"price": 2, "fair_value": 2, "gap_pct": 1, "gap_z": 2, "supply_shock_pct": 1, "score": 0, "p_up_pct": 0}
    table = analysis.summary().drop(columns=["contract"])
    for column, places in {**digits, "trust": 0}.items():
        table[column] = table[column].round(places) + 0.0
    return f"As of {analysis.as_of:%Y-%m-%d} (horizon {analysis.horizon_weeks}w)\n" + table.to_string()
