from __future__ import annotations

import math

import streamlit as st

from agri import config as C
from agri.data import load
from agri.pipeline import analyse_contract
from agri.report import figures_for, fmt
from agri.synthetic import generate

st.set_page_config(page_title="Agricultural commodities analytics", layout="wide")


@st.cache_resource(show_spinner=False)
def synthetic_bundle(seed: int):
    return generate(seed=seed)


@st.cache_resource(show_spinner=False)
def folder_bundle(folder: str):
    return load(folder)


@st.cache_resource(show_spinner="Running the models…")
def report_for(source: str, key: str, horizon: int, seed: int, folder: str):
    bundle = synthetic_bundle(seed) if source == "Synthetic demo" else folder_bundle(folder)
    return analyse_contract(bundle, key, horizon)


def main() -> None:
    st.title("Agricultural commodities analytics")
    with st.sidebar:
        source = st.radio("Data", ["Synthetic demo", "CSV folder"])
        seed, folder = 7, ""
        if source == "Synthetic demo":
            seed = int(st.number_input("Seed", value=7, step=1))
        else:
            folder = st.text_input("Folder with the input CSV files")
        key = st.selectbox("Contract", list(C.CONTRACTS), format_func=lambda k: C.CONTRACTS[k].name)
        horizon = int(st.slider("Horizon (weeks)", 1, 12, C.HORIZON_WEEKS))
    if source == "CSV folder" and not folder:
        st.info(
            "Enter the folder that holds prices.csv, stocks_to_use.csv, macro.csv, weather.csv, "
            "ndvi.csv, yields.csv and regions.csv."
        )
        return
    try:
        report = report_for(source, key, horizon, seed, folder)
    except (FileNotFoundError, KeyError) as error:
        st.error(f"Could not run the analysis: {error}")
        return
    if source == "Synthetic demo":
        st.caption("Synthetic data: the numbers illustrate the method, not real markets.")
    fv, score = report.fair_value, report.score
    columns = st.columns(5)
    columns[0].metric("Price", fmt(fv.price_now, 2))
    columns[1].metric("Fair value", fmt(fv.fv_now, 2), fmt(fv.gap_pct, 1, "% gap"), delta_color="off")
    columns[2].metric("Score", fmt(score.score, 0), score.verdict, delta_color="off")
    p_up = score.p_up * 100.0 if math.isfinite(score.p_up) else math.nan
    columns[3].metric(f"P(up {horizon}w)", fmt(p_up, 0, "%"))
    columns[4].metric("Trust", fmt(score.trust, 0))
    if report.notes:
        st.warning("; ".join(report.notes))
    for figure in figures_for(report):
        st.plotly_chart(figure)
    with st.expander("Trust breakdown"):
        st.table({name: [f"{value:.2f}"] for name, value in score.trust_parts.items()})


main()
