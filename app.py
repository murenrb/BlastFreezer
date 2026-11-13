"""Streamlit front end for the blast freezer cell model.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import json

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from blastfreezer import Cell, Pallet, Product, Scenario, SimSettings, simulate, summarize

st.set_page_config(page_title="Blast Freezer Cell Model", layout="wide")


# ---------------------------------------------------------------- inputs ----
def sidebar_inputs() -> Scenario:
    d = Scenario()
    sb = st.sidebar
    sb.header("Inputs")

    with sb.expander("Product", expanded=False):
        prod = Product(
            name=st.text_input("Name", d.product.name),
            water_fraction=st.number_input("Water fraction [-]", 0.0, 1.0, d.product.water_fraction, 0.01),
            cp_unfrozen=st.number_input("cp unfrozen [J/kg-K]", 500.0, 5000.0, d.product.cp_unfrozen, 50.0),
            cp_frozen=st.number_input("cp frozen [J/kg-K]", 500.0, 5000.0, d.product.cp_frozen, 50.0),
            T_initial_freeze=st.number_input("Initial freeze point [°C]", -10.0, 0.0, d.product.T_initial_freeze, 0.1),
        )

    with sb.expander("Pallet", expanded=False):
        pal = Pallet(
            mass=st.number_input("Product mass per pallet [kg]", 50.0, 3000.0, d.pallet.mass, 10.0),
            height=st.number_input("Height [m]", 0.1, 3.0, d.pallet.height, 0.05),
            exposed_area=st.number_input("Exposed area [m²]", 0.1, 50.0, d.pallet.exposed_area, 0.1),
            critical_length=st.number_input("Critical length [m]", 0.05, 3.0, d.pallet.critical_length, 0.05),
            k_effective=st.number_input("Effective k [W/m-K]", 0.01, 5.0, d.pallet.k_effective, 0.01),
        )

    with sb.expander("Cell", expanded=True):
        cell = Cell(
            n_pallets=int(st.number_input("Pallets in cell", 1, 500, d.cell.n_pallets, 1)),
            T_air=st.number_input("Supply air temp [°C]", -50.0, 0.0, d.cell.T_air, 0.5),
            htc=st.number_input("Air-side HTC [W/m²-K]", 1.0, 200.0, d.cell.htc, 1.0),
            refrigeration_capacity=1000.0 * st.number_input(
                "Refrigeration capacity [kW] (0 = unlimited)", 0.0, 5000.0,
                d.cell.refrigeration_capacity / 1000.0, 5.0),
        )

    with sb.expander("Simulation", expanded=True):
        sim = SimSettings(
            T_start=st.number_input("Load-in temp [°C]", -10.0, 40.0, d.sim.T_start, 0.5),
            T_target=st.number_input("Target temp [°C]", -40.0, 0.0, d.sim.T_target, 0.5),
            dt=st.number_input("Time step [s]", 1.0, 600.0, d.sim.dt, 1.0),
            t_max_hours=st.number_input("Max sim time [h]", 1.0, 240.0, d.sim.t_max_hours, 1.0),
        )

    return Scenario(product=prod, pallet=pal, cell=cell, sim=sim)


@st.cache_data(show_spinner="Running simulation…")
def run(scenario_dict: dict):
    s = Scenario.from_dict(scenario_dict)
    df = simulate(s)
    return df, summarize(df, s)


# ---------------------------------------------------------------- layout ----
st.title("Blast Freezer Cell Model")
st.caption("Placeholder physics: lumped-pallet model ported from legacy `pallet.py`.")

scenario = sidebar_inputs()
df, summary = run(scenario.to_dict())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Blast time", f"{summary['blast_time_h']:.1f} h" if summary["reached_target"] else "Not reached")
c2.metric("Peak load", f"{summary['peak_load_kW']:.0f} kW")
c3.metric("Average load", f"{summary['avg_load_kW']:.0f} kW")
c4.metric("Energy removed", f"{summary['energy_removed_kWh']:,.0f} kWh")

tab_plots, tab_data, tab_inputs = st.tabs(["Plots", "Data", "Scenario"])

with tab_plots:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Product temperature", "Cell heat removal"))
    fig.add_trace(go.Scatter(x=df.time_h, y=df.T_product, name="Product temp"), 1, 1)
    fig.add_trace(go.Scatter(x=df.time_h, y=df.Q_cell_kW, name="Heat removed"), 1, 2)
    fig.add_trace(go.Scatter(x=df.time_h, y=df.Q_demand_kW, name="Unconstrained demand",
                             line=dict(dash="dot")), 1, 2)
    if scenario.cell.refrigeration_capacity > 0:
        fig.add_hline(y=scenario.cell.refrigeration_capacity / 1000.0, line_dash="dash",
                      annotation_text="Capacity limit", row=1, col=2)
    fig.update_xaxes(title_text="Blasting time [h]")
    fig.update_yaxes(title_text="°C", row=1, col=1)
    fig.update_yaxes(title_text="kW", row=1, col=2)
    fig.update_layout(height=450, legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)

with tab_data:
    st.dataframe(df, use_container_width=True, height=400)
    st.download_button("Download CSV", df.to_csv(index=False), "blast_results.csv", "text/csv")

with tab_inputs:
    st.json(scenario.to_dict())
    st.download_button("Download scenario JSON", json.dumps(scenario.to_dict(), indent=2),
                       "scenario.json", "application/json")
