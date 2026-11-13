"""Streamlit front end for the blast freezer model.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import json

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from blastfreezer import (PACKAGING_PRESETS, Air, Cell, Packaging, PalletLoad, Product, Scenario,
                          SimSettings, simulate, summarize)

st.set_page_config(page_title="Blast Freezer Model", layout="wide")
D = Scenario()
M3S_TO_CFM = 2118.88


# ---------------------------------------------------------------- inputs ----
def sidebar_inputs() -> Scenario:
    sb = st.sidebar
    sb.header("Inputs")

    with sb.expander("Product", expanded=False):
        p = D.product
        product = Product(
            name=st.text_input("Name", p.name),
            water_fraction=st.number_input("Freezable water fraction [-]", 0.0, 1.0, p.water_fraction, 0.01),
            cp_unfrozen=st.number_input("cp unfrozen [J/kg·K]", 100.0, 5000.0, p.cp_unfrozen, 10.0),
            cp_frozen=st.number_input("cp frozen [J/kg·K]", 100.0, 5000.0, p.cp_frozen, 10.0),
            k_unfrozen=st.number_input("k unfrozen [W/m·K]", 0.01, 5.0, p.k_unfrozen, 0.01),
            k_frozen=st.number_input("k frozen [W/m·K]", 0.01, 5.0, p.k_frozen, 0.01),
            density=st.number_input("Product density, solid [kg/m³]", 100.0, 2000.0, p.density, 10.0),
            T_freeze=st.number_input("Initial freezing point [°C]", -15.0, 0.0, p.T_freeze, 0.1),
            latent_heat_water=p.latent_heat_water,
        )

    with sb.expander("Packaging", expanded=False):
        kind = st.selectbox("Type", list(PACKAGING_PRESETS), index=0)
        pre = {**PACKAGING_PRESETS[kind]}
        packaging = Packaging(
            kind=kind,
            package_height=st.number_input("Package height (heat-flow direction) [m]", 0.01, 2.0,
                                           D.packaging.package_height, 0.01),
            wall_thickness=st.number_input("Wall thickness [mm]", 0.0, 20.0, pre["wall_thickness"] * 1000,
                                           0.1, key=f"wt_{kind}") / 1000,
            k_wall=st.number_input("Wall k [W/m·K]", 0.001, 1.0, pre["k_wall"], 0.001, format="%.3f",
                                   key=f"kw_{kind}"),
            wall_density=st.number_input("Wall density [kg/m³]", 0.0, 2000.0, pre["wall_density"], 10.0,
                                         key=f"rw_{kind}"),
            cp_wall=pre["cp_wall"],
            air_gap=st.number_input("Headspace air gap per package [mm]", 0.0, 50.0, pre["air_gap"] * 1000,
                                    0.5, key=f"ag_{kind}") / 1000,
            k_air=D.packaging.k_air,
        )

    with sb.expander("Pallet", expanded=True):
        l = D.load
        use_spacers = st.checkbox("Blast spacers between layers", l.use_spacers)
        load = PalletLoad(
            mass=st.number_input("Product mass [kg]", 10.0, 5000.0, round(l.mass, 1), 10.0),
            length=st.number_input("Depth [m]", 0.1, 3.0, l.length, 0.01),
            width=st.number_input("Width [m]", 0.1, 3.0, l.width, 0.01),
            height=st.number_input("Stack height incl. spacers [m]", 0.1, 4.0, l.height, 0.01),
            use_spacers=use_spacers,
            n_spacers=int(st.number_input("Number of spacer layers", 1, 30, l.n_spacers, 1,
                                          disabled=not use_spacers)),
            spacer_thickness=st.number_input("Spacer thickness [mm]", 1.0, 200.0, l.spacer_thickness * 1000,
                                             1.0, disabled=not use_spacers) / 1000,
            geometry=st.selectbox("1-D geometry", ["auto", "plate", "sphere", "cylinder"],
                                  help="auto: plate if the block's smallest dimension is < 0.5× the "
                                       "middle one, otherwise an equivalent sphere (R = 3V/A)."),
        )

    with sb.expander("Air", expanded=True):
        a = D.air
        flow = st.number_input("Airflow per pallet [m³/s]", 0.01, 20.0, a.flow_per_pallet, 0.05)
        st.caption(f"≈ {flow * M3S_TO_CFM:,.0f} CFM")
        air = Air(
            T_in=st.number_input("Air temperature entering pallet [°C]", -60.0, 0.0, a.T_in, 0.5),
            flow_per_pallet=flow,
            htc=st.number_input("Surface HTC [W/m²·K]", 1.0, 200.0, a.htc, 1.0),
            cp=a.cp,
        )
        cell = Cell(n_pallets=int(st.number_input("Pallets in cell", 1, 500, D.cell.n_pallets, 1)))

    with sb.expander("Simulation", expanded=False):
        s = D.sim
        sim = SimSettings(
            T_start=st.number_input("Load-in temperature [°C]", -10.0, 40.0, s.T_start, 0.5),
            T_target=st.number_input("Target core temperature [°C]", -50.0, 0.0, s.T_target, 0.5),
            n_nodes=int(st.number_input("Nodes (core → surface)", 2, 200, s.n_nodes, 1)),
            dt_max=st.number_input("Max time step [s]", 1.0, 600.0, s.dt_max, 1.0,
                                   help="The solver uses the smaller of this and the explicit stability limit."),
            t_max_hours=st.number_input("Max simulated time [h]", 1.0, 500.0, s.t_max_hours, 1.0),
            record_every_s=s.record_every_s,
        )

    return Scenario(product, packaging, load, air, cell, sim)


@st.cache_data(show_spinner="Running simulation…")
def run(d: dict):
    s = Scenario.from_dict(d)
    r = simulate(s)
    return r, summarize(r, s)


# ---------------------------------------------------------------- layout ----
st.title("Blast Freezer Model: Single Pallet")
scenario = sidebar_inputs()
try:
    res, summ = run(scenario.to_dict())
except ValueError as e:
    st.error(str(e))
    st.stop()

h, info = res.history, res.info
c = st.columns(5)
c[0].metric("Freeze time", f"{summ['freeze_time_h']:.1f} h" if summ["reached_target"] else "Not reached")
c[1].metric("Peak load / pallet", f"{summ['peak_pallet_kW']:.1f} kW")
c[2].metric(f"Peak load / cell ({scenario.cell.n_pallets})", f"{summ['peak_cell_kW']:.0f} kW")
c[3].metric("Energy removed / pallet", f"{summ['energy_removed_kWh']:.1f} kWh")
c[4].metric("Effective k (unfrozen / frozen)", f"{info['k_eff_unfrozen']:.2f} / {info['k_eff_frozen']:.2f}")
st.caption(info["geometry_note"])

tab_T, tab_prof, tab_model, tab_data = st.tabs(["Temperatures & load", "Profiles", "Model details", "Data"])

with tab_T:
    col1, col2 = st.columns(2)
    fig = go.Figure()
    for name, col, dash in [("Core", "T_core", None), ("Mean", "T_mean", None),
                            ("Surface", "T_surface", None), ("Air in", "T_air_in", "dot"),
                            ("Air out", "T_air_out", "dot")]:
        fig.add_trace(go.Scatter(x=h.time_h, y=h[col], name=name, line=dict(dash=dash)))
    fig.add_hline(y=scenario.sim.T_target, line_dash="dash", annotation_text="Target")
    fig.update_layout(title="Temperatures", xaxis_title="Blasting time [h]", yaxis_title="°C",
                      height=420, legend=dict(orientation="h", y=-0.25))
    col1.plotly_chart(fig, width="stretch")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h.time_h, y=h.Q_pallet_kW, name="Per pallet"))
    fig.update_layout(title="Heat removal per pallet", xaxis_title="Blasting time [h]",
                      yaxis_title="kW", height=420)
    col2.plotly_chart(fig, width="stretch")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h.time_h, y=h.front_depth_mm, name="Front depth [mm]"))
    fig.add_trace(go.Scatter(x=h.time_h, y=100 * h.frozen_fraction, name="Frozen [%]", yaxis="y2"))
    fig.add_hline(y=info["R_mm"], line_dash="dot", annotation_text="Core")
    fig.update_layout(title="Freezing front", xaxis_title="Blasting time [h]",
                      yaxis=dict(title="Depth from surface [mm]"),
                      yaxis2=dict(title="Frozen [%]", overlaying="y", side="right", range=[0, 105]),
                      height=380, legend=dict(orientation="h", y=-0.3))
    st.plotly_chart(fig, width="stretch")

with tab_prof:
    prof = res.profiles
    x = np.array(prof.columns, dtype=float)
    n_lines = 10
    idx = np.unique(np.linspace(0, len(prof) - 1, n_lines).astype(int))
    fig = go.Figure()
    for i in idx:
        fig.add_trace(go.Scatter(x=x, y=prof.iloc[i].values, name=f"{prof.index[i]:.1f} h"))
    fig.add_hline(y=scenario.product.T_freeze, line_dash="dot", annotation_text="Freezing point")
    fig.update_layout(title="Temperature profile, core (0) → surface", xaxis_title="Distance from core [mm]",
                      yaxis_title="°C", height=420)
    st.plotly_chart(fig, width="stretch")

    fig = go.Figure(go.Heatmap(x=prof.index, y=x, z=prof.values.T, colorscale="RdBu_r",
                               colorbar=dict(title="°C")))
    fig.update_layout(title="Temperature field", xaxis_title="Blasting time [h]",
                      yaxis_title="Distance from core [mm]", height=420)
    st.plotly_chart(fig, width="stretch")

with tab_model:
    a, b = st.columns(2)
    a.subheader("Geometry & material")
    a.table({
        "Shape": info["shape"], "R (half-thickness / radius) [mm]": f"{info['R_mm']:.1f}",
        "Layers modeled as identical": str(info["n_layers"]),
        "Heat transfer area / pallet [m²]": f"{info['area_m2']:.2f}",
        "Product envelope volume [m³]": f"{info['envelope_volume_m3']:.3f}",
        "Bulk density [kg/m³]": f"{info['rho_bulk']:.0f}",
        "Volume: product / voids / walls / gap": (
            f"{info['phi_product']:.0%} / {info['phi_void']:.0%} / "
            f"{info['phi_wall']:.1%} / {info['phi_gap']:.1%}"),
        "k_eff unfrozen / frozen [W/m·K]": f"{info['k_eff_unfrozen']:.3f} / {info['k_eff_frozen']:.3f}",
        "Latent heat [MJ/m³]": f"{info['L_vol'] / 1e6:.1f}",
    })
    b.subheader("Air side & numerics")
    b.table({
        "Air mass flow [kg/s]": f"{info['m_dot_air']:.2f}",
        "HTC → effective HTC [W/m²·K]": f"{info['h']:.1f} → {info['h_eff']:.1f}",
        "NTU / effectiveness": f"{info['NTU']:.2f} / {info['eps']:.2f}",
        "Nodes / Δx [mm]": f"{info['n_nodes']} / {info['dx_mm']:.2f}",
        "Time step used / stability limit [s]": f"{info['dt_s']:.1f} / {info['dt_stable_s']:.1f}",
        "Steps": f"{info['n_steps']:,}",
        "Energy balance error": f"{info['energy_balance_error']:.1e}",
    })

with tab_data:
    st.dataframe(h, width="stretch", height=400)
    d1, d2, d3 = st.columns(3)
    d1.download_button("History CSV", h.to_csv(index=False), "history.csv", "text/csv")
    d2.download_button("Profiles CSV", res.profiles.to_csv(), "profiles.csv", "text/csv")
    d3.download_button("Scenario JSON", json.dumps(scenario.to_dict(), indent=2), "scenario.json",
                       "application/json")
