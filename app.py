"""Streamlit front end for the blast freezer model.

Run with:  streamlit run app.py

All inputs are stored in SI (st.session_state["si::<id>"]). Widgets show the
selected unit system and convert on the way in and out, so switching units
keeps the values.
"""

from __future__ import annotations

import json

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from blastfreezer import (CELL_FACES, PACKAGING_PRESETS, Air, CellSpec, Packaging, PalletLoad,
                          Product, Scenario, SimSettings, air_path, simulate, summarize)
from blastfreezer import units as U
from blastfreezer.geometry import representative_body

st.set_page_config(page_title="Blast Freezer Model", layout="wide")
D = Scenario()
ss = st.session_state


# ---------------------------------------------------------- unit helpers ----
def units() -> str:
    return ss.get("units", U.SI)


def qin(label: str, qid: str, q: str | None, default_si: float, lo_si: float, hi_si: float,
        si_scale: float = 1.0, **kw) -> float:
    """Number input shown in the active unit system; returns SI.

    si_scale converts the stored SI value to the quantity's SI display unit
    (e.g. 1000 for metres shown as mm).
    """
    sys_ = units()
    key_si = f"si::{qid}"
    if key_si not in ss:
        ss[key_si] = default_si
    if q is None:
        v = st.number_input(label, float(lo_si), float(hi_si), float(ss[key_si]), key=f"w::{qid}", **kw)
        ss[key_si] = v
        return v
    conv = lambda v: float(U.to_disp(q, v * si_scale, sys_))
    v = st.number_input(f"{label} [{U.unit(q, sys_)}]", conv(lo_si), conv(hi_si), conv(ss[key_si]),
                        float(U.step(q, sys_)), format=U.fmt(q, sys_), key=f"w::{qid}::{sys_}", **kw)
    ss[key_si] = U.from_disp(q, v, sys_) / si_scale
    return ss[key_si]


def iin(label: str, qid: str, default: int, lo: int, hi: int, **kw) -> int:
    return int(st.number_input(label, lo, hi, default, 1, key=f"w::{qid}", **kw))


def cv(q: str, v):
    """SI -> display."""
    return U.to_disp(q, v, units())


def ul(q: str) -> str:
    return U.unit(q, units())


# ---------------------------------------------------------------- inputs ----
def sidebar_inputs() -> Scenario:
    sb = st.sidebar
    sb.radio("Units", [U.SI, U.IP], horizontal=True, key="units")

    with sb.expander("Cell", expanded=True):
        c = D.cell
        st.caption("Size in pallet positions (PPs)")
        c1, c2, c3 = st.columns(3)
        with c1:
            n_deep = iin("Deep", "n_deep", c.n_deep, 1, 50)
        with c2:
            n_wide = iin("Wide", "n_wide", c.n_wide, 1, 50)
        with c3:
            n_high = iin("High", "n_high", c.n_high, 1, 20)
        inlet = st.selectbox("Air enters through", CELL_FACES, index=CELL_FACES.index(c.inlet_face),
                             key="w::inlet")
        cell = CellSpec(n_deep, n_wide, n_high, inlet)

    with sb.expander("Air", expanded=True):
        a = D.air
        T_in = qin("Supply air temperature", "T_in", "temp", a.T_in, -60, 0)
        flow = qin("Airflow per pallet", "flow", "flow", a.flow_per_pallet, 0.01, 20)
        st.caption(f"Cell total: {cv('flow', flow * cell.n_pallets):,.1f} {ul('flow')}")
        htc = qin("Surface HTC, free airflow", "htc", "htc", a.htc, 1, 200)
        ratio = qin("Spacer velocity ratio, flow across spacers", "ratio", None,
                    a.blocked_velocity_ratio, 0.01, 1.0, step=0.01,
                    help="Velocity in the spacers relative to free flow when air runs across them.")
        n_exp = qin("HTC velocity exponent n (h ∝ vⁿ)", "n_exp", None, a.htc_velocity_exponent,
                    0.3, 1.0, step=0.05, help="0.8 turbulent, 0.5 laminar.")
        air = Air(T_in, flow, htc, ratio, n_exp, a.cp)

    with sb.expander("Pallet", expanded=True):
        l = D.load
        mass = qin("Product mass", "mass", "mass", l.mass, 10, 3000)
        length = qin("Depth", "length", "length", l.length, 0.1, 3)
        width = qin("Width", "width", "length", l.width, 0.1, 3)
        height = qin("Product height (excl. spacers)", "height", "length", l.height, 0.1, 4)
        use_sp = st.checkbox("Blast spacers between layers", l.use_spacers, key="w::use_sp")
        n_sp = iin("Number of spacer layers", "n_sp", l.n_spacers, 1, 30, disabled=not use_sp)
        t_sp = qin("Spacer thickness", "t_sp", "small", l.spacer_thickness, 0.001, 0.2, si_scale=1000,
                   disabled=not use_sp)
        channels = st.selectbox("Spacer channels open along", ["both", "depth", "width"],
                                key="w::channels", disabled=not use_sp,
                                format_func={"both": "Depth and width", "depth": "Depth only",
                                             "width": "Width only"}.get)
        geom = st.selectbox("1-D geometry", ["auto", "plate", "sphere", "cylinder"], key="w::geom",
                            help="auto: plate if the block's smallest dimension is < 0.5× the "
                                 "middle one, otherwise an equivalent sphere (R = 3V/A).")
        load = PalletLoad(mass, length, width, height, use_sp, n_sp, t_sp, channels, geom)

    with sb.expander("Packaging", expanded=False):
        kind = st.selectbox("Type", list(PACKAGING_PRESETS), key="w::pkg")
        pre = PACKAGING_PRESETS[kind]
        pkg = Packaging(
            kind=kind,
            package_height=qin("Package height (heat-flow direction)", "pkg_h", "length",
                               D.packaging.package_height, 0.01, 2),
            wall_thickness=qin("Wall thickness", f"wt_{kind}", "small", pre["wall_thickness"], 0, 0.02,
                               si_scale=1000),
            k_wall=qin("Wall conductivity", f"kw_{kind}", "k", pre["k_wall"], 0.001, 1),
            wall_density=qin("Wall density", f"rw_{kind}", "density", pre["wall_density"], 0, 2000),
            cp_wall=pre["cp_wall"],
            air_gap=qin("Headspace air gap per package", f"ag_{kind}", "small", pre["air_gap"], 0, 0.05,
                        si_scale=1000),
            k_air=D.packaging.k_air,
            surface_wall=pre["surface_wall"],
        )
        if pkg.surface_wall:
            st.caption("Outer box wall is applied as a surface resistance.")

    with sb.expander("Product", expanded=False):
        p = D.product
        product = Product(
            name=st.text_input("Name", p.name, key="w::pname"),
            water_fraction=qin("Water content (mass fraction)", "xw", None, p.water_fraction, 0, 1, step=0.01),
            bound_water_fraction=qin("Bound (unfreezable) share of water", "xb", None,
                                     p.bound_water_fraction, 0, 1, step=0.01),
            cp_unfrozen=qin("cp unfrozen", "cpu", "cp", p.cp_unfrozen, 100, 5000),
            cp_frozen=qin("cp frozen", "cpf", "cp", p.cp_frozen, 100, 5000),
            k_unfrozen=qin("k unfrozen", "ku", "k", p.k_unfrozen, 0.01, 5),
            k_frozen=qin("k frozen", "kf", "k", p.k_frozen, 0.01, 5),
            density=qin("Product density (solid)", "rho", "density", p.density, 100, 2000),
            T_freeze=qin("Initial freezing point", "Tf", "temp", p.T_freeze, -15, 0),
            latent_heat_water=p.latent_heat_water,
        )

    with sb.expander("Simulation", expanded=False):
        s = D.sim
        sim = SimSettings(
            T_start=qin("Load-in temperature", "T0", "temp", s.T_start, -10, 40),
            T_target=qin("Target core temperature", "Ttgt", "temp", s.T_target, -50, 0),
            n_nodes=iin("Nodes (core → surface)", "nodes", s.n_nodes, 2, 200),
            dt_max=qin("Max time step [s]", "dtmax", None, s.dt_max, 1.0, 600.0, step=1.0,
                       help="The solver uses the smaller of this and the explicit stability limit."),
            t_max_hours=qin("Max simulated time [h]", "tmax", None, s.t_max_hours, 1.0, 1000.0, step=1.0),
            record_every_s=s.record_every_s,
        )

    return Scenario(product, pkg, load, air, cell, sim)


@st.cache_data(show_spinner="Running simulation…")
def run(d: dict):
    s = Scenario.from_dict(d)
    r = simulate(s)
    return r, summarize(r, s)


def fmt_h(t):
    return "—" if t is None else f"{t:.1f} h"


# ---------------------------------------------------------------- layout ----
st.title("Blast Freezer Model")
scenario = sidebar_inputs()

warnings = []
if scenario.load.use_spacers:
    layer = scenario.load.height / (scenario.load.n_spacers + 1)
    if scenario.packaging.package_height > layer * 1.001:
        warnings.append(f"Package height ({cv('length', scenario.packaging.package_height):.2f} "
                        f"{ul('length')}) is taller than the layer between spacers "
                        f"({cv('length', layer):.2f} {ul('length')}).")
for w in warnings:
    st.warning(w)

try:
    res, summ = run(scenario.to_dict())
except ValueError as e:
    st.error(str(e))
    st.stop()

info = res.info
c = st.columns(5)
c[0].metric("Cell freeze time (last pallet)", fmt_h(summ["cell_freeze_time_h"]),
            help="Time for the slowest pallet core to reach the target.")
c[1].metric("First pallet frozen", fmt_h(summ["first_freeze_time_h"]))
c[2].metric("Peak cell load", f"{cv('power', summ['peak_cell_kW']):,.0f} {ul('power')}")
c[3].metric("Energy removed (cell)", f"{cv('energy', summ['energy_removed_kWh']):,.0f} {ul('energy')}")
c[4].metric("Spacer HTC", f"{cv('htc', info['h']):.1f} {ul('htc')}", info["spacer_alignment"],
            delta_color="off")
if not summ["all_frozen"]:
    st.warning(f"Not every pallet reached the target within {summ['sim_end_h']:.0f} h. "
               "Increase the max simulated time under Simulation.")
st.caption(f"{info['geometry_note']}  Air: {info['inlet']} → {info['outlet']}, "
           f"{info['n_lanes']} lanes × {info['n_series']} pallets in series.")

tab_cell, tab_pal, tab_model, tab_data = st.tabs(["Cell", "Pallet detail", "Model details", "Data"])
P = res.pallets.copy()
for col in ["T_core", "T_surface", "T_mean", "T_air_in", "T_air_out"]:
    P[col] = cv("temp", P[col])
P["Q"] = cv("power", P["Q_kW"])
C = res.cell.copy()
for col in ["T_air_in", "T_air_out"]:
    C[col] = cv("temp", C[col])
C["Q"] = cv("power", C["Q_cell_kW"])
T_target_d = cv("temp", scenario.sim.T_target)
pos_label = lambda i: f"P{i} ({'inlet' if i == 1 else 'outlet' if i == info['n_series'] else 'mid'})"

with tab_cell:
    col1, col2 = st.columns(2)
    fig = go.Figure()
    for i, g in P.groupby("position"):
        fig.add_trace(go.Scatter(x=g.time_h, y=g.T_core, name=pos_label(i)))
    fig.add_hline(y=T_target_d, line_dash="dash", annotation_text="Target")
    fig.update_layout(title="Core temperature by series position", xaxis_title="Blasting time [h]",
                      yaxis_title=ul("temp"), height=420, legend=dict(orientation="h", y=-0.25))
    col1.plotly_chart(fig, width="stretch")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=C.time_h, y=C.Q, name="Cell load"))
    fig.update_layout(title="Cell heat removal", xaxis_title="Blasting time [h]",
                      yaxis_title=ul("power"), height=420)
    col2.plotly_chart(fig, width="stretch")

    col3, col4 = st.columns(2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=C.time_h, y=C.T_air_in, name="Cell inlet", line=dict(dash="dot")))
    for i, g in P.groupby("position"):
        fig.add_trace(go.Scatter(x=g.time_h, y=g.T_air_out, name=f"After {pos_label(i)}"))
    fig.update_layout(title="Air temperature along the flow path", xaxis_title="Blasting time [h]",
                      yaxis_title=ul("temp"), height=420, legend=dict(orientation="h", y=-0.25))
    col3.plotly_chart(fig, width="stretch")

    ft = res.freeze_times_h
    fig = go.Figure(go.Bar(x=[pos_label(i + 1) for i in range(len(ft))],
                           y=[t if t is not None else np.nan for t in ft],
                           text=[fmt_h(t) for t in ft], textposition="outside"))
    fig.update_layout(title=f"Freeze time by series position (inlet: {info['inlet']})",
                      yaxis_title="h", height=420)
    col4.plotly_chart(fig, width="stretch")

with tab_pal:
    n_s = info["n_series"]
    pos = st.selectbox("Series position", list(range(1, n_s + 1)), format_func=pos_label)
    g = P[P.position == pos]
    col1, col2 = st.columns(2)
    fig = go.Figure()
    for name, col, dash in [("Core", "T_core", None), ("Mean", "T_mean", None),
                            ("Surface", "T_surface", None), ("Air in", "T_air_in", "dot"),
                            ("Air out", "T_air_out", "dot")]:
        fig.add_trace(go.Scatter(x=g.time_h, y=g[col], name=name, line=dict(dash=dash)))
    fig.add_hline(y=T_target_d, line_dash="dash", annotation_text="Target")
    fig.update_layout(title=f"{pos_label(pos)} temperatures", xaxis_title="Blasting time [h]",
                      yaxis_title=ul("temp"), height=420, legend=dict(orientation="h", y=-0.25))
    col1.plotly_chart(fig, width="stretch")

    depth = cv("small", g.front_depth_mm) if units() == U.IP else g.front_depth_mm
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g.time_h, y=depth, name="Front depth"))
    fig.add_trace(go.Scatter(x=g.time_h, y=100 * g.frozen_fraction, name="Frozen [%]", yaxis="y2"))
    fig.update_layout(title="Freezing front", xaxis_title="Blasting time [h]",
                      yaxis=dict(title=f"Depth from surface [{ul('small')}]"),
                      yaxis2=dict(title="Frozen [%]", overlaying="y", side="right", range=[0, 105]),
                      height=420, legend=dict(orientation="h", y=-0.25))
    col2.plotly_chart(fig, width="stretch")

    prof = res.profiles[pos]
    x = np.array(prof.columns, dtype=float)
    x = cv("small", x) if units() == U.IP else x
    Tp = cv("temp", prof.values)
    idx = np.unique(np.linspace(0, len(prof) - 1, 10).astype(int))
    col3, col4 = st.columns(2)
    fig = go.Figure()
    for i in idx:
        fig.add_trace(go.Scatter(x=x, y=Tp[i], name=f"{prof.index[i]:.1f} h"))
    fig.add_hline(y=cv("temp", scenario.product.T_freeze), line_dash="dot", annotation_text="Freezing point")
    fig.update_layout(title="Temperature profile, core (0) → surface",
                      xaxis_title=f"Distance from core [{ul('small')}]", yaxis_title=ul("temp"), height=420)
    col3.plotly_chart(fig, width="stretch")
    fig = go.Figure(go.Heatmap(x=prof.index, y=x, z=Tp.T, colorscale="RdBu_r",
                               colorbar=dict(title=ul("temp"))))
    fig.update_layout(title="Temperature field", xaxis_title="Blasting time [h]",
                      yaxis_title=f"Distance from core [{ul('small')}]", height=420)
    col4.plotly_chart(fig, width="stretch")

with tab_model:
    a, b = st.columns(2)
    a.subheader("Pallet geometry & material")
    a.table({
        "Shape": info["shape"],
        "R (half-thickness / radius)": f"{cv('small', info['R_mm']) if units() == U.IP else info['R_mm']:.1f} {ul('small')}",
        "Layers modeled as identical": str(info["n_layers"]),
        "Heat transfer area / pallet": f"{cv('area', info['area_m2']):.2f} {ul('area')}",
        "Product envelope volume": f"{cv('volume', info['envelope_volume_m3']):.3f} {ul('volume')}",
        "Bulk density": f"{cv('density', info['rho_bulk']):.1f} {ul('density')}",
        "Volume: product / voids / walls / gap": (
            f"{info['phi_product']:.0%} / {info['phi_void']:.0%} / "
            f"{info['phi_wall']:.1%} / {info['phi_gap']:.1%}"),
        "k_eff unfrozen / frozen": f"{cv('k', info['k_eff_unfrozen']):.3f} / {cv('k', info['k_eff_frozen']):.3f} {ul('k')}",
        "Surface packaging resistance": f"{info['R_surface']:.3f} m²·K/W",
        "Latent heat (freezable water)": f"{cv('latent', info['L_vol'] / 1e6):.1f} {ul('latent')}",
    })
    b.subheader("Air path & numerics")
    b.table({
        "Flow": f"{info['inlet']} → {info['outlet']} (along {info['flow_axis']})",
        "Lanes × pallets in series": f"{info['n_lanes']} × {info['n_series']}",
        "Airflow cell / lane": f"{cv('flow', info['flow_total_m3s']):,.1f} / {cv('flow', info['flow_lane_m3s']):,.2f} {ul('flow')}",
        "Spacers": info["spacer_alignment"],
        "HTC applied → effective (ε-NTU)": f"{cv('htc', info['h']):.2f} → {cv('htc', info['h_eff']):.2f} {ul('htc')}",
        "NTU / effectiveness per pallet": f"{info['NTU']:.3f} / {info['eps']:.3f}",
        "Nodes / Δx": f"{info['n_nodes']} / {info['dx_mm']:.2f} mm",
        "Time step used / stability limit": f"{info['dt_s']:.1f} / {info['dt_stable_s']:.1f} s",
        "Steps": f"{info['n_steps']:,}",
        "Energy balance error": f"{info['energy_balance_error']:.1e}",
    })

with tab_data:
    st.caption("Downloads are in SI units.")
    st.dataframe(res.pallets, width="stretch", height=400)
    d1, d2, d3 = st.columns(3)
    d1.download_button("Pallet history CSV", res.pallets.to_csv(index=False), "pallets.csv", "text/csv")
    d2.download_button("Cell history CSV", res.cell.to_csv(index=False), "cell.csv", "text/csv")
    d3.download_button("Scenario JSON", json.dumps(scenario.to_dict(), indent=2), "scenario.json",
                       "application/json")
