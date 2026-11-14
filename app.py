"""Blast freezer cell model: Streamlit GUI.

Run with:  streamlit run app.py

Layout (see "gui example.png"):
  upper left   tabbed inputs (Pallet / Cell / Model / Boost) + Run solver
  upper right  cell view coloured by core temperature, airflow arrows, time slider
  bottom       time series of the run (Temperature / Power)

All inputs are stored in SI in st.session_state["si::<id>"]. Widgets show the
selected unit system and convert on the way in and out, so switching units
keeps the values.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from blastfreezer import (PACKAGING_PRESETS, Air, CellSpec, Packaging, PalletLoad,
                          Product, Scenario, SimSettings, simulate, summarize)
from blastfreezer import units as U
from blastfreezer.cell import Cell
from blastfreezer.config import REFRIGERANT_CHOICES, Cooling
from blastfreezer.pallet import air_density
from blastfreezer.refrigeration import Coil, RefSystem, design_return_balance
from blastfreezer.foods import CUSTOM, DEFAULT_FOOD, FOODS, product_kwargs
from blastfreezer.viz import GUI_FACES, cell_figure, power_figure, temperature_figure

FLOW_LABEL = {"left": "Left → right", "right": "Right → left", "bottom": "Bottom → top",
              "top": "Top → bottom"}

st.set_page_config(page_title="Blast Freezer Model", layout="wide")
D = Scenario()
ss = st.session_state


# ---------------------------------------------------------- unit helpers ----
def units() -> str:
    return ss.get("units", U.SI)


def cv(q: str, v):
    return U.to_disp(q, v, units())


def ul(q: str) -> str:
    return U.unit(q, units())


def qw(kind: str, label: str, qid: str, q: str | None, default_si: float, lo_si: float, hi_si: float,
       si_scale: float = 1.0, step: float | None = None, unit_label: str = "", **kw) -> float:
    """Slider or number input in the active unit system; returns SI.

    q=None means no physical unit; si_scale then scales for display (e.g. 100 for %).
    For q='small', si_scale=1000 shows metres as mm in SI.
    """
    sys_ = units()
    key_si = f"si::{qid}"
    if key_si not in ss:
        ss[key_si] = default_si
    if q is None:
        conv = lambda v: float(v * si_scale)
        back = lambda v: v / si_scale
        lab = f"{label} [{unit_label}]" if unit_label else label
        stp = step if step is not None else 0.01 * si_scale
        fmt = kw.pop("format", "%.2f")
    else:
        conv = lambda v: float(U.to_disp(q, v * si_scale, sys_))
        back = lambda v: U.from_disp(q, v, sys_) / si_scale
        lab = f"{label} [{U.unit(q, sys_)}]"
        stp = step if step is not None else U.step(q, sys_)
        fmt = kw.pop("format", U.fmt(q, sys_))
    lo, hi = sorted((conv(lo_si), conv(hi_si)))
    val = min(max(conv(ss[key_si]), lo), hi)
    widget = st.slider if kind == "slider" else st.number_input
    v = widget(lab, lo, hi, val, float(stp), format=fmt, key=f"w::{qid}::{sys_}", **kw)
    # write back only on a real change, so unit round-trips don't perturb stored SI values
    if not np.isclose(v, conv(ss[key_si]), rtol=1e-9, atol=1e-12):
        ss[key_si] = back(v)
    return ss[key_si]


def islider(label: str, qid: str, default: int, lo: int, hi: int, **kw) -> int:
    return int(st.slider(label, lo, hi, default, 1, key=f"w::{qid}", **kw))


# ---------------------------------------------------------------- inputs ----
def pallet_tab() -> tuple[Product, Packaging, PalletLoad, float, float]:
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown("**Product**")
        names = list(FOODS) + [CUSTOM]
        choice = st.selectbox("Product", names, index=names.index(DEFAULT_FOOD), key="w::product",
                              label_visibility="collapsed")
        if choice != CUSTOM:
            ss["custom_base"] = choice
            props = product_kwargs(choice)
            food = FOODS[choice]
            rows = [("Water content", f"{props['water_fraction']:.1%}"),
                    (f"cp unfrozen [{ul('cp')}]", f"{cv('cp', props['cp_unfrozen']):.3g}"),
                    (f"cp frozen [{ul('cp')}]", f"{cv('cp', props['cp_frozen']):.3g}"),
                    (f"k unfrozen [{ul('k')}]", f"{cv('k', props['k_unfrozen']):.3f}"),
                    (f"k frozen [{ul('k')}]", f"{cv('k', props['k_frozen']):.3f}"),
                    (f"Density [{ul('density')}]", f"{cv('density', props['density']):.0f}"),
                    (f"Initial freezing point [{ul('temp')}]", f"{cv('temp', props['T_freeze']):.1f}")]
            st.dataframe(pd.DataFrame(rows, columns=["Property", "Value"]), hide_index=True,
                         width="stretch", height=document_height(len(rows)))
            st.caption(f"Source: {food.source}" + ("  ⚠️ estimated" if food.estimated else ""))
        else:
            base = ss.get("custom_base", DEFAULT_FOOD)
            b = product_kwargs(base)
            st.caption(f"Starting from {base}")
            k = f"cust_{base}_"
            props = dict(
                name=f"Custom ({base})",
                water_fraction=qw("number", "Water content", k + "xw", None, b["water_fraction"], 0, 1,
                                  si_scale=100, unit_label="%", step=0.5, format="%.1f"),
                cp_unfrozen=qw("number", "cp unfrozen", k + "cpu", "cp", b["cp_unfrozen"], 500, 5000),
                cp_frozen=qw("number", "cp frozen", k + "cpf", "cp", b["cp_frozen"], 500, 5000),
                k_unfrozen=qw("number", "k unfrozen", k + "ku", "k", b["k_unfrozen"], 0.05, 3),
                k_frozen=qw("number", "k frozen", k + "kf", "k", b["k_frozen"], 0.05, 3),
                density=qw("number", "Density (solid)", k + "rho", "density", b["density"], 300, 2000),
                T_freeze=qw("number", "Initial freezing point", k + "Tf", "temp", b["T_freeze"], -10, 0),
            )
        bound = qw("slider", "Bound (unfreezable) water", "xb", None, D.product.bound_water_fraction, 0, 0.3,
                   si_scale=100, unit_label="% of water", step=1.0, format="%.0f")
        product = Product(bound_water_fraction=bound, **props)

        st.markdown("**Packaging**")
        kind = st.selectbox("Packaging", list(PACKAGING_PRESETS), key="w::pkg", label_visibility="collapsed")
        pre = PACKAGING_PRESETS[kind]
        pkg_h = qw("number", "Package height", "pkg_h", "length", D.packaging.package_height, 0.02, 2.0)
        with st.expander("Packaging details"):
            wall = qw("number", "Wall thickness", f"wt_{kind}", "small", pre["wall_thickness"], 0, 0.02,
                      si_scale=1000)
            k_wall = qw("number", "Wall conductivity", f"kw_{kind}", "k", pre["k_wall"], 0.001, 1)
            rho_wall = qw("number", "Wall density", f"rw_{kind}", "density", pre["wall_density"], 0, 2000)
            gap = qw("number", "Headspace air gap per package", f"ag_{kind}", "small", pre["air_gap"], 0, 0.05,
                     si_scale=1000)
            if pre["surface_wall"]:
                st.caption("Outer box wall is applied as a surface resistance.")
        pkg = Packaging(kind, pkg_h, wall, k_wall, rho_wall, pre["cp_wall"], gap, D.packaging.k_air,
                        pre["surface_wall"])

    with c2:
        st.markdown("**Pallet**")
        l = D.load
        mass = qw("slider", "Product mass", "mass", "mass", l.mass, 100, 1200, step=5 if units() == U.SI else 10)
        d1, d2, d3 = st.columns(3)
        with d1:
            length = qw("number", "Depth", "length", "length", l.length, 0.2, 3)
        with d2:
            width = qw("number", "Width", "width", "length", l.width, 0.2, 3)
        with d3:
            height = qw("number", "Height", "height", "length", l.height, 0.1, 4,
                        help="Product height, excluding spacers.")
        use_sp = st.toggle("Blast spacers between layers", l.use_spacers, key="w::use_sp")
        n_sp = islider("Spacer layers", "n_sp", l.n_spacers, 1, 15, disabled=not use_sp)
        t_sp = qw("number", "Spacer thickness", "t_sp", "small", l.spacer_thickness, 0.005, 0.2,
                  si_scale=1000, disabled=not use_sp)
        load = PalletLoad(mass, length, width, height, use_sp, n_sp, t_sp, "both",
                          ss.get("w::geom", "auto"))
        if use_sp:
            st.caption(f"{n_sp + 1} layers of {cv('length', height / (n_sp + 1)):.2f} {ul('length')}; "
                       f"stack height {cv('length', load.stack_height):.2f} {ul('length')}")

        st.markdown("**Temperatures**")
        T0 = qw("slider", "Load-in temperature", "T0", "temp", D.sim.T_start, -5, 40)
        Tt = qw("slider", "Target core temperature", "Ttgt", "temp", D.sim.T_target, -40, -5)
    return product, pkg, load, T0, Tt


def document_height(n_rows: int) -> int:
    return 38 + 35 * n_rows


def cell_tab() -> tuple[CellSpec, float]:
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown("**Size (pallet positions)**")
        c = D.cell
        n_deep = islider("Deep", "n_deep", c.n_deep, 1, 10)
        n_wide = islider("Wide", "n_wide", c.n_wide, 1, 10)
        n_high = islider("High", "n_high", c.n_high, 1, 8)
        st.caption(f"{n_deep * n_wide * n_high} pallet positions")
    with c2:
        st.markdown("**Air**")
        inlet = st.selectbox("Air flow direction", GUI_FACES, index=GUI_FACES.index(c.inlet_face),
                             key="w::inlet", format_func=lambda f: FLOW_LABEL[f])
        flow = qw("slider", "Total airflow", "flow_total", "flow", D.air.flow_total, 1.0, 120.0,
                  step=0.5 if units() == U.SI else 1000.0)
        axis_n = {"left": n_wide, "right": n_wide, "bottom": n_high, "top": n_high}[inlet]
        n_lanes = n_deep * n_wide * n_high // axis_n
        st.caption(f"{n_lanes} lanes × {axis_n} pallets in series · "
                   f"{cv('flow', flow / n_lanes):,.{2 if units() == U.SI else 0}f} {ul('flow')} per lane")
    return CellSpec(n_deep, n_wide, n_high, inlet), flow


def cooling_tab(flow_total: float) -> Cooling:
    c = D.cooling
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown("**Coil (design point)**")
        coil_Q = qw("number", "Design cooling power", "coil_Q", "power", c.coil_Q_design, 1_000, 2_000_000,
                    si_scale=1e-3)
        coil_Ts = qw("number", "Design supply air temp", "coil_Ts", "temp", c.coil_T_supply_design, -60, 0)
        coil_Tr = qw("number", "Design return air temp", "coil_Tr", "temp", c.coil_T_return_design, -60, 10)
        coil_SST = qw("number", "Design SST", "coil_SST", "temp", c.coil_SST_design, -70, 0)
    with c2:
        st.markdown("**Refrigeration system (design point)**")
        ref_Q = qw("number", "Design refrigeration power", "ref_Q", "power", c.ref_Q_design, 1_000, 2_000_000,
                   si_scale=1e-3)
        ref_SST = qw("number", "Design SST", "ref_SST", "temp", c.ref_SST_design, -70, 0)
        refr = st.selectbox("Refrigerant", REFRIGERANT_CHOICES, index=REFRIGERANT_CHOICES.index(c.refrigerant),
                            key="w::refrigerant")
    cooling = Cooling(coil_Q, coil_Ts, coil_Tr, coil_SST, ref_Q, ref_SST, refr)

    # derived design numbers
    try:
        coil = Coil(coil_Q, coil_Ts, coil_Tr, coil_SST)
        ref = RefSystem(ref_Q, ref_SST, refr)
    except ValueError as e:
        st.error(str(e))
        return cooling
    C_air = flow_total * air_density(coil_Ts) * D.air.cp
    flow_d = coil.design_C_air / D.air.cp / air_density(coil_Ts)
    bal = design_return_balance(coil, ref, C_air)
    T = lambda v: f"{cv('temp', v):.1f} {ul('temp')}"
    P = lambda w: f"{cv('power', w / 1000):,.1f} {ul('power')}"
    d1, d2 = st.columns(2, gap="medium")
    d1.caption(f"Coil UA {cv('ua', coil.UA / 1000):,.{2 if units() == U.SI else 0}f} {ul('ua')} · "
               f"design airflow {cv('flow', flow_d):,.{1 if units() == U.SI else 0}f} {ul('flow')} "
               f"(cell: {cv('flow', flow_total):,.{1 if units() == U.SI else 0}f})")
    d2.caption(f"{refr} · compressor displacement {cv('flow', ref.displacement):,.{3 if units() == U.SI else 0}f} "
               f"{ul('flow')} · {P(float(ref.capacity(coil_SST)))} at coil design SST")
    st.caption(f"At the coil design return air ({T(coil_Tr)}) and the cell airflow, the system balances at "
               f"SST {T(bal.SST)}, supply air {T(bal.T_supply)}, {P(bal.Q)}.")
    if abs(flow_total / flow_d - 1) > 0.2:
        st.warning("Cell airflow differs from the airflow implied by the coil design point "
                   "(design power ÷ air temperature drop), so the coil won't deliver its design "
                   "power at the design return temperature.")
    return cooling


def model_tab() -> tuple[SimSettings, float, float]:
    c1, c2 = st.columns(2, gap="medium")
    s, a = D.sim, D.air
    with c1:
        st.radio("Units", [U.SI, U.IP], horizontal=True, key="units")
        st.markdown("**Heat transfer**")
        htc = qw("slider", "Surface HTC", "htc", "htc", a.htc, 5, 80)
        reduction = qw("slider", "Reduction of HTC when spacers are perpendicular to airflow",
                       "htc_reduction", None, a.spacer_htc_reduction, 0.0, 1.0, si_scale=100, unit_label="%",
                       step=1.0, format="%.0f",
                       help="When airflow is not aligned with spacers, heat transfer is reduced by this "
                            "amount because the spacers don't see good airflow.")
    with c2:
        st.markdown("**Solver**")
        n_nodes = islider("Nodes, core → surface", "nodes", s.n_nodes, 10, 100,
                          help="40 gives ~1–2 % freeze-time error.")
        st.selectbox("1-D geometry", ["auto", "plate", "sphere", "cylinder"], key="w::geom",
                     help="auto: plate if the block's smallest dimension is < 0.5× the middle one, "
                          "otherwise an equivalent sphere (R = 3V/A).")
        t_max = qw("slider", "Max simulated time", "tmax", None, s.t_max_hours, 12, 500, unit_label="h",
                   step=12.0, format="%.0f")
        dt_max = qw("number", "Max time step", "dtmax", None, s.dt_max, 1, 600, unit_label="s", step=1.0,
                    format="%.0f", help="The solver uses the smaller of this and the stability limit.")
        rec = qw("number", "Output interval", "rec", None, s.record_every_s, 60, 3600, si_scale=1 / 60,
                 unit_label="min", step=1.0, format="%.0f", help="Resolution of the time slider and plots.")
    return SimSettings(s.T_start, s.T_target, n_nodes, dt_max, t_max, rec), htc, reduction


# ---------------------------------------------------------------- layout ----
left, right = st.columns([5, 6], gap="large")

with left:
    st.subheader("Inputs")
    t_pal, t_cell, t_cool, t_model, t_boost = st.tabs(["Pallet", "Cell", "Cooling", "Model", "Boost"])
    with t_model:
        sim, htc, reduction = model_tab()
    with t_pal:
        product, pkg, load, T0, Tt = pallet_tab()
    with t_cell:
        cell, flow_total = cell_tab()
    with t_cool:
        cooling = cooling_tab(flow_total)
    with t_boost:
        st.info("Boost inputs will go here.")
    sim.T_start, sim.T_target = T0, Tt
    air = Air(flow_total, htc, reduction, D.air.cp)
    scenario = Scenario(product, pkg, load, air, cell, sim, cooling)

    if load.use_spacers and pkg.package_height > load.height / (load.n_spacers + 1) * 1.001:
        st.warning("Package height is taller than the layer between spacers.")
    if st.button("Run solver", type="primary", width="stretch"):
        try:
            with st.spinner("Solving…"):
                res = simulate(scenario)
            ss["res"], ss["res_scn"] = res, scenario.to_dict()
        except ValueError as e:
            st.error(str(e))

res = ss.get("res")
if res is not None and Scenario.from_dict(ss["res_scn"]).cell.inlet_face not in GUI_FACES:
    res = None                                    # solution from an older layout option
shown = Scenario.from_dict(ss["res_scn"]) if res is not None else scenario
stale = res is not None and ss["res_scn"] != scenario.to_dict()

with right:
    st.subheader("Cell")
    if stale:
        st.info("Inputs changed since the last solve. Showing the last solution; click **Run solver** to update.")
    c = shown.cell
    flow_text = f"{cv('flow', shown.air.flow_total):,.1f} {ul('flow')}"
    if res is None:
        n_ser = {"bottom": c.n_high, "top": c.n_high, "left": c.n_wide, "right": c.n_wide}[c.inlet_face]
        T_frames, times, T_in, T_out = [[shown.sim.T_start] * n_ser], [0.0], [None], [None]
        try:                                  # load-in balance as a preview
            c0 = Cell(shown)
            T_in, T_out = [c0.balance.T_supply], [c0.balance.T_return]
        except ValueError:
            pass
    else:
        P = res.pallets.pivot(index="time_h", columns="position", values="T_core").sort_index()
        T_frames, times = P.values, list(P.index)
        Cw = res.cell.set_index("time_h").loc[P.index]
        T_in, T_out = Cw["T_air_in"].tolist(), Cw["T_air_out"].tolist()
    st.plotly_chart(
        cell_figure(c.n_deep, c.n_wide, c.n_high, c.inlet_face, T_frames, times, T_out,
                    shown.sim.T_start, shown.sim.T_target, cv, ul, T_in, flow_text, height=480),
        width="stretch", config={"displayModeBar": False}, key="cell_fig")
    if res is None:
        st.caption("Run the solver, then drag the time slider (or press ▶) to step through the blast.")
    else:
        summ = summarize(res, shown)
        m = st.columns(4)
        fmt_h = lambda t: "—" if t is None else f"{t:.1f} h"
        m[0].metric("Cell frozen (last pallet)", fmt_h(summ["cell_freeze_time_h"]))
        m[1].metric("First pallet frozen", fmt_h(summ["first_freeze_time_h"]))
        m[2].metric("Peak cooling", f"{cv('power', summ['peak_cell_kW']):,.0f} {ul('power')}")
        m[3].metric("Energy removed", f"{cv('energy', summ['energy_removed_kWh']):,.0f} {ul('energy')}")
        if not summ["all_frozen"]:
            st.warning(f"Not every pallet reached the target within {summ['sim_end_h']:.0f} h "
                       "(Model tab → Max simulated time).")

st.divider()
t_temp, t_pow = st.tabs(["Temperature", "Power"])
if res is None:
    for t in (t_temp, t_pow):
        t.info("Run the solver to see the time series.")
else:
    with t_temp:
        st.plotly_chart(temperature_figure(res, shown.sim.T_target, cv, ul), width="stretch")
        st.caption("Warmest / coldest = last / first pallet to reach the target. "
                   "Surface node = outermost finite volume.")
    with t_pow:
        st.plotly_chart(power_figure(res, cv, ul, design_kW=shown.cooling.ref_Q_design / 1000), width="stretch")
        st.caption("Cooling power = coil load = refrigeration capacity at the balanced SST. "
                   "Fan power is not modeled yet.")

    with st.expander("Model details & data"):
        info = res.info
        a, b = st.columns(2)
        a.dataframe(pd.DataFrame([
            ("Pallet geometry", info["geometry_note"]),
            ("Bulk density", f"{cv('density', info['rho_bulk']):.0f} {ul('density')}"),
            ("Volume: product / voids / walls / gap",
             f"{info['phi_product']:.0%} / {info['phi_void']:.0%} / {info['phi_wall']:.1%} / {info['phi_gap']:.1%}"),
            ("k_eff unfrozen / frozen",
             f"{cv('k', info['k_eff_unfrozen']):.3f} / {cv('k', info['k_eff_frozen']):.3f} {ul('k')}"),
            ("Surface packaging resistance", f"{info['R_surface']:.3f} m²·K/W"),
            ("Latent heat (freezable water)", f"{cv('latent', info['L_vol'] / 1e6):.1f} {ul('latent')}"),
        ], columns=["Item", "Value"]), hide_index=True, width="stretch")
        b.dataframe(pd.DataFrame([
            ("Flow", f"{info['inlet']} → {info['outlet']}, {info['n_lanes']} lanes × {info['n_series']} in series"),
            ("Spacers", info["spacer_alignment"]),
            ("HTC applied → effective (ε-NTU)",
             f"{cv('htc', info['h']):.2f} → {cv('htc', info['h_eff']):.2f} {ul('htc')}"),
            ("Airflow per lane", f"{cv('flow', info['flow_lane_m3s']):,.2f} {ul('flow')}"),
            ("Nodes / time step", f"{info['n_nodes']} / {info['dt_s']:.1f} s ({info['n_steps']:,} steps)"),
            ("Energy balance error", f"{info['energy_balance_error']:.1e}"),
        ], columns=["Item", "Value"]), hide_index=True, width="stretch")
        if info.get("cooling_mode") == "coil":
            C = res.cell
            st.dataframe(pd.DataFrame([
                ("Coil UA / LMTD", f"{cv('ua', info['coil_UA_W_K'] / 1000):,.2f} {ul('ua')} / {info['coil_LMTD']:.2f} K"),
                ("Coil effectiveness at cell airflow", f"{info['coil_eps']:.3f}"),
                ("Compressor displacement", f"{cv('flow', info['ref_displacement_m3s']):,.3f} {ul('flow')}"),
                ("SST range over the blast", f"{cv('temp', C.SST.max()):.1f} → {cv('temp', C.SST.min()):.1f} {ul('temp')}"),
                ("Supply air range", f"{cv('temp', C.T_air_in.max()):.1f} → {cv('temp', C.T_air_in.min()):.1f} {ul('temp')}"),
                ("Steps at SST limit", f"{(C.limit != '').sum()} of {len(C)} recorded"),
            ], columns=["Cooling", "Value"]), hide_index=True, width="stretch")
        ft = pd.DataFrame({"Series position": [f"P{i + 1}" for i in range(len(res.freeze_times_h))],
                           "Freeze time [h]": [None if t is None else round(t, 2) for t in res.freeze_times_h]})
        st.dataframe(ft, hide_index=True)
        d1, d2, d3 = st.columns(3)
        d1.download_button("Pallet history CSV (SI)", res.pallets.to_csv(index=False), "pallets.csv", "text/csv")
        d2.download_button("Cell history CSV (SI)", res.cell.to_csv(index=False), "cell.csv", "text/csv")
        d3.download_button("Scenario JSON", json.dumps(shown.to_dict(), indent=2), "scenario.json",
                           "application/json")
