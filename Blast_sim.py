"""Blast freezer cell model: Streamlit GUI.

Run with:  streamlit run Blast_sim.py

Layout (see "gui example.png"):
  upper left   tabbed inputs (Pallet / Cell / Model / Boost) + Run solver
  upper right  cell view coloured by core temperature, airflow arrows, time slider
  bottom       time series of the run (Temperature / Power)

All inputs are stored in SI in st.session_state["si::<id>"]. Widgets show the
selected unit system and convert on the way in and out, so switching units
keeps the values. Save / load configuration writes and reads that state as JSON
(blastfreezer/config_file.py).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import streamlit as st

from blastfreezer import (PACKAGING_PRESETS, Air, CellSpec, Packaging, PalletLoad,
                          Product, Scenario, SimSettings, simulate_pair, summarize)
from blastfreezer import config_file
from blastfreezer import units as U
from blastfreezer.cell import Cell
from blastfreezer.config import REFRIGERANT_CHOICES, Boost, Cooling, coil_airflow, coil_return
from blastfreezer.pallet import air_density
from blastfreezer.refrigeration import Coil, RefSystem, design_return_balance
from blastfreezer.foods import CUSTOM, DEFAULT_FOOD, FOODS, product_kwargs
from blastfreezer.viz import (GUI_FACES, air_temperature_figure, cell_figure, cooling_figure,
                              product_temperature_figure, ref_temperature_figure)

FLOW_LABEL = {"left": "Left → right", "right": "Right → left", "bottom": "Bottom → top",
              "top": "Top → bottom"}

st.set_page_config(page_title="Blast Freezing Simulator", layout="wide")
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
    key = f"w::{qid}"
    if key in ss:                                   # e.g. from a loaded configuration file
        ss[key] = min(max(int(ss[key]), lo), hi)
    return int(st.slider(label, lo, hi, default, 1, key=key, **kw))


# ---------------------------------------------------------------- inputs ----
MAX_LAYERS = 20
GEOMETRIES = ["auto", "plate", "sphere", "cylinder"]
DRIVERS = ["Return air", "Airflow"]


def pallet_tab() -> tuple[Product, Packaging, PalletLoad, float, float]:
    c1, c2 = st.columns(2, gap="medium")
    l = D.load
    # Pallet column first: package height and spacers in the packaging details depend on its height.
    with c2:
        st.markdown("**Pallet**")
        mass = qw("slider", "Total product weight on pallet", "mass", "mass", l.mass, 100, 1200, step=5 if units() == U.SI else 10)
        d1, d2, d3 = st.columns(3)
        with d1:
            length = qw("number", "Depth", "length", "length", l.length, 0.2, 3)
        with d2:
            width = qw("number", "Width", "width", "length", l.width, 0.2, 3)
        with d3:
            height = qw("number", "Height", "height", "length", l.height, 0.1, 4,
                        help="Product height, excluding spacers.")
        use_sp = st.toggle("Blast spacers between layers", l.use_spacers, key="w::use_sp",
                           help="A spacer between every layer of packages. Count and thickness are in "
                                "Packaging details.")
        summary = st.empty()

        st.markdown("**Temperatures**")
        T0 = qw("slider", "Load-in temperature", "T0", "temp", D.sim.T_start, -5, 40)
        Tt = qw("slider", "Target core temperature", "Ttgt", "temp", D.sim.T_target, -40, -5)

    with c1:
        st.markdown("**Product**")
        names = list(FOODS) + [CUSTOM]
        choice = st.selectbox("Product", names, index=names.index(DEFAULT_FOOD), key="w::product",
                              label_visibility="collapsed")
        with st.expander("Product details", expanded=choice == CUSTOM):
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
        with st.expander("Packaging details"):
            n_layers = st.select_slider(
                f"Package height [{ul('length')}]", options=list(range(MAX_LAYERS, 0, -1)),
                value=min(max(int(ss.setdefault("n_layers", round(l.height / D.packaging.package_height))), 1),
                          MAX_LAYERS),
                key=f"w::n_layers::{units()}",       # per unit system, like qw(); value kept in ss
                # labels must be unique: the widget maps its value back to an option by label
                format_func=lambda n: f"{cv('length', height / n):.2f} ({n} layer{'s' * (n > 1)})",
                help=f"Pallet height divided into a whole number of layers (1 to {MAX_LAYERS}).")
            ss["n_layers"] = n_layers
            pkg_h = height / n_layers
            n_sp = n_layers - 1
            st.number_input("Spacer layers", value=n_sp, disabled=True,
                            help="Calculated: one spacer between every pair of package layers "
                                 "(layers − 1). Spacers are never skipped on a row.")
            t_sp = qw("number", "Spacer thickness", "t_sp", "small", l.spacer_thickness, 0.005, 0.2,
                      si_scale=1000, disabled=not use_sp)
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

    load = PalletLoad(mass, length, width, height, use_sp, n_sp, t_sp, "both", ss.get("w::geom", "auto"))
    layers = f"{n_layers} layer{'s' * (n_layers > 1)} of {cv('length', pkg_h):.2f} {ul('length')}"
    if use_sp and n_sp > 0:
        summary.caption(f"{layers}, {n_sp} spacer{'s' * (n_sp > 1)} of {cv('small', t_sp * 1000):.{1 if units() == U.SI else 2}f} "
                        f"{ul('small')}; stack height {cv('length', load.stack_height):.2f} {ul('length')}")
    else:
        summary.caption(layers + (" (single layer, so no spacers)" if use_sp else ", no spacers"))
    return product, pkg, load, T0, Tt


def document_height(n_rows: int) -> int:
    return 38 + 35 * n_rows


def cell_tab(airflow: float, boost_airflow: float) -> CellSpec:
    c1, c2 = st.columns(2, gap="medium")
    c = D.cell
    with c1:
        st.markdown("**Size (pallet positions)**")
        n_deep = islider("Deep", "n_deep", c.n_deep, 1, 10)
        n_wide = islider("Wide", "n_wide", c.n_wide, 1, 10)
        n_high = islider("High", "n_high", c.n_high, 1, 8)
        st.caption(f"{n_deep * n_wide * n_high} pallet positions")
    with c2:
        st.markdown("**Air**")
        inlet = st.selectbox("Air flow direction", GUI_FACES, index=GUI_FACES.index(c.inlet_face),
                             key="w::inlet", format_func=lambda f: FLOW_LABEL[f])
        axis_n = {"left": n_wide, "right": n_wide, "bottom": n_high, "top": n_high}[inlet]
        n_lanes = n_deep * n_wide * n_high // axis_n
        dec = 2 if units() == U.SI else 0
        st.caption(f"Airflow {cv('flow', airflow):,.{dec}f} {ul('flow')} (Cooling tab), plus "
                   f"{cv('flow', boost_airflow):,.{dec}f} {ul('flow')} while the boost runs (Boost tab) · "
                   f"{n_lanes} lanes × {axis_n} pallets in series · "
                   f"{cv('flow', airflow / n_lanes):,.{dec}f} {ul('flow')} per lane without boost")
    return CellSpec(n_deep, n_wide, n_high, inlet)


def _readonly(label: str, q: str, si_value: float, help: str, si_scale: float = 1.0, pfx: str = ""):
    """Greyed-out number showing a calculated value. Keyed by prefix + value: a keyed widget ignores
    later changes to `value`, so a new value must mean a new widget."""
    v = float(cv(q, si_value * si_scale))
    st.number_input(f"{label} [{ul(q)}]", value=v, format=U.fmt(q, units()), disabled=True, help=help,
                    key=f"ro::{pfx}{label}::{v:.9g}")


def _forget_widget(qid: str):
    for sys_ in (U.SI, U.IP):
        ss.pop(f"w::{qid}::{sys_}", None)


HELP = {
    "coil_Q": (
        "Coil capacity from the supplier's cut sheet at the design point below. Together with the design "
        "supply air, return air and SST it sets the coil UA (UA = Q / LMTD), which is then held constant. "
        "At any other return air temperature and SST during the blast, the coil load is "
        "ε · ṁcp · (T_return − SST), with ε = 1 − exp(−UA / ṁcp)."),
    "coil_Ts": (
        "Air temperature leaving the coil at the design point, before fan heat. Used to back out the coil "
        "UA, and as the reference temperature for converting airflow (volume) to mass flow."),
    "coil_SST": (
        "Saturated suction temperature on the coil cut sheet at the design point. Only used to back out "
        "the coil UA; during the blast the SST comes from the balance between the coil, the refrigeration "
        "system and the controls. It does not need to match the refrigeration system's design SST."),
    "driver": (
        "The design cooling power links airflow and the air temperature drop across the coil: "
        "Q = ṁ · cp · (T_return − T_supply). Pick which one you enter; the other is calculated and greyed "
        "out. Switching keeps the current values."),
    "coil_Tr": (
        "Air entering the coil (leaving the pallets) at the design point. A smaller temperature drop "
        "across the coil means more airflow for the same design power; blast coils typically run a "
        "4–8 °F (2–4 K) drop."),
    "coil_flow": (
        "Total cell airflow through the coil, measured at the design supply air temperature. It is split "
        "evenly across the pallet lanes and held constant through the blast (fixed-speed fans)."),
    "calc_flow": "Calculated: design cooling power ÷ (cp × air density × (return − supply)).",
    "calc_Tr": "Calculated: supply air + design cooling power ÷ (cp × air density × airflow).",
    "fan_spec": (
        "Fan power per unit of airflow (default 1 hp per 1000 CFM). Total fan power = this × airflow. "
        "All of it ends up as heat in the cell air: it is added to the air leaving the coil, so the coil "
        "must remove pallet load + fan heat. More airflow improves heat transfer but adds fan heat."),
    "ref_Q": (
        "Refrigeration capacity available to this coil at the refrigeration design SST (compressor cut "
        "sheet or plant allocation). The compressor is treated as fixed displacement: the suction volume "
        "flow V = Q / (ρ_vapor · h_fg) at the design SST is held constant, so capacity at any other SST "
        "scales with vapor density × latent heat. Higher SST → denser suction gas → more capacity."),
    "ref_SST": (
        "Suction temperature at which the design refrigeration power is rated. Independent of the coil "
        "design SST, so cut sheets from different suppliers can be combined."),
    "refrigerant": (
        "Sets how capacity changes with SST, using vapor density and latent heat from CoolProp. Liquid is "
        "assumed to reach the coil saturated at SST (recirculated / low-pressure receiver). CO2 cannot go "
        "below its triple point (−69.8 °F / −56.6 °C)."),
    "SST_max": (
        "Highest suction temperature the plant allows. Under heavy load (right after loading), a "
        "fixed-displacement compressor lets SST rise and its capacity climb; in a central plant, other "
        "compressors hold suction pressure instead. When the balance would push SST above this value, "
        "SST is held here and the coil gets whatever capacity it needs. Set it equal to the design SST "
        "for a stiff central plant, or high to model a dedicated, constrained system."),
    "T_set": (
        "Lowest allowed supply air temperature (mixed air entering the pallets, after fan heat). The "
        "system pulls down until supply air reaches the setpoint; from then on refrigeration capacity is "
        "reduced (SST rises, like an evaporator pressure regulator) to hold supply air there, while the "
        "return air keeps falling toward it as the pallets finish freezing. Airflow stays fixed. With the "
        "boost coil running, the boost is throttled first, then the main coil."),
    "boost_off": (
        "When the boost coil switches off: never, at a set blasting time, or once the warmest pallet "
        "(the last to freeze) reaches a core temperature. Off means its fans and refrigeration stop, so "
        "its airflow leaves the cell and only the main coil's airflow remains. It stays off."),
}


def coil_ref_inputs(p: str, c) -> dict:
    """Coil and Refrigeration system expanders. p prefixes the widget ids ("" main, "b_" boost);
    c supplies the defaults (Cooling or Boost)."""
    with st.expander("Coil (design point)"):
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            coil_Q = qw("number", "Design cooling power", p + "coil_Q", "power", c.coil_Q_design, 1_000,
                        2_000_000, si_scale=1e-3, help=HELP["coil_Q"])
            coil_Ts = qw("number", "Design supply air temp", p + "coil_Ts", "temp", c.coil_T_supply_design,
                         -60, 0, help=HELP["coil_Ts"])
            coil_SST = qw("number", "Design SST", p + "coil_SST", "temp", c.coil_SST_design, -70, 0,
                          help=HELP["coil_SST"])
        with c2:
            driver = st.segmented_control("Set by", DRIVERS, default="Return air",
                                          key=f"w::{p}coil_driver", help=HELP["driver"])
            driver = driver or "Return air"
            # same order either way (return air, then airflow); only which one is greyed out changes
            if driver == "Return air":
                Tr = qw("number", "Design return air temp", p + "coil_Tr", "temp", c.coil_T_return_design,
                        -60, 10, help=HELP["coil_Tr"])
                if Tr <= coil_Ts:
                    st.error("Return air must be warmer than supply air.")
                    Tr = coil_Ts + 0.1
                airflow = coil_airflow(coil_Q, coil_Ts, Tr)
                ss[f"si::{p}coil_flow"] = airflow
                _forget_widget(p + "coil_flow")
                _readonly("Airflow", "flow", airflow, HELP["calc_flow"] + " " + HELP["coil_flow"], pfx=p)
            else:
                tr_slot = st.empty()                    # filled once airflow is known
                airflow = qw("number", "Airflow", p + "coil_flow", "flow", c.airflow, 0.5, 200,
                             help=HELP["coil_flow"])
                Tr = coil_return(coil_Q, coil_Ts, airflow)
                ss[f"si::{p}coil_Tr"] = Tr
                _forget_widget(p + "coil_Tr")
                with tr_slot.container():
                    _readonly("Design return air temp", "temp", Tr, HELP["calc_Tr"] + " " + HELP["coil_Tr"],
                              pfx=p)
            fan_spec = qw("number", "Fan power", p + "fan_spec", "fanspec", c.fan_specific_power, 0, 10_000,
                          si_scale=1e-3, help=HELP["fan_spec"])
            st.caption(f"Fans: {cv('power', fan_spec * airflow / 1000):,.1f} {ul('power')} of heat "
                       f"({fan_spec * airflow / 745.7:,.0f} hp)")
    with st.expander("Refrigeration system (design point)"):
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            ref_Q = qw("number", "Design refrigeration power", p + "ref_Q", "power", c.ref_Q_design, 1_000,
                       2_000_000, si_scale=1e-3, help=HELP["ref_Q"])
            ref_SST = qw("number", "Design SST", p + "ref_SST", "temp", c.ref_SST_design, -70, 0,
                         help=HELP["ref_SST"])
        with c2:
            refr = st.selectbox("Refrigerant", REFRIGERANT_CHOICES,
                                index=REFRIGERANT_CHOICES.index(c.refrigerant), key=f"w::{p}refrigerant",
                                help=HELP["refrigerant"])
    return dict(coil_Q_design=coil_Q, coil_T_supply_design=coil_Ts, coil_SST_design=coil_SST, airflow=airflow,
                fan_specific_power=fan_spec, ref_Q_design=ref_Q, ref_SST_design=ref_SST, refrigerant=refr,
                _Tr=Tr)


def design_caption(v: dict):
    """Coil UA, displacement and the coil-vs-refrigeration meeting point at the design return air."""
    try:
        coil = Coil(v["coil_Q_design"], v["coil_T_supply_design"], v["_Tr"], v["coil_SST_design"])
        ref = RefSystem(v["ref_Q_design"], v["ref_SST_design"], v["refrigerant"])
    except ValueError as e:
        st.error(str(e))
        return
    bal = design_return_balance(coil, ref, coil.design_C_air)
    T = lambda x: f"{cv('temp', x):.1f} {ul('temp')}"
    P = lambda w: f"{cv('power', w / 1000):,.1f} {ul('power')}"
    st.caption(f"Coil UA {cv('ua', coil.UA / 1000):,.{2 if units() == U.SI else 0}f} {ul('ua')} · "
               f"{v['refrigerant']} compressor displacement "
               f"{cv('flow', ref.displacement):,.{3 if units() == U.SI else 0}f} {ul('flow')} · at the design "
               f"return air, coil and refrigeration meet at SST {T(bal.SST)}, supply air {T(bal.T_supply)}, "
               f"{P(bal.Q)} (before fan heat and controls).")


def cooling_tab() -> Cooling:
    c = D.cooling
    v = coil_ref_inputs("", c)
    with st.expander("Controls"):
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            SST_max = qw("number", "Max SST", "SST_max", "temp", c.SST_max, -70, 10, help=HELP["SST_max"])
        with c2:
            T_set = qw("number", "Supply air setpoint", "T_sup_set", "temp", c.T_supply_setpoint, -70, 0,
                       help=HELP["T_set"])
    design_caption(v)
    return Cooling(**{k: x for k, x in v.items() if not k.startswith("_")}, SST_max=SST_max,
                   T_supply_setpoint=T_set)


BOOST_OFF_LABEL = {"never": "Never", "time": "At a set time", "core": "When the warmest pallet core reaches"}


def boost_tab() -> Boost:
    b = D.boost
    st.caption("A second coil with its own fans and refrigeration system, in parallel with the main coil. "
               "Return air splits between the coils and their outlets mix into the supply air. Every solve "
               "runs twice, without and with the boost, to show how much it shortens the blast.")
    v = coil_ref_inputs("b_", b)
    with st.expander("Controls"):
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            SST_max = qw("number", "Max SST", "b_SST_max", "temp", b.SST_max, -70, 10, help=HELP["SST_max"])
        with c2:
            mode = st.selectbox("Boost shutoff", list(BOOST_OFF_LABEL), key="w::b_off_mode",
                                index=list(BOOST_OFF_LABEL).index(b.off_mode),
                                format_func=lambda m: BOOST_OFF_LABEL[m], help=HELP["boost_off"])
            off_time = qw("slider", "Shut off at", "b_off_time", None, b.off_time_h, 0.5, 72, unit_label="h",
                          step=0.5, format="%.1f") if mode == "time" else ss.get("si::b_off_time", b.off_time_h)
            off_T = qw("slider", "Shut off when the warmest core reaches", "b_off_T", "temp", b.off_core_T,
                       -40, 10) if mode == "core" else ss.get("si::b_off_T", b.off_core_T)
        st.caption("The supply air setpoint (Cooling tab → Controls) applies to the mixed supply air; "
                   "the boost is throttled first, then the main coil.")
    when = {"never": "runs for the whole blast", "time": f"shuts off at {off_time:.1f} h",
            "core": f"shuts off when the warmest pallet core reaches {cv('temp', off_T):.1f} {ul('temp')}"}[mode]
    st.caption(f"**Boost {when}.**")
    design_caption(v)
    return Boost(**{k: x for k, x in v.items() if not k.startswith("_")}, SST_max=SST_max, off_mode=mode,
                 off_time_h=off_time, off_core_T=off_T)


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
        st.selectbox("1-D geometry", GEOMETRIES, key="w::geom",
                     help="auto: plate if the block's smallest dimension is < 0.5× the middle one, "
                          "otherwise an equivalent sphere (R = 3V/A).")
        t_max = qw("slider", "Max simulated time", "tmax", None, s.t_max_hours, 12, 500, unit_label="h",
                   step=12.0, format="%.0f")
        dt_max = qw("number", "Max time step", "dtmax", None, s.dt_max, 1, 600, unit_label="s", step=1.0,
                    format="%.0f", help="The solver uses the smaller of this and the stability limit.")
        rec = qw("number", "Output interval", "rec", None, s.record_every_s, 60, 3600, si_scale=1 / 60,
                 unit_label="min", step=1.0, format="%.0f", help="Resolution of the time slider and plots.")
    return SimSettings(s.T_start, s.T_target, n_nodes, dt_max, t_max, rec), htc, reduction


# ------------------------------------------------------- config file ----
CONFIG_OPTIONS = {          # allowed values of selection inputs in a loaded file
    "units": [U.SI, U.IP], "w::product": list(FOODS) + [CUSTOM], "custom_base": list(FOODS),
    "w::pkg": list(PACKAGING_PRESETS), "w::geom": GEOMETRIES, "w::inlet": GUI_FACES,
    "w::refrigerant": REFRIGERANT_CHOICES, "w::b_refrigerant": REFRIGERANT_CHOICES,
    "w::coil_driver": DRIVERS + [None], "w::b_coil_driver": DRIVERS + [None],
    "w::b_off_mode": list(BOOST_OFF_LABEL), "w::use_sp": [True, False],
}


def _upload_key() -> str:
    return f"cfg_upload_{ss.get('cfg_n', 0)}"


def _load_config():
    """Uploader callback: runs before the script, so every widget is rebuilt from the loaded state."""
    f = ss.get(_upload_key())
    if f is None:
        return
    try:
        notes = config_file.apply(ss, config_file.parse(f.getvalue()), CONFIG_OPTIONS)
    except ValueError as e:
        ss["cfg_msg"] = ("error", f"Could not load {f.name}: {e}")
    else:
        ss["cfg_msg"] = ("warning" if notes else "success",
                         f"Loaded {f.name}." + ("  \n" + "  \n".join(f"• {n}" for n in notes) if notes else ""))
    ss["cfg_n"] = ss.get("cfg_n", 0) + 1            # fresh, empty uploader: the same file can be loaded again


def config_section(scenario: Scenario):
    with st.expander("Save / load configuration"):
        st.download_button("Save configuration", config_file.dumps(ss, scenario.to_dict()),
                           f"blast_config_{datetime.now():%Y-%m-%d_%H%M}.json", "application/json",
                           icon=":material/download:", width="stretch",
                           help="Downloads every input as a JSON file (values in SI, whatever units are "
                                "shown). Load it again below to pick up where you left off.")
        st.file_uploader("Load configuration", type="json", key=_upload_key(), on_change=_load_config,
                         help="Replaces all inputs with the ones in a saved configuration file. Results "
                              "stay until you run the solver again.")
        kind, msg = ss.pop("cfg_msg", (None, None))
        if kind:
            getattr(st, kind)(msg)


# ---------------------------------------------------------------- layout ----
LOGO = Path(__file__).parent / "resources" / "rebound_logo_2C_A big.png"
h_title, h_logo = st.columns([22, 3], vertical_alignment="center")      # logo ~60% of a 1/5 column
h_title.title("Blast Freezing Simulator")
if LOGO.exists():
    h_logo.image(str(LOGO), width="stretch")

left, right = st.columns([5, 6], gap="large")

with left:
    st.subheader("Inputs")
    t_pal, t_cell, t_cool, t_model, t_boost = st.tabs(["Pallet", "Cell", "Cooling", "Model", "Boost"])
    with t_model:
        sim, htc, reduction = model_tab()
    with t_pal:
        product, pkg, load, T0, Tt = pallet_tab()
    with t_cool:
        cooling = cooling_tab()
    with t_boost:
        boost = boost_tab()
    with t_cell:
        cell = cell_tab(cooling.airflow, boost.airflow)
    sim.T_start, sim.T_target = T0, Tt
    air = Air(htc, reduction, D.air.cp)
    scenario = Scenario(product, pkg, load, air, cell, sim, cooling, boost)

    if st.button("Run solver", type="primary", width="stretch"):
        try:
            with st.spinner("Solving twice: without and with the boost coil…"):
                off, on = simulate_pair(scenario)
            ss["res"], ss["res_scn"] = {"off": off, "on": on}, scenario.to_dict()
        except ValueError as e:
            st.error(str(e))
    config_section(scenario)

res = ss.get("res")
if res is not None and (not isinstance(res, dict) or "flow_m3s" not in res["on"].cell
                        or Scenario.from_dict(ss["res_scn"]).cell.inlet_face not in GUI_FACES):
    res = None                                    # solution from an older app version
shown = Scenario.from_dict(ss["res_scn"]) if res is not None else scenario
stale = res is not None and ss["res_scn"] != scenario.to_dict()
dec_flow = 1 if units() == U.SI else 0
fmt_flow = lambda q: f"{cv('flow', q):,.{dec_flow}f} {ul('flow')}"
fmt_h = lambda t: "—" if t is None else f"{t:.1f} h"

with right:
    st.subheader("Cell")
    if stale:
        st.info("Inputs changed since the last solve. Showing the last solution; click **Run solver** to update.")
    c = shown.cell
    if res is None:
        n_ser = {"bottom": c.n_high, "top": c.n_high, "left": c.n_wide, "right": c.n_wide}[c.inlet_face]
        T_frames, times = [[shown.sim.T_start] * n_ser], [0.0]
        T_sup, T_ret, SST, SST_b, on_frames = [None], [None], [None], [None], [True]
        flows = [fmt_flow(shown.cooling.airflow + shown.boost.airflow)]
        try:                                  # load-in balance (boost running) as a preview
            b0 = Cell(shown, boost=True).balance
            T_sup, T_ret, SST = [b0.T_supply], [b0.T_return], [b0.SST]
            SST_b = [b0.coils[1].SST] if len(b0.coils) > 1 else [None]
        except ValueError:
            pass
    else:
        view = st.segmented_control("Show", ["With boost", "Without boost"], default="With boost",
                                    key="w::view_run", label_visibility="collapsed") or "With boost"
        r = res["on"] if view == "With boost" else res["off"]
        P = r.pallets.pivot(index="time_h", columns="position", values="T_core").sort_index()
        T_frames, times = P.values, list(P.index)
        Cw = r.cell.set_index("time_h").loc[P.index]
        T_sup, T_ret, SST = Cw["T_supply"].tolist(), Cw["T_return"].tolist(), Cw["SST"].tolist()
        SST_b, on_frames = Cw["SST_boost"].tolist(), Cw["boost_on"].tolist()
        flows = [fmt_flow(q) for q in Cw["flow_m3s"]]
    st.plotly_chart(
        cell_figure(c.n_deep, c.n_wide, c.n_high, c.inlet_face, T_frames, times, T_ret,
                    shown.sim.T_start, shown.sim.T_target, cv, ul, T_sup, flows, height=560,
                    SST_frames=SST, boost_frames=on_frames, SST_boost_frames=SST_b),
        width="stretch", config={"displayModeBar": False}, key="cell_fig")
    if res is None:
        st.caption("Run the solver, then drag the time slider (or press ▶) to step through the blast.")
    else:
        s_off, s_on = summarize(res["off"], shown), summarize(res["on"], shown)
        t_off, t_on = s_off["cell_freeze_time_h"], s_on["cell_freeze_time_h"]
        m = st.columns(4)
        m[0].metric("Cell frozen with boost", fmt_h(t_on),
                    delta=None if None in (t_on, t_off) else f"{t_on - t_off:+.1f} h vs no boost",
                    delta_color="inverse", help="Warmest pallet core reaches the target.")
        m[1].metric("Cell frozen without boost", fmt_h(t_off))
        m[2].metric("Peak coil load (with boost)", f"{cv('power', s_on['peak_coil_kW']):,.0f} {ul('power')}",
                    help="Main + boost coil load, including fan heat.")
        m[3].metric("Boost switched off", "Never" if s_on["boost_off_h"] is None else fmt_h(s_on["boost_off_h"]))
        for name, sm in (("without", s_off), ("with", s_on)):
            if not sm["all_frozen"]:
                st.warning(f"Run {name} boost: not every pallet reached the target within {sm['sim_end_h']:.0f} h "
                           "(Model tab → Max simulated time).")

st.divider()
t_prod, t_air, t_ref, t_cool = st.tabs(["Product temperature", "Air temperature", "Ref temperature", "Cooling"])
if res is None:
    for t in (t_prod, t_air, t_ref, t_cool):
        t.info("Run the solver to see the time series.")
else:
    off, on = res["off"], res["on"]
    with t_prod:
        st.plotly_chart(product_temperature_figure(off, on, cv, ul), width="stretch")
        st.caption("Dashed = without boost, solid = with boost. Warmest pallet = last to reach the target; "
                   "surface node = its outermost finite volume.")
    with t_air:
        st.plotly_chart(air_temperature_figure(off, on, cv, ul), width="stretch")
        st.caption("Dashed = without boost, solid = with boost. Supply = mixed air leaving the coils, after fan "
                   "heat; return = air leaving the pallets, entering the coils.")
    with t_ref:
        st.plotly_chart(ref_temperature_figure(off, on, cv, ul), width="stretch")
        st.caption("Dashed = without boost, solid = with boost. Saturated suction temperature of each coil; "
                   "the boost SST ends when the boost shuts off.")
    with t_cool:
        st.plotly_chart(cooling_figure(off, on, cv, ul, design_kW=shown.cooling.ref_Q_design / 1000),
                        width="stretch")
        st.caption("Areas: with boost, stacked to the total coil load (refrigeration load). Fan heat at the "
                   "bottom; the rest (the pallet load) split between the main and boost coils in proportion "
                   "to their coil loads. Dashed line: total coil load without boost. Raw per-coil loads are in "
                   "the cell history CSV.")

    with st.expander("Model details & data"):
        info = on.info
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
            ("HTC applied", f"{cv('htc', info['h']):.2f} {ul('htc')}"),
            ("Effective HTC (ε-NTU), no boost", f"{cv('htc', off.info['h_eff']):.2f} {ul('htc')}"),
            ("Airflow per lane, no boost", f"{cv('flow', info['flow_lane_m3s']):,.2f} {ul('flow')}"),
            ("Nodes / time step (boost run)",
             f"{info['n_nodes']} / {info['dt_s']:.1f} s ({info['n_steps']:,} steps)"),
            ("Energy balance error", f"{max(info['energy_balance_error'], off.info['energy_balance_error']):.1e}"),
        ], columns=["Item", "Value"]), hide_index=True, width="stretch")
        if info.get("cooling_mode") == "coil":
            Co, Cn = off.cell, on.cell
            rng = lambda x: f"{cv('temp', x.max()):.1f} → {cv('temp', x.min()):.1f} {ul('temp')}"
            rows = [
                ("Main coil UA / LMTD", f"{cv('ua', info['coil_UA_W_K'] / 1000):,.2f} {ul('ua')} / "
                                        f"{info['coil_LMTD']:.2f} K"),
                ("Main coil effectiveness", f"{info['coil_eps']:.3f}"),
                ("Main compressor displacement", f"{cv('flow', info['ref_displacement_m3s']):,.3f} {ul('flow')}"),
            ]
            if "boost_UA_W_K" in info:
                rows += [
                    ("Boost coil UA / LMTD", f"{cv('ua', info['boost_UA_W_K'] / 1000):,.2f} {ul('ua')} / "
                                             f"{info['boost_LMTD']:.2f} K"),
                    ("Boost coil effectiveness", f"{info['boost_eps']:.3f}"),
                    ("Boost compressor displacement",
                     f"{cv('flow', info['boost_displacement_m3s']):,.3f} {ul('flow')}"),
                    ("Boost SST range", rng(Cn.SST_boost.dropna()) if Cn.SST_boost.notna().any() else "—"),
                ]
            rows += [
                ("Main SST range: no boost / boost", f"{rng(Co.SST)} / {rng(Cn.SST)}"),
                ("Supply air range: no boost / boost", f"{rng(Co.T_supply)} / {rng(Cn.T_supply)}"),
                ("Records at a control limit (boost run)",
                 f"main {(Cn.limit != '').sum()}, boost {(Cn.limit_boost != '').sum()} of {len(Cn)}"),
            ]
            st.dataframe(pd.DataFrame(rows, columns=["Cooling", "Value"]), hide_index=True, width="stretch")
        rnd = lambda t: None if t is None else round(t, 2)
        st.dataframe(pd.DataFrame({
            "Series position": [f"P{i + 1}" for i in range(len(on.freeze_times_h))],
            "Freeze time, no boost [h]": [rnd(t) for t in off.freeze_times_h],
            "Freeze time, boost [h]": [rnd(t) for t in on.freeze_times_h]}), hide_index=True)
        both = lambda key: pd.concat([getattr(off, key).assign(run="no boost"),
                                      getattr(on, key).assign(run="boost")]).to_csv(index=False)
        d1, d2, d3 = st.columns(3)
        d1.download_button("Pallet history CSV (SI)", both("pallets"), "pallets.csv", "text/csv")
        d2.download_button("Cell history CSV (SI)", both("cell"), "cell.csv", "text/csv")
        d3.download_button("Solved scenario JSON", json.dumps(shown.to_dict(), indent=2), "scenario.json",
                           "application/json", help="Model inputs of this solution, for scripting. To reopen "
                           "the inputs in the app, use Save / load configuration instead.")
