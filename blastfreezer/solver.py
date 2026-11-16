"""Run a blast: step the cell until every pallet core reaches the target."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .cell import Cell
from .config import Scenario

CELL_COLUMNS = ["time_h", "T_supply", "T_return", "Q_cell_kW", "Q_coil_kW", "Q_main_kW", "Q_boost_kW", "fan_kW",
                "SST", "SST_boost", "T_coil_out", "T_boost_out", "flow_m3s", "boost_on", "limit", "limit_boost",
                "T_core_warmest", "n_frozen_positions"]
PALLET_COLUMNS = ["time_h", "position", "T_core", "T_surface", "T_mean", "T_air_in", "T_air_out",
                  "Q_kW", "frozen_fraction", "front_depth_mm"]


@dataclass
class Result:
    cell: pd.DataFrame            # cell totals vs time
    pallets: pd.DataFrame         # long format: one row per (time, series position)
    profiles: dict                # position -> DataFrame(index time_h, columns distance from core [mm])
    freeze_times_h: list          # per series position (None if not reached)
    info: dict


def simulate(s: Scenario, boost: bool = False) -> Result:
    """One blast. boost=True runs the boost coil from s.boost alongside the main coil."""
    cell = Cell(s, boost=boost)
    lane = cell.lanes[0]
    dt = min(s.sim.dt_max, cell.dt_stable)
    t_max = s.sim.t_max_hours * 3600.0
    every = max(1, int(round(s.sim.record_every_s / dt)))
    x_mm = np.round(lane.pallets[0].grid.x_centers * 1000.0, 2)

    cell_rows, pal_rows = [], []
    prof = {i + 1: [] for i in range(len(lane.pallets))}
    times = []

    def record(initial=False):
        th = cell.t / 3600.0
        times.append(th)
        n_frozen = sum(p.t_frozen is not None for p in lane.pallets)
        bal = cell.balance        # balance used for the step just taken (T_supply = mixed supply air)
        units = {c.name: c for c in bal.coils}
        main, bst = units.get("main"), units.get("boost")
        nan = float("nan")
        flow = sum(c.C_air for c in bal.coils) / s.air.cp if bal.coils else cell.m_dot_total
        cell_rows.append((th, bal.T_supply, cell.T_air_out, cell.Q / 1000.0, bal.Q / 1000.0,
                          (main.Q if main else bal.Q) / 1000.0, (bst.Q if bst else 0.0) / 1000.0,
                          bal.Q_fan / 1000.0, bal.SST, bst.SST if bst else nan, bal.T_coil_out,
                          bst.T_out if bst else nan, flow * s.cooling.airflow / cell.m_main,
                          bst is not None, bal.limited, bst.limited if bst else "",
                          cell.T_core_warmest, n_frozen))
        for i, p in enumerate(lane.pallets):
            Q = p.Q_now if initial else p.Q
            T_out = p.T_air_out if initial else p.T_air_out_step
            pal_rows.append((th, i + 1, p.T_core, p.T_surface, p.T_mean, p.T_air_in, T_out,
                             Q / 1000.0, p.frozen_fraction, p.front_depth * 1000.0))
            prof[i + 1].append(p.T_nodes.copy())

    record(initial=True)
    i = 0
    while cell.t < t_max and not cell.all_frozen:
        cell.step(dt)
        i += 1
        if i % every == 0 or cell.all_frozen:
            record()

    profiles = {k: pd.DataFrame(np.array(v), index=pd.Index(times, name="time_h"), columns=x_mm)
                for k, v in prof.items()}
    p0, path, m = lane.pallets[0], cell.path, lane.pallets[0].mat
    info = {
        # geometry & material (identical for every pallet)
        "geometry_note": p0.body.note, "shape": p0.body.shape, "R_mm": p0.body.R * 1000.0,
        "n_layers": p0.body.n_layers, "area_m2": p0.body.area_pallet,
        "envelope_volume_m3": p0.body.volume_pallet, "rho_bulk": m.rho_bulk,
        "phi_product": m.phi_product, "phi_void": m.phi_void, "phi_wall": m.phi_wall,
        "phi_gap": m.phi_gap, "R_surface": m.R_surface,
        "k_eff_unfrozen": m.k_unfrozen, "k_eff_frozen": m.k_frozen, "L_vol": m.L_vol,
        # air path
        "flow_axis": path.axis, "inlet": path.inlet, "outlet": path.outlet,
        "n_series": path.n_series, "n_lanes": path.n_lanes, "n_pallets": s.cell.n_pallets,
        "spacer_alignment": path.spacer_alignment, "h": path.h, "h_eff": p0.h_eff,
        "NTU": p0.NTU, "eps": p0.eps, "flow_total_m3s": path.flow_total,
        "m_dot_total": path.m_dot_total, "m_dot_lane": path.m_dot_lane,
        "flow_lane_m3s": path.flow_total / path.n_lanes,
        # cooling
        **_cooling_info(cell, s),
        # boost
        "boost": cell.boost_enabled,
        "boost_off_h": None if cell.t_boost_off is None else cell.t_boost_off / 3600.0,
        # numerics
        "dt_s": dt, "dt_stable_s": cell.dt_stable, "n_steps": i, "n_nodes": p0.grid.n,
        "dx_mm": p0.grid.dx * 1000.0,
        "energy_removed_kWh": sum(p.E_removed for p in lane.pallets) * path.n_lanes / 3.6e6,
        "energy_balance_error": max(abs(p.energy_balance_error) for p in lane.pallets),
    }
    freeze = [None if p.t_frozen is None else p.t_frozen / 3600.0 for p in lane.pallets]
    return Result(pd.DataFrame(cell_rows, columns=CELL_COLUMNS),
                  pd.DataFrame(pal_rows, columns=PALLET_COLUMNS), profiles, freeze, info)


def _cooling_info(cell: Cell, s: Scenario) -> dict:
    if cell.fixed:
        return {"cooling_mode": "fixed supply"}
    from .refrigeration import design_return_balance
    coil, ref = cell.coil, cell.ref
    from .pallet import air_density
    bal = design_return_balance(coil, ref, cell.main_unit.C_air)
    return {
        "cooling_mode": "coil",
        "coil_UA_W_K": coil.UA, "coil_LMTD": coil.LMTD,
        "coil_design_flow_m3s": coil.design_C_air / s.air.cp / air_density(coil.T_supply_design),
        "coil_eps": coil.effectiveness(cell.main_unit.C_air),
        "ref_displacement_m3s": ref.displacement,
        "ref_capacity_at_coil_SST_kW": float(ref.capacity(coil.SST_design)) / 1000.0,
        "design_return_SST": bal.SST, "design_return_T_supply": bal.T_supply,
        "design_return_Q_kW": bal.Q / 1000.0,
        **({"boost_UA_W_K": cell.boost_coil.UA, "boost_LMTD": cell.boost_coil.LMTD,
            "boost_eps": cell.boost_unit.eps, "boost_displacement_m3s": cell.boost_ref.displacement}
           if cell.boost_enabled else {}),
    }


def summarize(r: Result, s: Scenario) -> dict:
    ft = r.freeze_times_h
    done = [t for t in ft if t is not None]
    return {
        "all_frozen": len(done) == len(ft),
        "cell_freeze_time_h": max(done) if len(done) == len(ft) else None,
        "first_freeze_time_h": min(done) if done else None,
        "peak_cell_kW": float(r.cell["Q_cell_kW"].max()),
        "peak_coil_kW": float(r.cell["Q_coil_kW"].max()),
        "fan_kW": float(r.cell["fan_kW"].iloc[-1]),
        "energy_removed_kWh": r.info["energy_removed_kWh"],
        "sim_end_h": float(r.cell["time_h"].iloc[-1]),
        "boost_off_h": r.info.get("boost_off_h"),
        "peak_boost_kW": float(r.cell["Q_boost_kW"].max()),
    }


def simulate_pair(s: Scenario) -> tuple[Result, Result]:
    """(without boost, with boost): the same scenario run twice to show what the boost buys."""
    return simulate(s, boost=False), simulate(s, boost=True)
