"""Run a single-pallet blast until the core reaches the target temperature."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Scenario
from .pallet import Pallet

HISTORY_COLUMNS = [
    "time_h", "T_core", "T_surface", "T_mean", "T_air_in", "T_air_out",
    "Q_pallet_kW", "Q_cell_kW", "frozen_fraction", "front_depth_mm",
]


@dataclass
class Result:
    history: pd.DataFrame        # one row per recorded time
    profiles: pd.DataFrame       # node temperatures: index time_h, columns = distance from core [mm]
    info: dict                   # geometry, material, numerics


def _record(p: Pallet, n_pallets: int) -> tuple:
    Q = p.Q_now
    return (p.t / 3600.0, p.T_core, p.T_surface, p.T_mean, p.T_air_in, p.T_air_out,
            Q / 1000.0, Q * n_pallets / 1000.0, p.frozen_fraction, p.front_depth * 1000.0)


def simulate(s: Scenario) -> Result:
    p = Pallet(s)
    dt = min(s.sim.dt_max, p.dt_stable)
    t_max = s.sim.t_max_hours * 3600.0
    every = max(1, int(round(s.sim.record_every_s / dt)))
    n_pal = s.cell.n_pallets

    rows, prof, times = [_record(p, n_pal)], [p.T_nodes.copy()], [0.0]
    i = 0
    while p.t < t_max and p.T_core > s.sim.T_target:
        p.step(dt)
        i += 1
        if i % every == 0 or p.T_core <= s.sim.T_target:
            rows.append(_record(p, n_pal))
            prof.append(p.T_nodes.copy())
            times.append(p.t / 3600.0)

    history = pd.DataFrame(rows, columns=HISTORY_COLUMNS)
    profiles = pd.DataFrame(np.array(prof), index=pd.Index(times, name="time_h"),
                            columns=np.round(p.grid.x_centers * 1000.0, 2))
    m = p.mat
    info = {
        "geometry_note": p.body.note,
        "shape": p.body.shape,
        "R_mm": p.body.R * 1000.0,
        "n_layers": p.body.n_layers,
        "area_m2": p.body.area_pallet,
        "envelope_volume_m3": p.body.volume_pallet,
        "rho_bulk": m.rho_bulk,
        "phi_product": m.phi_product, "phi_void": m.phi_void,
        "phi_wall": m.phi_wall, "phi_gap": m.phi_gap,
        "k_eff_unfrozen": m.k_unfrozen, "k_eff_frozen": m.k_frozen,
        "C_unfrozen": m.C_unfrozen, "C_frozen": m.C_frozen, "L_vol": m.L_vol,
        "h": s.air.htc, "h_eff": p.h_eff, "NTU": p.NTU, "eps": p.eps,
        "m_dot_air": p.m_dot,
        "dt_s": dt, "dt_stable_s": p.dt_stable, "n_steps": i,
        "n_nodes": p.grid.n, "dx_mm": p.grid.dx * 1000.0,
        "energy_removed_kWh": p.E_removed / 3.6e6,
        "energy_balance_error": p.energy_balance_error,
    }
    return Result(history, profiles, info)


def summarize(r: Result, s: Scenario) -> dict:
    h = r.history
    reached = bool(h["T_core"].iloc[-1] <= s.sim.T_target)
    return {
        "reached_target": reached,
        "freeze_time_h": float(h["time_h"].iloc[-1]),
        "peak_pallet_kW": float(h["Q_pallet_kW"].max()),
        "peak_cell_kW": float(h["Q_cell_kW"].max()),
        "energy_removed_kWh": r.info["energy_removed_kWh"],
        "frozen_fraction_end": float(h["frozen_fraction"].iloc[-1]),
    }
