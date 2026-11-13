"""Transient solver for a blast freezer cell.

PLACEHOLDER PHYSICS: this is a port of the legacy lumped model in
`pallet.py`. Each pallet is a single node with a series resistance
(air film + internal conduction) and an effective cp(T). All pallets in the
cell are identical and loaded at once. Refrigeration capacity caps the total
heat removal rate.

The interface (`simulate(scenario) -> pandas.DataFrame`) is what the app relies
on. When we build the real physics (1-D/2-D conduction, enthalpy method,
staggered loading, air-side energy balance), keep this signature and the
output columns, and the UI won't need to change.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import Scenario
from .properties import EffectiveCp

OUTPUT_COLUMNS = [
    "time_h",        # elapsed time [h]
    "T_product",     # pallet temperature [degC]
    "Q_pallet_kW",   # heat removed per pallet [kW]
    "Q_cell_kW",     # heat removed, whole cell [kW]
    "Q_demand_kW",   # heat the product could give up if capacity were unlimited [kW]
    "cp_eff",        # effective cp [J/kg-K]
]


def _thermal_resistance(s: Scenario) -> float:
    """Air film + internal conduction resistance for one pallet [K/W]."""
    p, c = s.pallet, s.cell
    R_air = 1.0 / (p.exposed_area * c.htc)
    L = p.critical_length
    # Legacy cylindrical-style conduction term
    R_cond = (L / 2.0 - L / 8.0) / (2.0 * math.pi * p.height * p.k_effective)
    return R_air + R_cond


def simulate(s: Scenario) -> pd.DataFrame:
    cp_eff = EffectiveCp(s.product)
    R = _thermal_resistance(s)
    m = s.pallet.mass
    n = s.cell.n_pallets
    cap = s.cell.refrigeration_capacity
    dt = s.sim.dt
    n_steps = int(s.sim.t_max_hours * 3600.0 / dt)

    T = s.sim.T_start
    rows = []
    for i in range(n_steps + 1):
        Q_demand = (T - s.cell.T_air) / R            # W per pallet
        Q = Q_demand
        if cap > 0 and Q * n > cap:
            Q = cap / n
        cp = float(cp_eff(T))
        rows.append((i * dt / 3600.0, T, Q / 1000.0, Q * n / 1000.0, Q_demand * n / 1000.0, cp))
        if T <= s.sim.T_target:
            break
        T = T - Q * dt / (m * cp)

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def summarize(df: pd.DataFrame, s: Scenario) -> dict:
    reached = df["T_product"].iloc[-1] <= s.sim.T_target
    return {
        "reached_target": bool(reached),
        "blast_time_h": float(df["time_h"].iloc[-1]),
        "peak_load_kW": float(df["Q_cell_kW"].max()),
        "avg_load_kW": float(df["Q_cell_kW"].mean()),
        "energy_removed_kWh": float(np.trapezoid(df["Q_cell_kW"], df["time_h"])),
    }
