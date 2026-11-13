"""Single-pallet freezing model: 1-D finite-volume enthalpy method, explicit Euler.

Grid: cell-centred, fixed. Cell 0 is the core (symmetry / zero flux), cell
N-1 is at the surface. Conductance between neighbours uses the series
(harmonic) combination of the two half-cells; the surface cell connects to
the air through its half-cell plus the effective air-side film.

Air side: the pallet sees air at T_in with mass flow m_dot. With a uniform
surface temperature, the heat removed is
    Q = eps * m_dot * cp * (T_surface - T_in),   eps = 1 - exp(-NTU),
    NTU = h * A / (m_dot * cp),
which is the same as a film coefficient h_eff = eps * m_dot * cp / A applied
against T_in. Air leaving the pallet is T_out = T_in + Q / (m_dot * cp).

The freezing front is not a grid feature. Its position is the depth from the
surface that holds the current frozen volume, sum(f_i * V_i).
"""

from __future__ import annotations

import math

import numpy as np

from .config import Scenario
from .geometry import Body, Grid, make_grid, representative_body
from .materials import Material, build_material

R_AIR = 287.05          # J/kg-K
P_ATM = 101_325.0       # Pa
STABILITY_SAFETY = 0.9


class Pallet:
    def __init__(self, s: Scenario):
        self.s = s
        self.body: Body = representative_body(s.load)
        self.grid: Grid = make_grid(self.body.shape, self.body.R, s.sim.n_nodes)
        self.mat: Material = build_material(s.product, s.packaging, s.load.mass, self.body.volume_pallet)
        # unit-body -> whole-pallet multiplier (same for areas and volumes by construction)
        self.scale = self.body.area_pallet / self.grid.A_surface

        self.H = np.full(self.grid.n, float(self.mat.enthalpy(s.sim.T_start)))
        self.H0_total = self._total_enthalpy()
        self.t = 0.0
        self.E_removed = 0.0      # J, cumulative heat removed from the pallet
        self.Q = 0.0              # W, last step
        self.T_air_in = s.air.T_in
        self.set_air(s.air.T_in, s.air.flow_per_pallet)
        self.dt_stable = self._stable_dt()

    # ---- setup ------------------------------------------------------------
    def set_air(self, T_in: float, flow_m3s: float):
        """Set inlet air conditions (the cell model will call this every step later)."""
        a = self.s.air
        self.T_air_in = T_in
        rho = P_ATM / (R_AIR * (T_in + 273.15))
        self.m_dot = rho * flow_m3s
        self.C_air = self.m_dot * a.cp                              # W/K
        self.NTU = a.htc * self.body.area_pallet / self.C_air if self.C_air > 0 else math.inf
        self.eps = 1.0 - math.exp(-self.NTU)
        self.h_eff = self.eps * self.C_air / self.body.area_pallet if self.C_air > 0 else a.htc

    def _stable_dt(self) -> float:
        """Explicit Euler limit: dt <= V_i * C_min / sum(G_i) for every cell."""
        g, k = self.grid, self.mat.k_max
        G = np.zeros(g.n)
        G_int = g.A_faces[1:-1] * k / g.dx
        G[:-1] += G_int
        G[1:] += G_int
        G[-1] += g.A_surface / (g.dx / (2 * k) + 1.0 / self.h_eff)
        return STABILITY_SAFETY * float(np.min(g.V_cells * self.mat.C_min / G))

    # ---- time stepping ----------------------------------------------------
    def step(self, dt: float) -> float:
        """Advance one explicit Euler step. Returns heat removed from the pallet [W]."""
        g, mat = self.grid, self.mat
        T, f = mat.state(self.H)
        k = mat.k(f)

        R_int = g.dx / (2 * k[:-1]) + g.dx / (2 * k[1:])
        q_int = g.A_faces[1:-1] / R_int * (T[:-1] - T[1:])              # W outward, per unit body
        G_s = g.A_surface / (g.dx / (2 * k[-1]) + 1.0 / self.h_eff)
        q_s = G_s * (T[-1] - self.T_air_in)                               # W to air, per unit body

        net = np.zeros(g.n)
        net[:-1] -= q_int
        net[1:] += q_int
        net[-1] -= q_s
        self.H = self.H + dt * net / g.V_cells

        self.Q = q_s * self.scale
        self._q_s_unit, self._k_surf, self._T_surf_cell = q_s, k[-1], T[-1]
        self.E_removed += self.Q * dt
        self.t += dt
        return self.Q

    # ---- derived quantities ----------------------------------------------
    def _total_enthalpy(self) -> float:
        return float(np.sum(self.H * self.grid.V_cells) * self.scale)

    @property
    def T_nodes(self) -> np.ndarray:
        return self.mat.state(self.H)[0]

    @property
    def frozen_nodes(self) -> np.ndarray:
        return self.mat.state(self.H)[1]

    @property
    def T_core(self) -> float:
        return float(self.T_nodes[0])

    @property
    def T_surface(self) -> float:
        T, f = self.mat.state(self.H)
        q = self._current_surface_flux(T, f)
        return float(T[-1] - q / self.grid.A_surface * self.grid.dx / (2 * self.mat.k(f[-1])))

    def _current_surface_flux(self, T, f) -> float:
        k = self.mat.k(f[-1])
        G_s = self.grid.A_surface / (self.grid.dx / (2 * k) + 1.0 / self.h_eff)
        return G_s * (T[-1] - self.T_air_in)

    @property
    def Q_now(self) -> float:
        """Heat removal rate at the current state [W]."""
        T, f = self.mat.state(self.H)
        return self._current_surface_flux(T, f) * self.scale

    @property
    def T_mean(self) -> float:
        return float(np.sum(self.T_nodes * self.grid.V_cells) / self.grid.V_total)

    @property
    def frozen_fraction(self) -> float:
        return float(np.sum(self.frozen_nodes * self.grid.V_cells) / self.grid.V_total)

    @property
    def front_depth(self) -> float:
        """Depth of the freezing front from the surface [m]."""
        g = self.grid
        V_frozen = float(np.sum(self.frozen_nodes * g.V_cells))
        r_front = g.radius_of_volume(g.cumulative_volume(g.R) - V_frozen)
        return g.R - r_front

    @property
    def T_air_out(self) -> float:
        return self.T_air_in + self.Q_now / self.C_air if self.C_air > 0 else self.T_air_in

    @property
    def energy_balance_error(self) -> float:
        """(enthalpy lost - heat removed) / heat removed; should be ~1e-12."""
        lost = self.H0_total - self._total_enthalpy()
        return (lost - self.E_removed) / max(abs(self.E_removed), 1e-30)
