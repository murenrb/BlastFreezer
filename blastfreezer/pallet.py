"""Single-pallet freezing model: 1-D finite-volume enthalpy method, explicit Euler.

Grid: cell-centred, fixed. Cell 0 is the core (symmetry / zero flux), cell
N-1 is at the surface. Conductance between neighbours uses the series
(harmonic) combination of the two half-cells. The surface cell connects to
the air through its half-cell, the packaging surface resistance (boxes), and
the effective air film.

Air side: the pallet sees air at T_in with mass flow m_dot. With a uniform
surface temperature, the heat removed is
    Q = eps * m_dot * cp * (T_surface - T_in),   eps = 1 - exp(-NTU),
    NTU = h * A / (m_dot * cp),
which equals a film coefficient h_eff = eps * m_dot * cp / A against T_in.
Air leaving the pallet is T_out = T_in + Q / (m_dot * cp).

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


def air_density(T_c: float) -> float:
    return P_ATM / (R_AIR * (T_c + 273.15))


class Pallet:
    def __init__(self, s: Scenario, name: str = "pallet", T_in: float | None = None,
                 m_dot: float | None = None, h: float | None = None):
        """Standalone defaults: inlet = cell inlet, all of the cell airflow, h = free-flow HTC."""
        self.s = s
        self.name = name
        self.body: Body = representative_body(s.load)
        self.grid: Grid = make_grid(self.body.shape, self.body.R, s.sim.n_nodes)
        self.mat: Material = build_material(s.product, s.packaging, s.load.mass,
                                            self.body.volume_pallet, self.body.R)
        # unit-body -> whole-pallet multiplier (same for areas and volumes by construction)
        self.scale = self.body.area_pallet / self.grid.A_surface

        self.H = np.full(self.grid.n, float(self.mat.enthalpy(s.sim.T_start)))
        self.H0_total = self._total_enthalpy()
        self.t = 0.0
        self.E_removed = 0.0      # J, cumulative heat removed from the pallet
        self.Q = 0.0              # W, during the last step
        self.T_air_out_step = None
        self.t_frozen = None      # s, when the core first reached T_target

        T_ref = s.cooling.T_air_reference
        T_in = T_ref if T_in is None else T_in
        if m_dot is None:
            m_dot = air_density(T_ref) * s.cooling.airflow
        self.set_air(T_in, m_dot, s.air.htc if h is None else h)
        self.dt_stable = self._stable_dt()

    # ---- air side ---------------------------------------------------------
    def set_air(self, T_in: float, m_dot: float, h: float):
        """Set inlet air for the next step. The cell calls this every step."""
        self.T_air_in = T_in
        if (getattr(self, "m_dot", None), getattr(self, "h", None)) != (m_dot, h):
            self.m_dot, self.h = m_dot, h
            self.C_air = m_dot * self.s.air.cp                              # W/K
            A = self.body.area_pallet
            self.NTU = h * A / self.C_air if self.C_air > 0 else math.inf
            self.eps = 1.0 - math.exp(-self.NTU)
            self.h_eff = self.eps * self.C_air / A if self.C_air > 0 else h
            self.R_film = 1.0 / self.h_eff + self.mat.R_surface             # m^2-K/W outside the product

    def _stable_dt(self) -> float:
        """Explicit Euler limit: dt <= V_i * C_min / sum(G_i) for every cell."""
        g, k = self.grid, self.mat.k_max
        G = np.zeros(g.n)
        G_int = g.A_faces[1:-1] * k / g.dx
        G[:-1] += G_int
        G[1:] += G_int
        G[-1] += g.A_surface / (g.dx / (2 * k) + self.R_film)
        return STABILITY_SAFETY * float(np.min(g.V_cells * self.mat.C_min / G))

    # ---- time stepping ----------------------------------------------------
    def step(self, dt: float) -> float:
        """Advance one explicit Euler step. Returns heat removed from the pallet [W]."""
        g, mat = self.grid, self.mat
        T, f = mat.state(self.H)
        k = mat.k(f)

        R_int = g.dx / (2 * k[:-1]) + g.dx / (2 * k[1:])
        q_int = g.A_faces[1:-1] / R_int * (T[:-1] - T[1:])              # W outward, per unit body
        q_s = g.A_surface / (g.dx / (2 * k[-1]) + self.R_film) * (T[-1] - self.T_air_in)

        net = np.zeros(g.n)
        net[:-1] -= q_int
        net[1:] += q_int
        net[-1] -= q_s
        self.H = self.H + dt * net / g.V_cells

        self.Q = q_s * self.scale
        self.T_air_out_step = self.T_air_in + self.Q / self.C_air
        self.E_removed += self.Q * dt
        self.t += dt
        if self.t_frozen is None and self.T_core <= self.s.sim.T_target:
            self.t_frozen = self.t
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
        H0 = self.H[0]
        m = self.mat
        if H0 < 0:
            return m.T_freeze + H0 / m.C_frozen
        if H0 > m.L_vol:
            return m.T_freeze + (H0 - m.L_vol) / m.C_unfrozen
        return m.T_freeze

    def surface_coupling(self) -> tuple[float, float]:
        """(G, T_N): heat to air = G * (T_N - T_air_in) for the whole pallet, at the current state."""
        T, f = self.mat.state(self.H[-1:])
        k = self.mat.k(f[0])
        G = self.grid.A_surface / (self.grid.dx / (2 * k) + self.R_film) * self.scale
        return G, float(T[0])

    def _surface_flux(self, T, f) -> float:
        """Heat flow to the air at the current state, per unit body [W]."""
        k = self.mat.k(f[-1])
        return self.grid.A_surface / (self.grid.dx / (2 * k) + self.R_film) * (T[-1] - self.T_air_in)

    @property
    def T_surface(self) -> float:
        """Product surface temperature (inside any surface packaging)."""
        T, f = self.mat.state(self.H)
        q = self._surface_flux(T, f)
        return float(T[-1] - q / self.grid.A_surface * self.grid.dx / (2 * self.mat.k(f[-1])))

    @property
    def Q_now(self) -> float:
        """Heat removal rate at the current state and current inlet air [W]."""
        T, f = self.mat.state(self.H)
        return self._surface_flux(T, f) * self.scale

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
