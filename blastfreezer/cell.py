"""Blast cell: a Cartesian grid of pallet positions with series airflow.

Axes: depth (front -> back), width (left -> right), height (bottom -> top).
Air enters one face and leaves the opposite face, moving in straight lines.
A "lane" is one line of pallets along the flow; each lane's pallets are
solved in flow order, with each pallet's air-out becoming the next pallet's
air-in within the same time step.

With even distribution and a constant inlet temperature, every lane is
identical, so the cell holds one representative Lane with a multiplicity of
n_lanes. Uneven distribution later just means several Lanes with their own
flows.

Supply air comes from the coil / refrigeration balance (refrigeration.py),
re-solved every time step.

Spacers: if the flow runs along the spacer channels, the surface HTC is used
as given. If it runs across them (vertical flow, or horizontal flow across
one-way channels), the HTC is reduced by `spacer_htc_reduction`:
h = htc * (1 - reduction).
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Scenario
from .pallet import Pallet, air_density
from .refrigeration import Balance, Coil, RefSystem, solve_balance

# face -> (axis, axis label)
FACE_AXIS = {"bottom": "height", "top": "height", "front": "depth", "back": "depth",
             "left": "width", "right": "width"}
OPPOSITE = {"bottom": "top", "top": "bottom", "front": "back", "back": "front",
            "left": "right", "right": "left"}


@dataclass(frozen=True)
class AirPath:
    axis: str                  # depth | width | height
    inlet: str
    outlet: str
    n_series: int              # pallets in each lane
    n_lanes: int
    spacer_alignment: str      # "no spacers" | "along channels" | "across channels"
    h: float                   # W/m^2-K applied to each pallet
    flow_total: float          # m^3/s at inlet temperature
    m_dot_total: float         # kg/s
    m_dot_lane: float          # kg/s


def air_path(s: Scenario) -> AirPath:
    c, ld, a = s.cell, s.load, s.air
    axis = FACE_AXIS[c.inlet_face]
    n = {"depth": c.n_deep, "width": c.n_wide, "height": c.n_high}
    n_series = n[axis]
    n_lanes = c.n_pallets // n_series

    if not (ld.use_spacers and ld.n_spacers > 0):
        alignment, h = "no spacers", a.htc
    elif axis != "height" and ld.spacer_channels in ("both", axis):
        alignment, h = "along channels", a.htc
    else:
        alignment = "across channels"
        h = a.htc * (1.0 - a.spacer_htc_reduction)

    flow_total = s.cooling.airflow
    m_dot_total = air_density(s.cooling.T_air_reference) * flow_total
    return AirPath(axis, c.inlet_face, OPPOSITE[c.inlet_face], n_series, n_lanes, alignment, h,
                   flow_total, m_dot_total, m_dot_total / n_lanes)


class Lane:
    """Pallets in series along the airflow, solved in flow order."""

    def __init__(self, s: Scenario, path: AirPath, multiplicity: int):
        self.m_dot = path.m_dot_lane
        self.h = path.h
        self.multiplicity = multiplicity
        self.pallets = [Pallet(s, name=f"P{i + 1}", m_dot=self.m_dot, h=self.h)
                        for i in range(path.n_series)]
        self.T_in = s.cooling.T_air_reference
        self.T_out = self.T_in
        self.update_air()

    def linear_return(self) -> tuple[float, float]:
        """(a, b) with lane outlet T_out = a + b * T_in for the current pallet states.

        Each pallet: Q = G (T_N - T_in), T_out = T_in + Q / C  ->  T_out = alpha T_in + beta.
        """
        a, b = 0.0, 1.0
        C = self.m_dot * self.pallets[0].s.air.cp
        for p in self.pallets:
            G, T_N = p.surface_coupling()
            alpha, beta = 1.0 - G / C, G * T_N / C
            a, b = alpha * a + beta, alpha * b
        return a, b

    def update_air(self):
        """Propagate inlet air through the lane using the current pallet states (no stepping)."""
        T = self.T_in
        for p in self.pallets:
            p.set_air(T, self.m_dot, self.h)
            T = p.T_air_out
        self.T_out = T

    def step(self, dt: float) -> float:
        T, Q = self.T_in, 0.0
        for p in self.pallets:
            p.set_air(T, self.m_dot, self.h)
            Q += p.step(dt)
            T = p.T_air_out_step
        self.T_out = T
        return Q


class Cell:
    """Pallet lanes plus the coil/refrigeration loop that sets the supply air.

    Each step: with the pallet states frozen, cell return air is linear in supply
    air; the coil + refrigeration balance is solved for SST and supply air; then
    every lane is stepped with that supply air.
    """

    def __init__(self, s: Scenario):
        self.s = s
        self.path = air_path(s)
        self.lanes = [Lane(s, self.path, multiplicity=self.path.n_lanes)]
        self.t = 0.0
        c = s.cooling
        self.fixed = c.fixed_supply_T is not None
        if not self.fixed:
            self.coil = Coil(c.coil_Q_design, c.coil_T_supply_design, c.coil_T_return_design, c.coil_SST_design)
            self.ref = RefSystem(c.ref_Q_design, c.ref_SST_design, c.refrigerant)
        self.C_air = self.path.m_dot_total * s.air.cp
        self.balance = self._solve_supply()
        for l in self.lanes:
            l.T_in = self.balance.T_supply
            l.update_air()
        self.Q = self.balance.Q - self.balance.Q_fan     # pallet load

    def _linear_return(self) -> tuple[float, float]:
        """Mixed cell return air T_r = a + b * T_supply."""
        w = sum(l.multiplicity * l.m_dot for l in self.lanes)
        a = b = 0.0
        for l in self.lanes:
            la, lb = l.linear_return()
            a += l.multiplicity * l.m_dot * la / w
            b += l.multiplicity * l.m_dot * lb / w
        return a, b

    def _solve_supply(self) -> Balance:
        a, b = self._linear_return()
        if self.fixed:
            Ts = self.s.cooling.fixed_supply_T
            Tr = a + b * Ts
            return Balance(float("nan"), Ts, Tr, self.C_air * (Tr - Ts), "fixed supply", Ts, 0.0)
        c = self.s.cooling
        return solve_balance(a, b, self.C_air, self.coil, self.ref, fan_W=c.fan_power,
                             SST_max=c.SST_max, T_return_min=c.T_return_setpoint)

    @property
    def pallets(self):
        return [p for l in self.lanes for p in l.pallets]

    @property
    def dt_stable(self) -> float:
        return min(p.dt_stable for p in self.pallets)

    def step(self, dt: float) -> float:
        self.balance = self._solve_supply()
        for l in self.lanes:
            l.T_in = self.balance.T_supply
        self.Q = sum(l.multiplicity * l.step(dt) for l in self.lanes)
        self.t += dt
        return self.Q

    @property
    def T_supply(self) -> float:
        return self.balance.T_supply

    @property
    def T_air_out(self) -> float:
        """Mixed outlet temperature (all lanes share m_dot today)."""
        w = sum(l.multiplicity * l.m_dot for l in self.lanes)
        return sum(l.multiplicity * l.m_dot * l.T_out for l in self.lanes) / w

    @property
    def all_frozen(self) -> bool:
        return all(p.t_frozen is not None for p in self.pallets)
