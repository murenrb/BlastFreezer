"""Coil and refrigeration system, and the air/refrigerant balance each time step.

Coil
----
The refrigerant side is isothermal at SST, so the coil is a heat exchanger
with C_min = air. UA is backed out of the design point with the LMTD:
    UA = Q_d / LMTD,  LMTD = (dT1 - dT2) / ln(dT1 / dT2),
    dT1 = T_return_d - SST_d,  dT2 = T_supply_d - SST_d.
At any condition:  eps = 1 - exp(-UA / (m_dot cp)),
                   Q_coil = eps * m_dot cp * (T_return - SST).

Refrigeration system
--------------------
Fixed compressor displacement (volumetric flow at suction), sized at the
design point:  V_dot = Q_d / (rho_g(SST_d) * h_fg(SST_d)).
At any SST:    Q_ref = V_dot * rho_g(SST) * h_fg(SST).
Liquid is assumed to reach the evaporator saturated at SST (recirculated /
low-pressure receiver), so the refrigeration effect is h_fg. Properties come
from CoolProp, tabulated once per run.

Balance
-------
Within a time step the pallet states are frozen, so cell return air is linear
in supply air: T_r = a + b * T_s (see Lane.linear_return). One or more coil
units (coil + its own fans + its own refrigeration system) run in parallel:
each takes return air, and their outlets mix into the supply air.

For one unit at a fixed airflow, the free balance Q_coil = Q_ref fixes SST as a
function of return air alone:  C eps (T_r - SST) = Q_ref(SST), i.e.
T_r = SST + Q_ref(SST) / (C eps), which is tabulated once per unit and inverted
by interpolation (exact for the tabulated Q_ref). Each unit's outlet is then
T_c = T_r - eps (T_r - SST) + fan heat; the mixed supply is the C-weighted mean,
and T_r = a + b * T_s(T_r) has one root (the right side rises slower than T_r),
found with brentq.

Controls: a unit whose free SST is above its SST_max is held at SST_max and the
coil gets what it needs. Each unit has its own supply-air setpoint (air leaving
that unit, after its fans): if its free outlet would be colder, its SST is raised
(down to zero load) to hold the outlet at the setpoint. So the boost can deliver
-50 while the main coil holds -30; the supply air entering the pallets is their
C-weighted mix. Each unit's outlet stays continuous and rises no faster than
return air, so the single brentq root above covers the controlled case too. As
the pallets cool, return air keeps falling while SST rises.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq

REFRIGERANTS = {"NH3": "Ammonia", "CO2": "CO2", "R404A": "R404A", "R134a": "R134a"}
SST_TABLE_MAX = 30.0        # degC, below CO2's critical point
SST_TABLE_STEP = 0.25       # K


@dataclass(frozen=True)
class Coil:
    Q_design: float          # W
    T_supply_design: float   # degC
    T_return_design: float   # degC
    SST_design: float        # degC

    def __post_init__(self):
        if not (self.T_return_design > self.T_supply_design > self.SST_design):
            raise ValueError("Coil design needs return air > supply air > SST.")
        if self.Q_design <= 0:
            raise ValueError("Coil design cooling power must be positive.")

    @property
    def LMTD(self) -> float:
        d1 = self.T_return_design - self.SST_design
        d2 = self.T_supply_design - self.SST_design
        return (d1 - d2) / math.log(d1 / d2)

    @property
    def UA(self) -> float:
        """W/K"""
        return self.Q_design / self.LMTD

    @property
    def design_C_air(self) -> float:
        """Air capacity rate implied by the design point, W/K."""
        return self.Q_design / (self.T_return_design - self.T_supply_design)

    def effectiveness(self, C_air: float) -> float:
        return 1.0 - math.exp(-self.UA / C_air)


@lru_cache(maxsize=16)
def _volumetric_capacity_table(refrigerant: str):
    """(SST grid degC, rho_g * h_fg J/m^3) from CoolProp."""
    import CoolProp.CoolProp as CP

    fluid = REFRIGERANTS[refrigerant]
    T_lo = CP.PropsSI("Ttriple", fluid) - 273.15 + 0.5
    T = np.arange(math.ceil(T_lo / SST_TABLE_STEP) * SST_TABLE_STEP, SST_TABLE_MAX + 1e-9, SST_TABLE_STEP)
    Tk = T + 273.15
    rho = CP.PropsSI("D", "T", Tk, "Q", 1, fluid)
    hfg = CP.PropsSI("H", "T", Tk, "Q", 1, fluid) - CP.PropsSI("H", "T", Tk, "Q", 0, fluid)
    return T, np.asarray(rho) * np.asarray(hfg)


@dataclass(frozen=True)
class RefSystem:
    Q_design: float          # W, refrigeration capacity at design SST
    SST_design: float        # degC
    refrigerant: str

    def __post_init__(self):
        if self.refrigerant not in REFRIGERANTS:
            raise ValueError(f"Unknown refrigerant {self.refrigerant}")
        T, _ = _volumetric_capacity_table(self.refrigerant)
        if not (T[0] <= self.SST_design <= T[-1]):
            raise ValueError(f"Design SST outside {self.refrigerant} range "
                             f"({T[0]:.1f} to {T[-1]:.1f} C).")
        if self.Q_design <= 0:
            raise ValueError("Refrigeration design power must be positive.")

    @property
    def SST_min(self) -> float:
        return float(_volumetric_capacity_table(self.refrigerant)[0][0])

    @property
    def SST_max(self) -> float:
        return float(_volumetric_capacity_table(self.refrigerant)[0][-1])

    def volumetric_capacity(self, SST):
        T, vc = _volumetric_capacity_table(self.refrigerant)
        return np.interp(SST, T, vc)

    @property
    def displacement(self) -> float:
        """Suction volumetric flow, m^3/s."""
        return self.Q_design / float(self.volumetric_capacity(self.SST_design))

    def capacity(self, SST):
        return self.displacement * self.volumetric_capacity(SST)


@dataclass(frozen=True)
class CoilState:
    """One coil unit in a balance."""
    name: str
    SST: float
    Q: float                 # W, coil load
    T_out: float             # air leaving the coil, before fan heat
    Q_fan: float             # W
    C_air: float             # W/K
    limited: str = ""

    @property
    def T_supply(self) -> float:
        """Air leaving this unit, after its fan heat."""
        return self.T_out + self.Q_fan / self.C_air


@dataclass(frozen=True)
class Balance:
    SST: float               # first (main) unit
    T_supply: float          # mixed air entering the pallets (coil outlets + fan heat)
    T_return: float          # air leaving the pallets = entering every coil
    Q: float                 # W, total coil load (= pallet load + fan heat)
    limited: str = ""        # first unit: "" | "SST max" | "supply setpoint" | "SST floor" | "SST ceiling"
    T_coil_out: float = float("nan")   # first unit, before fan heat
    Q_fan: float = 0.0       # W, all units
    coils: tuple = ()        # CoilState per unit


class CoilUnit:
    """A coil with its own fans and refrigeration system, at a fixed airflow."""

    def __init__(self, coil: Coil, ref: RefSystem, C_air: float, fan_W: float = 0.0,
                 SST_max: float | None = None, name: str = "main", T_set: float | None = None):
        """T_set: lowest allowed air temperature leaving this unit (after its fans); None = no limit."""
        self.name, self.C_air, self.fan_W, self.SST_max, self.T_set = name, C_air, fan_W, SST_max, T_set
        self.eps = coil.effectiveness(C_air)
        self.d = fan_W / C_air
        S, vc = _volumetric_capacity_table(ref.refrigerant)
        self._S = S
        self._Tr = S + ref.displacement * vc / (self.eps * C_air)   # return air at which free SST = S

    def free_SST(self, T_r: float) -> tuple[float, str]:
        if T_r <= self._Tr[0]:
            S, lim = float(self._S[0]), "SST floor"
        elif T_r >= self._Tr[-1]:
            S, lim = float(self._S[-1]), "SST ceiling"
        else:
            S, lim = float(np.interp(T_r, self._Tr, self._S)), ""
        if self.SST_max is not None and S > self.SST_max:
            S, lim = self.SST_max, "SST max"
        return S, lim

    def T_coil(self, T_r: float, SST: float) -> float:
        """Air leaving the coil, before fan heat."""
        return T_r - self.eps * (T_r - SST)

    def operate(self, T_r: float) -> tuple[float, str]:
        """SST with controls: free balance, then held at the supply setpoint if the outlet would be colder.

        Throttling stops at zero load (SST = return air): the coil never heats the air.
        """
        S, lim = self.free_SST(T_r)
        if self.T_set is None or self.T_coil(T_r, S) + self.d >= self.T_set:
            return S, lim
        S_hold = T_r - (T_r + self.d - self.T_set) / self.eps      # outlet + fan heat = T_set
        return min(S_hold, T_r), "supply setpoint"


def solve_parallel(a: float, b: float, units: list[CoilUnit]) -> Balance:
    """Close coils + refrigeration + load with cell return air T_r = a + b * T_supply.

    Each unit runs at its own controls (CoilUnit.operate); the supply air is their C-weighted mix.
    """
    C_tot = sum(u.C_air for u in units)

    def states(T_r):
        return [u.operate(T_r) for u in units]

    def T_mix(T_r, SSTs):
        return sum(u.C_air * (u.T_coil(T_r, S) + u.d) for u, S in zip(units, SSTs)) / C_tot

    def h(T_r):
        return a + b * T_mix(T_r, [S for S, _ in states(T_r)]) - T_r

    lo, hi = -150.0, 150.0
    while h(lo) < 0.0:
        lo -= 100.0
    while h(hi) > 0.0:
        hi += 100.0
    T_r = brentq(h, lo, hi, xtol=1e-10)
    st = states(T_r)
    SSTs, labels = [S for S, _ in st], [lim for _, lim in st]

    coils = []
    for u, S, lim in zip(units, SSTs, labels):
        Tc = u.T_coil(T_r, S)
        coils.append(CoilState(u.name, float(S), float(u.C_air * (T_r - Tc)), float(Tc), float(u.fan_W),
                               float(u.C_air), lim))
    T_s = T_mix(T_r, SSTs)
    m = coils[0]
    return Balance(m.SST, float(T_s), float(T_r), float(sum(c.Q for c in coils)), m.limited, m.T_out,
                   float(sum(c.Q_fan for c in coils)), tuple(coils))


def solve_balance(a: float, b: float, C_air: float, coil: Coil, ref: RefSystem, *,
                  fan_W: float = 0.0, SST_max: float | None = None,
                  T_supply_min: float | None = None) -> Balance:
    """Single coil unit: see solve_parallel. Fan heat is added to the air leaving the coil."""
    return solve_parallel(a, b, [CoilUnit(coil, ref, C_air, fan_W, SST_max, T_set=T_supply_min)])


def design_return_balance(coil: Coil, ref: RefSystem, C_air: float) -> Balance:
    """Coil vs refrigeration only: return air held at the coil's design return, no fans or controls."""
    return solve_balance(coil.T_return_design, 0.0, C_air, coil, ref)
