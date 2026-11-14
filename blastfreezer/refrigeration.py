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
in supply air: T_r = a + b * T_s (see Lane.linear_return). With the coil,
    T_s = (1 - eps) T_r + eps SST   ->   T_s(SST) = ((1-eps) a + eps SST) / (1 - (1-eps) b)
and Q_coil(SST) decreases with SST while Q_ref(SST) increases, so
Q_coil(SST) = Q_ref(SST) has one root, found with brentq.
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
class Balance:
    SST: float
    T_supply: float          # air entering the pallets (coil outlet + fan heat)
    T_return: float          # air leaving the pallets = entering the coil
    Q: float                 # W, coil load (= pallet load + fan heat)
    limited: str = ""        # "" | "SST max" | "return setpoint" | "SST floor" | "SST ceiling"
    T_coil_out: float = float("nan")
    Q_fan: float = 0.0


def solve_balance(a: float, b: float, C_air: float, coil: Coil, ref: RefSystem, *,
                  fan_W: float = 0.0, SST_max: float | None = None,
                  T_return_min: float | None = None) -> Balance:
    """Close coil + refrigeration + load, with cell return air T_r = a + b * T_supply.

    Fan heat is added to the air leaving the coil (dT = fan_W / C_air).
    SST_max: the plant holds suction at or below this; above it the coil gets what it needs.
    T_return_min: return air setpoint; when the free balance would go colder, capacity is
    reduced (SST rises) to hold the return air at the setpoint.
    """
    eps = coil.effectiveness(C_air)
    d = fan_W / C_air
    a2 = a + b * d                       # return air as a function of coil outlet: T_r = a2 + b T_c
    den = 1.0 - (1.0 - eps) * b

    def T_c(sst):
        return ((1.0 - eps) * a2 + eps * sst) / den

    def f(sst):
        return C_air * (a2 + (b - 1.0) * T_c(sst)) - float(ref.capacity(sst))

    lo, hi = ref.SST_min, ref.SST_max
    limited = ""
    if f(lo) <= 0.0:
        sst, limited = lo, "SST floor"
    elif f(hi) >= 0.0:
        sst, limited = hi, "SST ceiling"
    else:
        sst = brentq(f, lo, hi, xtol=1e-6)
    if SST_max is not None and sst > SST_max:
        sst, limited = SST_max, "SST max"
    Tc = T_c(sst)
    Tr = a2 + b * Tc
    if T_return_min is not None and Tr < T_return_min:
        Tr = T_return_min
        Tc = (Tr - a2) / b
        sst = (Tc - (1.0 - eps) * Tr) / eps
        limited = "return setpoint"
    return Balance(float(sst), float(Tc + d), float(Tr), float(C_air * (Tr - Tc)), limited,
                   float(Tc), float(fan_W))


def design_return_balance(coil: Coil, ref: RefSystem, C_air: float) -> Balance:
    """Coil vs refrigeration only: return air held at the coil's design return, no fans or controls."""
    return solve_balance(coil.T_return_design, 0.0, C_air, coil, ref)
