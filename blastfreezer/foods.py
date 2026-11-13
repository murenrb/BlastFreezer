"""Food property library.

Sources
-------
- Water content and specific heats: the "Food Prop Lookup" table of the
  Initial Proposal Generator, plus Beef (carcass, choice) and Frankfurter from
  ASHRAE Refrigeration Handbook (2006), Ch. 9, Table 3.
- Initial freezing point and composition (protein, fat, carbohydrate, fiber,
  ash): ASHRAE Table 3 rows the values correspond to.
- Thermal conductivity and solid density: computed from composition with the
  Choi & Okos (1986) component models (ASHRAE Tables 1 and 2), using
  volume-weighted (parallel) mixing:
      unfrozen at +5 C
      frozen at -18 C, ice fraction from Tchigeov (ASHRAE Eq. 5).

Notes
-----
- "Sliced peaches" in the proposal generator carries the ASHRAE *Apricots*
  values (86.35 % water, cp 3.87 / 1.95). Kept as-is; composition and freezing
  point use the Apricots row to stay consistent.
- The cooked chicken items (52.79 % water) have no ASHRAE row; they use raw
  chicken's cp and freezing point, with chicken's solids scaled to the lower
  water content.
- Plant based protein has no ASHRAE row; composition and freezing point are
  estimates (marked `estimated=True`).
- ASHRAE's carbohydrate column includes fiber, so fiber is subtracted from it.
- ASHRAE lists chicken's frozen cp as 3.32 kJ/kg-K, which is high compared to
  other meats (~2.2). Kept as published.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Food:
    name: str
    water: float          # mass fraction
    cp_unfrozen: float    # J/kg-K
    cp_frozen: float      # J/kg-K
    T_freeze: float       # degC
    protein: float        # mass fractions of the unfrozen food
    fat: float
    carb: float
    fiber: float
    ash: float
    source: str
    estimated: bool = False

    # derived (Choi-Okos)
    @property
    def k_unfrozen(self) -> float:
        return conductivity(self, 5.0)

    @property
    def k_frozen(self) -> float:
        return conductivity(self, -18.0)

    @property
    def density(self) -> float:
        return density(self, 5.0)


# ---- Choi & Okos (1986) component models, t in degC ------------------------
def _k(c: str, t: float) -> float:
    return {
        "protein": 1.7881e-1 + 1.1958e-3 * t - 2.7178e-6 * t * t,
        "fat": 1.8071e-1 - 2.7604e-4 * t - 1.7749e-7 * t * t,
        "carb": 2.0141e-1 + 1.3874e-3 * t - 4.3312e-6 * t * t,
        "fiber": 1.8331e-1 + 1.2497e-3 * t - 3.1683e-6 * t * t,
        "ash": 3.2962e-1 + 1.4011e-3 * t - 2.9069e-6 * t * t,
        "water": 5.7109e-1 + 1.7625e-3 * t - 6.7036e-6 * t * t,
        "ice": 2.2196 - 6.2489e-3 * t + 1.0154e-4 * t * t,
    }[c]


def _rho(c: str, t: float) -> float:
    return {
        "protein": 1.3299e3 - 5.1840e-1 * t,
        "fat": 9.2559e2 - 4.1757e-1 * t,
        "carb": 1.5991e3 - 3.1046e-1 * t,
        "fiber": 1.3115e3 - 3.6589e-1 * t,
        "ash": 2.4238e3 - 2.8063e-1 * t,
        "water": 9.9718e2 + 3.1439e-3 * t - 3.7574e-3 * t * t,
        "ice": 9.1689e2 - 1.3071e-1 * t,
    }[c]


def ice_fraction(f: Food, t: float) -> float:
    """Tchigeov (ASHRAE Eq. 5); zero above the initial freezing point."""
    if t >= f.T_freeze:
        return 0.0
    return min(f.water, 1.105 * f.water / (1.0 + 0.7138 / math.log(f.T_freeze - t + 1.0)))


def _mass_fractions(f: Food, t: float) -> dict:
    x_ice = ice_fraction(f, t)
    return {"water": f.water - x_ice, "ice": x_ice, "protein": f.protein, "fat": f.fat,
            "carb": f.carb, "fiber": f.fiber, "ash": f.ash}


def density(f: Food, t: float) -> float:
    x = _mass_fractions(f, t)
    return 1.0 / sum(xi / _rho(c, t) for c, xi in x.items() if xi > 0)


def conductivity(f: Food, t: float) -> float:
    x = _mass_fractions(f, t)
    vols = {c: xi / _rho(c, t) for c, xi in x.items() if xi > 0}
    V = sum(vols.values())
    return sum(v / V * _k(c, t) for c, v in vols.items())


# ---- library ----------------------------------------------------------------
_PG = "Initial Proposal Generator, Food Prop Lookup"
_CHICKEN_SOLIDS = (0.1860, 0.1506, 0.0, 0.0, 0.0079)   # ASHRAE raw chicken p, f, c, fiber, ash


def _scaled_chicken(water: float) -> tuple:
    s = sum(_CHICKEN_SOLIDS)
    return tuple(v * (1.0 - water) / s for v in _CHICKEN_SOLIDS)


FOODS: dict[str, Food] = {f.name: f for f in [
    Food("Beef, carcass (choice)", 0.5726, 3240, 2310, -2.2, 0.1732, 0.2405, 0.0, 0.0, 0.0081,
         "ASHRAE Table 3"),
    Food("Sausage, frankfurter", 0.5387, 3150, 2310, -1.7, 0.1128, 0.2915, 0.0255, 0.0, 0.0315,
         "ASHRAE Table 3"),
    Food("Uncooked chicken", 0.6599, 4340, 3320, -2.8, *_CHICKEN_SOLIDS, f"{_PG}; ASHRAE chicken"),
    Food("Steamed wings", 0.52792, 4340, 3320, -2.8, *_scaled_chicken(0.52792),
         f"{_PG}; composition scaled from ASHRAE chicken"),
    Food("Chicken patty", 0.52792, 4340, 3320, -2.8, *_scaled_chicken(0.52792),
         f"{_PG}; composition scaled from ASHRAE chicken"),
    Food("Cooked bone-in wing", 0.52792, 4340, 3320, -2.8, *_scaled_chicken(0.52792),
         f"{_PG}; composition scaled from ASHRAE chicken"),
    Food("Plant based protein product", 0.70, 3520, 2070, -1.5, 0.20, 0.05, 0.04, 0.0, 0.01,
         f"{_PG}; composition and freezing point estimated", estimated=True),
    Food("Sliced peaches", 0.8635, 3870, 1950, -1.1, 0.0140, 0.0039, 0.1112 - 0.0240, 0.0240, 0.0075,
         f"{_PG} (matches ASHRAE apricots)"),
    Food("Blueberries", 0.8461, 3830, 2060, -1.6, 0.0067, 0.0038, 0.1413 - 0.0270, 0.0270, 0.0021,
         f"{_PG}; ASHRAE Table 3"),
    Food("Raspberries", 0.8657, 3960, 1910, -0.6, 0.0091, 0.0055, 0.1157 - 0.0680, 0.0680, 0.0040,
         f"{_PG}; ASHRAE Table 3"),
]}

DEFAULT_FOOD = "Beef, carcass (choice)"
CUSTOM = "Custom"


def product_kwargs(name: str) -> dict:
    """Fields for config.Product from a library entry."""
    f = FOODS[name]
    return dict(name=f.name, water_fraction=f.water, cp_unfrozen=float(f.cp_unfrozen),
                cp_frozen=float(f.cp_frozen), k_unfrozen=round(f.k_unfrozen, 3),
                k_frozen=round(f.k_frozen, 3), density=round(f.density, 1), T_freeze=f.T_freeze)
