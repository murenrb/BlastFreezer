"""Input definitions for a blast freezer cell simulation.

Everything the solver needs lives in these dataclasses. The Streamlit UI fills
them in, and the solver only ever reads them. That keeps the physics testable
without Streamlit.

Units: SI throughout (m, kg, s, W, J), except temperatures in degC.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class Product:
    """Thermal properties of the product being frozen."""

    name: str = "Generic meat"
    water_fraction: float = 0.688          # mass fraction water [-]
    cp_unfrozen: float = 3200.0            # J/kg-K, above initial freezing point
    cp_frozen: float = 1600.0              # J/kg-K, fully frozen
    k_unfrozen: float = 0.45               # W/m-K
    k_frozen: float = 1.4                  # W/m-K
    density: float = 1000.0                # kg/m^3
    latent_heat_water: float = 334_000.0   # J/kg of water
    T_initial_freeze: float = -1.8         # degC, initial freezing point


@dataclass
class Pallet:
    """Geometry and loading of one pallet."""

    mass: float = 900.0                    # kg of product per pallet
    length: float = 1.2                    # m
    width: float = 1.0                     # m
    height: float = 1.5                    # m
    exposed_area: float = 10.0             # m^2, area exposed to airflow (placeholder, uncalibrated)
    critical_length: float = 1.0           # m, characteristic conduction length
    k_effective: float = 6.0               # W/m-K, effective lumped value (placeholder, uncalibrated)


@dataclass
class Cell:
    """The blast cell: air side and capacity."""

    n_pallets: int = 24
    T_air: float = -30.0                   # degC, supply air temperature
    htc: float = 25.0                      # W/m^2-K, air-side heat transfer coefficient
    refrigeration_capacity: float = 150_000.0  # W, coil/refrigeration limit (0 = unlimited)


@dataclass
class SimSettings:
    T_start: float = 11.0                  # degC, product temperature at load-in
    T_target: float = -18.0                # degC, end-of-blast core temperature
    dt: float = 60.0                       # s, time step
    t_max_hours: float = 72.0              # h, hard stop


@dataclass
class Scenario:
    """A complete simulation case."""

    product: Product = field(default_factory=Product)
    pallet: Pallet = field(default_factory=Pallet)
    cell: Cell = field(default_factory=Cell)
    sim: SimSettings = field(default_factory=SimSettings)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Scenario":
        return cls(
            product=Product(**d.get("product", {})),
            pallet=Pallet(**d.get("pallet", {})),
            cell=Cell(**d.get("cell", {})),
            sim=SimSettings(**d.get("sim", {})),
        )
