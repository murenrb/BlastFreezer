"""Input definitions for a blast freezer simulation.

Everything the solver needs lives in these dataclasses. The Streamlit UI fills
them in; the physics only reads them. Defaults come from the legacy Tar Heel
scripts (Old Sources/TPHL - Tar Heel.py) unless marked NEW, which means the
legacy code had no equivalent and the value is a placeholder to be confirmed.

Units: SI throughout (m, kg, s, W, J), temperatures in degC.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Product:
    """Thermal properties of the product itself (not the packed bulk)."""

    name: str = "Legacy Tar Heel product"
    water_fraction: float = 0.57           # mass fraction freezable water [-]        (legacy)
    cp_unfrozen: float = 2470.0            # J/kg-K                                    (legacy)
    cp_frozen: float = 1470.0              # J/kg-K                                    (legacy)
    k_unfrozen: float = 0.50               # W/m-K                                     NEW (typical meat)
    k_frozen: float = 1.40                 # W/m-K                                     (legacy k_meat)
    density: float = 1050.0                # kg/m^3, solid product, no voids           NEW (typical meat)
    T_freeze: float = -1.9                 # degC, initial freezing point              (peak of legacy cp curve)
    latent_heat_water: float = 334_000.0   # J/kg of water


@dataclass
class Packaging:
    """Packaging between the product and the air.

    `kind` is just a label; the numbers below are what the model uses. The UI
    fills them from PACKAGING_PRESETS when the kind changes.
    """

    kind: str = "Corrugated boxes"
    package_height: float = 0.26           # m, height of one box/bag in the heat-flow direction  NEW
    wall_thickness: float = 0.004          # m, cardboard/film thickness per wall       NEW
    k_wall: float = 0.064                  # W/m-K, corrugated board                    NEW
    wall_density: float = 120.0            # kg/m^3, effective density of the wall      NEW
    cp_wall: float = 1400.0                # J/kg-K                                     NEW
    air_gap: float = 0.005                 # m, headspace air gap per package           NEW
    k_air: float = 0.023                   # W/m-K


PACKAGING_PRESETS: dict[str, dict] = {
    "Corrugated boxes": dict(wall_thickness=0.004, k_wall=0.064, wall_density=120.0,
                             cp_wall=1400.0, air_gap=0.005),
    "Bags (no box)": dict(wall_thickness=0.0001, k_wall=0.20, wall_density=920.0,
                          cp_wall=1900.0, air_gap=0.002),
    "Unpackaged": dict(wall_thickness=0.0, k_wall=0.064, wall_density=0.0,
                       cp_wall=1400.0, air_gap=0.0),
}


@dataclass
class PalletLoad:
    """Geometry and loading of one pallet (product envelope, excluding the wooden deck)."""

    mass: float = 1349.0 * 0.4536          # kg of product                             (legacy)
    length: float = 1.02                   # m                                         (legacy)
    width: float = 1.22                    # m                                         (legacy)
    height: float = 2.0                    # m, product stack height incl. spacers     (legacy)
    use_spacers: bool = True
    n_spacers: int = 6                     # spacer layers between box layers          (legacy)
    spacer_thickness: float = 0.03         # m                                         NEW
    geometry: str = "auto"                 # "auto" | "plate" | "sphere" | "cylinder"


@dataclass
class Air:
    """Air delivered to one pallet (the cell distributes air evenly)."""

    T_in: float = -45.0                    # degC, air entering the pallet             (legacy)
    flow_per_pallet: float = 1.0           # m^3/s (~2100 CFM)                         NEW
    htc: float = 30.0                      # W/m^2-K, surface heat transfer coeff.     (legacy)
    cp: float = 1006.0                     # J/kg-K


@dataclass
class Cell:
    n_pallets: int = 24                    # pallets per load                          (legacy)


@dataclass
class SimSettings:
    T_start: float = 11.0                  # degC, uniform product temp at load-in     (legacy)
    T_target: float = -17.0                # degC, stop when core reaches this         (legacy)
    n_nodes: int = 40                      # finite volumes core->surface (~1-2% freeze-time error)
    dt_max: float = 60.0                   # s, upper limit; solver uses min(dt_max, stable dt)
    t_max_hours: float = 96.0              # h, hard stop
    record_every_s: float = 300.0          # s, output interval


@dataclass
class Scenario:
    product: Product = field(default_factory=Product)
    packaging: Packaging = field(default_factory=Packaging)
    load: PalletLoad = field(default_factory=PalletLoad)
    air: Air = field(default_factory=Air)
    cell: Cell = field(default_factory=Cell)
    sim: SimSettings = field(default_factory=SimSettings)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Scenario":
        return cls(
            product=Product(**d.get("product", {})),
            packaging=Packaging(**d.get("packaging", {})),
            load=PalletLoad(**d.get("load", {})),
            air=Air(**d.get("air", {})),
            cell=Cell(**d.get("cell", {})),
            sim=SimSettings(**d.get("sim", {})),
        )
