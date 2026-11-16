"""Input definitions for a blast freezer simulation.

Everything the solver needs lives in these dataclasses. The Streamlit UI fills
them in; the physics only reads them. Product defaults come from foods.py;
other defaults from the legacy scripts or confirmed by the user.

Units: SI throughout (m, kg, s, W, J), temperatures in degC.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .foods import DEFAULT_FOOD, product_kwargs


_D = product_kwargs(DEFAULT_FOOD)       # default product: beef carcass (choice), see foods.py


@dataclass
class Product:
    """Thermal properties of the product itself (not the packed bulk)."""

    name: str = _D["name"]
    water_fraction: float = _D["water_fraction"]     # total water, mass fraction of product [-]
    bound_water_fraction: float = 0.05               # fraction of that water that never freezes [-]
    cp_unfrozen: float = _D["cp_unfrozen"]           # J/kg-K
    cp_frozen: float = _D["cp_frozen"]               # J/kg-K
    k_unfrozen: float = _D["k_unfrozen"]             # W/m-K
    k_frozen: float = _D["k_frozen"]                 # W/m-K
    density: float = _D["density"]                   # kg/m^3, solid product, no voids
    T_freeze: float = _D["T_freeze"]                 # degC, initial freezing point
    latent_heat_water: float = 334_000.0             # J/kg of water


@dataclass
class Packaging:
    """Packaging between the product and the air.

    `kind` is a label; the numbers are what the model uses. The UI fills them
    from PACKAGING_PRESETS when the kind changes.
    """

    kind: str = "Corrugated boxes"
    package_height: float = 0.17           # m, one box/bag in the heat-flow direction (1.2 m / 7 tiers)
    wall_thickness: float = 0.004          # m, cardboard/film thickness per wall
    k_wall: float = 0.064                  # W/m-K
    wall_density: float = 120.0            # kg/m^3
    cp_wall: float = 1400.0                # J/kg-K
    air_gap: float = 0.005                 # m, headspace air gap per package
    k_air: float = 0.023                   # W/m-K
    surface_wall: bool = True              # outer wall modeled as a surface resistance


PACKAGING_PRESETS: dict[str, dict] = {
    "Corrugated boxes": dict(wall_thickness=0.004, k_wall=0.064, wall_density=120.0,
                             cp_wall=1400.0, air_gap=0.005, surface_wall=True),
    "Bags (no box)": dict(wall_thickness=0.0001, k_wall=0.20, wall_density=920.0,
                          cp_wall=1900.0, air_gap=0.002, surface_wall=False),
    "Unpackaged": dict(wall_thickness=0.0, k_wall=0.064, wall_density=0.0,
                       cp_wall=1400.0, air_gap=0.0, surface_wall=False),
}

SPACER_CHANNELS = ("both", "depth", "width")   # horizontal direction(s) air can pass through spacers


@dataclass
class PalletLoad:
    """One pallet. `height` is product only; spacers add to the stack height."""

    mass: float = 1349.0 * 0.45359237      # kg of product
    length: float = 1.02                   # m, along cell depth
    width: float = 1.22                    # m, along cell width
    height: float = 1.2                    # m, product height EXCLUDING spacers
    use_spacers: bool = True
    n_spacers: int = 6                     # spacer layers between box layers
    spacer_thickness: float = 0.03         # m
    spacer_channels: str = "both"          # see SPACER_CHANNELS
    geometry: str = "auto"                 # "auto" | "plate" | "sphere" | "cylinder"

    @property
    def stack_height(self) -> float:
        return self.height + (self.n_spacers * self.spacer_thickness if self.use_spacers else 0.0)


@dataclass
class Air:
    htc: float = 30.0                      # W/m^2-K, surface HTC with air flowing freely
    spacer_htc_reduction: float = 0.9      # HTC reduction when airflow is perpendicular to the spacers
    cp: float = 1006.0                     # J/kg-K


def _F(f: float) -> float:
    return (f - 32.0) / 1.8


CP_AIR = 1006.0                            # J/kg-K
HP = 745.7                                 # W
CFM = 1.0 / 2118.880003                    # m^3/s


def rho_air(T_c: float) -> float:
    return 101_325.0 / (287.05 * (T_c + 273.15))


def coil_airflow(Q: float, T_supply: float, T_return: float) -> float:
    """Airflow [m^3/s at supply temperature] that carries Q across the coil air temperature drop."""
    return Q / (CP_AIR * rho_air(T_supply) * (T_return - T_supply))


def coil_return(Q: float, T_supply: float, airflow: float) -> float:
    return T_supply + Q / (CP_AIR * rho_air(T_supply) * airflow)


TON = 3516.853                             # W per ton of refrigeration
REFRIGERANT_CHOICES = ("NH3", "CO2", "R404A", "R134a")


@dataclass
class Cooling:
    """Coil, fans and refrigeration system. Supply air comes from the balance each step.

    The coil design return air and the airflow are linked by the design power:
    airflow = Q / (cp * rho(T_supply) * (T_return - T_supply)). Airflow is stored;
    the design return air is derived from it.
    """

    coil_Q_design: float = 32 * TON        # W
    coil_T_supply_design: float = _F(-20.0)  # degC, air leaving the coil
    coil_SST_design: float = _F(-30.0)       # degC
    airflow: float = coil_airflow(32 * TON, _F(-20.0), _F(-12.0))  # m^3/s at supply temp (fixed)
    fan_specific_power: float = HP / (1000 * CFM)  # W per m^3/s (1 hp per 1000 CFM); heat goes to the air
    ref_Q_design: float = 32 * TON         # W, refrigeration capacity at design SST
    ref_SST_design: float = _F(-30.0)      # degC
    refrigerant: str = "NH3"
    SST_max: float = _F(-20.0)             # degC, plant holds suction at or below this
    T_return_setpoint: float = _F(-30.0)   # degC, return air is not allowed below this (frost)
    fixed_supply_T: float | None = None    # degC; if set, bypasses the coil (verification/comparison)

    @property
    def coil_T_return_design(self) -> float:
        return coil_return(self.coil_Q_design, self.coil_T_supply_design, self.airflow)

    @property
    def fan_power(self) -> float:
        return self.fan_specific_power * self.airflow if self.fixed_supply_T is None else 0.0

    @property
    def T_air_reference(self) -> float:
        """Temperature used to turn volumetric airflow into mass flow."""
        return self.fixed_supply_T if self.fixed_supply_T is not None else self.coil_T_supply_design


BOOST_OFF_MODES = ("never", "time", "core")


@dataclass
class Boost:
    """Boost coil: a second coil with its own fans and refrigeration system, in parallel with
    the main coil. Return air splits between the coils and their outlets mix into the supply.

    Same inputs as the main coil (defaults copy it). The return-air setpoint on Cooling applies
    to both; the boost is throttled first. It switches off (fans and refrigeration, so its
    airflow stops) per off_mode: "never", "time" (at off_time_h), or "core" (when the warmest
    pallet core reaches off_core_T). Once off it stays off.
    """

    coil_Q_design: float = 32 * TON
    coil_T_supply_design: float = _F(-20.0)
    coil_SST_design: float = _F(-30.0)
    airflow: float = coil_airflow(32 * TON, _F(-20.0), _F(-12.0))
    fan_specific_power: float = HP / (1000 * CFM)
    ref_Q_design: float = 32 * TON
    ref_SST_design: float = _F(-30.0)
    refrigerant: str = "NH3"
    SST_max: float = _F(-20.0)
    off_mode: str = "never"
    off_time_h: float = 8.0
    off_core_T: float = -5.0

    @property
    def coil_T_return_design(self) -> float:
        return coil_return(self.coil_Q_design, self.coil_T_supply_design, self.airflow)

    @property
    def fan_power(self) -> float:
        return self.fan_specific_power * self.airflow

    @property
    def T_air_reference(self) -> float:
        return self.coil_T_supply_design


CELL_FACES = ("bottom", "top", "front", "back", "left", "right")


@dataclass
class Cell:
    """Cartesian grid of pallet positions (PPs). Air enters one face, leaves the opposite one."""

    n_deep: int = 4
    n_wide: int = 2
    n_high: int = 4
    inlet_face: str = "left"

    @property
    def n_pallets(self) -> int:
        return self.n_deep * self.n_wide * self.n_high


@dataclass
class SimSettings:
    T_start: float = 11.0                  # degC, uniform product temp at load-in
    T_target: float = -17.0                # degC, core target
    n_nodes: int = 40                      # finite volumes core->surface (~1-2% freeze-time error)
    dt_max: float = 60.0                   # s, upper limit; solver uses min(dt_max, stable dt)
    t_max_hours: float = 240.0             # h, hard stop
    record_every_s: float = 600.0          # s, output interval


@dataclass
class Scenario:
    product: Product = field(default_factory=Product)
    packaging: Packaging = field(default_factory=Packaging)
    load: PalletLoad = field(default_factory=PalletLoad)
    air: Air = field(default_factory=Air)
    cell: Cell = field(default_factory=Cell)
    sim: SimSettings = field(default_factory=SimSettings)
    cooling: Cooling = field(default_factory=Cooling)
    boost: Boost = field(default_factory=Boost)

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
            cooling=Cooling(**d.get("cooling", {})),
            boost=Boost(**d.get("boost", {})),
        )
