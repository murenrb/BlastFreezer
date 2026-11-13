"""Homogenized ("smeared") properties of the packed product.

The packed load is treated as one uniform material per unit envelope volume,
built from three parts in series along the heat-flow path:

  1. Package walls (cardboard / film): 2 walls per package height
  2. Headspace air gap: 1 gap per package height
  3. Package contents: product with air voids, via Maxwell-Eucken
     (product continuous, air dispersed)

The void fraction comes from the pallet mass vs. the available volume, so a
loosely packed pallet automatically gets lower k and lower heat capacity.

Phase change uses an isothermal enthalpy method. H is J/m^3 with H = 0 at
"fully frozen, at T_freeze":

    H < 0          frozen,   T = Tf + H / C_frozen
    0 <= H <= L    freezing, T = Tf,  frozen fraction f = 1 - H / L
    H > L          unfrozen, T = Tf + (H - L) / C_unfrozen

k is blended linearly with the frozen fraction, k = (1-f) k_u + f k_f.
Latent heat is handled exactly by H, so no effective-cp spike is needed.
To use a gradual freezing curve later (e.g. Schwartzberg), replace
`enthalpy` and `state`; nothing else needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Packaging, Product


@dataclass(frozen=True)
class Material:
    T_freeze: float
    C_unfrozen: float      # J/m^3-K, volumetric heat capacity
    C_frozen: float
    L_vol: float           # J/m^3, latent heat per envelope volume
    k_unfrozen: float      # W/m-K, effective
    k_frozen: float
    rho_bulk: float        # kg/m^3, product + packaging per envelope volume
    phi_product: float     # volume fractions of the envelope
    phi_void: float
    phi_wall: float
    phi_gap: float

    # ---- enthalpy <-> state -------------------------------------------------
    def enthalpy(self, T):
        T = np.asarray(T, dtype=float)
        dT = T - self.T_freeze
        return np.where(dT > 0, self.L_vol + self.C_unfrozen * dT, self.C_frozen * dT)

    def state(self, H):
        """Return (T, frozen_fraction) for an enthalpy array."""
        H = np.asarray(H, dtype=float)
        T = np.where(H < 0.0, self.T_freeze + H / self.C_frozen,
                     np.where(H > self.L_vol, self.T_freeze + (H - self.L_vol) / self.C_unfrozen,
                              self.T_freeze))
        if self.L_vol > 0:
            f = np.clip(1.0 - H / self.L_vol, 0.0, 1.0)
        else:
            f = (H <= 0.0).astype(float)
        return T, f

    def k(self, f):
        return (1.0 - f) * self.k_unfrozen + f * self.k_frozen

    @property
    def k_max(self) -> float:
        return max(self.k_unfrozen, self.k_frozen)

    @property
    def C_min(self) -> float:
        return min(self.C_unfrozen, self.C_frozen)


def maxwell_eucken(k_cont: float, k_disp: float, phi_disp: float) -> float:
    num = 2 * k_cont + k_disp - 2 * phi_disp * (k_cont - k_disp)
    den = 2 * k_cont + k_disp + phi_disp * (k_cont - k_disp)
    return k_cont * num / den


def build_material(product: Product, pkg: Packaging, mass: float, envelope_volume: float) -> Material:
    h = pkg.package_height
    phi_wall = 2.0 * pkg.wall_thickness / h if h > 0 else 0.0
    phi_gap = pkg.air_gap / h if h > 0 else 0.0
    phi_contents = 1.0 - phi_wall - phi_gap
    if phi_contents <= 0:
        raise ValueError("Package walls and air gap fill the whole package height.")

    phi_product = mass / product.density / envelope_volume
    if phi_product > phi_contents + 1e-9:
        raise ValueError(
            f"Product does not fit: needs {phi_product:.0%} of the envelope volume, "
            f"only {phi_contents:.0%} is available inside packaging. Check mass, "
            f"density and dimensions.")
    void_in_contents = max(0.0, 1.0 - phi_product / phi_contents)

    def k_series(k_prod: float) -> float:
        k_c = maxwell_eucken(k_prod, pkg.k_air, void_in_contents)
        r = phi_contents / k_c
        if phi_wall > 0:
            r += phi_wall / pkg.k_wall
        if phi_gap > 0:
            r += phi_gap / pkg.k_air
        return 1.0 / r

    rho_prod = mass / envelope_volume                 # product mass per envelope volume
    rho_wall = phi_wall * pkg.wall_density
    C_wall = rho_wall * pkg.cp_wall

    return Material(
        T_freeze=product.T_freeze,
        C_unfrozen=rho_prod * product.cp_unfrozen + C_wall,
        C_frozen=rho_prod * product.cp_frozen + C_wall,
        L_vol=rho_prod * product.water_fraction * product.latent_heat_water,
        k_unfrozen=k_series(product.k_unfrozen),
        k_frozen=k_series(product.k_frozen),
        rho_bulk=rho_prod + rho_wall,
        phi_product=phi_product,
        phi_void=phi_contents - phi_product,
        phi_wall=phi_wall,
        phi_gap=phi_gap,
    )
