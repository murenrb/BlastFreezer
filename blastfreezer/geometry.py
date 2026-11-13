"""Geometry: reduce a pallet to a 1-D representative body and build its grid.

A pallet is mapped to one of three 1-D shapes:

    plate     heat flows through two faces (half-thickness R, symmetric)
    cylinder  radial heat flow (radius R)            -- available, not auto-selected yet
    sphere    radial heat flow (radius R)

The finite-volume solver only sees cell volumes and face areas, so adding a
new geometry means adding a branch in `representative_body` (and, if it is a
new 1-D shape, an exponent in SHAPE_EXPONENT). Nothing in the solver changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import PalletLoad

# A(r) ∝ r**p : plate p=0, cylinder p=1, sphere p=2
SHAPE_EXPONENT = {"plate": 0, "cylinder": 1, "sphere": 2}

# Auto-selection: a block is treated as a plate when its smallest dimension is
# less than this fraction of its middle dimension.
PLATE_ASPECT_LIMIT = 0.5


@dataclass(frozen=True)
class Body:
    """The 1-D body that represents the pallet."""

    shape: str            # plate | cylinder | sphere
    R: float              # m, half-thickness (plate) or radius
    area_pallet: float    # m^2, heat transfer area of the whole pallet
    volume_pallet: float  # m^3, product envelope volume of the whole pallet (no spacer gaps)
    n_layers: int         # identical layers the pallet is split into
    block_dims: tuple     # (a, b, c) of the block that was reduced, sorted ascending
    note: str


def representative_body(load: PalletLoad) -> Body:
    if load.use_spacers and load.n_spacers > 0:
        n_layers = load.n_spacers + 1
        h_block = load.height / n_layers      # height is product only; spacers sit between layers
    else:
        n_layers = 1
        h_block = load.height

    a, b, c = sorted((load.length, load.width, h_block))
    V_block = a * b * c
    shape = load.geometry
    if shape == "auto":
        shape = "plate" if a < PLATE_ASPECT_LIMIT * b else "sphere"

    if shape == "plate":
        # Two large faces exposed; edges ignored (conservative for heat load,
        # negligible for the core of a thin layer).
        R = a / 2.0
        area_block = 2.0 * b * c
        note = (f"Plate: {n_layers} layer(s) of {a*1000:.0f} mm thickness, cooled on both "
                f"{b:.2f} × {c:.2f} m faces (edges ignored).")
    elif shape == "sphere":
        # Equivalent sphere with the block's volume-to-surface ratio: R = 3V/A.
        area_block = 2.0 * (a * b + b * c + a * c)
        R = 3.0 * V_block / area_block
        note = (f"Sphere: radius {R*1000:.0f} mm, same volume/surface ratio as the "
                f"{a:.2f} × {b:.2f} × {c:.2f} m block.")
    elif shape == "cylinder":
        # Infinite cylinder along the longest axis, same V/A over the side surface: R = 2V/A.
        area_block = 2.0 * (a + b) * c
        R = 2.0 * V_block / area_block
        note = f"Cylinder: radius {R*1000:.0f} mm along the {c:.2f} m axis (ends ignored)."
    else:
        raise ValueError(f"Unknown geometry '{shape}'")

    return Body(shape=shape, R=R, area_pallet=area_block * n_layers,
                volume_pallet=V_block * n_layers, n_layers=n_layers,
                block_dims=(a, b, c), note=note)


@dataclass(frozen=True)
class Grid:
    """Uniform cell-centred 1-D finite-volume grid for a unit body.

    'Unit body' = 1 m^2 of face (plate), 1 m of length (cylinder), or the whole
    sphere. Index 0 is the core, index N-1 is the surface cell.
    """

    shape: str
    R: float
    dx: float
    x_centers: np.ndarray   # (N,)   distance from core
    x_faces: np.ndarray     # (N+1,)
    A_faces: np.ndarray     # (N+1,) face areas
    V_cells: np.ndarray     # (N,)   cell volumes

    @property
    def n(self) -> int:
        return len(self.V_cells)

    @property
    def A_surface(self) -> float:
        return float(self.A_faces[-1])

    @property
    def V_total(self) -> float:
        return float(self.V_cells.sum())

    def cumulative_volume(self, r):
        """Volume inside radius/half-thickness r (unit body)."""
        p = SHAPE_EXPONENT[self.shape]
        return {0: lambda r: r, 1: lambda r: np.pi * r**2,
                2: lambda r: 4.0 / 3.0 * np.pi * r**3}[p](r)

    def radius_of_volume(self, v):
        p = SHAPE_EXPONENT[self.shape]
        v = max(v, 0.0)
        return {0: lambda v: v, 1: lambda v: np.sqrt(v / np.pi),
                2: lambda v: (3.0 * v / (4.0 * np.pi)) ** (1.0 / 3.0)}[p](v)


def make_grid(shape: str, R: float, n: int) -> Grid:
    if n < 2:
        raise ValueError("Need at least 2 nodes.")
    p = SHAPE_EXPONENT[shape]
    x_f = np.linspace(0.0, R, n + 1)
    if p == 0:
        A = np.ones_like(x_f)
    elif p == 1:
        A = 2.0 * np.pi * x_f
    else:
        A = 4.0 * np.pi * x_f**2
    g = Grid(shape, R, R / n, 0.5 * (x_f[:-1] + x_f[1:]), x_f, A, np.empty(0))
    V = np.diff(g.cumulative_volume(x_f))
    return Grid(shape, R, R / n, g.x_centers, x_f, A, V)
