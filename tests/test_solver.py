"""Verification tests for the single-pallet freezing model.

1. Pure conduction (no phase change) vs. the exact series solution for a
   plate and a sphere with convective boundary.
2. Plank's freezing-time equation in the low-Stefan-number limit.
3. Exact energy conservation of the explicit scheme.
4. Geometry bookkeeping (spacers -> layers, sphere radius).
"""

import math

import numpy as np
import pytest
from scipy.optimize import brentq

from blastfreezer import Pallet, Scenario
from blastfreezer.geometry import representative_body


def _solid_block_scenario(shape: str, dims, n_nodes=40, T_freeze=-200.0, water=0.0,
                          cp_frozen=1500.0, k=0.5, h=30.0) -> Scenario:
    """Unpackaged, void-free block with huge airflow (h_eff -> h)."""
    s = Scenario()
    L, W, H = dims
    rho = 1000.0
    s.product.density = rho
    s.product.water_fraction = water
    s.product.bound_water_fraction = 0.0
    s.product.T_freeze = T_freeze
    s.product.cp_unfrozen = 3000.0
    s.product.cp_frozen = cp_frozen
    s.product.k_unfrozen = k
    s.product.k_frozen = k
    s.packaging.wall_thickness = 0.0
    s.packaging.air_gap = 0.0
    s.load.use_spacers = False
    s.load.geometry = shape
    s.load.length, s.load.width, s.load.height = L, W, H
    s.load.mass = rho * L * W * H
    s.air.htc = h
    s.air.flow_total = 1e6
    s.sim.n_nodes = n_nodes
    return s


def _plate_roots(Bi, n=30):
    return [brentq(lambda z: z * math.tan(z) - Bi, i * math.pi + 1e-9, i * math.pi + math.pi / 2 - 1e-9)
            for i in range(n)]


def _sphere_roots(Bi, n=30):
    return [brentq(lambda z: 1 - z / math.tan(z) - Bi, i * math.pi + 1e-9, (i + 1) * math.pi - 1e-9)
            for i in range(n)]


@pytest.mark.parametrize("shape", ["plate", "sphere"])
def test_conduction_matches_analytic(shape):
    T0, Tinf, k, h, C = 10.0, -40.0, 0.5, 30.0, 1000.0 * 3000.0
    dims = (0.2, 2.0, 2.0) if shape == "plate" else (0.6, 0.6, 0.6)
    s = _solid_block_scenario(shape, dims, n_nodes=40, k=k, h=h)
    s.sim.T_start, s.cooling.fixed_supply_T = T0, Tinf
    p = Pallet(s)
    R = p.body.R
    Bi, alpha = h * R / k, k / C
    zs = _plate_roots(Bi) if shape == "plate" else _sphere_roots(Bi)

    def exact(x, t):
        Fo, th = alpha * t / R**2, 0.0
        for z in zs:
            if shape == "plate":
                Cn = 4 * math.sin(z) / (2 * z + math.sin(2 * z))
                th += Cn * math.exp(-z * z * Fo) * math.cos(z * x / R)
            else:
                Cn = 4 * (math.sin(z) - z * math.cos(z)) / (2 * z - math.sin(2 * z))
                th += Cn * math.exp(-z * z * Fo) * math.sin(z * x / R) / (z * x / R)
        return Tinf + (T0 - Tinf) * th

    dt = p.dt_stable
    for t_check in (2 * 3600, 8 * 3600):
        while p.t + dt <= t_check:
            p.step(dt)
        x = p.grid.x_centers
        err = np.abs(p.T_nodes - np.array([exact(xi, p.t) for xi in x]))
        assert err.max() < 0.25, f"{shape} t={t_check/3600}h max error {err.max():.3f} K"


def test_plank_freezing_time_plate():
    """Low Stefan number, start at the freezing point -> Plank's equation."""
    Tf, Ta, k, h = -1.0, -41.0, 1.4, 30.0
    s = _solid_block_scenario("plate", (0.1, 2.0, 2.0), n_nodes=40, T_freeze=Tf, water=1.0,
                              cp_frozen=100.0, k=k, h=h)
    s.cooling.fixed_supply_T = Ta
    s.sim.T_start = Tf + 1e-6
    p = Pallet(s)
    a = 0.1
    t_plank = 1000.0 * 334_000.0 / (Tf - Ta) * (a / (2 * h) + a * a / (8 * k))
    dt = p.dt_stable
    while p.frozen_nodes.min() < 1.0:
        p.step(dt)
    assert abs(p.t - t_plank) / t_plank < 0.03, f"model {p.t:.0f}s vs Plank {t_plank:.0f}s"


def test_energy_conserved_exactly():
    s = Scenario()
    s.sim.n_nodes = 15
    p = Pallet(s)
    for _ in range(3000):
        p.step(p.dt_stable)
    assert abs(p.energy_balance_error) < 1e-9


def test_standalone_pallet_core_and_front_monotonic():
    s = Scenario()
    s.load.use_spacers = True
    p = Pallet(s)
    T_prev, d_prev = p.T_core, p.front_depth
    while p.t_frozen is None:
        p.step(p.dt_stable)
        assert p.T_core <= T_prev + 1e-9 and p.front_depth >= d_prev - 1e-9
        T_prev, d_prev = p.T_core, p.front_depth
    assert p.frozen_fraction == pytest.approx(1.0)


def test_spacer_geometry():
    s = Scenario()
    s.load.use_spacers = True     # 1.2 m of product, 6 spacers -> 7 layers
    b = representative_body(s.load)
    assert b.shape == "plate" and b.n_layers == 7
    assert b.R == pytest.approx(1.2 / 7 / 2)
    assert b.area_pallet == pytest.approx(2 * 1.02 * 1.22 * 7)
    assert s.load.stack_height == pytest.approx(1.2 + 6 * 0.03)


def test_sphere_radius_is_3V_over_A():
    s = Scenario()
    s.load.use_spacers = False
    b = representative_body(s.load)
    L, W, H = 1.02, 1.22, 1.2
    assert b.shape == "sphere"
    assert b.R == pytest.approx(3 * L * W * H / (2 * (L * W + W * H + L * H)))


def test_product_that_does_not_fit_is_rejected():
    s = Scenario()
    s.load.mass = 5000.0
    with pytest.raises(ValueError, match="does not fit"):
        Pallet(s)


def test_scenario_roundtrip():
    s = Scenario()
    assert Scenario.from_dict(s.to_dict()) == s
