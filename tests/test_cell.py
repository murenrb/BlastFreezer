"""Tests for packaging/bound-water changes, the cell air path, and units."""

import pytest

from blastfreezer import Cell, Pallet, Scenario, air_path, simulate, summarize
from blastfreezer import units as U
from blastfreezer.config import PACKAGING_PRESETS


def _spacers(s: Scenario) -> Scenario:
    s.load.use_spacers = True
    return s


# ---- materials --------------------------------------------------------------
def test_bound_water_reduces_latent_heat():
    a, b = Scenario(), Scenario()
    a.product.bound_water_fraction = 0.0
    b.product.bound_water_fraction = 0.05
    assert Pallet(b).mat.L_vol == pytest.approx(0.95 * Pallet(a).mat.L_vol)


def test_box_outer_wall_is_surface_resistance():
    s = Scenario()
    p = Pallet(s)
    assert p.mat.R_surface == pytest.approx(0.004 / 0.064)
    bags = Scenario()
    bags.packaging.kind = "Bags (no box)"
    for k, v in PACKAGING_PRESETS["Bags (no box)"].items():
        setattr(bags.packaging, k, v)
    assert Pallet(bags).mat.R_surface == 0.0


# ---- air path ---------------------------------------------------------------
@pytest.mark.parametrize("face,axis,n_series,n_lanes", [
    ("bottom", "height", 4, 8), ("top", "height", 4, 8),
    ("front", "depth", 2, 16), ("left", "width", 4, 8)])
def test_lanes_follow_inlet_face(face, axis, n_series, n_lanes):
    s = Scenario()
    s.cell.n_deep, s.cell.n_wide, s.cell.n_high = 2, 4, 4
    s.cell.inlet_face = face
    p = air_path(s)
    assert (p.axis, p.n_series, p.n_lanes) == (axis, n_series, n_lanes)
    assert p.m_dot_total == pytest.approx(p.m_dot_lane * n_lanes)


def test_spacer_alignment_sets_htc():
    s = _spacers(Scenario())
    s.cell.inlet_face = "bottom"
    assert air_path(s).h == pytest.approx(30.0 * (1 - 0.9))       # vertical: across spacers
    s.cell.inlet_face = "front"
    assert air_path(s).h == pytest.approx(30.0)                      # along channels
    s.load.spacer_channels = "width"
    assert air_path(s).h == pytest.approx(30.0 * (1 - 0.9))       # depth flow across width-only channels
    s.load.use_spacers = False
    assert air_path(s).h == pytest.approx(30.0)                      # no spacers: plain surface


def test_air_warms_by_heat_removed_along_lane():
    s = _spacers(Scenario())
    c = Cell(s)
    lane = c.lanes[0]
    for _ in range(200):
        Q = lane.step(c.dt_stable)
        assert lane.T_out - lane.T_in == pytest.approx(Q / (lane.m_dot * s.air.cp), rel=1e-9)
        for up, down in zip(lane.pallets, lane.pallets[1:]):
            assert down.T_air_in == pytest.approx(up.T_air_out_step)


def test_single_position_cell_matches_standalone_pallet():
    s = _spacers(Scenario())
    s.cell.n_deep = s.cell.n_wide = s.cell.n_high = 1
    s.cell.inlet_face = "front"          # along spacer channels -> full h, same as standalone
    s.cooling.fixed_supply_T = -45.0     # constant supply air, like the standalone pallet
    r = simulate(s)
    p = Pallet(s)
    dt = min(s.sim.dt_max, p.dt_stable)
    while p.t_frozen is None:
        p.step(dt)
    assert r.freeze_times_h[0] == pytest.approx(p.t_frozen / 3600.0)


def test_downstream_pallets_freeze_later_and_across_is_slower():
    s = _spacers(Scenario())
    s.cooling.airflow = 8.0                 # low flow (1 m^3/s per lane) so air warming is visible
    s.cell.n_deep, s.cell.n_wide, s.cell.n_high = 2, 4, 4
    s.cell.inlet_face = "bottom"            # 2 x 4 x 4: 8 lanes of 4
    across = simulate(s)
    ft = across.freeze_times_h
    assert all(a < b for a, b in zip(ft, ft[1:]))
    s.cell.inlet_face = "left"              # 8 lanes of 4 along the width, along channels
    along = simulate(s)
    assert max(along.freeze_times_h) < max(ft)
    assert summarize(along, s)["all_frozen"]


def test_cell_energy_balance():
    r = simulate(_spacers(Scenario()))
    assert r.info["energy_balance_error"] < 1e-9


# ---- units ------------------------------------------------------------------
@pytest.mark.parametrize("q", list(U.Q))
def test_units_roundtrip(q):
    assert U.from_disp(q, U.to_disp(q, 12.34, U.IP), U.IP) == pytest.approx(12.34)


def test_unit_spot_checks():
    assert U.to_disp("temp", -40.0, U.IP) == pytest.approx(-40.0)
    assert U.to_disp("flow", 1.0, U.IP) == pytest.approx(2118.88, rel=1e-5)
    assert U.to_disp("mass", 1349 * 0.45359237, U.IP) == pytest.approx(1349.0)
    assert U.to_disp("power", 3.516853, U.IP) == pytest.approx(1.0)
