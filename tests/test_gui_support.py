"""Tests for the food library and the cell-view helpers."""

import numpy as np
import pytest

from blastfreezer import Scenario, simulate
from blastfreezer import units as U
from blastfreezer.config import CELL_FACES
from blastfreezer.foods import FOODS, ice_fraction, product_kwargs
from blastfreezer.viz import (GUI_FACES, cell_figure, extreme_positions, power_figure, series_position,
                              temperature_figure)

cv = lambda q, v: U.to_disp(q, v, U.SI)
ul = lambda q: U.unit(q, U.SI)


# ---- foods ------------------------------------------------------------------
def test_library_has_requested_products():
    for name in ["Steamed wings", "Chicken patty", "Uncooked chicken", "Cooked bone-in wing",
                 "Plant based protein product", "Sliced peaches", "Blueberries", "Raspberries",
                 "Beef, carcass (choice)", "Sausage, frankfurter"]:
        assert name in FOODS


@pytest.mark.parametrize("name", list(FOODS))
def test_composition_and_properties_are_physical(name):
    f = FOODS[name]
    assert f.water + f.protein + f.fat + f.carb + f.fiber + f.ash == pytest.approx(1.0, abs=0.01)
    assert 0.3 < f.k_unfrozen < 0.7
    assert 1.0 < f.k_frozen < 2.2
    assert 950 < f.density < 1150
    assert 0.0 < ice_fraction(f, -18.0) < f.water


def test_ashrae_values_carried_exactly():
    beef = product_kwargs("Beef, carcass (choice)")
    assert (beef["water_fraction"], beef["cp_unfrozen"], beef["cp_frozen"], beef["T_freeze"]) == \
        (0.5726, 3240.0, 2310.0, -2.2)
    frank = product_kwargs("Sausage, frankfurter")
    assert (frank["water_fraction"], frank["cp_unfrozen"], frank["cp_frozen"], frank["T_freeze"]) == \
        (0.5387, 3150.0, 2310.0, -1.7)


def test_choi_okos_matches_ashrae_lean_beef_measurement():
    # ASHRAE Table 5: lean beef (75 % water) k = 0.506 W/m-K at 3 C
    from blastfreezer.foods import Food, conductivity
    lean = Food("lean", 0.75, 3500, 2100, -1.7, 0.21, 0.03, 0.0, 0.0, 0.01, "test")
    assert conductivity(lean, 3.0) == pytest.approx(0.506, rel=0.08)


# ---- viz ----------------------------------------------------------------------
def test_series_position_counts_from_inlet():
    n = {"depth": 2, "width": 3, "height": 4}
    idx = {"depth": 0, "width": 0, "height": 0}
    assert series_position("bottom", idx, n) == 0
    assert series_position("top", idx, n) == 3
    assert series_position("front", idx, n) == 0
    assert series_position("back", idx, n) == 1
    assert series_position("right", idx, n) == 2


def test_extreme_positions():
    assert extreme_positions([5.0, 6.0, 7.0], [-20, -19, -18]) == (1, 3)
    assert extreme_positions([5.0, None, 7.0], [-20, -5, -18]) == (1, 2)


N_SERIES = {"bottom": 4, "top": 4, "left": 3, "right": 3}


@pytest.mark.parametrize("face", GUI_FACES)
def test_cell_figure_draws_every_pallet(face):
    fig = cell_figure(2, 3, 4, face, [list(range(N_SERIES[face]))], [0.0], [None], 11, -17, cv, ul, [None])
    fills = [t for t in fig.data if t.fill == "toself"]
    assert len(fills) == 2 * 3 * 4
    assert len([a for a in fig.layout.annotations if a.showarrow]) == 2
    assert not fig.frames and not fig.layout.sliders


def test_cell_figure_animation_frames():
    n_t, face = 500, "left"
    T = np.linspace(11, -20, n_t)[:, None] + np.zeros((1, N_SERIES[face]))
    times = list(np.linspace(0, 25, n_t))
    fig = cell_figure(2, 3, 4, face, T, times, list(np.full(n_t, -44.0)), 11, -17, cv, ul,
                      list(np.full(n_t, -45.0)))
    assert 2 <= len(fig.frames) <= 240
    assert len(fig.layout.sliders[0].steps) == len(fig.frames)
    first, last = fig.frames[0], fig.frames[-1]
    assert first.data[0].fillcolor != last.data[0].fillcolor      # colours change over time
    assert fig.layout.sliders[0].steps[-1].label == "25.0"         # last time step included


def test_cell_figure_rejects_depth_flow():
    with pytest.raises(ValueError):
        cell_figure(2, 3, 4, "front", [[0, 0]], [0.0], [None], 11, -17, cv, ul, [None])


def test_time_series_figures_build():
    s = Scenario()
    s.load.use_spacers = True
    s.cell.n_deep, s.cell.n_wide, s.cell.n_high = 1, 1, 3
    s.cell.inlet_face = "bottom"          # 3 pallets in series -> distinct warmest / coldest
    r = simulate(s)
    assert len(temperature_figure(r, s.sim.T_target, cv, ul).data) == 7      # incl. SST
    assert len(power_figure(r, cv, ul).data) == 3
