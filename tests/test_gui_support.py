"""Tests for the food library and the cell-view helpers."""

import numpy as np
import pytest

from blastfreezer import Scenario, simulate
from blastfreezer import units as U
from blastfreezer.config import CELL_FACES
from blastfreezer.foods import FOODS, ice_fraction, product_kwargs
from blastfreezer.viz import (GUI_FACES, air_temperature_figure, cell_figure, cooling_bands, cooling_figure,
                              extreme_positions, product_temperature_figure, ref_temperature_figure,
                              series_position)

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
    heads = [t for t in fig.data if t.name == "arrowhead"]
    fills = [t for t in fig.data if t.fill == "toself" and t.name not in ("arrowhead", "coil box")]
    assert len(fills) == 2 * 3 * 4
    assert len(heads) == 5                         # 2 on the boost branch, supply in, return out, into coil
    sup = heads[2]
    tip, base = (sup.x[0], sup.y[0]), ((sup.x[1] + sup.x[2]) / 2, (sup.y[1] + sup.y[2]) / 2)
    into = {"left": tip[0] > base[0], "right": tip[0] < base[0],      # first head points into the inlet face
            "bottom": tip[1] > base[1], "top": tip[1] < base[1]}
    assert into[face]
    lines = [t for t in fig.data if t.mode == "lines" and t.fill is None]
    supply_line = lines[2]                         # boost in, boost out, supply, return
    end = (supply_line.x[-1], supply_line.y[-1])   # line stops inside the head, short of the tip
    assert min(abs(end[0] - tip[0]), abs(end[1] - tip[1])) < 1e-9 < max(abs(end[0] - tip[0]), abs(end[1] - tip[1]))
    assert "Main Coil" in {a.text for a in fig.layout.annotations}
    boost_text = [t for t in fig.data if t.mode == "text" and t.text and "Boost Coil" in t.text][0]
    assert boost_text.textfont.color == "#c3c7cc"  # greyed out: no boost in this figure
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
    assert fig.data[first.traces[0]].fill == "toself"             # frames point at the pallet traces
    assert fig.layout.sliders[0].steps[-1].label == "25.0"         # last time step included


def test_cell_figure_boost_turns_grey_when_off():
    n_t, face = 6, "left"
    T = np.linspace(11, -20, n_t)[:, None] + np.zeros((1, N_SERIES[face]))
    on = [True, True, True, False, False, False]
    fig = cell_figure(2, 3, 4, face, T, list(range(n_t)), [-10.0] * n_t, 11, -17, cv, ul, [-30.0] * n_t,
                      [f"{q} m3/s" for q in (34, 34, 34, 17, 17, 17)], boost_frames=on,
                      SST_boost_frames=[-40.0, -40, -40, None, None, None])
    box = [i for i, t in enumerate(fig.data) if t.name == "coil box"][0]
    styles = []
    for f in fig.frames:
        d = dict(zip(f.traces, f.data))
        styles.append(d[box].line.color)
        assert ("34 m3/s" if on[int(f.name)] else "17 m3/s") in "".join(t for t in sum(
            [list(x.text) for x in f.data if x.text is not None and not isinstance(x.text, str)], []))
    assert styles[:3] == ["#2b3a4a"] * 3 and styles[3:] == ["#d0d4d8"] * 3


def test_cell_figure_rejects_depth_flow():
    with pytest.raises(ValueError):
        cell_figure(2, 3, 4, "front", [[0, 0]], [0.0], [None], 11, -17, cv, ul, [None])


def test_time_series_figures_build():
    s = Scenario()
    s.cell.n_deep, s.cell.n_wide, s.cell.n_high = 1, 1, 3
    s.cell.inlet_face = "bottom"          # 3 pallets in series
    s.boost.off_mode, s.boost.off_time_h = "time", 2.0
    off, on = simulate(s), simulate(s, boost=True)
    prod, air = product_temperature_figure(off, on, cv, ul), air_temperature_figure(off, on, cv, ul)
    ref = ref_temperature_figure(off, on, cv, ul)
    assert len(prod.data) == 2 + 2         # core, surface per run
    assert len(air.data) == 2 + 2          # supply, return per run
    assert len(ref.data) == 1 + 2          # main SST; + boost SST on the boost run
    assert all("warmest" in t.name for t in prod.data)
    assert all(t.name.startswith("SST") for t in ref.data)
    lines = prod.data + air.data + ref.data
    for t in lines:                        # dashed without boost, solid with
        assert t.line.dash == ("solid" if t.name.endswith("· boost") else "dash")
    colors = {t.legendgroup: t.line.color for t in lines}
    assert len(set(colors.values())) == len(colors)          # one colour per quantity
    assert len(product_temperature_figure(off, None, cv, ul).data) == 2
    assert len(air_temperature_figure(off, None, cv, ul).data) == 2
    assert len(ref_temperature_figure(off, None, cv, ul).data) == 1

    cool = cooling_figure(off, on, cv, ul, design_kW=100.0)
    stacked = [t for t in cool.data if t.stackgroup]
    assert [t.legendgroup for t in stacked] == ["fan", "main", "boost"]       # bottom to top
    assert [t.line.dash for t in cool.data if not t.stackgroup] == ["dash"]   # total without boost


def test_cooling_bands_stack_to_total_load():
    s = Scenario()                         # full default cell: a tiny cell throttles the boost to zero
    s.sim.t_max_hours = 4.0
    s.boost.off_mode, s.boost.off_time_h = "time", 2.0
    C = simulate(s, boost=True).cell
    b = cooling_bands(C)
    assert np.allclose(b["fan"] + b["main"] + b["boost"], C.Q_coil_kW)
    assert all((v >= 0).all() for v in b.values())
    after = (C.time_h > 2.0).to_numpy()
    assert after.any() and (b["boost"][after] == 0).all()                    # boost off: band gone
    assert (b["boost"][~after][1:] > 0).all()
