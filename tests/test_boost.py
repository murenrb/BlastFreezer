"""Boost coil: parallel coil balance, mixing, throttling order, shutoff."""

import numpy as np
import pytest

from blastfreezer import Scenario, simulate, simulate_pair, summarize
from blastfreezer.cell import Cell
from blastfreezer.config import TON, _F
from blastfreezer.refrigeration import Coil, CoilUnit, RefSystem, solve_balance, solve_parallel


def _unit(scale=1.0, name="main", fan_W=5000.0, refrigerant="NH3", SST_max=None, T_set=None):
    coil = Coil(32 * TON * scale, _F(-20), _F(0), _F(-30))
    ref = RefSystem(32 * TON * scale, _F(-30), refrigerant)
    return CoilUnit(coil, ref, coil.design_C_air, fan_W * scale, SST_max, name, T_set=T_set)


def test_two_identical_units_equal_one_double_unit():
    """Two identical coils in parallel = one coil of twice the size, flow, fans and refrigeration."""
    a, b = -5.0, 0.7
    pair = solve_parallel(a, b, [_unit(name="main"), _unit(name="boost")])
    one = solve_parallel(a, b, [_unit(scale=2.0)])
    assert pair.T_supply == pytest.approx(one.T_supply, abs=1e-8)
    assert pair.T_return == pytest.approx(one.T_return, abs=1e-8)
    assert pair.Q == pytest.approx(one.Q, rel=1e-9)
    assert pair.coils[0].SST == pytest.approx(one.SST, abs=1e-8) == pair.coils[1].SST


def test_outlets_mix_by_capacity_rate():
    units = [_unit(), _unit(scale=0.5, name="boost", refrigerant="CO2")]
    bal = solve_parallel(-5.0, 0.7, units)
    mix = sum(c.C_air * (c.T_out + c.Q_fan / c.C_air) for c in bal.coils) / sum(c.C_air for c in bal.coils)
    assert bal.T_supply == pytest.approx(mix, abs=1e-10)
    assert bal.T_return == pytest.approx(-5.0 + 0.7 * bal.T_supply, abs=1e-8)
    for u, c in zip(units, bal.coils):                     # each coil meets its own refrigeration
        assert c.Q == pytest.approx(c.C_air * u.eps * (bal.T_return - c.SST), rel=1e-9)
    assert bal.Q == pytest.approx(sum(c.Q for c in bal.coils))


def test_single_unit_matches_legacy_wrapper():
    u = _unit()
    coil = Coil(32 * TON, _F(-20), _F(0), _F(-30))
    ref = RefSystem(32 * TON, _F(-30), "NH3")
    w = solve_balance(-5.0, 0.7, coil.design_C_air, coil, ref, fan_W=5000.0)
    p = solve_parallel(-5.0, 0.7, [u])
    assert (w.SST, w.T_supply, w.Q) == pytest.approx((p.SST, p.T_supply, p.Q))


def test_each_coil_holds_its_own_supply_setpoint():
    """Main and boost at different setpoints: each coil throttles to its own outlet; the supply is their mix."""
    a, b = -15.0, 0.5                                      # return air ~ -34 C
    T_main, T_boost = _F(-30), -38.0
    held = solve_parallel(a, b, [_unit(T_set=T_main), _unit(name="boost", T_set=T_boost)])
    main, boost = held.coils
    assert main.T_supply == pytest.approx(T_main) and boost.T_supply == pytest.approx(T_boost)
    assert main.limited == boost.limited == "supply setpoint" and main.Q > 0 and boost.Q > main.Q
    assert held.T_supply == pytest.approx((main.C_air * T_main + boost.C_air * T_boost) / (main.C_air + boost.C_air))
    assert held.T_return == pytest.approx(a + b * held.T_supply)
    assert held.Q == pytest.approx(sum(c.C_air * (held.T_return - c.T_out) for c in held.coils))


def test_cold_boost_unloads_the_main_coil():
    """Once boost air pulls the return below the main setpoint, the main coil has nothing to do: zero load,
    its outlet is return air plus fan heat (it never heats)."""
    a, b = -20.0, 0.5
    held = solve_parallel(a, b, [_unit(T_set=_F(-30)), _unit(name="boost", T_set=_F(-50))])
    main, boost = held.coils
    assert main.Q == pytest.approx(0.0, abs=1e-6) and main.SST == pytest.approx(held.T_return)
    assert main.T_supply == pytest.approx(held.T_return + main.Q_fan / main.C_air) and main.T_supply < _F(-30)
    assert boost.T_supply == pytest.approx(_F(-50)) and boost.Q > 0


def test_setpoint_only_throttles_its_own_coil():
    a, b = -4.0, 0.9
    free = solve_parallel(a, b, [_unit(), _unit(name="boost")])
    T_set = free.coils[1].T_supply + 1.0                   # boost setpoint just above its free outlet
    held = solve_parallel(a, b, [_unit(), _unit(name="boost", T_set=T_set)])
    main, boost = held.coils
    assert boost.T_supply == pytest.approx(T_set) and boost.limited == "supply setpoint"
    assert main.limited != "supply setpoint" and main.T_supply < T_set
    # pallets near -40 C: -30 C can't be held even fully unloaded, so the air floats (never heats)
    held2 = solve_parallel(a, b, [_unit(T_set=-30.0), _unit(name="boost", T_set=-30.0)])
    assert all(c.Q == pytest.approx(0.0, abs=1e-6) and c.SST <= held2.T_return + 1e-9 for c in held2.coils)
    assert held2.T_supply < -30.0


def test_colder_boost_setpoint_shortens_the_blast():
    from dataclasses import replace
    s = Scenario()
    base = simulate(s, boost=True)
    cold = simulate(replace(s, boost=replace(s.boost, T_supply_setpoint=_F(-50))), boost=True)
    C = cold.cell[cold.cell.boost_on]
    assert C.T_boost_supply.min() >= _F(-50) - 1e-6
    below = C.T_main_supply < _F(-30) - 1e-6               # only when the main coil is fully unloaded
    assert np.allclose(C.Q_main_kW[below], 0.0, atol=1e-6)
    assert (C.T_supply >= C[["T_main_supply", "T_boost_supply"]].min(axis=1) - 1e-9).all()
    t = lambda r: summarize(r, s)["cell_freeze_time_h"]
    assert t(cold) < t(base)


def test_boost_off_at_time_drops_airflow():
    s = Scenario()
    s.boost.off_mode, s.boost.off_time_h = "time", 3.0
    r = simulate(s, boost=True)
    C = r.cell
    assert r.info["boost_off_h"] == pytest.approx(3.0, abs=0.01)
    on, off = C[C.time_h < 2.9], C[C.time_h > 3.2]
    assert on.boost_on.all() and not off.boost_on.any()
    assert on.flow_m3s.iloc[-1] == pytest.approx(s.cooling.airflow + s.boost.airflow, rel=1e-6)
    assert off.flow_m3s.iloc[0] == pytest.approx(s.cooling.airflow, rel=1e-6)
    assert off.SST_boost.isna().all() and (off.Q_boost_kW == 0).all()


def test_boost_off_at_warmest_core():
    s = Scenario()
    s.boost.off_mode, s.boost.off_core_T = "core", -5.0
    r = simulate(s, boost=True)
    C = r.cell
    t_off = r.info["boost_off_h"]
    assert t_off is not None
    before = C[C.time_h < t_off - 0.2]
    assert (before.T_core_warmest > -5.0).all() and before.boost_on.all()
    assert C[C.time_h >= t_off].T_core_warmest.iloc[0] <= -5.0 + 0.05


def test_boost_speeds_up_the_blast_and_conserves_energy():
    s = Scenario()
    off, on = simulate_pair(s)
    t_off, t_on = summarize(off, s)["cell_freeze_time_h"], summarize(on, s)["cell_freeze_time_h"]
    assert t_on < t_off
    assert on.info["energy_balance_error"] < 1e-10
    assert on.cell.T_supply.min() >= s.cooling.T_supply_setpoint - 1e-6
    assert off.info["boost_off_h"] is None and not off.cell.boost_on.any()


def test_stable_step_follows_airflow():
    s = Scenario()
    c0, c1 = Cell(s), Cell(s, boost=True)
    assert c1.dt_stable <= c0.dt_stable
    assert c1.lanes[0].m_dot == pytest.approx(c0.lanes[0].m_dot * (1 + c1.m_boost / c1.m_main))


def test_average_and_peak_cooling():
    s = Scenario()
    s.boost.off_mode, s.boost.off_time_h = "time", 4.0
    s.sim.t_max_hours = 8.0                                # stop early: averages cover whatever ran
    r = simulate(s, boost=True)
    sm, C = summarize(r, s), r.cell
    t = C.time_h.to_numpy()
    E_boost = np.trapezoid(C.Q_boost_kW.to_numpy(), t)
    assert sm["avg_boost_kW"] == pytest.approx(E_boost / t[-1])
    assert sm["avg_coil_kW"] == pytest.approx(sm["avg_main_kW"] + sm["avg_boost_kW"])
    assert 0 < sm["avg_boost_kW"] < sm["peak_boost_kW"] <= sm["peak_coil_kW"]
    # boost ran 4 of 8 h, so its average is well under its average while running
    running = C[C.time_h <= 4.0]
    assert sm["avg_boost_kW"] < 0.6 * running.Q_boost_kW.mean()
