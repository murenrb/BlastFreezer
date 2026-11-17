"""Boost coil: parallel coil balance, mixing, throttling order, shutoff."""

import numpy as np
import pytest

from blastfreezer import Scenario, simulate, simulate_pair, summarize
from blastfreezer.cell import Cell
from blastfreezer.config import TON, _F
from blastfreezer.refrigeration import Coil, CoilUnit, RefSystem, solve_balance, solve_parallel


def _unit(scale=1.0, name="main", fan_W=5000.0, refrigerant="NH3", SST_max=None):
    coil = Coil(32 * TON * scale, _F(-20), _F(0), _F(-30))
    ref = RefSystem(32 * TON * scale, _F(-30), refrigerant)
    return CoilUnit(coil, ref, coil.design_C_air, fan_W * scale, SST_max, name)


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


def test_supply_setpoint_throttles_boost_first():
    a, b = -4.0, 0.9                                       # light load: free air ~ -60 C
    units = [_unit(), _unit(name="boost")]
    free = solve_parallel(a, b, units)
    T_set = free.T_supply + 1.0                            # small correction: boost alone can absorb it
    held = solve_parallel(a, b, units, T_supply_min=T_set)
    main, boost = held.coils
    assert held.T_supply == pytest.approx(T_set) and held.T_return == pytest.approx(a + b * T_set)
    assert boost.limited == "supply setpoint" and main.limited != "supply setpoint"
    assert boost.SST > main.SST                            # boost unloaded, main still at its free SST
    held2 = solve_parallel(a, b, units, T_supply_min=-42.0)  # big correction: boost fully off, then main
    assert held2.coils[1].Q == pytest.approx(0.0, abs=1e-6)
    assert held2.coils[0].limited == "supply setpoint" and held2.coils[0].Q > 0
    assert held2.T_supply == pytest.approx(-42.0)
    # pallets here sit near -40 C: a -30 C supply can't be held even with both coils unloaded,
    # so the air floats (coils never turn into heaters)
    held3 = solve_parallel(a, b, units, T_supply_min=-30.0)
    assert all(c.Q == pytest.approx(0.0, abs=1e-6) for c in held3.coils)
    assert held3.T_supply < -30.0


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
