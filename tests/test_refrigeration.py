"""Coil, refrigeration system, and the coupled cell balance."""

import math

import pytest

from blastfreezer import Scenario, simulate
from blastfreezer.cell import Cell
from blastfreezer.config import TON, _F
from blastfreezer.refrigeration import (REFRIGERANTS, Coil, RefSystem, design_return_balance,
                                        solve_balance)


def _coil():
    return Coil(32 * TON, _F(-20), _F(0), _F(-30))


def test_coil_ua_from_lmtd():
    c = _coil()
    lmtd = (30 - 10) / 1.8 / math.log(30 / 10)            # dT1 = 30 F, dT2 = 10 F
    assert c.LMTD == pytest.approx(lmtd)
    assert c.UA == pytest.approx(32 * TON / lmtd)


@pytest.mark.parametrize("refrigerant", list(REFRIGERANTS))
def test_design_point_is_reproduced(refrigerant):
    """Design airflow + design return air + matching ref -> design SST, supply air and load."""
    c = _coil()
    r = RefSystem(32 * TON, _F(-30), refrigerant)
    b = design_return_balance(c, r, c.design_C_air)
    assert b.SST == pytest.approx(_F(-30), abs=1e-3)
    assert b.T_supply == pytest.approx(_F(-20), abs=1e-3)
    assert b.Q == pytest.approx(32 * TON, rel=1e-5)


@pytest.mark.parametrize("refrigerant", list(REFRIGERANTS))
def test_ref_capacity_rises_with_sst(refrigerant):
    r = RefSystem(32 * TON, _F(-30), refrigerant)
    assert float(r.capacity(_F(-30))) == pytest.approx(32 * TON)
    assert r.capacity(_F(-20)) > r.capacity(_F(-30)) > r.capacity(_F(-40))


def test_balance_closes_all_three_heat_flows():
    c, r = _coil(), RefSystem(32 * TON, _F(-30), "NH3")
    C_air = 40_000.0
    a, b = -5.0, 0.7                                      # some lane: T_r = a + b T_s
    bal = solve_balance(a, b, C_air, c, r)
    eps = c.effectiveness(C_air)
    assert bal.T_return == pytest.approx(a + b * bal.T_supply)
    assert bal.Q == pytest.approx(eps * C_air * (bal.T_return - bal.SST), rel=1e-6)
    assert bal.Q == pytest.approx(float(r.capacity(bal.SST)), rel=1e-6)


def test_cell_return_matches_linear_prediction():
    s = Scenario()
    cell = Cell(s)
    for _ in range(50):
        cell.step(cell.dt_stable)
        bal = cell.balance
        assert cell.T_air_out == pytest.approx(bal.T_return, abs=1e-9)
        assert cell.Q + bal.Q_fan == pytest.approx(bal.Q, rel=1e-9)       # coil = pallets + fan heat
        assert bal.T_supply - bal.T_coil_out == pytest.approx(bal.Q_fan / cell.C_air)


def test_airflow_and_design_return_are_linked():
    from blastfreezer.config import Cooling, coil_airflow
    c = Cooling()
    assert c.coil_T_return_design == pytest.approx(_F(-12.0))
    assert coil_airflow(c.coil_Q_design, c.coil_T_supply_design, c.coil_T_return_design) == \
        pytest.approx(c.airflow)
    assert c.fan_power == pytest.approx(745.7 * c.airflow * 2118.880003 / 1000)   # 1 hp per 1000 CFM


def test_sst_max_caps_suction():
    c, r = _coil(), RefSystem(32 * TON, _F(-30), "NH3")
    free = solve_balance(5.0, 0.5, c.design_C_air, c, r)
    capped = solve_balance(5.0, 0.5, c.design_C_air, c, r, SST_max=free.SST - 3.0)
    assert capped.SST == pytest.approx(free.SST - 3.0) and capped.limited == "SST max"
    assert capped.Q > free.Q                    # plant supplies more to hold suction down


def test_return_setpoint_holds_return_air():
    c, r = _coil(), RefSystem(32 * TON, _F(-30), "NH3")
    a, b = -4.0, 0.9                            # light load: free balance goes to about -61 C
    free = solve_balance(a, b, c.design_C_air, c, r)
    held = solve_balance(a, b, c.design_C_air, c, r, T_return_min=free.T_return + 2.0)
    assert held.T_return == pytest.approx(free.T_return + 2.0)
    assert held.SST > free.SST and held.limited == "return setpoint"
    assert held.Q == pytest.approx(c.design_C_air * (held.T_return - held.T_coil_out))


def test_default_run_respects_controls():
    s = Scenario()
    C = simulate(s).cell
    assert C.T_return.min() >= s.cooling.T_return_setpoint - 1e-6
    assert C.SST.max() <= s.cooling.SST_max + 1e-6
    assert (C.limit == "return setpoint").any()


def test_more_refrigeration_freezes_faster():
    s = Scenario()
    base = max(simulate(s).freeze_times_h)
    s.cooling.ref_Q_design = s.cooling.coil_Q_design = 64 * TON
    assert max(simulate(s).freeze_times_h) < base


def test_co2_sst_stays_above_triple_point():
    s = Scenario()
    s.cooling.refrigerant = "CO2"
    r = simulate(s)
    assert r.cell.SST.min() >= -56.6
    assert all(t is not None for t in r.freeze_times_h)


def test_invalid_coil_design_rejected():
    with pytest.raises(ValueError):
        Coil(32 * TON, _F(0), _F(-20), _F(-30))           # supply warmer than return


def test_return_setpoint_cannot_push_sst_past_zero_load():
    """Throttling stops at zero coil load (SST = return air); it never turns the coil into a heater."""
    c, r = _coil(), RefSystem(32 * TON, _F(-30), "NH3")
    held = solve_balance(-4.0, 0.9, c.design_C_air, c, r, T_return_min=-30.0)
    assert held.SST <= held.T_return + 1e-9 and held.Q >= -1e-6
