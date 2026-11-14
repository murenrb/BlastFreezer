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
        assert cell.T_air_out == pytest.approx(cell.balance.T_return, abs=1e-9)
        assert cell.Q == pytest.approx(cell.balance.Q, rel=1e-9)


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
