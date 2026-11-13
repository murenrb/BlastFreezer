from blastfreezer import Scenario, simulate, summarize
from blastfreezer.solver import OUTPUT_COLUMNS


def test_default_scenario_reaches_target():
    s = Scenario()
    df = simulate(s)
    assert list(df.columns) == OUTPUT_COLUMNS
    assert summarize(df, s)["reached_target"]


def test_temperature_monotonically_decreases():
    df = simulate(Scenario())
    assert (df["T_product"].diff().dropna() <= 1e-9).all()


def test_capacity_limit_respected():
    s = Scenario()
    s.cell.refrigeration_capacity = 50_000.0
    df = simulate(s)
    assert df["Q_cell_kW"].max() <= 50.0 + 1e-6


def test_scenario_roundtrip():
    s = Scenario()
    assert Scenario.from_dict(s.to_dict()) == s
