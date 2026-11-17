"""Save / load of the GUI configuration file."""

import json
from pathlib import Path

import pytest

from blastfreezer import config_file as cf

APP = str(Path(__file__).resolve().parents[1] / "Blast_sim.py")


def gui_state():
    return {"units": "Imperial", "si::mass": 812.5, "si::T0": 4.0, "w::product": "Beef, lean",
            "w::inlet": "bottom", "w::n_deep": 3, "w::coil_driver": "Airflow", "w::use_sp": False,
            "n_layers": 9, "custom_base": "Beef, lean",
            "w::mass::Imperial": 1791.3, "w::view_run": "With boost", "res": object(), "cfg_n": 2,
            "ro::Airflow::12": 12.0}


def test_collect_keeps_inputs_only():
    d = cf.collect(gui_state(), {"x": 1})
    assert d["app"] == cf.APP_NAME and d["scenario"] == {"x": 1}
    inp = d["inputs"]
    assert inp["si::mass"] == 812.5 and inp["w::n_deep"] == 3 and inp["n_layers"] == 9
    for k in ("w::mass::Imperial", "w::view_run", "res", "cfg_n", "ro::Airflow::12"):
        assert k not in inp
    json.dumps(d)                                   # serialisable


def test_round_trip_replaces_state():
    saved = cf.dumps(gui_state())
    state = {"si::mass": 300.0, "si::stale": 1.0, "w::mass::SI": 300.0, "w::product": "Chicken",
             "w::view_run": "Without boost", "res": "keep"}
    notes = cf.apply(state, cf.parse(saved))
    assert notes == []
    assert state["si::mass"] == 812.5 and state["units"] == "Imperial" and state["w::inlet"] == "bottom"
    assert "si::stale" not in state and "w::mass::SI" not in state      # widgets rebuild from si:: values
    assert state["res"] == "keep" and state["w::view_run"] == "Without boost"


def test_bad_values_are_skipped():
    inputs = {"w::product": "Unobtainium", "si::mass": "heavy", "si::T0": True, "bogus": 1, "w::inlet": "top"}
    state = {}
    notes = cf.apply(state, inputs, {"w::product": ["Beef, lean"], "w::inlet": ["top", "bottom"]})
    assert state == {"w::inlet": "top"} and len(notes) == 4


@pytest.mark.parametrize("raw, msg", [
    (b"not json", "valid JSON"),
    (b"[1, 2]", "configuration file"),
    (json.dumps({"product": {}, "cooling": {}}), "solved-scenario"),
    (json.dumps({"format": 99, "inputs": {}}), "newer version"),
])
def test_parse_errors(raw, msg):
    with pytest.raises(ValueError, match=msg):
        cf.parse(raw)


def test_loaded_config_drives_app():
    """A loaded state reaches the widgets and the scenario the app builds."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    saved = cf.collect(at.session_state.to_dict())
    inputs = dict(saved["inputs"], **{"si::mass": 650.0, "w::inlet": "bottom", "w::n_deep": 2,
                                      "w::b_off_mode": "time", "si::b_off_time": 6.0})
    state = {}
    cf.apply(state, inputs)
    at2 = AppTest.from_file(APP, default_timeout=60)
    for k, v in state.items():
        at2.session_state[k] = v
    at2.run()
    assert not at2.exception
    again = cf.collect(at2.session_state.to_dict())["inputs"]
    for k in ("si::mass", "w::inlet", "w::n_deep", "w::b_off_mode", "si::b_off_time"):
        assert again[k] == inputs[k], k
    assert at2.slider(key="w::b_off_time::SI").value == 6.0
