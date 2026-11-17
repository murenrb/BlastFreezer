"""Save and load the GUI configuration as JSON (local file, nothing leaves the machine).

The file stores the GUI's own state, so loading it puts every input back exactly
as it was, including choices the model never sees (unit system, which coil value
is typed in vs calculated, custom product values for every base product):

    {"app": "Blast Freezing Simulator", "format": 1,
     "inputs":   {"units": ..., "si::<id>": SI value, "w::<id>": widget value, ...},
     "scenario": Scenario.to_dict()}        # for reading / scripting only; ignored on load

Numeric inputs are stored in SI ("si::<id>"), so a file saved in Imperial loads
correctly in SI and vice versa.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, MutableMapping
from datetime import datetime

APP_NAME = "Blast Freezing Simulator"
FORMAT = 1

PLAIN_KEYS = ("units", "n_layers", "custom_base")       # non-widget state worth keeping
UNIT_SUFFIXES = ("::SI", "::Imperial")                  # per-unit-system widgets: rebuilt from si:: values
SKIP_WIDGETS = ("w::view_run",)                          # result view, not an input


def _is_input_key(k: str) -> bool:
    if k in PLAIN_KEYS or k.startswith("si::"):
        return True
    return k.startswith("w::") and not k.endswith(UNIT_SUFFIXES) and k not in SKIP_WIDGETS


def _jsonable(v) -> bool:
    return v is None or isinstance(v, (bool, int, float, str))


def collect(state: Mapping, scenario: dict | None = None) -> dict:
    """Configuration dict from the GUI session state."""
    inputs = {k: state[k] for k in sorted(state) if isinstance(k, str) and _is_input_key(k)
              and _jsonable(state[k])}
    out = {"app": APP_NAME, "format": FORMAT, "saved": datetime.now().isoformat(timespec="seconds"),
           "inputs": inputs}
    if scenario is not None:
        out["scenario"] = scenario
    return out


def dumps(state: Mapping, scenario: dict | None = None) -> str:
    return json.dumps(collect(state, scenario), indent=2)


def parse(raw: bytes | str) -> dict:
    """Inputs from a saved file. Raises ValueError with a readable message."""
    try:
        data = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"Not a valid JSON file ({e}).") from None
    if not isinstance(data, dict) or not isinstance(data.get("inputs"), dict):
        if isinstance(data, dict) and "product" in data and "cooling" in data:
            raise ValueError("This is a solved-scenario export, not a configuration file. "
                             "Use “Save configuration” to make a file that can be loaded.")
        raise ValueError("Not a Blast Freezing Simulator configuration file.")
    if data.get("format", FORMAT) > FORMAT:
        raise ValueError("This configuration was saved by a newer version of the app.")
    return data["inputs"]


def apply(state: MutableMapping, inputs: Mapping, options: Mapping[str, list] | None = None) -> list[str]:
    """Replace the GUI inputs in `state` with `inputs`. Returns notes on anything skipped.

    options: allowed values per key (selectboxes); a value not in the list is skipped and the
    input falls back to its default, e.g. a product removed from the library.
    """
    options = options or {}
    for k in [k for k in state if isinstance(k, str) and (k in PLAIN_KEYS or k.startswith(("si::", "w::")))
              and k not in SKIP_WIDGETS]:
        del state[k]                    # includes per-unit widgets, so they rebuild from the loaded si:: values
    notes = []
    for k, v in inputs.items():
        if not (isinstance(k, str) and _is_input_key(k)) or not _jsonable(v):
            notes.append(f"ignored unknown entry {k!r}")
            continue
        if k.startswith("si::") and (isinstance(v, bool) or not isinstance(v, (int, float))):
            notes.append(f"ignored non-numeric {k[4:]!r}")
            continue
        if k in options and v not in options[k]:
            notes.append(f"{v!r} is not available for {k.split('::')[-1]!r}; using the default")
            continue
        state[k] = v
    return notes
