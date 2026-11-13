# BlastFreezer

Streamlit app for transient modeling of a blast freezing cell.

## Layout

```
app.py                  Streamlit UI (inputs, plots, downloads)
blastfreezer/
  config.py             Input dataclasses (Product, Pallet, Cell, SimSettings, Scenario)
  properties.py         Temperature-dependent product properties (effective cp)
  solver.py             Transient solver: simulate(scenario) -> DataFrame
tests/                  pytest tests for the solver
```

The UI only talks to `simulate()` and `summarize()`, so the physics can be replaced
without touching the app.

## Setup

```bash
python3 -m venv ~/.venvs/blastfreezer        # keep the venv out of OneDrive
source ~/.venvs/blastfreezer/bin/activate
pip install -r requirements-dev.txt
```

## Run

```bash
streamlit run app.py
pytest
```

## Status

Physics is a placeholder: a lumped-pallet model ported from the legacy `pallet.py`.
