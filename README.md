# BlastFreezer

Streamlit app for transient modeling of a blast freezing cell.

## Layout

```
app.py                  Streamlit UI (inputs, plots, downloads)
blastfreezer/
  config.py             Input dataclasses (Product, Packaging, PalletLoad, Air, Cell, SimSettings)
  geometry.py           Pallet -> 1-D representative body (plate / sphere / cylinder) + FV grid
  materials.py          Homogenized packed-product properties, enthalpy <-> temperature
  pallet.py             Pallet class: explicit-Euler finite-volume enthalpy solver
  solver.py             simulate(scenario) -> Result(history, profiles, info)
tests/                  Verification vs. analytic conduction, Plank's equation, energy balance
```

## Model (single pallet)

- Spacers split the stack into identical layers; one layer is modeled as an infinite plate
  (two faces cooled). Without spacers the block becomes an equivalent sphere, R = 3V/A.
- The packed load is homogenized: product + voids (Maxwell-Eucken), in series with
  package walls and the headspace air gap.
- Isothermal enthalpy method on a fixed grid; the freezing front comes from the frozen volume.
- Air side uses epsilon-NTU with the airflow through the pallet; air-out temperature is reported.
- Runs until the core node reaches the target temperature.

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

Single-pallet physics in place. Next: cell-level air side and refrigeration coupling.
