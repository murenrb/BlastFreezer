# BlastFreezer

Streamlit app for transient modeling of a blast freezing cell.

## Layout

```
app.py                  Streamlit UI (inputs, plots, downloads)
blastfreezer/
  config.py             Input dataclasses (Product, Packaging, PalletLoad, Air, Cell, SimSettings)
  units.py              SI <-> Imperial display conversions (model is SI internally)
  cell.py               Cell of pallet positions; lanes of pallets in series along the airflow
  geometry.py           Pallet -> 1-D representative body (plate / sphere / cylinder) + FV grid
  materials.py          Homogenized packed-product properties, enthalpy <-> temperature
  pallet.py             Pallet class: explicit-Euler finite-volume enthalpy solver
  solver.py             simulate(scenario) -> Result(cell, pallets, profiles, freeze_times_h, info)
tests/                  Verification vs. analytic conduction, Plank's equation, energy balance
```

## Model

**Pallet**
- Spacers split the product height into identical layers; one layer is modeled as an infinite
  plate (two faces cooled). Without spacers the block becomes an equivalent sphere, R = 3V/A.
- The packed load is homogenized: product + voids (Maxwell-Eucken), in series with internal
  package walls and the headspace air gap. For boxes, the outer wall is a surface resistance.
- Bound water (share of the water that never freezes) is excluded from latent heat.
- Isothermal enthalpy method on a fixed grid, explicit Euler; the freezing front comes from the
  frozen volume.
- Air side uses epsilon-NTU with the airflow through the pallet; air-out temperature is reported.

**Cell**
- Grid of pallet positions (deep x wide x high). Air enters one face and leaves the opposite one
  in straight lanes; pallets in a lane are solved in flow order, each one's air-out feeding the
  next one's air-in.
- Total airflow = airflow per pallet x number of pallets, split evenly across lanes.
- Flow along spacer channels uses the surface HTC; flow across them (vertical, or across one-way
  channels) reduces spacer velocity by a ratio (default 0.1) with h proportional to v^n (n = 0.8).
- Runs until every pallet core reaches the target.

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

Single pallet and series-flow cell in place.

## Current assumptions / deferred items

- Inlet air temperature is constant and evenly distributed. Airflow per pallet is an input;
  later it will come from a fan input split evenly per lane.
- Air moves in straight lines from the inlet face to the opposite face (no turns).
- Spacer channels are open in both horizontal directions by default.
- Surface HTC is a constant input (better sources pending); only the across-spacer velocity
  drop changes it.
- With spacers, pallet side faces are ignored (conservative, especially for flow across spacers).
- All pallets start at the load-in temperature at t = 0, so the peak load is an initial spike.
  Loading time and coil/refrigeration limits come with the coil model.
