# BlastFreezer

Streamlit app for transient modeling of a blast freezing cell.

## Layout

```
app.py                  Streamlit GUI: tabbed inputs, cell view with time slider, time series
blastfreezer/
  config.py             Input dataclasses (Product, Packaging, PalletLoad, Air, Cell, SimSettings)
  units.py              SI <-> Imperial display conversions (model is SI internally)
  foods.py              Product property library (ASHRAE / proposal generator, Choi-Okos k and density)
  viz.py                Plotly figures: cell view, temperature and power time series
  cell.py               Cell of pallet positions; lanes of pallets in series; coil/ref balance per step
  refrigeration.py      Coil (UA from design point) and fixed-displacement refrigeration (CoolProp)
  geometry.py           Pallet -> 1-D representative body (plate / sphere / cylinder) + FV grid
  materials.py          Homogenized packed-product properties, enthalpy <-> temperature
  pallet.py             Pallet class: explicit-Euler finite-volume enthalpy solver
  solver.py             simulate(scenario) -> Result(cell, pallets, profiles, freeze_times_h, info)
tests/                  Verification vs. analytic conduction, Plank's equation, energy balance
```

## GUI

- Upper left: inputs in tabs (Pallet, Cell, Model, Boost) and **Run solver**.
- Upper right: the cell (front view), one square per pallet position, coloured by core temperature
  from the load-in temperature (red) to the target (blue), with the inlet/outlet air arrows
  (left/right or bottom/top). The time slider and play button run in the browser, so the cell
  updates while dragging.
- Bottom: Temperature (warmest and coldest pallet core and surface node, air in and out) and
  Power (cooling, fan) time series.

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
- Total cell airflow is an input, split evenly across lanes.
- Flow along spacer channels uses the surface HTC; flow across them (vertical, or across one-way
  channels) reduces the HTC by an input percentage (default 90 %).
- Runs until every pallet core reaches the target.

**Cooling**
- Coil: UA = design power / LMTD from the design supply, return and SST. At any condition the
  refrigerant side is isothermal at SST: Q = eps * m_dot cp * (T_return - SST).
- Refrigeration: fixed compressor displacement sized at the design point; capacity at any SST =
  displacement * rho_g(SST) * h_fg(SST) (CoolProp; NH3, CO2, R404A, R134a). Liquid is assumed
  saturated at SST.
- Each time step, with pallet states frozen, cell return air is linear in supply air, so coil load =
  refrigeration capacity = pallet load is solved exactly for SST and supply air.

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

- Supply air comes from the coil/refrigeration balance and is evenly distributed. Total airflow is
  an input (fan input later), split evenly per lane; mass airflow is held constant.
- Coil UA is constant (no airflow dependence, no frost). Compressor displacement is constant (no
  volumetric-efficiency change, no motor limit, no capacity control), so SST floats with load.
- No fan heat, wall, infiltration or defrost loads yet.
- Air moves in straight lines from the inlet face to the opposite face (no turns).
- Spacer channels are open in both horizontal directions by default.
- Surface HTC is a constant input (better sources pending); the only change is the reduction
  applied when airflow is perpendicular to the spacers.
- With spacers, pallet side faces are ignored (conservative, especially for flow across spacers).
- All pallets start at the load-in temperature at t = 0, so the peak load is an initial spike.
  Loading time is not modeled.
