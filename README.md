# BlastFreezer

Streamlit app for transient modeling of a blast freezing cell.

## Layout

```
Blast_sim.py            Streamlit GUI: tabbed inputs, cell view with time slider, time series
blastfreezer/
  config.py             Input dataclasses (Product, Packaging, PalletLoad, Air, Cell, SimSettings)
  config_file.py        Save / load the GUI inputs as a JSON configuration file
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

- Upper left: inputs in tabs (Pallet, Cell, Cooling, Model, Boost) and **Run solver**. Detailed
  inputs sit in collapsible sections: Product details (properties, bound water), Packaging details
  (package height, spacers, packaging walls), and on the Cooling tab Coil, Refrigeration system and
  Controls.
- Package height is picked from the pallet height divided into 1 to 20 whole layers. With spacers on,
  there is one spacer between every pair of layers (layers − 1), so the count is calculated.
- Upper right: the cell (front view), one square per pallet position, coloured by core temperature
  from the load-in temperature (red) to the target (blue). The air loop runs from the main coil
  (above left, showing its SST) through the supply leg into the cell and back through the return
  leg; arrows follow the chosen flow direction (left/right or bottom/top). The boost coil is drawn
  above the main coil as a parallel branch (black while it runs, grey when off). The time
  slider and play button run in the browser, so the cell, air temperatures and SST update while
  dragging.
- **Save / load configuration** (under Run solver) downloads every input as a JSON file and loads
  one back, replacing all inputs. Values are stored in SI, so a file saved in Imperial loads the same
  in SI. Everything stays on your machine. "Solved scenario JSON" (Model details) is the model-input
  export of a solution, for scripting; it cannot be loaded back into the GUI.
- Every solve runs twice, without and with the boost coil. The metrics show both cell freeze times
  and the time saved; the cell view toggles between the two runs (the boost coil is drawn black while
  it runs and greys out when it shuts off).
- Bottom: Temperature (warmest pallet core and surface node, supply and return air, main and boost
  SST) and Power (all-coil load, boost coil load, pallet load, fans). One colour per quantity;
  dashed = without boost, solid = with boost. All "frozen" decisions use the warmest pallet.

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
- Coil airflow and design return air are linked by the design power (enter either one). Airflow
  is fixed during the blast.
- Fans: specific power input (default 1 hp per 1000 CFM); all fan power is added to the air
  leaving the coil, so coil load = pallet load + fan heat.
- Controls: the plant holds SST at or below a max SST (above it, the coil gets what it needs).
  Return air is not allowed below a setpoint (frost); once reached, SST rises to hold it.
- Each time step, with pallet states frozen, cell return air is linear in supply air, so the coil,
  refrigeration, fans and pallets are balanced exactly for SST and supply air.

**Boost coil**
- A second coil with its own fans and refrigeration system (same inputs as the main coil), in
  parallel with it. Return air splits between the coils; their outlets (plus each one's fan heat)
  mix by capacity rate into the supply air. While it runs, its airflow is added to every lane; the
  surface HTC stays constant (the extra flow acts through air capacity in the eps-NTU air side).
- Balance: for a fixed airflow each unit's free SST depends only on return air, so it is tabulated
  once per unit; the mixed balance is then one monotonic equation in return air, solved each step.
- Controls: each unit has its own max SST. The return-air setpoint applies to both: the boost is
  throttled first (down to zero load), then the main coil. Coils never act as heaters.
- Shutoff: never, at a set time, or when the warmest pallet core reaches a temperature. Off means
  fans and refrigeration stop, so the cell drops back to the main airflow. It stays off.

## Setup

```bash
python3 -m venv ~/.venvs/blastfreezer        # keep the venv out of OneDrive
source ~/.venvs/blastfreezer/bin/activate
pip install -r requirements-dev.txt
```

## Run

```bash
streamlit run Blast_sim.py
pytest
```

## Status

Single pallet and series-flow cell in place.

## Current assumptions / deferred items

- Supply air comes from the coil/refrigeration balance and is evenly distributed. Airflow is set
  on the coil, split evenly per lane; mass airflow is held constant.
- Coil UA is constant (no airflow dependence, no frost). Compressor displacement is constant (no
  volumetric-efficiency change); max SST stands in for plant limits.
- No wall, infiltration or defrost loads yet.
- Air moves in straight lines from the inlet face to the opposite face (no turns).
- Spacer channels are open in both horizontal directions by default.
- Surface HTC is a constant input (better sources pending); the only change is the reduction
  applied when airflow is perpendicular to the spacers.
- With spacers, pallet side faces are ignored (conservative, especially for flow across spacers).
- All pallets start at the load-in temperature at t = 0, so the peak load is an initial spike.
  Loading time is not modeled.
