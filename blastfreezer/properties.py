"""Temperature-dependent product properties.

Currently holds the empirical effective-cp curve ported from the legacy
`pallet.py` (cp_eff_baseline, kJ/kg-K vs degC), scaled to a product by the
ratio of total enthalpy change, exactly as the legacy code did.

TODO(physics): replace with a proper enthalpy method (e.g. Chen / Schwartzberg
frozen-fraction model) so latent heat release follows the product's freezing
curve instead of a fixed baseline shape.
"""

from __future__ import annotations

import numpy as np

from .config import Product

# Legacy baseline: temperature [degC] -> effective cp [kJ/kg-K]
_BASELINE = {
    50.0: 4.0, 10.9448922222222: 109.280386782833, 10.7609216666667: 66.559401762603,
    10.3746455555556: 21.3274240638697, 10.0105622222222: 12.4459650996361,
    9.49026722222222: 10.8539125341504, 9.03224944444444: 8.70478850082306,
    8.421915: 8.63128946643635, 7.74155944444444: 7.47054134920692,
    7.09990611111111: 8.03957777453537, 6.33337222222222: 7.10418469781388,
    5.61784555555556: 7.3481857206733, 5.00362388888889: 6.91362276163797,
    4.36197888888889: 6.84234526662861, 3.71293388888889: 8.5577694415518,
    2.94253444444445: 7.50743121861474, 2.37193611111111: 7.14012290269118,
    1.73029166666667: 8.63097946287184, 1.02250444444444: 9.7648668376957,
    0.415678888888888: 9.93243986665158, -0.112003888888888: 9.73294672179293,
    -0.56967: 15.3180893294744, -1.12514055555556: 14.4735348068238,
    -1.34042555555556: 22.4634125765274, -1.59406444444445: 35.5233951164055,
    -1.72316222222222: 102.112537923754, -1.81674555555556: 164.654696156624,
    -1.87935722222222: 219.980735560125, -2.00459388888889: 98.3015488517449,
    -2.88124111111111: 0.71043770634201, -4.20817055555556: 0.334920340983551,
    -5.35145666666667: 0.370634896867755, -6.43987666666667: 1.32930648227173,
    -7.856865: 1.21246336466285, -9.10498277777778: 1.4012794310616,
    -10.3608494444444: 1.32570425579227, -11.2515638888889: 2.23757577714335,
    -12.7698666666667: 1.54214024109491, -14.6248138888889: 1.67376942338437,
    -15.6917761111111: 1.76452289214927, -16.7911: 1.62737573191861,
    -17.7855872222222: 1.69269950575155, -50.0: 1.69269950575155,
}
_T = np.array(sorted(_BASELINE))
_CP = np.array([_BASELINE[t] for t in _T]) * 1000.0  # -> J/kg-K

# Baseline product the curve was derived from (legacy constants)
_BASE_CP_LIQ, _BASE_CP_SOL, _BASE_WATER = 3200.0, 1600.0, 0.688


def _enthalpy_span(cp_liq: float, cp_sol: float, water: float) -> float:
    """Legacy scaling metric: 11 K sensible above, latent, 17 K sensible below."""
    return cp_liq * 11.0 + water * 330_000.0 + cp_sol * 17.0


class EffectiveCp:
    """Callable effective cp(T) [J/kg-K] for a given product."""

    def __init__(self, product: Product):
        scale = _enthalpy_span(product.cp_unfrozen, product.cp_frozen, product.water_fraction) / \
            _enthalpy_span(_BASE_CP_LIQ, _BASE_CP_SOL, _BASE_WATER)
        self.T = _T
        self.cp = _CP * scale

    def __call__(self, T):
        return np.interp(T, self.T, self.cp)
