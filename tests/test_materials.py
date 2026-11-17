"""Package-contents conductivity with partial air segregation."""

import numpy as np
import pytest

from blastfreezer import Scenario
from blastfreezer.materials import contents_k, maxwell_eucken
from blastfreezer.pallet import Pallet

KP, KA, V = 0.45, 0.023, 0.5


def test_segregation_limits():
    assert contents_k(KP, KA, V, 0.0) == pytest.approx(maxwell_eucken(KP, KA, V))          # all mixed in
    assert contents_k(KP, KA, V, 1.0) == pytest.approx(maxwell_eucken(KA, KP, 1.0 - V))    # all around
    assert contents_k(KP, KA, 0.0, 0.7) == pytest.approx(KP)                               # no air


def test_segregation_lowers_k_monotonically():
    k = [contents_k(KP, KA, V, s) for s in np.linspace(0, 1, 21)]
    assert all(a > b for a, b in zip(k, k[1:]))
    assert maxwell_eucken(KA, KP, 1 - V) < k[10] < maxwell_eucken(KP, KA, V)


def test_segregation_slows_freezing_and_defaults():
    s = Scenario()
    assert s.product.name.startswith("Sausage") and s.product.air_segregation == 0.75
    k = {}
    for seg in (0.0, 0.5, 1.0):
        s.product.air_segregation = seg
        k[seg] = Pallet(s).mat.k_frozen
    assert k[0.0] > k[0.5] > k[1.0]
