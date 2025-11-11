import importlib
import numpy as _np


def test_inscribed_circ_and_annulus_basis_roundtrip():
    # Import the actual module and run a small round-trip check. Tests assume
    # the developer has installed package dependencies in the test env.
    if "dorito.bases" in importlib.sys.modules:
        importlib.reload(importlib.sys.modules["dorito.bases"])

    mod = importlib.import_module("dorito.bases")

    # Create a small basis and get window
    basis, window = mod.inscribed_circ_basis(7, return_window=True)
    assert basis.size == 7
    coeffs = basis.to_basis(_np.ones((7, 7)))
    recon = basis.from_basis(coeffs)
    assert recon.shape == (7, 7)

    # Annulus
    basis_a, window_a = mod.inscribed_annulus_basis(7, iterations=1, return_window=True)
    assert basis_a.size == 7
    coeffs_a = basis_a.to_basis(_np.ones((7, 7)))
    recon_a = basis_a.from_basis(coeffs_a)
    assert recon_a.shape == (7, 7)
