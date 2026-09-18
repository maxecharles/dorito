# import importlib
# import numpy as _np

# Old Test
# def test_inscribed_circ_and_annulus_basis_roundtrip():
#     # Import the actual module and run a small round-trip check. Tests assume
#     # the developer has installed package dependencies in the test env.
#     if "dorito.bases" in importlib.sys.modules:
#         importlib.reload(importlib.sys.modules["dorito.bases"])

#     mod = importlib.import_module("dorito.bases")

#     # Create a small basis and get window
#     basis, window = mod.inscribed_circ_basis(7, return_window=True)
#     assert basis.size == 7
#     coeffs = basis.to_basis(_np.ones((7, 7)))
#     recon = basis.from_basis(coeffs)
#     assert recon.shape == (7, 7)

#     # Annulus
#     basis_a, window_a = mod.inscribed_annulus_basis(7, iterations=1, return_window=True)
#     assert basis_a.size == 7
#     coeffs_a = basis_a.to_basis(_np.ones((7, 7)))
#     recon_a = basis_a.from_basis(coeffs_a)
#     assert recon_a.shape == (7, 7)

# New Tests
import equinox as eqx
import jax.numpy as np
import jax.random as jr
 
from dorito.bases import (
    LatentBasis,
    LinearBasis,
    inscribed_annulus_basis,
    inscribed_circ_basis,
)
 
SIZE = 7

# ---------------------------------------------------------------- circ basis
def test_circ_basis_shapes():
    basis, window = inscribed_circ_basis(SIZE)
    window = np.array(window)
    n_pix = int(np.sum(window != 0))
 
    assert basis.size == SIZE
    assert basis.n_basis == n_pix
    assert basis.M.shape == (SIZE**2, n_pix)
    assert basis.M_inv.shape == (n_pix, SIZE**2)
    assert window.shape == (SIZE, SIZE)
    # An inscribed circle should drop the corners but keep most of the square.
    assert 0 < n_pix < SIZE**2
 
 
def test_circ_basis_is_a_masking_projection():
    basis, window = inscribed_circ_basis(SIZE)
    mask = np.array(window) != 0
    img = jr.normal(jr.PRNGKey(0), (SIZE, SIZE))
 
    coeffs = basis.to_basis(img)
    recon = basis.from_basis(coeffs)
 
    assert coeffs.shape == (basis.n_basis,)
    assert recon.shape == (SIZE, SIZE)
    # M is a column selection of the identity, so the round trip keeps the
    # in-window pixels untouched and zeroes everything else.
    assert np.allclose(recon, np.where(mask, img, 0.0))
 
 
def test_circ_basis_coeff_roundtrip_is_exact():
    basis, _ = inscribed_circ_basis(SIZE)
    coeffs = jr.normal(jr.PRNGKey(1), (basis.n_basis,))
    assert np.allclose(basis.to_basis(basis.from_basis(coeffs)), coeffs)
 
 
def test_circ_basis_without_window():
    out = inscribed_circ_basis(SIZE, return_window=False)
    assert isinstance(out, LinearBasis)
    assert out.size == SIZE
 
 
# ------------------------------------------------------------- annulus basis
def test_annulus_basis_excludes_centre():
    basis, window = inscribed_annulus_basis(SIZE, iterations=1)
    window = np.array(window)
    c = SIZE // 2
 
    assert basis.size == SIZE
    assert window[c, c] == 0
    assert basis.n_basis == int(np.sum(window != 0))
    # No pixel should be dilated outside the top-hat, so nothing goes negative.
    assert np.all(window >= 0)
 
 
def test_annulus_is_smaller_than_circle():
    circ, _ = inscribed_circ_basis(SIZE)
    ann, _ = inscribed_annulus_basis(SIZE, iterations=1)
    assert ann.n_basis < circ.n_basis
 
 
def test_annulus_hole_grows_with_iterations():
    small, _ = inscribed_annulus_basis(SIZE, iterations=1)
    large, _ = inscribed_annulus_basis(SIZE, iterations=2)
    assert large.n_basis < small.n_basis
 
 
def test_annulus_basis_roundtrip():
    basis, window = inscribed_annulus_basis(SIZE, iterations=1)
    mask = np.array(window) != 0
    img = np.ones((SIZE, SIZE))
 
    recon = basis.from_basis(basis.to_basis(img))
    assert recon.shape == (SIZE, SIZE)
    assert np.allclose(recon, np.where(mask, img, 0.0))
 
 
# -------------------------------------------------------------- LinearBasis
def test_non_orthogonal_basis_uses_pinv():
    M = jr.normal(jr.PRNGKey(2), (16, 4))
    basis = LinearBasis(M)
 
    assert basis.size == 4  # sqrt(16)
    assert np.allclose(basis.M_inv @ basis.M, np.eye(4), atol=1e-4)
 
    coeffs = jr.normal(jr.PRNGKey(3), (4,))
    assert np.allclose(basis.to_basis(basis.from_basis(coeffs)), coeffs, atol=1e-4)
 
 
def test_ortho_flag_transposes_instead_of_inverting():
    M = np.eye(16)[:, :4]
    basis = LinearBasis(M, ortho=True)
    assert np.allclose(basis.M_inv, basis.M.T)
 
 
def test_n_basis_truncates_columns():
    M = jr.normal(jr.PRNGKey(4), (16, 6))
    basis = LinearBasis(M, n_basis=3)
 
    assert basis.n_basis == 3
    assert basis.M.shape == (16, 3)
    assert np.allclose(basis.M, M[:, :3])
 
 
# -------------------------------------------------------------- LatentBasis
class _DummyModel(eqx.Module):
    """Minimal encoder / passthrough / decoder stack for LatentBasis."""
 
    modules: list
 
    def __init__(self, key, n_pix=9, n_latent=3):
        k1, k2, k3 = jr.split(key, 3)
        self.modules = [
            eqx.nn.Linear(n_pix, n_latent, key=k1),
            eqx.nn.Linear(n_latent, n_latent, key=k2),  # intermediate, unused
            eqx.nn.Linear(n_latent, n_pix, key=k3),
        ]
 
 
def test_latent_basis_picks_first_and_last_modules():
    model = _DummyModel(jr.PRNGKey(5))
    basis = LatentBasis(model)
 
    assert basis.encoder is model.modules[0]
    assert basis.decoder is model.modules[-1]
 
 
def test_latent_basis_roundtrip_shapes():
    model = _DummyModel(jr.PRNGKey(6))
    basis = LatentBasis(model)
    img = jr.normal(jr.PRNGKey(7), (9,))
 
    coeffs = basis.to_basis(img)
    recon = basis.from_basis(coeffs)
 
    assert coeffs.shape == (3,)
    assert recon.shape == (9,)
    # Encoder and decoder are independent, so this is a projection at best.
    assert np.allclose(coeffs, model.modules[0](img))
    assert np.allclose(recon, model.modules[-1](coeffs))