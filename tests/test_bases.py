import importlib
import sys
import types
import numpy as _np


def _inject_bases_stubs():
    # jaxtyping.Array
    jt = types.ModuleType("jaxtyping")
    jt.Array = object
    sys.modules["jaxtyping"] = jt

    # zodiax.Base
    zodiax = types.ModuleType("zodiax")

    class Base:
        pass

    zodiax.Base = Base
    sys.modules["zodiax"] = zodiax

    # equinox.eqx.field -> simple callable returning None
    eqx = types.ModuleType("equinox")
    eqx.field = lambda **k: None
    sys.modules["equinox"] = eqx

    # jax.numpy -> numpy
    jax = types.ModuleType("jax")
    jax.numpy = _np
    sys.modules["jax"] = jax
    sys.modules["jax.numpy"] = _np

    # photutils.psf.matching.TopHatWindow
    photutils = types.ModuleType("photutils")
    psf = types.ModuleType("photutils.psf")
    matching = types.ModuleType("photutils.psf.matching")

    class TopHatWindow:
        def __init__(self, radius):
            self.radius = radius

        def __call__(self, shape):
            h, w = shape
            cy, cx = h // 2, w // 2
            Y, X = _np.ogrid[:h, :w]
            r = _np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            return r <= max(h, w) / 2 * 0.99

    matching.TopHatWindow = TopHatWindow
    psf.matching = matching
    photutils.psf = psf
    sys.modules["photutils"] = photutils
    sys.modules["photutils.psf"] = psf
    sys.modules["photutils.psf.matching"] = matching

    # scipy.ndimage.binary_dilation -> use numpy morphological dilation approximation
    scipy = types.ModuleType("scipy")
    nd = types.ModuleType("scipy.ndimage")

    def binary_dilation(arr, iterations=1):
        out = _np.copy(arr)
        for _ in range(iterations):
            # simple dilation: set border True if any neighbor True
            padded = _np.pad(out, 1, mode="constant", constant_values=0)
            new = _np.zeros_like(out)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    new |= padded[1 + dy : 1 + dy + out.shape[0], 1 + dx : 1 + dx + out.shape[1]]
            out = new
        return out

    nd.binary_dilation = binary_dilation
    scipy.ndimage = nd
    sys.modules["scipy"] = scipy
    sys.modules["scipy.ndimage"] = nd


def test_inscribed_circ_and_annulus_basis_roundtrip():
    # Prepare stubs and import the module cleanly
    _inject_bases_stubs()

    if "dorito.bases" in sys.modules:
        importlib.reload(sys.modules["dorito.bases"])

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
