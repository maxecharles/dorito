"""Test fixtures and environment shims.

This conftest injects lightweight stubs for heavy external dependencies when
they are not available in the test environment. It only creates stubs when a
module is not importable to avoid overriding real installations.
"""
import importlib
import importlib.util
import sys
import types
import numpy as _np


def _ensure_module(name, factory):
    """Ensure a module with `name` exists in sys.modules; if not, create it
    using the provided factory callable which returns a module object.
    """
    if importlib.util.find_spec(name) is None and name not in sys.modules:
        sys.modules[name] = factory()


def _make_jax():
    m = types.ModuleType("jax")
    m.Array = object
    m.numpy = _np
    # simple tree shim
    jtu = types.SimpleNamespace(map=lambda f, x: x)
    sys.modules["jax.tree"] = jtu
    m.tree = jtu
    return m


def _make_amigo():
    m = types.ModuleType("amigo")

    # amigo.model_fits
    mf = types.ModuleType("amigo.model_fits")

    class ModelFit:
        def __init__(self, *a, **k):
            pass

        def initialise_params(self, optics):
            return {}

    mf.ModelFit = ModelFit
    sys.modules["amigo.model_fits"] = mf

    # amigo.vis_models
    vm = types.ModuleType("amigo.vis_models")
    vm.vis_to_im = lambda *a, **k: (None, None)
    sys.modules["amigo.vis_models"] = vm

    # amigo.vis_analysis
    va = types.ModuleType("amigo.vis_analysis")
    va.AmigoOIData = type("AmigoOIData", (object,), {})
    sys.modules["amigo.vis_analysis"] = va

    # amigo.core_models minimal
    core = types.ModuleType("amigo.core_models")
    class BaseModeller:
        def __init__(self, params=None):
            self.params = params or {}

    class AmigoModel:
        def __init__(self, *a, **k):
            pass

    core.BaseModeller = BaseModeller
    core.AmigoModel = AmigoModel
    sys.modules["amigo.core_models"] = core

    # expose package
    return m


def _make_dlux():
    m = types.ModuleType("dLux")
    utils = types.ModuleType("dLux.utils")
    utils.MFT = lambda *a, **k: _np.zeros((1, 1))
    utils.downsample = lambda arr, *a, **k: arr
    utils.deg2rad = lambda x: x
    m.utils = utils
    sys.modules["dLux.utils"] = utils
    return m


def _make_photutils():
    m = types.ModuleType("photutils")
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
    m.psf = psf
    sys.modules["photutils.psf.matching"] = matching
    sys.modules["photutils.psf"] = psf
    return m


def _make_zodiax():
    m = types.ModuleType("zodiax")
    class Base:
        pass

    m.Base = Base
    return m


def _make_equinox():
    m = types.ModuleType("equinox")
    m.field = lambda **k: None
    return m


# Ensure stubs only when real packages are not present
_ensure_module("jax", _make_jax)
_ensure_module("amigo", _make_amigo)
_ensure_module("dLux", _make_dlux)
_ensure_module("photutils", _make_photutils)
_ensure_module("zodiax", _make_zodiax)
_ensure_module("equinox", _make_equinox)
import types
import sys


def _make_module(name):
    m = types.ModuleType(name)
    m.__file__ = f"<fake {name}>"
    return m


def pytest_configure():
    """No-op: use the real `amigo` package installed in the environment.

    Previously we injected lightweight stubs for `amigo` so imports would
    succeed in environments without the real package. The real `amigo`
    dependency is now installed in CI and local dev environments, so tests
    should exercise the actual package. Keep this hook as a no-op to avoid
    interfering with the real installation.
    """
    return
