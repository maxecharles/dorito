import importlib
import sys
import types
import numpy as _np


def _inject_models_stubs():
    # Minimal jax shim
    jax = types.ModuleType("jax")
    jax.Array = _np.ndarray
    jax.numpy = _np
    jax.tree = types.SimpleNamespace(map=lambda f, x: x)
    sys.modules["jax"] = jax
    sys.modules["jax.numpy"] = _np

    # amigo.core_models.BaseModeller / AmigoModel
    amigo = types.ModuleType("amigo")
    core = types.ModuleType("amigo.core_models")

    class BaseModeller:
        def __init__(self, params=None):
            self.params = params or {}

    class AmigoModel:
        def __init__(self, *a, **k):
            pass

    core.BaseModeller = BaseModeller
    core.AmigoModel = AmigoModel
    sys.modules["amigo"] = amigo
    sys.modules["amigo.core_models"] = core

    # dLux utils stub
    dlu = types.ModuleType("dLux")
    utils = types.ModuleType("dLux.utils")
    utils.arcsec2rad = lambda x: x * (_np.pi / 180.0) / 3600.0
    dlu.utils = utils
    sys.modules["dLux"] = dlu
    sys.modules["dLux.utils"] = utils


def test_models_module_smoke():
    _inject_models_stubs()

    # reload if already imported
    if "dorito.models" in sys.modules:
        importlib.reload(sys.modules["dorito.models"])

    mod = importlib.import_module("dorito.models")

    # Basic smoke checks
    assert hasattr(mod, "ResolvedAmigoModel")
    assert hasattr(mod, "TransformedResolvedModel")
    assert hasattr(mod, "ResolvedDiscoModel")
