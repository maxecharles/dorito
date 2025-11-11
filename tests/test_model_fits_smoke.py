import importlib
import sys
import types


def _inject_stubs():
    """Insert lightweight stub modules for `amigo`, `dLux` and `jax` so
    we can import `dorito.model_fits` without heavy third-party deps.
    """
    # jax -> use numpy as a backend for tests
    import numpy as _np

    jax = types.ModuleType("jax")
    jax.numpy = _np
    sys.modules["jax"] = jax
    sys.modules["jax.numpy"] = _np

    # amigo minimal surface API
    amigo = types.ModuleType("amigo")
    amigo.model_fits = types.ModuleType("amigo.model_fits")
    class _ModelFit:
        def __init__(self, *a, **k):
            pass

    amigo.model_fits.ModelFit = _ModelFit
    amigo.vis_models = types.ModuleType("amigo.vis_models")
    amigo.vis_models.vis_to_im = lambda *a, **k: (None, None)
    amigo.vis_analysis = types.ModuleType("amigo.vis_analysis")
    amigo.vis_analysis.AmigoOIData = type("AmigoOIData", (object,), {})
    amigo.misc = types.ModuleType("amigo.misc")
    amigo.misc.interp = lambda arr, knots, samps, method=None: arr

    sys.modules["amigo"] = amigo
    sys.modules["amigo.model_fits"] = amigo.model_fits
    sys.modules["amigo.vis_models"] = amigo.vis_models
    sys.modules["amigo.vis_analysis"] = amigo.vis_analysis
    sys.modules["amigo.misc"] = amigo.misc

    # dLux minimal
    dlu = types.ModuleType("dLux")
    dlu.utils = types.ModuleType("dLux.utils")
    dlu.utils.pixel_coords = lambda *a, **k: None
    dlu.utils.rotate_coords = lambda *a, **k: None
    dlu.utils.deg2rad = lambda x: x
    dlu.utils.downsample = lambda arr, *a, **k: arr

    class MFT:
        def __init__(self, *a, **k):
            pass

    dlu.utils.MFT = MFT
    sys.modules["dLux"] = dlu
    sys.modules["dLux.utils"] = dlu.utils


def test_model_fits_import_smoke():
    # Prepare and inject stubs, then import the module to ensure it can be
    # imported and that the key classes are present.
    _inject_stubs()

    # Ensure a clean import
    if "dorito.model_fits" in sys.modules:
        importlib.reload(sys.modules["dorito.model_fits"])

    mod = importlib.import_module("dorito.model_fits")

    # Expect key classes to be present (smoke test only)
    assert hasattr(mod, "ResolvedFit")
    assert hasattr(mod, "ResolvedOIFit")
    assert hasattr(mod, "TransformedResolvedFit")
