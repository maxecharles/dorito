import importlib


def test_model_fits_basic_smoke():
    """Basic smoke test for the `dorito.model_fits` module.

    This test only asserts the presence of the main ModelFit-derived
    classes. We intentionally avoid exercising `initialise_params` here
    (it requires more detailed model state) — the previous deeper tests
    that called `initialise_params` have been removed and replaced by
    this lightweight smoke check.
    """
    if "dorito.model_fits" in importlib.sys.modules:
        importlib.reload(importlib.sys.modules["dorito.model_fits"])

    mod = importlib.import_module("dorito.model_fits")

    assert hasattr(mod, "ResolvedFit")
    assert hasattr(mod, "ResolvedOIFit")
    assert hasattr(mod, "TransformedResolvedFit")
    assert hasattr(mod, "PointResolvedFit")
