import importlib


def test_models_module_smoke():
    # Reload module if already imported
    if "dorito.models" in importlib.sys.modules:
        importlib.reload(importlib.sys.modules["dorito.models"])

    mod = importlib.import_module("dorito.models")

    # Basic smoke checks
    assert hasattr(mod, "ResolvedAmigoModel")
    assert hasattr(mod, "TransformedResolvedModel")
    assert hasattr(mod, "ResolvedDiscoModel")
