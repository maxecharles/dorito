def test_package_import():
    """Basic smoke test: package imports and exposes a version."""
    import importlib

    pkg = importlib.import_module("dorito")
    assert hasattr(pkg, "__version__"), "package has no __version__"


def test_submodules_present():
    """Ensure core submodules are present on the top-level package."""
    import dorito

    expected = ("misc", "models", "model_fits", "plotting", "stats", "bases")
    for name in expected:
        assert hasattr(dorito, name), f"dorito.{name} not present"
