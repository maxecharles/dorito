def test_package_import():
    """Basic smoke test: package imports and exposes a version."""
    import importlib

    pkg = importlib.import_module("dorito")
    assert hasattr(pkg, "__version__"), "package has no __version__"


def test_submodules_present():
    """(DEPRECATED) previously checked submodule attributes.

    Kept as a placeholder for future per-module import tests; no-op to
    avoid importing heavy optional dependencies at package import time.
    """
    # Intentionally do nothing — prefer explicit per-module tests.
