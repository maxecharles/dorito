"""dorito

An extension of the amigo package for image reconstruction for JWST/AMI.

The top-level package exposes submodules: ``misc``, ``model_fits``, ``models``,
``plotting``, ``stats`` and ``bases``.  Docstrings in each submodule describe
their public API; see those modules for details.
"""

import importlib
import importlib.metadata
from typing import Any

__version__ = importlib.metadata.version("dorito")

# Lazily import heavy submodules on attribute access to avoid importing
# large/optional dependencies at package import time (helps tests and
# lightweight tools that only need small parts of the package).
_SUBMODULES = ["misc", "model_fits", "models", "plotting", "stats", "bases"]


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        module = importlib.import_module(f"dorito.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + _SUBMODULES)


__all__ = _SUBMODULES
