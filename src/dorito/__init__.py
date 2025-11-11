"""dorito

An extension of the amigo package for image reconstruction for JWST/AMI.

The top-level package exposes submodules: ``misc``, ``model_fits``, ``models``,
``plotting``, ``stats`` and ``bases``.  Docstrings in each submodule describe
their public API; see those modules for details.
"""

import importlib.metadata

__version__ = importlib.metadata.version("dorito")

from . import misc
from . import model_fits
from . import models
from . import plotting
from . import stats
from . import bases

# from . import wavelets

__all__ = [
    misc,
    model_fits,
    models,
    plotting,
    stats,
    bases,
]
