import importlib.metadata

__version__ = importlib.metadata.version("dorito")

from . import build_model
from . import misc
from . import model_fits
from . import models
from . import plotting
from . import stats

__all__ = [
    build_model,
    misc,
    model_fits,
    models,
    plotting,
    stats,
]