import importlib.metadata

__version__ = importlib.metadata.version("dorito")

from . import misc
from . import model_fits
from . import models
from . import plotting
from . import stats

# from . import wavelets

__all__ = [
    misc,
    model_fits,
    models,
    plotting,
    stats,
    # wavelets,
]
