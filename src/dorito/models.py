from amigo.core_models import BaseModeller
from amigo.optical_models import AMIOptics
from jax import numpy as np
import dLux as dl


class ResolvedAmigoModel(BaseModeller):
    filters: dict
    # dispersion: dict
    # contrast: float
    optics: AMIOptics
    visibilities: None
    detector: None
    ramp: None
    read: None

    def __init__(self, params, optics, ramp, detector, read, filters, visibilities=None):
        self.filters = filters
        self.optics = optics
        self.detector = detector
        self.ramp = ramp
        self.read = read
        self.visibilities = visibilities

        super().__init__(params)

        # if "spectral_coeffs" in params:
        #     if (
        #         len(params["spectral_coeffs"].keys())
        #         != list(self.filters.values())[0].shape[0]
        #     ):
        #         raise ValueError(
        #             "The number of spectral coefficients must match the number of "
        #             "filters."
        #         )

    @property
    def distribution(self, exposure):
        distribution = 10 ** self.log_distribution[exposure.get_key("log_distribution")]
        return distribution / distribution.sum()

    def model(self, exposure, **kwargs):
        return exposure.fit(self, exposure, **kwargs)

    def __getattr__(self, key):
        if key in self.params:
            return self.params[key]
        for k, val in self.params.items():
            if hasattr(val, key):
                return getattr(val, key)
        if hasattr(self.optics, key):
            return getattr(self.optics, key)
        if hasattr(self.detector, key):
            return getattr(self.detector, key)
        if hasattr(self.ramp, key):
            return getattr(self.ramp, key)
        if hasattr(self.read, key):
            return getattr(self.read, key)
        if hasattr(self.visibilities, key):
            return getattr(self.visibilities, key)
        raise AttributeError(f"{self.__class__.__name__} has no attribute " f"{key}.")
