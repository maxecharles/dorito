from amigo.core_models import BaseModeller
from amigo.optical_models import AMIOptics

# core jax
from jax import Array


class ResolvedAmigoModel(BaseModeller):
    filters: dict
    optics: AMIOptics
    detector: None
    ramp: None
    read: None
    visibilities: None
    # TODO WAVELETS IS ONE OF THESE THINGS
    # EMPTY CLASSES THAT GET THE BITS POPULATED WHEN YOU CALL MODEL AND THE THINGS THAT MAP TO THE RIGHT PLACE INTHE MODEL PARAMS IS DETERMIEND BY THE MODEL FITS WHICCH ALLOWS YOU THAT FINE GRAINED CONTROL
    # MODEL FITS JUST MAPS YOU FROM THE PARAMEERS DICTIONARIY

    def __init__(self, params, optics, ramp, detector, read, filters):
        self.filters = filters
        self.optics = optics
        self.detector = detector
        self.ramp = ramp
        self.read = read
        self.visibilities = None

        super().__init__(params)

    def _get_distribution_from_key(self, exp_key) -> Array:
        """
        Returns the normalised intensity distribution of the source
        from the key of the parameter.
        """
        log_dist = self.params["log_distribution"][exp_key]

        return 10**log_dist

    def get_distribution(self, exposure) -> Array:
        """
        Returns the normalised intensity distribution of the source
        from the exposure object.
        """
        return self._get_distribution_from_key(exposure.get_key("log_distribution"))

    # def model(self, fit, **kwargs):
    #     return fit(self, **kwargs)

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


class WaveletModel(ResolvedAmigoModel):

    # TODO: wavelets : None

    def __init__(
        self,
        params,
        optics,
        ramp,
        detector,
        read,
        filters,
    ):

        if "wavelets" not in params:
            raise ValueError("Wavelets must be provided in params.")

        super().__init__(
            params=params,
            optics=optics,
            ramp=ramp,
            detector=detector,
            read=read,
            filters=filters,
        )

    def _get_distribution_from_key(self, exp_key):
        wavelets = self.params["wavelets"][exp_key]
        return wavelets.distribution

    def get_distribution(self, exposure) -> Array:
        """
        Returns the normalised intensity distribution of the source.
        """
        return _get_distribution_from_key(exposure.get_key("wavelets"))
