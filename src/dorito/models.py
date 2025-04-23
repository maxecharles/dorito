from amigo.core_models import BaseModeller
from amigo.optical_models import AMIOptics

# core jax
from jax import Array
import dLux.utils as dlu


class ResolvedAmigoModel(BaseModeller):
    # filters: dict
    optics: AMIOptics
    detector: None
    read: None
    rotate: bool = False
    source_oversample: int = 1

    # TODO WAVELETS IS ONE OF THESE THINGS
    # EMPTY CLASSES THAT GET THE BITS POPULATED WHEN YOU CALL MODEL AND THE THINGS THAT MAP TO THE RIGHT PLACE INTHE MODEL PARAMS IS DETERMIEND BY THE MODEL FITS WHICCH ALLOWS YOU THAT FINE GRAINED CONTROL
    # MODEL FITS JUST MAPS YOU FROM THE PARAMEERS DICTIONARIY

    def __init__(
        self,
        source_size,
        exposures,
        optics,
        detector,
        read,
        rotate=False,
        rolls_dict=None,
        source_oversample=1.0,
    ):

        self.optics = optics
        self.detector = detector
        self.read = read
        self.rotate = rotate
        self.source_oversample = source_oversample

        params = {}
        for exp in exposures:
            param_dict = exp.initialise_params(optics, source_size, rolls_dict)
            for param, (key, value) in param_dict.items():
                if param not in params.keys():
                    params[param] = {}
                params[param][key] = value
        self.params = params

        super().__init__(params)

    def _get_distribution_from_key(self, exp_key) -> Array:
        """
        Returns the normalised intensity distribution of the source
        from the key of the parameter.
        """
        log_dist = self.params["log_distribution"][exp_key]

        return 10**log_dist

    def _get_pa_from_key(self, exp_key) -> Array:
        """
        Returns the normalised intensity distribution of the source
        from the key of the parameter.
        """
        pa_deg = self.params["position_angles"][exp_key]

        return dlu.deg2rad(pa_deg)

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
        # if hasattr(self.visibilities, key):
        #     return getattr(self.visibilities, key)
        raise AttributeError(f"{self.__class__.__name__} has no attribute " f"{key}.")


class WaveletModel(ResolvedAmigoModel):

    wavelets: None

    def __init__(
        self,
        params,
        optics,
        ramp,
        detector,
        read,
        filters,
        wavelets,
    ):
        self.wavelets = wavelets

        super().__init__(
            params=params,
            optics=optics,
            ramp=ramp,
            detector=detector,
            read=read,
            filters=filters,
        )

    def get_distribution(self, exposure) -> Array:
        """
        Returns the normalised intensity distribution of the source
        from the exposure object.
        """
        return exposure.get_distribution(self)

    # def __getattr__(self, key):
    #     if hasattr(self.wavelets, key):
    #         return getattr(self.wavelets, key)
    #     return super().__getattr__(key)
