from amigo.core_models import BaseModeller
from amigo.optical_models import AMIOptics

# core jax
from jax import Array
import dLux.utils as dlu


class ResolvedAmigoModel(BaseModeller):
    """
    A class for resolved source models in the AMIGO framework.
    This class is designed to handle the parameters and distributions
    of resolved sources, including their optical properties and detector characteristics.
    It inherits from BaseModeller and provides methods to retrieve source distributions
    and position angles based on exposure keys.
    """

    optics: AMIOptics
    detector: None
    read: None
    rotate: bool = False
    source_oversample: int = 1

    def __init__(
        self,
        source_size,
        exposures,
        optics,
        detector,
        read,
        rotate=False,
        rolls_dict=None,
        source_oversample=1,
    ):
        """
        Initialises the ResolvedAmigoModel with the given parameters.
        Args:
            source_size (float): Size of the source in arcseconds.
            exposures (list): List of exposure objects to initialise parameters to.
            optics (AMIOptics): Optical model for the AMIGO instrument.
            detector: Detector model for the AMIGO instrument.
            read: Readout model for the AMIGO instrument.
            rotate (bool): Whether to rotate the source distribution.
            rolls_dict (dict): Dictionary containing roll angles for exposures.
            source_oversample (int): Oversampling factor for the source distribution.
        """

        self.optics = optics
        self.detector = detector
        self.read = read
        self.rotate = rotate
        self.source_oversample = source_oversample

        # Initialising the parameters for each exposure
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
    """
    A class for resolved source models in the AMIGO framework that includes wavelet transforms.
    This class extends the ResolvedAmigoModel to incorporate wavelet transforms
    for more complex source distributions.
    It inherits from BaseModeller and provides methods to retrieve source distributions
    and position angles based on exposure keys, while also handling wavelet transforms.
    """

    wavelets: None

    def __init__(
        self,
        source_size,
        exposures,
        optics,
        detector,
        read,
        rotate=False,
        rolls_dict=None,
        source_oversample=1,
        wavelets=None,
    ):
        self.wavelets = wavelets

        super().__init__(
            source_size,
            exposures,
            optics,
            detector,
            read,
            rotate=rotate,
            rolls_dict=rolls_dict,
            source_oversample=source_oversample,
        )


class ResolvedDiscoModel(BaseModeller):
    """
    A class to hold the parameters of a resolved source model to be used in fitting
    to DISCO data.
    """

    uv_npixels: int
    uv_pscale: float
    oversample: float
    psf_pixel_scale: float

    def __init__(
        self,
        ois: list,
        distribution: Array,
        uv_npixels: int,
        uv_pscale: float,
        oversample: float = 1.0,
        psf_pixel_scale: float = 0.065524085,  # arcsec/pixel
    ):

        self.uv_npixels = uv_npixels
        self.oversample = oversample
        self.uv_pscale = uv_pscale
        self.psf_pixel_scale = psf_pixel_scale

        params = {}
        for oi in ois:
            param_dict = oi.initialise_params(self, distribution)
            for param, (key, value) in param_dict.items():
                if param not in params.keys():
                    params[param] = {}
                params[param][key] = value

        super().__init__(params)

    @property
    def pscale_in(self):
        """
        The pixel scale of the image plane, in radians per pixel.
        """
        return dlu.arcsec2rad(self.psf_pixel_scale / self.oversample)

    def get_distribution(self, exposure):
        """
        Get the distribution from the exposure.

        Args:
            exposure: The exposure object containing the distribution key.
        Returns:
            Array: The intensity distribution of the source.
        """
        log_dist = self.params["log_dist"][exposure.get_key("log_dist")]
        return 10**log_dist
