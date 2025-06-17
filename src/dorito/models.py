from amigo.core_models import BaseModeller, AmigoModel

# core jax
from jax import Array
import dLux.utils as dlu


class ResolvedAmigoModel(AmigoModel):
    """
    Amigo model for resolved sources.
    """

    rotate: bool = True
    source_oversample: int = 1

    def __init__(
        self,
        exposures,
        optics,
        detector,
        ramp_model,
        read,
        state,
        source_oversample=1,
        rotate=True,
    ):

        self.rotate = rotate
        self.source_oversample = source_oversample

        super().__init__(exposures, optics, detector, ramp_model, read, state)

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
