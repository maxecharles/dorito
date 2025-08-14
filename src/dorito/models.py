from amigo.core_models import BaseModeller, AmigoModel

# core jax
from jax import Array, numpy as np, tree as jtu
import dLux.utils as dlu
from scipy.ndimage import binary_dilation


class BaseResolvedModel:

    def get_distribution(self, exposure, rotate=True):
        """
        Get the distribution from the exposure.

        Args:
            exposure: The exposure object containing the distribution key.
        Returns:
            Array: The intensity distribution of the source.
        """
        distribution = 10 ** self.params["log_dist"][exposure.get_key("log_dist")]
        if rotate:
            distribution = exposure.rotate(distribution)

        return distribution

    def __call__(self, exposure, rotate=True):
        """ """
        return self.get_distribution(exposure, rotate=rotate)


# Rewriting the parameter initialisation to allow for custom initialisers
class _AmigoModel(AmigoModel):

    def __init__(
        self, exposures, optics, detector, ramp_model, read, state=None, param_initers: dict = None
    ):
        if state is not None:
            optics = optics.set("transmission", state["transmission"])
            detector = detector.set("jitter", state["jitter"])
            ramp_model = ramp_model.set(
                ["FF", "SRF", "nn_weights"], [state["FF"], state["SRF"], state["nn_weights"]]
            )
            read = read.set(
                ["dark_current", "non_linearity"],
                [state["dark_current"], state["non_linearity"]],
            )

        params = {}
        for exp in exposures:
            #######################################################################################
            # NOTE: This is the only different bit from the original AmigoModel __init__ method
            # You could still in theory pass the vis_model as an kwarg, but it is not used here
            if param_initers is not None:
                if exp.calibrator:
                    ps = {}
                else:
                    ps = param_initers
                param_dict = exp.initialise_params(optics, **ps)
            #######################################################################################
            else:
                param_dict = exp.initialise_params(optics)
            for param, (key, value) in param_dict.items():
                if param not in params.keys():
                    params[param] = {}
                params[param][key] = value

        if state is not None:
            params["defocus"] = state["defocus"]

            abb = {}
            for key in params["aberrations"].keys():
                prog, filt = key.split("_")
                abb[key] = state["aberrations"][filt]

            params["aberrations"] = abb  # jtu.map(lambda x: abb, params["aberrations"])

        # This seems to fix some recompile issues
        def fn(x):
            if isinstance(x, Array):
                if "i" in x.dtype.str:
                    return x
                return np.array(x, dtype=float)
            return x

        self.params = jtu.map(lambda x: fn(x), params)
        self.optics = jtu.map(lambda x: fn(x), optics)
        self.detector = jtu.map(lambda x: fn(x), detector)
        self.ramp_model = jtu.map(lambda x: fn(x), ramp_model)
        self.read = jtu.map(lambda x: fn(x), read)
        # self.vis_model = jtu.map(lambda x: fn(x), vis_model)

        # NOTE: okay I changed this too
        self.vis_model = None


class ResolvedAmigoModel(_AmigoModel, BaseResolvedModel):
    """
    Amigo model for resolved sources.
    """

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
        param_initers: dict = None,
    ):

        self.source_oversample = source_oversample

        super().__init__(exposures, optics, detector, ramp_model, read, state, param_initers)


class MCAModel(ResolvedAmigoModel):

    size: int = None
    moat: np.ndarray = None  # Placeholder for the moat attribute
    star: np.ndarray = None  # Placeholder for the star attribute

    def __init__(
        self,
        exposures: list,
        optics,
        detector,
        ramp_model,
        read,
        state,
        param_initers: dict,
        source_oversample=1,
        moat_width: int = 0,
    ):

        dist_shape = param_initers["distribution"].shape
        zeros = np.zeros(dist_shape).flatten()
        star = zeros.at[zeros.size // 2].set(True)

        if moat_width == 0:
            moat_mask = np.array(star, dtype=bool)
        else:
            moat_mask = binary_dilation(star.reshape(dist_shape), iterations=moat_width).flatten()

        # Precompute safe indices for use in JIT-compiled code
        self.moat = np.where(~moat_mask)[0]  # shape (N,)

        self.star = star
        self.size = dist_shape[0]

        param_initers["moat"] = self.moat

        super().__init__(
            exposures=exposures,
            optics=optics,
            detector=detector,
            ramp_model=ramp_model,
            read=read,
            state=state,
            source_oversample=source_oversample,
            param_initers=param_initers,
        )

    def get_distribution(self, exposure, rotate=True, with_star=True):
        """
        Get the distribution from the exposure.

        Args:
            exposure: The exposure object containing the distribution key.
        Returns:
            Array: The intensity distribution of the source.
        """

        zeros = np.zeros(self.size * self.size)
        values = 10 ** self.params["log_dist"][exposure.get_key("log_dist")]
        contrast = self.params["contrast"][exposure.get_key("contrast")]

        resolved_component = zeros.at[self.moat].set(values).reshape((self.size, self.size))

        if with_star:
            star_component = contrast * self.star.reshape((self.size, self.size))
            distribution = resolved_component + star_component
        else:
            distribution = resolved_component

        if rotate:
            distribution = exposure.rotate(distribution)

        return distribution


# class WaveletModel(ResolvedAmigoModel):
#     """
#     A class for resolved source models in the AMIGO framework that includes wavelet transforms.
#     This class extends the ResolvedAmigoModel to incorporate wavelet transforms
#     for more complex source distributions.
#     It inherits from BaseModeller and provides methods to retrieve source distributions
#     and position angles based on exposure keys, while also handling wavelet transforms.
#     """

#     wavelets: None

#     def __init__(
#         self,
#         source_size,
#         exposures,
#         optics,
#         detector,
#         read,
#         rotate=False,
#         rolls_dict=None,
#         source_oversample=1,
#         wavelets=None,
#     ):
#         self.wavelets = wavelets

#         super().__init__(
#             source_size,
#             exposures,
#             optics,
#             detector,
#             read,
#             rotate=rotate,
#             rolls_dict=rolls_dict,
#             source_oversample=source_oversample,
#         )


class ResolvedDiscoModel(BaseResolvedModel):
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


class MCADiscoModel(ResolvedDiscoModel):

    size: int = None
    moat: np.ndarray = None  # Placeholder for the moat attribute
    star: np.ndarray = None  # Placeholder for the star attribute

    def __init__(
        self,
        ois: list,
        distribution: np.ndarray,
        uv_npixels: int,
        uv_pscale: float,
        oversample: float = 1.0,
        psf_pixel_scale: float = 0.065524085,  # arcsec/pixel
        moat_width: int = 3,
    ):

        dist_shape = distribution.shape
        zeros = np.zeros(dist_shape).flatten()
        star = zeros.at[zeros.size // 2].set(True)

        moat_mask = binary_dilation(star.reshape(dist_shape), iterations=moat_width).flatten()
        # Precompute safe indices for use in JIT-compiled code
        self.moat = np.where(~moat_mask)[0]  # shape (N,)

        self.star = star
        self.size = dist_shape[0]

        super().__init__(
            ois,
            distribution,
            uv_npixels,
            uv_pscale,
            oversample,
            psf_pixel_scale,
        )

    def get_distribution(self, exposure, rotate=False, with_star=True):
        """
        Get the distribution from the exposure.

        Args:
            exposure: The exposure object containing the distribution key.
        Returns:
            Array: The intensity distribution of the source.
        """

        zeros = np.zeros(self.size * self.size)
        values = 10 ** self.params["log_dist"][exposure.get_key("log_dist")]
        contrast = self.params["contrast"][exposure.get_key("contrast")]

        resolved_component = zeros.at[self.moat].set(values).reshape((self.size, self.size))

        if with_star:
            star_component = contrast * self.star.reshape((self.size, self.size))
            distribution = resolved_component + star_component
        else:
            distribution = resolved_component

        if rotate:
            distribution = exposure.rotate(distribution)

        return distribution
