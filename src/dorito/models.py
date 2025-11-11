from jax import Array, numpy as np, tree as jtu
from amigo.core_models import BaseModeller, AmigoModel
import dLux.utils as dlu
from .bases import ImageBasis


class _BaseResolvedModel(BaseModeller):

    rotate: bool = True

    def get_distribution(self, exposure, rotate: bool = None):
        """
        Get the distribution from the exposure.

        Args:
            exposure: The exposure object containing the distribution key.
        Returns:
            Array: The intensity distribution of the source.
        """
        distribution = 10 ** self.params["log_dist"][exposure.get_key("log_dist")]
        if rotate is None:
            rotate = self.rotate
        if rotate:
            distribution = exposure.rotate(distribution)

        return distribution

    def __call__(self, exposure):
        """ """
        return self.get_distribution(exposure)


class _AmigoModel(AmigoModel):
    """
    Rewriting the parameter initialisation to allow for custom initialisers
    """

    def __init__(
        self,
        exposures,
        optics,
        detector,
        ramp_model,
        read,
        state=None,
        param_initers: dict = {},
    ):
        if state is not None:
            optics = optics.set("transmission", state["transmission"])
            detector = detector.set("jitter", state["jitter"])
            ramp_model = ramp_model.set(
                ["FF", "SRF", "nn_weights"],
                [state["FF"], state["SRF"], state["nn_weights"]],
            )
            read = read.set(
                ["dark_current", "non_linearity"],
                [state["dark_current"], state["non_linearity"]],
            )

        params = {}
        for exp in exposures:
            #######################################################################################
            # NOTE: This is the only different bit from the original AmigoModel __init__ method
            # You could still in theory pass the vis_model as a kwarg, but it is not used here):

            # For Calibrator Exposures
            if exp.calibrator:
                param_dict = exp.initialise_params(optics)
            else:
                param_dict = exp.initialise_params(optics, **param_initers)

            for param, (key, value) in param_dict.items():
                if param not in params.keys():
                    params[param] = {}
                params[param][key] = value
            #######################################################################################

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


class ResolvedAmigoModel(_AmigoModel, _BaseResolvedModel):
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
        rotate: bool = True,
        source_oversample=1,
        param_initers: dict = None,
    ):
        self.rotate = rotate
        self.source_oversample = source_oversample

        super().__init__(exposures, optics, detector, ramp_model, read, state, param_initers)


# class MCAModel(ResolvedAmigoModel):

#     size: int = None
#     moat: np.ndarray = None  # Placeholder for the moat attribute
#     star: np.ndarray = None  # Placeholder for the star attribute

#     def __init__(
#         self,
#         exposures: list,
#         optics,
#         detector,
#         ramp_model,
#         read,
#         state,
#         param_initers: dict,
#         rotate=True,
#         source_oversample=1,
#         moat_width: int = 0,
#     ):

#         dist_shape = param_initers["distribution"].shape
#         zeros = np.zeros(dist_shape).flatten()
#         star = zeros.at[zeros.size // 2].set(True)

#         if moat_width == 0:
#             moat_mask = np.array(star, dtype=bool)
#         else:
#             moat_mask = binary_dilation(star.reshape(dist_shape), iterations=moat_width).flatten()

#         # Precompute safe indices for use in JIT-compiled code
#         self.moat = np.where(~moat_mask)[0]  # shape (N,)

#         self.star = star
#         self.size = dist_shape[0]

#         param_initers["moat"] = self.moat

#         super().__init__(
#             exposures=exposures,
#             optics=optics,
#             detector=detector,
#             ramp_model=ramp_model,
#             read=read,
#             state=state,
#             rotate=rotate,
#             source_oversample=source_oversample,
#             param_initers=param_initers,
#         )

#     def get_distribution(self, exposure, rotate: bool = None, with_star: bool = True):
#         """
#         Get the distribution from the exposure.

#         Args:
#             exposure: The exposure object containing the distribution key.
#         Returns:
#             Array: The intensity distribution of the source.
#         """

#         zeros = np.zeros(self.size * self.size)
#         values = 10 ** self.params["log_dist"][exposure.get_key("log_dist")]
#         contrast = self.params["contrast"][exposure.get_key("contrast")]

#         resolved_component = zeros.at[self.moat].set(values).reshape((self.size, self.size))

#         if with_star:
#             star_component = contrast * self.star.reshape((self.size, self.size))
#             distribution = resolved_component + star_component
#         else:
#             distribution = resolved_component

#         if rotate is None:
#             rotate = self.rotate
#         if rotate:
#             distribution = exposure.rotate(distribution)

#         return distribution


class ResolvedDiscoModel(_BaseResolvedModel):
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
        rotate: bool = True,
    ):

        self.uv_npixels = uv_npixels
        self.oversample = oversample
        self.uv_pscale = uv_pscale
        self.psf_pixel_scale = psf_pixel_scale
        self.rotate = rotate

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


# class MCADiscoModel(ResolvedDiscoModel):

#     size: int = None
#     moat: np.ndarray = None  # Placeholder for the moat attribute
#     star: np.ndarray = None  # Placeholder for the star attribute

#     def __init__(
#         self,
#         ois: list,
#         distribution: np.ndarray,
#         uv_npixels: int,
#         uv_pscale: float,
#         oversample: float = 1.0,
#         psf_pixel_scale: float = 0.065524085,  # arcsec/pixel
#         moat_width: int = 3,
#         rotate=True,
#     ):

#         dist_shape = distribution.shape
#         zeros = np.zeros(dist_shape).flatten()
#         star = zeros.at[zeros.size // 2].set(True)

#         moat_mask = binary_dilation(star.reshape(dist_shape), iterations=moat_width).flatten()
#         # Precompute safe indices for use in JIT-compiled code
#         self.moat = np.where(~moat_mask)[0]  # shape (N,)

#         self.star = star
#         self.size = dist_shape[0]

#         super().__init__(
#             ois,
#             distribution,
#             uv_npixels,
#             uv_pscale,
#             oversample,
#             psf_pixel_scale,
#             rotate,
#         )

#     def get_distribution(self, exposure, rotate: bool = None, with_star: bool = True):
#         """
#         Get the distribution from the exposure.

#         Args:
#             exposure: The exposure object containing the distribution key.
#         Returns:
#             Array: The intensity distribution of the source.
#         """

#         zeros = np.zeros(self.size * self.size)
#         values = 10 ** self.params["log_dist"][exposure.get_key("log_dist")]
#         contrast = self.params["contrast"][exposure.get_key("contrast")]

#         resolved_component = zeros.at[self.moat].set(values).reshape((self.size, self.size))

#         if with_star:
#             star_component = contrast * self.star.reshape((self.size, self.size))
#             distribution = resolved_component + star_component
#         else:
#             distribution = resolved_component

#         if rotate is None:
#             rotate = self.rotate
#         if rotate:
#             distribution = exposure.rotate(distribution)

#         return distribution


import jax.tree as jtu
from dorito.models import ResolvedAmigoModel


class TransformedResolvedModel(ResolvedAmigoModel):

    basis: None
    window: Array

    def __init__(
        self,
        exposures,
        optics,
        detector,
        ramp_model,
        read,
        basis: ImageBasis,
        state,
        source_oversample=1,
        window: Array = None,
        param_initers: dict = {},
        rotate: bool = True,
    ):

        # This seems to fix some recompile issues
        def fn(x):
            if isinstance(x, Array):
                if "i" in x.dtype.str:
                    return x
                return np.array(x, dtype=float)
            return x

        self.basis = jtu.map(lambda x: fn(x), basis)
        self.window = window

        if "distribution" in param_initers.keys():
            init_log_dist = np.log10(param_initers["distribution"])
            init_coeffs = self.basis.to_basis(init_log_dist)
            param_initers["coeffs"] = init_coeffs
            del param_initers["distribution"]

        super().__init__(
            exposures,
            optics,
            detector,
            ramp_model,
            read,
            state,
            rotate,
            source_oversample,
            param_initers,
        )

    def get_distribution(
        self,
        exposure,
        rotate: bool = None,
        exponentiate=True,
        window=True,
    ):

        coeffs = self.params["log_dist"][exposure.get_key("log_dist")]

        # exponentiation
        if exponentiate:
            distribution = 10 ** self.basis.from_basis(coeffs)
        else:
            distribution = self.basis.from_basis(coeffs)

        # windowing
        if self.window is not None and window:
            distribution *= self.window

        # rotation
        if rotate is None:
            rotate = self.rotate
        if rotate:
            distribution = exposure.rotate(distribution)

        return distribution
