"""Models and Amigo model subclasses used for resolved sources.

This module implements resolved-source models for both image plane
fitting and interferometric data (DISCO).
"""

from jax import Array, numpy as np, tree as jtu
from amigo.core_models import BaseModeller, AmigoModel
import dLux.utils as dlu
from .bases import ImageBasis

__all__ = [
    "ResolvedAmigoModel",
    "ResolvedDiscoModel",
    "TransformedResolvedModel",
]


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
    """Amigo model specialised for resolved (image-plane) sources.

    This class composes the internal `_AmigoModel` parameter initialisation
    behaviour with the `_BaseResolvedModel` helpers for retrieving a
    resolved-source distribution. It is the primary model used in the
    examples to represent a resolved astronomical source for use with the
    amigo fitting machinery.

    Parameters
    ----------
    exposures : sequence
        Sequence of exposure / fit objects describing each observation.
    optics, detector, ramp_model, read
        Amigo-style objects used to build the forward model (see amigo
        documentation for expected types).
    state : mapping, optional
        Optional calibration state used to override optics/detector/ramp
        initial values.
    rotate : bool, optional
        If True (default) apply exposure rotation to distributions.
    source_oversample : int, optional
        Oversampling factor for source-plane operations. When setting an
        oversampling factor > 1, the optics model must be initialised
        with an oversample to match (e.g. 3 times source_oversample).
    param_initers : dict, optional
        Parameter initialiser values forwarded to exposure initialisation.
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


class TransformedResolvedModel(ResolvedAmigoModel):
    """Resolved model that stores and operates in a compact image basis.

    This class wraps a provided ``ImageBasis`` object and stores the
    source distribution as basis coefficients. When initialising, if a
    ``distribution`` is provided in ``param_initers`` it is converted to
    basis coefficients and stored under the ``coeffs`` initialiser key.

    Parameters
    ----------
    basis : ImageBasis
        Basis object providing ``to_basis`` / ``from_basis`` conversions.
    window : Array, optional
        Optional multiplicative window applied to reconstructed images.
    source_oversample : int, optional
        Oversampling factor for source-plane operations.
    param_initers : dict, optional
        Parameter initialisers; accepts a ``distribution`` entry which will
        be converted to ``coeffs`` via the supplied ``basis``.
    """

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


class ResolvedDiscoModel(_BaseResolvedModel):
    """Resolved-source model container for DISCO / interferometric fits.

    This lightweight container holds parameters required to transform an
    image-plane distribution into the complex visibilities used by the
    DISCO-style fitting code. The constructor collects per-oi parameters by
    calling each `oi.initialise_params(self, distribution)` and assembling a
    parameter mapping compatible with the rest of the amigo pipeline.

    Parameters
    ----------
    ois : list
        List of OI-like exposure objects providing `initialise_params` and
        other data accessors used during fitting.
    distribution : Array
        Initial image-plane distribution used to derive initial parameters.
    uv_npixels : int
        Number of pixels in the output u/v plane used for transforms.
    uv_pscale : float
        Pixel scale in the u/v plane.
    oversample : float, optional
        Image-plane oversampling factor used by the model (default: 1.0).
    psf_pixel_scale : float, optional
        PSF pixel scale in arcseconds per pixel (default chosen for examples).
    rotate : bool, optional
        If True, model-dirty images and transforms will be rotated by the
        exposure parallactic angle when requested.

    Attributes
    ----------
    pscale_in : float
        Property returning the image-plane pixel scale in radians per pixel.
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
