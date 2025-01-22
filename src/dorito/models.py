from amigo.core_models import BaseModeller
from amigo.optical_models import AMIOptics

from . import build_model

# core jax
import jax
from jax import vmap, Array, numpy as np, tree as jtu

# wavelets
import jaxshearlab.pyShearLab2D as jsl
import jaxwt

from abc import abstractmethod


class ResolvedAmigoModel(BaseModeller):
    filters: dict
    optics: AMIOptics
    detector: None
    ramp: None
    read: None
    visibilities: None

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


class BaseWaveletModel(ResolvedAmigoModel):
    shapes: list
    sizes: list
    starts: list
    tree_def: None

    def __init__(
        self,
        params,
        optics,
        ramp,
        detector,
        read,
        filters,
        exposures=None,
        wavelets=None,
    ):
        if exposures is None:
            raise ValueError("Exposures must be provided.")

        waveleaves, tree_def = jtu.flatten(wavelets)
        coeffs = np.concatenate([val.flatten() for val in waveleaves])

        self.shapes = [v.shape for v in waveleaves]
        self.sizes = [int(v.size) for v in waveleaves]
        self.starts = [int(i) for i in np.cumsum(np.array([0] + self.sizes))]
        self.tree_def = tree_def

        wavelet_dict = {}
        for exp in exposures:
            if not exp.calibrator:
                key = exp.get_key("wavelets")
                wavelet_dict[key] = coeffs

        params["wavelets"] = wavelet_dict

        super().__init__(
            params=params,
            optics=optics,
            ramp=ramp,
            detector=detector,
            read=read,
            filters=filters,
        )

    def flatten_wavelets(self, wavelets):
        waveleaves, _ = jtu.flatten(wavelets)
        return np.concatenate([val.flatten() for val in waveleaves])

    def _wavelet_coeffs_from_key(self, exp_key):
        wavalues = self.params["wavelets"][exp_key]

        waveleaves = [
            jax.lax.dynamic_slice(wavalues, (start,), (size,)).reshape(shape)
            for start, size, shape in zip(self.starts, self.sizes, self.shapes)
        ]
        return jtu.unflatten(self.tree_def, waveleaves)

    def wavelet_coeffs(self, exposure):
        return self._wavelet_coeffs_from_key(exposure.get_key("wavelets"))

    @abstractmethod
    def _get_distribution_from_key(self, exp_key) -> Array:
        """
        Returns the normalised intensity distribution of the source.
        """

    def get_distribution(self, exposure) -> Array:
        """
        Returns the normalised intensity distribution of the source.
        """
        return self._get_distribution_from_key(exposure.get_key("wavelets"))

    @abstractmethod
    def wavelet_transform(self, exposure) -> Array:
        """
        Returns the wavelet coefficients of for a given distribution.
        """


class WaveletModel(BaseWaveletModel):
    wavelet: str = "db2"
    level: int = None

    def __init__(
        self,
        params,
        optics,
        ramp,
        detector,
        read,
        filters,
        exposures=None,
        wavelets=None,
        source_size=98,
        level=None,
        wavelet="db2",
    ):
        if wavelets is None:
            if source_size is None or level is None:
                raise ValueError("Source size and level must be provided if wavelets are not.")
            wavelets = build_model.wavelet_prior(source_size, level=level)

        self.wavelet = wavelet
        self.level = level

        super().__init__(
            params=params,
            optics=optics,
            ramp=ramp,
            detector=detector,
            read=read,
            filters=filters,
            exposures=exposures,
            wavelets=wavelets,
        )

    def _get_distribution_from_key(self, exp_key) -> Array:
        """
        Returns the normalised intensity distribution of the source.
        """
        wavelet_coeffs = self._wavelet_coeffs_from_key(exp_key)
        distribution = jaxwt.waverec2(wavelet_coeffs, self.wavelet)[0]
        return distribution

    def wavelet_transform(self, array) -> Array:
        # converting to wavelet coefficients
        return jaxwt.conv_fwt_2d.wavedec2(
            array,
            self.wavelet,
            level=self.level,
        )


class ShearletModel(BaseWaveletModel):
    shearletSystem: None

    def __init__(
        self,
        params,
        optics,
        ramp,
        detector,
        read,
        filters,
        exposures=None,
        source_size=98,
        wavelets=None,
        shearletSystem=None,
        nScales=2,
    ):
        if wavelets is None:
            if shearletSystem is not None:
                wavelets, _ = build_model.shearlet_prior(
                    source_size,
                    shearletSystem=shearletSystem,
                    nScales=nScales,
                )

            elif shearletSystem is None:
                wavelets, shearletSystem = build_model.shearlet_prior(
                    source_size,
                    nScales=nScales,
                )

        self.shearletSystem = shearletSystem

        super().__init__(
            params=params,
            optics=optics,
            ramp=ramp,
            detector=detector,
            read=read,
            filters=filters,
            exposures=exposures,
            wavelets=wavelets,
        )

    def _get_distribution_from_key(self, exp_key) -> Array:
        """
        Returns the normalised intensity distribution of the source.
        """
        wavelet_coeffs = self._wavelet_coeffs_from_key(exp_key)
        return jsl.SLshearrec2D(wavelet_coeffs, self.shearletSystem)

    def wavelet_transform(self, array) -> Array:
        return jsl.SLsheardec2D(array, self.shearletSystem)
