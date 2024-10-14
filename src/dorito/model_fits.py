from amigo.model_fits import ModelFit
from jax import Array, numpy as np, vmap


class ResolvedFit(ModelFit):
    """
    Model fit for resolved sources. Adds the log_distribution parameter to the
    model and uses it to calculate the intensity distribution of the source.
    """

    def get_distribution(self, model, exposure) -> Array:
        """
        Returns the normalised intensity distribution of the source.
        """
        distribution = 10 ** model.log_distribution[self.get_key(exposure, "log_distribution")]
        return distribution / distribution.sum()

    def _eval_weight(self, model, exposure, wavelength: Array) -> Array:
        """
        Evaluates the polynomial function at the supplied wavelength.

        Parameters
        ----------
        wavelength : Array, metres
            The wavelength at which to evaluate the polynomial function.

        Returns
        -------
        weight : Array
            The relative weight of the supplied wavelength.
        """
        coeffs = model.spectral_coeffs[self.get_key(exposure, "spectral_coeffs")]

        return np.array([coeffs[i] * wavelength**i for i in range(len(coeffs))]).sum()

    def spectral_weights(self, model, exposure) -> Array:
        """
        Gets the relative spectral weights by evaluating the polynomial function at the
        internal wavelengths. Output weights are automatically normalised to a sum of
        1.

        Returns
        -------
        weights : Array
            The normalised relative weights of each wavelength.
        """
        wavelengths = model.filters[exposure.filter][0]
        weights = vmap(self._eval_weight, in_axes=(None, None, 0))(model, exposure, wavelengths)
        return weights / weights.sum()

    def get_spectra(self, model, exposure):
        wavels, filt_weights = model.filters[exposure.filter]
        source_weights = self.spectral_weights(model, exposure)
        weights = filt_weights * (source_weights / source_weights.sum())
        return wavels, weights / weights.sum()

    def get_key(self, exposure, param):
        match param:
            case "log_distribution":
                return exposure.filter
            case "spectral_coeffs":
                return exposure.filter

        return super().get_key(exposure, param)

    def map_param(self, exposure, param):
        match param:
            case "log_distribution":
                return f"{param}.{exposure.get_key(param)}"
            case "spectral_coeffs":
                return f"{param}.{exposure.get_key(param)}"
            # case "aberrations":
            #     return exposure.key

        return super().map_param(exposure, param)

    def __call__(self, model, exposure):
        psf = self.model_psf(model, exposure)
        image = psf.convolve(self.get_distribution(model, exposure), method="fft")
        image = self.model_detector(image, model, exposure)
        ramp = self.model_ramp(image, model, exposure)
        return self.model_read(ramp, model, exposure)


class DynamicResolvedFit(ResolvedFit):
    """
    Model fit for resolved sources where each exposure has a different
    intensity distribution.
    """

    def get_key(self, exposure, param):
        match param:
            case "log_distribution":
                return "_".join([exposure.key, exposure.filter])

        return super().get_key(exposure, param)
