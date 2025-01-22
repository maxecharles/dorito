from amigo.model_fits import ModelFit
from jax import Array, numpy as np, vmap


class ResolvedFit(ModelFit):
    """
    Model fit for resolved sources. Adds the log_distribution parameter to the
    model and uses it to calculate the intensity distribution of the source.
    """

    def get_key(self, param):
        match param:
            case "log_distribution":
                return self.filter
            case "spectral_coeffs":
                return self.filter

        return super().get_key(param)

    def map_param(self, param):
        match param:
            case "log_distribution":
                return f"{param}.{self.get_key(param)}"
            case "spectral_coeffs":
                return f"{param}.{self.get_key(param)}"

        return super().map_param(param)

    def _eval_weight(self, model, wavelength: Array) -> Array:
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
        coeffs = model.spectral_coeffs[self.get_key("spectral_coeffs")]

        return np.array([coeffs[i] * wavelength**i for i in range(len(coeffs))]).sum()

    def spectral_weights(self, model) -> Array:
        """
        Gets the relative spectral weights by evaluating the polynomial function at the
        internal wavelengths. Output weights are automatically normalised to a sum of
        1.

        Returns
        -------
        weights : Array
            The normalised relative weights of each wavelength.
        """
        wavelengths = model.filters[self.filter][0]
        weights = vmap(self._eval_weight, in_axes=(None, 0))(model, wavelengths)
        return weights / weights.sum()

    def get_spectra(self, model):
        """
        Gets the wavelengths and normalised spectral weights
        for the given exposure.

        Returns
        -------
        wavels : Array
            The wavelengths of the spectral weights.
        weights : Array
            The normalised spectral weights.
        """
        wavels, filt_weights = model.filters[self.filter]
        source_weights = self.spectral_weights(model)
        weights = filt_weights * (source_weights / source_weights.sum())
        return wavels, weights / weights.sum()

    def get_distribution(self, model) -> Array:
        """
        Returns the normalised intensity distribution of the source
        from the exposure object.
        """
        return model._get_distribution_from_key(self.get_key("log_distribution"))

    def simulate(self, model, return_paths=False):
        psf = self.model_psf(model)
        image = psf.convolve(
            self.get_distribution(model),
            method="fft",
        )
<<<<<<< HEAD
        image = self.model_detector(image, model, exposure)
        ramp = self.model_ramp(image, model, exposure)
        # return self.model_read(ramp, model, exposure)
        return np.diff(self.model_read(ramp, model, exposure).data, axis=0)
=======
        illuminance = self.model_illuminance(image, model)
        if return_paths:
            ramp, latent_path = self.model_ramp(illuminance, model, return_paths=return_paths)
            return self.model_read(ramp, model), latent_path
        else:
            ramp = self.model_ramp(illuminance, model)
            return self.model_read(ramp, model)
>>>>>>> 4b42509 (updating to latest amigo)


class DynamicResolvedFit(ResolvedFit):
    """
    Model fit for resolved sources where each exposure has a different
    intensity distribution.
    """

    def get_key(self, param):
        match param:
            case "log_distribution":
                return "_".join([self.key, self.filter])

        return super().get_key(param)


class WaveletFit(ResolvedFit):
    def get_key(self, exposure, param):
        match param:
            case "wavelets":
                return exposure.filter

        return super().get_key(exposure, param)

    def map_param(self, exposure, param):
        match param:
            case "wavelets":
                return f"{param}.{exposure.get_key(param)}"

        return super().map_param(exposure, param)

    def __call__(self, model, exposure):
        psf = self.model_psf(model, exposure)
        image = psf.convolve(model.get_distribution(exposure), method="fft")
        image = self.model_detector(image, model, exposure)
        ramp = self.model_ramp(image, model, exposure)
        return self.model_read(ramp, model, exposure)
