from amigo.model_fits import ModelFit
from jax import Array, numpy as np, vmap
import dLux.utils as dlu


class _ModelFit(ModelFit):
    actual_dither: int

    def __init__(self, file, **kwargs):
        super().__init__(file, **kwargs)
        self.actual_dither = int(file[0].header["PATT_NUM"])

    def get_key(self, param):
        match param:
            case "aberrations":
                return "_".join([self.filter, str(self.actual_dither)])
        return super().get_key(param)


class PointFit(_ModelFit):
    pass


class ResolvedFit(_ModelFit):
    """
    Model fit for resolved sources. Adds the log_distribution parameter to the
    model and uses it to calculate the intensity distribution of the source.
    """

    def get_key(self, param):
        match param:
            case "log_distribution":
                return self.filter
            case "position_angles":
                return self.filename

        return super().get_key(param)

    def map_param(self, param):
        match param:
            case "log_distribution":
                return f"{param}.{self.get_key(param)}"
            case "position_angles":
                return f"{param}.{self.get_key(param)}"

        return super().map_param(param)

    def initialise_params(self, optics, source_size, rolls_dict=None):

        params = super().initialise_params(optics)

        # log distribution
        params["log_distribution"] = (
            self.get_key("log_distribution"),
            np.log10(np.ones((source_size, source_size)) / source_size**2),
        )

        # position angles
        key = self.get_key("position_angles")
        if rolls_dict is not None:
            params["position_angles"] = (key, rolls_dict[key])
        else:
            params["position_angles"] = (key, np.array([0.0]))

        return params

    def get_distribution(self, model, rotate=False):
        """
        Returns the normalised intensity distribution of the source
        from the exposure object.
        """
        dist = model._get_distribution_from_key(self.get_key("log_distribution"))

        if rotate:
            if type(rotate) == bool:
                pa = model._get_pa_from_key(self.get_key("position_angles"))
            else:
                pa = rotate
            dist = dlu.rotate(dist, angle=-pa)

        return dist

    def simulate(self, model):

        psf = self.model_psf(model)

        # convolve source with PSF
        image = psf.convolve(self.get_distribution(model, rotate=model.rotate), method="fft")

        # downsample the image to the 3x oversample for the detector/ramp
        image = image.downsample(model.source_oversample)

        illuminance = self.model_illuminance(image, model)
        ramp = self.model_ramp(illuminance, model)
        return self.model_read(ramp, model)


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

    def update_wavelets(self, model):
        """
        Updates the wavelet coefficients for the given model.
        """
        wavelets = model.wavelets

        if "approx" in model.params.keys() and "details" in model.params.keys():
            approx = 10 ** model.approx[self.get_key("approx")]  # try fitting log approx
            # approx = model.approx[self.get_key("approx")]
            details = model.details[self.get_key("details")]

            wavelets = wavelets.set(["approx", "values"], [approx, details])

        return wavelets

    def get_distribution(self, model) -> Array:
        """
        Returns the normalised intensity distribution of the source
        from the exposure object.
        """
        wavelets = self.update_wavelets(model)
        return wavelets.distribution

    def get_key(self, param):
        match param:
            case "approx":
                return self.filter
            case "details":
                return self.filter

        return super().get_key(param)

    def map_param(self, param):
        match param:
            case "approx":
                return f"{param}.{self.get_key(param)}"
            case "details":
                return f"{param}.{self.get_key(param)}"

        return super().map_param(param)

    # def __call__(self, model, exposure):
    #     psf = self.model_psf(model, exposure)
    #     image = psf.convolve(model.get_distribution(exposure), method="fft")
    #     image = self.model_detector(image, model, exposure)
    #     ramp = self.model_ramp(image, model, exposure)
    #     return self.model_read(ramp, model, exposure)
