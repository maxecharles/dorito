from amigo.model_fits import ModelFit
from amigo.vis_models import vis_to_im
from amigo.vis_analysis import AmigoOIData
from jax import Array, numpy as np, vmap
import dLux.utils as dlu
import equinox as eqx
import zodiax as zdx


class _ModelFit(ModelFit):
    actual_dither: int

    def __init__(self, file, **kwargs):
        super().__init__(file, **kwargs)
        self.actual_dither = int(file[0].header["PATT_NUM"])

    def simulate(self, model, return_bleed=False):
        psf = self.model_psf(model)
        psf = psf.downsample(model.source_oversample)
        illuminance = self.model_illuminance(psf, model)
        if return_bleed:
            ramp, latent_path = self.model_ramp(illuminance, model, return_bleed=return_bleed)
            return self.model_read(ramp, model), latent_path
        else:
            ramp = self.model_ramp(illuminance, model)
            return self.model_read(ramp, model)


class PointFit(_ModelFit):
    pass


class UniqueAbFit(PointFit):
    def get_key(self, param):
        match param:
            case "aberrations":
                return "_".join([self.filter, str(self.actual_dither)])
        return super().get_key(param)


class ResolvedFit(PointFit):
    """
    Model fit for resolved sources. Adds the log_distribution parameter to the
    model and uses it to calculate the intensity distribution of the source.
    """

    unique_abs: bool = False

    def __init__(self, file, unique_abs=False, **kwargs):
        super().__init__(file, **kwargs)
        self.unique_abs = unique_abs

    def get_key(self, param):
        match param:
            case "log_distribution":
                return self.filter
            case "position_angles":
                return self.filename
            case "aberrations":
                if self.unique_abs:
                    return "_".join([self.filter, str(self.actual_dither)])
                else:
                    return self.filter

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


class ResolvedOIFit(AmigoOIData):
    """
    Amigo OI Fit class
    """

    key: str
    # filter: str

    def __init__(self, oi_data, key):
        self.key = key
        # self.filter = filter
        super().__init__(oi_data)

    def initialise_params(self, model, distribution):
        params = {}

        # centre = distribution.shape[0] // 2
        # centre_val = distribution[centre, centre]
        # distribution /= centre_val

        distribution /= distribution.sum()

        params["log_dist"] = self.get_key("log_dist"), np.log10(distribution)

        params["base_uv"] = self.get_key("base_uv"), self.get_base_uv(model, distribution.shape[0])

        return params

    def get_key(self, param):

        match param:
            case "log_dist":
                return self.key
            case "base_uv":
                return self.key

    def map_param(self, param):
        """
        The `key` argument will return only the _key_ extension of the parameter path,
        which is required for object initialisation.
        """

        # Map the appropriate parameter to the correct key
        if param in ["log_dist", "base_uv"]:
            return f"{param}.{self.get_key(param)}"

        # Else its global
        return param

    def get_base_uv(self, model, n_pix):
        """
        Get the base uv for normalisation
        """
        ind = n_pix // 2
        base_dist = np.zeros((n_pix, n_pix)).at[ind, ind].set(1.0)

        # base uv for normalisation
        base_uv = self.to_otf(model, base_dist)
        return base_uv

    def to_otf(self, model, distribution):

        # distribution = np.pad(distribution, 5 * distribution.shape[0])

        return dlu.MFT(
            phasor=distribution + 0j,
            wavelength=self.wavel,  # If focal length is None this doesn't do anything
            pixel_scale_in=model.pscale_in,
            npixels_out=model.uv_npixels,
            pixel_scale_out=model.uv_pscale,
        )

    def to_logvis(self, model, distribution):
        """
        Get the log visibilities
        """
        # Getting regular visibilities
        uv = self.to_otf(model, distribution)

        # Normalise the visibilities, take log and get amp and phase
        uv /= model.params["base_uv"][self.get_key("base_uv")]
        uv = dlu.downsample(uv, 2, mean=True)
        vis = uv.flatten()[: uv.size // 2]
        vis = np.log(vis)
        return vis.real, vis.imag

    def model_sicko(self, model, distribution):

        # Getting log visibilities
        log_amp, phase = self.to_logvis(model, distribution)

        # Transform to the sicko basis
        sicko_amp = np.dot(log_amp, self.vis_mat)
        sicko_phase = np.dot(phase, self.phi_mat)

        return sicko_amp, sicko_phase
        # return np.concatenate([sicko_amp, sicko_phase])
        # return np.vstack([sicko_amp, sicko_phase])

    def dirty_image(self, model, npix=None):
        """
        Get the dirty image via MFT.
        """

        if npix is None:
            npix = model.get_distribution(self).shape[0]

        # converting to u,v visibilities
        log_vis = np.dot(self.vis, np.linalg.pinv(self.vis_mat))
        phase = np.dot(self.phi, np.linalg.pinv(self.phi_mat))
        vis_im, phase_im = vis_to_im(log_vis, phase, (51, 51))

        # exponentiating
        uv = np.exp(vis_im + 1j * phase_im)

        # Getting the dirty image
        dirty_image = dlu.MFT(
            phasor=uv,
            wavelength=self.wavel,  # If focal length is None this doesn't do anything
            pixel_scale_in=2 * model.uv_pscale,
            npixels_out=npix,
            pixel_scale_out=model.pscale_in,
            inverse=True,
        )

        # Taking amplitudes
        dirty_image = np.abs(dirty_image)

        # Normalise the image
        return dirty_image / dirty_image.sum()

    def __call__(self, model):
        """
        Call the model with the given parameters.
        """
        # NOTE: Distribution must be odd number of pixels in one axis
        distribution = model.get_distribution(self)

        # rotate distribution by the parallactic angle
        knots = dlu.pixel_coords(distribution.shape[0], 1.0)
        samps = dlu.rotate_coords(knots, -dlu.deg2rad(self.parang))

        distribution = amigo.misc.interp(
            distribution,
            knots,
            samps,
            method="cubic",
        )

        # clipping to enforce positivity
        distribution = np.clip(distribution, min=0.0, max=None)

        return self.model_sicko(model, distribution=distribution)
