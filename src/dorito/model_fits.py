from amigo.model_fits import ModelFit
from amigo.core_models import BaseModeller
from amigo.vis_models import vis_to_im
from amigo.vis_analysis import AmigoOIData
from amigo.misc import interp
from jax import Array, numpy as np
import dLux.utils as dlu


class BaseResolvedFit:

    def rotate(self, distribution, clip=True):
        """
        Rotate the distribution by the parallactic angle.
        This method rotates the distribution using the dLux utility functions.
        Args:
            distribution: The distribution of the resolved source.
            clip: If True, clips the distribution to enforce positivity.
        Returns:
            Array: The rotated distribution, optionally clipped to enforce positivity.
        """
        knots = dlu.pixel_coords(distribution.shape[0], 1.0)
        samps = dlu.rotate_coords(knots, dlu.deg2rad(self.parang))

        distribution = interp(
            distribution,
            knots,
            samps,
            method="linear",
        )

        # clipping to enforce positivity
        if clip:
            return np.clip(distribution, min=0.0, max=None)

        return distribution


class ResolvedFit(ModelFit, BaseResolvedFit):
    """
    Model fit for resolved sources. This adds the log distribution parameter.
    """

    def __init__(self, file, use_cov=True):
        return super().__init__(file, use_cov=use_cov)

    def get_key(self, param):
        match param:
            case "log_dist":
                return self.filter

        return super().get_key(param)

    def map_param(self, param):
        match param:
            case "log_dist":
                return f"{param}.{self.get_key(param)}"

        return super().map_param(param)

    def initialise_params(self, optics, distribution):
        """
        Initialise the parameters for the resolved source model fit.
        The log distribution is set to a uniform distribution specified by the source size.

        Args:
            optics: The optics object (to pass to the parent class).
            source_size: The size of the source distribution (assumed square).

        Returns:
            params: A dictionary containing the initialised parameters for the model fit.
        """

        params = super().initialise_params(optics)

        # log distribution
        params["log_dist"] = (
            self.get_key("log_dist"),
            np.log10(distribution / distribution.sum()),
        )

        return params

    def simulate(self, model, return_slopes=True, rotate=True):
        # model = self.nuke_pixel_grads(model)
        psf = self.model_psf(model)

        # convolve source with PSF
        image = psf.convolve(model.get_distribution(self, rotate=rotate), method="fft")

        # downsample the image to the 3x oversample for the detector/ramp
        image = image.downsample(model.source_oversample)

        illuminance = self.model_illuminance(image, model)
        ramp = self.model_ramp(illuminance, model)
        ramp = self.model_read(ramp, model)

        if return_slopes:
            return ramp.set("data", np.diff(ramp.data, axis=0))
        return ramp


class MCAFit(ResolvedFit):

    def initialise_params(self, optics, moat, distribution, contrast):

        params = ModelFit.initialise_params(self, optics)

        # resolved component distributino
        distribution = (distribution.flatten())[~moat]
        distribution *= (1 - contrast) / distribution.sum()  # normalise the distribution

        log_dist = np.log10(distribution)
        params["log_dist"] = self.get_key("log_dist"), log_dist

        # the star component
        params["contrast"] = self.get_key("contrast"), np.array(contrast)

        return params

    def get_key(self, param):

        match param:
            case "contrast":
                return self.filter

        return super().get_key(param)

    def map_param(self, param):

        # Map the appropriate parameter to the correct key
        if param in ["contrast"]:
            return f"{param}.{self.get_key(param)}"

        # Else its global
        return super().map_param(param)


# class WaveletFit(ResolvedFit):

#     def get_key(self, param):
#         match param:
#             case "approx":
#                 return self.filter
#             case "details":
#                 return self.filter

#         return super().get_key(param)

#     def map_param(self, param):
#         match param:
#             case "approx":
#                 return f"{param}.{self.get_key(param)}"
#             case "details":
#                 return f"{param}.{self.get_key(param)}"

#         return super().map_param(param)

#     def update_wavelets(self, model):
#         """
#         Updates the wavelet coefficients for the given model.
#         """
#         wavelets = model.wavelets

#         if "approx" in model.params.keys() and "details" in model.params.keys():
#             approx = 10 ** model.approx[self.get_key("approx")]  # try fitting log approx
#             # approx = model.approx[self.get_key("approx")]
#             details = model.details[self.get_key("details")]

#             wavelets = wavelets.set(["approx", "values"], [approx, details])

#         return wavelets

#     def get_distribution(self, model) -> Array:
#         """
#         Returns the normalised intensity distribution of the source
#         from the exposure object.
#         """
#         wavelets = self.update_wavelets(model)
#         return wavelets.distribution


class OIFit(AmigoOIData):
    """
    Repurposing the AmigoOIData class to act as an Exposure/ModelFit amigo class.
    This is useful for fitting to this OI data.
    It requires a key and filter to be specified, which are used to identify the
    parameters in the model.

    Args:
        oi_data: The OI data to fit.
        key: The key to identify the parameters in the model.
        filter: The filter to use for the OI data. Must be one of "F380M", "F430M", or "F480M".
    """

    key: str
    filter: str

    def __init__(self, oi_data, key, filter):
        self.key = key

        if filter not in ["F380M", "F430M", "F480M"]:
            raise ValueError(
                f"Filter {filter} is not supported. Use 'F380M', 'F430M', or 'F480M'."
            )

        self.filter = filter
        super().__init__(oi_data)

    def initialise_params(self, model, distribution):
        pass

    def get_key(self, param):
        pass

    def map_param(self, param):
        pass


class ResolvedOIFit(OIFit, BaseResolvedFit):
    """
    Extending the OIFit class to include methods for handling resolved source fits.
    """

    def initialise_params(self, model, distribution):
        """
        Initialise the parameters for the resolved OI fit.
        This method sets up the log distribution and base UV parameters.
        The log distribution is the logarithm of the resolved source distribution,
        normalised to sum to 1. The base UV is the Fourier transform of a delta function
        at the centre of the distribution, which is used for normalisation.

        Args:
            model: The model object containing the parameters.
            distribution: The distribution of the resolved source.

        Returns:
            params: A dictionary containing the initialised parameters for the model fit.
        """

        params = {}  # Initialise an empty dictionary for parameters
        distribution /= distribution.sum()  # normalise the distribution

        params["log_dist"] = self.get_key("log_dist"), np.log10(distribution)
        params["base_uv"] = self.get_key("base_uv"), self.get_base_uv(model, distribution.shape[0])

        return params

    def get_key(self, param):

        match param:
            case "log_dist":
                return self.filter
            case "base_uv":
                return self.filter  # this is probably unnecessary

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

        Args:
            model: The model object containing the parameters.
            n_pix: The number of pixels in one axis of the distribution.
        Returns:
            Array: The base UV for normalisation, which is the Fourier transform of a delta function.
        """
        ind = n_pix // 2
        base_dist = np.zeros((n_pix, n_pix)).at[ind, ind].set(1.0)

        # base uv for normalisation
        base_uv = self.to_otf(model, base_dist)
        return base_uv

    def to_otf(self, model, distribution):
        """
        Transform the distribution to the OTF plane (Optical Transfer Function).
        This method performs a Matrix Fourier Transform of the distribution and returns the
        resulting visibilities in the OTF format.
        Args:
            model: The model object containing the parameters.
            distribution: The distribution of the resolved source.
        Returns:
            dlu.MFT: The OTF visibilities as a dLux MFT object.
        """

        return dlu.MFT(
            phasor=distribution + 0j,
            wavelength=self.wavel,
            pixel_scale_in=model.pscale_in,
            npixels_out=model.uv_npixels,
            pixel_scale_out=model.uv_pscale,
            inverse=True,
        )

    def to_cvis(self, model, distribution):

        # Perform MFT and move to OTF plane
        uv = self.to_otf(model, distribution)  # shape (102, 102)

        # Normalise the complex u,v plane
        uv /= model.params["base_uv"][self.get_key("base_uv")]

        # Downsample to the desired u,v resolution
        uv = dlu.downsample(uv, 2, mean=True)  # shape (51, 51)

        # flatten and take first half (u,v symmetry)
        cvis = uv.flatten()[: uv.size // 2]

        return cvis

    def model_disco(self, model, distribution):
        """
        Compute the model visibilities and phases for the given model object.
        """
        cvis = self.to_cvis(model, distribution)
        return self.flatten_model(cvis)

    def dirty_image(
        self, model, npix=None, rotate=True, otf_support=None, pad=None, pad_value=1 + 0j
    ):
        """
        Get the dirty image via MFT. This is the image that would be obtained
        if the visibilities were directly transformed back to the image plane.

        Args:
            model: The model object containing the parameters.
            npix: The number of pixels in one axis of the dirty image.
                    If None, uses the same size as the model source distribution.
            rotate: If True, rotates the dirty image by the parallactic angle.
                    If a float, rotates by that (-'ve) angle in radians.
        Returns:
            Array: The dirty image, normalised to sum to 1.
        """

        if npix is None:
            npix = model.get_distribution(self).shape[0]

        # converting to u,v visibilities
        log_vis = np.dot(np.linalg.pinv(self.vis_mat), self.vis)
        phase = np.dot(np.linalg.pinv(self.phi_mat), self.phi)
        vis_im, phase_im = vis_to_im(log_vis, phase, (51, 51))

        # exponentiating
        uv = np.exp(vis_im + 1j * phase_im)

        if pad is not None:
            # Pad the uv visibilities if a pad is specified
            uv = np.pad(uv, pad_width=pad, mode="constant", constant_values=pad_value)

        # If an OTF support is provided, apply it to the uv visibilities
        if otf_support is not None:
            uv *= otf_support

        # Getting the dirty image
        dirty_image = dlu.MFT(
            phasor=uv,
            wavelength=self.wavel,
            pixel_scale_in=2 * model.uv_pscale,
            npixels_out=npix,
            pixel_scale_out=model.pscale_in,
            inverse=True,
        )

        # Taking amplitudes
        dirty_image = np.abs(dirty_image)

        # Optional rotation of the dirty image
        if rotate:
            dirty_image = self.rotate(dirty_image)

        # Normalise the image
        return dirty_image / dirty_image.sum()

    def __call__(self, model, rotate=True):
        """
        Simulate the DISCOs from the resolved source distribution.
        This method retrieves the distribution from the model, optionally rotates it,
        and then computes the DISCOs using the model_disco method.
        Args:
            model: The model object containing the parameters.
            rotate: If True, rotates the distribution by the parallactic angle.
                    If a float, rotates by that (-'ve) angle in radians.
        Returns:
            tuple: A tuple containing the amplitudes and phases in the DISCO basis.
        """
        # NOTE: Distribution must be odd number of pixels in one axis
        distribution = model.get_distribution(self, rotate=rotate)

        return self.model_disco(model, distribution=distribution)


class MCAOIFit(ResolvedOIFit):

    def initialise_params(self, model, distribution, contrast=0.949):

        params = {}  # Initialise an empty dictionary for parameters

        # resolved component distributino
        distribution = (distribution.flatten())[~model.moat]
        distribution *= (1 - contrast) / distribution.sum()  # normalise the distribution

        log_dist = np.log10(distribution)
        params["log_dist"] = self.get_key("log_dist"), log_dist

        # the star component
        params["contrast"] = self.get_key("contrast"), np.array(contrast)

        # Fourier normalisation (not to be optimised)
        params["base_uv"] = self.get_key("base_uv"), self.get_base_uv(model, distribution.shape[0])
        params
        return params

    def get_key(self, param):

        match param:
            case "contrast":
                return self.filter

        return super().get_key(param)

    def map_param(self, param):

        # Map the appropriate parameter to the correct key
        match param:
            case "contrast":
                return f"{param}.{self.get_key(param)}"

        # Else its global
        return super().map_param(param)

    def __call__(self, model, rotate=False):
        return super().__call__(model, rotate=rotate)


class TransformedResolvedFit(ResolvedFit):
    """
    Model fit for resolved sources. This adds the log distribution parameter.
    """

    def initialise_params(self, optics, coeffs):
        """
        Initialise the parameters for the resolved source model fit.
        The log distribution is set to a uniform distribution specified by the source size.

        Args:
            optics: The optics object (to pass to the parent class).
            source_size: The size of the source distribution (assumed square).

        Returns:
            params: A dictionary containing the initialised parameters for the model fit.
        """

        params = ModelFit.initialise_params(self, optics)

        # log distribution
        params["log_dist"] = (self.get_key("log_dist"), coeffs)

        return params