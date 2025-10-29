from amigo.model_fits import ModelFit
from amigo.vis_models import vis_to_im
from amigo.vis_analysis import AmigoOIData
from amigo.misc import interp
from jax import lax, numpy as np
import dLux as dl
import dLux.utils as dlu
import equinox as eqx


class BaseResolvedFit:

    def rotate(self, distribution, clip=True, interp_method="linear"):
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
            method=interp_method,
        )

        # clipping to enforce positivity
        if clip:
            return np.clip(distribution, min=0.0, max=None)

        return distribution

    def simulate(self, model, return_slopes: bool = True, **kwargs):
        # model = self.nuke_pixel_grads(model)
        psf = self.model_psf(model)

        image = self.model_interferogram(psf, model, **kwargs)

        # downsample the image to the 3x oversample for the detector/ramp
        image = image.downsample(model.source_oversample)

        illuminance = self.model_illuminance(image, model)
        ramp = self.model_ramp(illuminance, model)
        ramp = self.model_read(ramp, model)

        if return_slopes:
            return ramp.set("data", np.diff(ramp.data, axis=0))
        return ramp

    def model_interferogram(self, psf, model, **kwargs):
        pass


class ResolvedFit(BaseResolvedFit, ModelFit):
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

        source_id: str = (None,)

    def model_interferogram(
        self,
        psf,
        model,
        rotate: bool = None,
        source_id: str = None,
    ):
        return psf.convolve(
            model.get_distribution(self, rotate=rotate, source_id=source_id), method="fft"
        )


class DynamicResolvedFit(ResolvedFit):
    """
    Model fit for resolved sources where each exposure has a different
    intensity distribution.
    """

    def get_key(self, param):
        match param:
            case "log_dist":
                return "_".join([self.key, self.filter])

        return super().get_key(param)


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

    def __call__(self, model, rotate: bool = None):
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


class PointResolvedFit(TransformedResolvedFit):
    """
    Model fit for resolved sources. This adds the log distribution parameter.
    """

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

    def initialise_params(self, optics, coeffs, contrast):
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
        params["contrast"] = self.get_key("contrast"), np.array(contrast)

        return params

    def model_interferogram(
        self,
        psf,
        model,
        rotate: bool = None,
        source_id: str = None,
    ):

        if source_id is None:
            key = self.get_key("contrast")
        else:
            key = "_".join((self.get_key("contrast"), source_id))
        contrast = model.params["contrast"][key]
        psf1 = psf * (1 - contrast)
        psf2 = psf * (contrast)

        # convolve source with PSF
        resolved_component = model.get_distribution(self, rotate=rotate, source_id=source_id)
        return psf1 + psf2.convolve(resolved_component, method="fft").data


class MultiSourceFit(ModelFit):

    exposures: dict
    calibrator: bool = None  # this will mean multi fit
    unique_params: list = None

    def __init__(self, file, exp_dict, unique_params=None):

        super().__init__(file)

        for source_id, exp in exp_dict.items():
            if not isinstance(exp, ModelFit):
                raise ValueError(
                    f"All exposures must be ModelFit instances, got {type(exp)} for source_id {source_id}."
                )

        # Assert all exposures have the same filter
        filenames = [exp.filename for exp in exp_dict.values()]
        assert len(set(filenames)) == 1

        self.exposures = exp_dict
        self.calibrator = None  # this will mean multi fit
        if unique_params is None:
            unique_params = [
                "positions",
                "fluxes",
                "spectra",
                "log_dist",
                "contrast",
            ]
        self.unique_params = unique_params

    def get_key(self, param, source_id=None):

        if source_id is None:
            print("Warning: source_id is None in get_key, taking the first exposure.")
            source_id = list(self.exposures.keys())[0]

        exp = self.exposures[source_id]

        if param in self.unique_params:
            return "_".join([exp.get_key(param), source_id])

        return exp.get_key(param)

    def map_param(self, param, source_id=None):

        if source_id is None:
            print("Warning: source_id is None in map_param, taking the first exposure.")
            source_id = list(self.exposures.keys())[0]

        # Map the appropriate parameter to the correct key
        if param in self.unique_params:
            return f"{param}.{self.get_key(param, source_id)}"

        # Else its global
        return self.exposures[source_id].map_param(param)

    def get_spectra(self, model, source_id):
        wavels, filt_weights = model.filters[self.filter]
        xs = np.linspace(-1, 1, len(wavels), endpoint=True)
        spectra_slopes = 1 + model.get(self.map_param("spectra", source_id)) * xs
        weights = filt_weights * spectra_slopes
        weights = np.where(weights < 0, 0.0, weights)
        return wavels, weights / weights.sum()

    def model_wfs(self, model, source_id):
        pos = dlu.arcsec2rad(model.positions[self.get_key("positions", source_id)])
        wavels, weights = self.get_spectra(model, source_id)

        optics = self.update_optics(model, source_id)
        wfs = eqx.filter_jit(optics.propagate)(wavels, pos, weights, return_wf=True)

        # Convert Cartesian to Angular wf
        if wfs.units == "Cartesian":
            wfs = wfs.multiply("pixel_scale", 1 / optics.focal_length)
            wfs = wfs.set(["plane", "units"], ["Focal", "Angular"])
        return wfs

    def model_psf(self, model, source_id):
        wfs = self.model_wfs(model, source_id)
        return dl.PSF(wfs.psf.sum(0), wfs.pixel_scale.mean(0))

    def model_illuminance(self, psf, model, source_id):
        flux = self.ngroups * 10 ** model.fluxes[self.get_key("fluxes", source_id)]
        psf = eqx.filter_jit(model.detector.apply)(psf)
        return psf.multiply("data", flux)

    def update_optics(self, model, source_id):
        optics = model.optics
        if "aberrations" in model.params.keys():
            coefficients = model.aberrations[self.get_key("aberrations", source_id)]

            # Nuke the piston gradient to prevent degeneracy
            fixed_piston = lax.stop_gradient(coefficients[0, 0])
            coefficients = coefficients.at[0, 0].set(fixed_piston)

            # Stop gradient for science targets
            if not self.calibrator:
                coefficients = lax.stop_gradient(coefficients)
            optics = optics.set("pupil_mask.abb_coeffs", coefficients)

        if hasattr(model, "reflectivity"):
            coefficients = model.reflectivity[self.get_key("reflectivity", source_id)]
            optics = optics.set("pupil_mask.amp_coeffs", coefficients)

        # Set the defocus
        optics = optics.set("defocus", model.defocus[self.get_key("defocus", source_id)])

        return optics

    def simulate(self, model, return_slopes: bool = True, **kwargs):

        # model/propagate the PSF of each source separately!
        illuminances = []
        for source_id, exp in self.exposures.items():
            psf = self.model_psf(model, source_id)
            if isinstance(exp, BaseResolvedFit):
                image = exp.model_interferogram(psf, model, source_id=source_id, **kwargs)
            else:
                image = psf
            image = image.downsample(model.source_oversample)
            illuminance = self.model_illuminance(image, model, source_id)
            illuminances.append(illuminance.data)

        illuminance = dl.PSF(np.array(illuminances).sum(axis=0), image.pixel_scale)

        # Just grab any old exposure to get the detector methods
        exp = list(self.exposures.values())[0]
        ramp = exp.model_ramp(illuminance, model)
        ramp = exp.model_read(ramp, model)

        if return_slopes:
            return ramp.set("data", np.diff(ramp.data, axis=0))
        return ramp

    # def rotate(self, distribution, clip=True, interp_method="linear"):
    #     pass

    def print_summary(self):
        for source_id, exp in self.exposures.items():
            print(f"Source ID: {source_id}")
            exp.print_summary()
            print()

    # def __getattr__(self, name):
    #     # called only if attribute not found normally
    #     print(f"Delegating {name} to inner B")
    #     return getattr(list(self.exposures.values())[0], name)

    def __call__(self, model, return_slopes=True):
        return self.simulate(model, return_slopes=return_slopes).data
