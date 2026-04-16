"""Fitting helpers and ModelFit subclasses for resolved sources.

This module provides ModelFit classes for fitting resolved sources using
amigo, including utilities to handle the log distribution parameter,
rotation by parallactic angle, and simulation of resolved source interferograms.
"""

from amigo.model_fits import ModelFit
from amigo.vis_models import vis_to_im
from amigo.vis_analysis import AmigoOIData
from amigo.misc import interp, find_position
from jax import numpy as np, lax
import dLux as dl
import dLux.utils as dlu
import equinox as eqx
from .models import ResolvedAmigoModel


__all__ = [
    # "BaseResolvedFit",
    "ResolvedFit",
    "DynamicResolvedFit",
    "TransformedResolvedFit",
    "PointResolvedFit",
    # "OIFit",
    "ResolvedOIFit",
]


class _BaseResolvedFit:

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


class ResolvedFit(_BaseResolvedFit, ModelFit):
    """Model fit for resolved (extended) sources.

    This class extends :class:`amigo.model_fits.ModelFit` to add support for a
    spatial distribution parameter (kept as its base-10 logarithm, stored
    under the key ``log_dist``). It supplies sensible default initialisation
    and maps the ``log_dist`` parameter into the expected keyed parameter
    namespace for per-filter fitting.

    Parameters
    ----------
    file : str or path-like
        Path to the data file or exposure to be passed to :class:`ModelFit`.
    use_cov : bool, optional
        Whether to use the covariance information from the data, by default
        ``True``.
    """

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

        params = super().initialise_params(optics)

        # log distribution
        params["log_dist"] = (
            self.get_key("log_dist"),
            np.log10(distribution / distribution.sum()),
        )

        return params

    def model_interferogram(
        self,
        psf,
        model,
        rotate: bool = None,
    ):
        return psf.convolve(model.get_distribution(self, rotate=rotate), method="fft")


class DynamicResolvedFit(ResolvedFit):
    """Resolved fit where each exposure has its own distribution.

    For time-series or sequence data where every exposure can have an
    independent resolved-source distribution, this class modifies the
    parameter keying so that distribution parameters are unique per
    exposure (the key includes the exposure ``self.key`` and the filter).

    Notes
    -----
    Only the keying behaviour differs from :class:`ResolvedFit` — the
    underlying parameter representation and simulation pipeline remain the
    same.
    """

    def get_key(self, param):
        match param:
            case "log_dist":
                return "_".join([self.key, self.filter])

        return super().get_key(param)


class TransformedResolvedFit(ResolvedFit):
    """Resolved-source fit using coefficients describing a transformed basis.

    This variant initialises the ``log_dist`` parameter from a set of
    coefficients (for example a set of basis coefficients or a compressed
    representation) rather than from an explicit full image distribution.

    """

    def initialise_params(self, optics, coeffs):

        params = ModelFit.initialise_params(self, optics)

        # log distribution
        params["log_dist"] = (self.get_key("log_dist"), coeffs)

        return params


class PointResolvedFit(TransformedResolvedFit):
    """Resolved fit combining an unresolved point-like component and an extended component.

    This fit represents the source as a superposition of a point source
    component and a resolved (extended) component. This is useful for
    modelling systems like young stars with extended protoplanetary disks.

    Notes
    -----
    Parameters for building the transformed/resolved component are described
    on :meth:`initialise_params` (for example ``optics``, ``coeffs`` and
    ``contrast`` are the arguments used when initialising parameters).
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
    ):

        contrast = model.params["contrast"][self.get_key("contrast")]
        psf1 = psf * (1 - contrast)
        psf2 = psf * (contrast)

        # convolve source with PSF
        resolved_component = model.get_distribution(self, rotate=rotate)
        return psf1 + psf2.convolve(resolved_component, method="fft").data


class _OIFit(AmigoOIData):
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


class ResolvedOIFit(_OIFit, _BaseResolvedFit):
    """OI-data backed resolved-source fit utilities.

    This class mixes the OI-data wrapper behaviour from :class:`_OIFit` with
    the resolved-source helpers in :class:`_BaseResolvedFit`. It provides
    methods to convert distributions into OTFs/visibilities, produce model
    DISCO outputs, and compute dirty images usable for visualisation and
    normalisation.

    Methods
    -------
    initialise_params(model, distribution)
        Prepare ``log_dist`` and ``base_uv`` parameters for an OI fit.
    to_otf(model, distribution)
        Return a dLux MFT representing the distribution in OTF space.
    to_cvis(model, distribution)
        Convert an image distribution into flattened complex visibilities
        suitable for DISCO-style modelling.
    dirty_image(...)
        Compute a dirty image from the underlying observed OI visibilities.
    __call__(model, rotate=None)
        Produce the amplitudes/phases used by DISCO from the model
        distribution.
    """

    def initialise_params(self, model, distribution):

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
        """Convert an image distribution into complex visibilities for DISCO.

        The pipeline performed here is:
        1. Transform the image distribution to the OTF plane via
           :meth:`to_otf` (a dLux MFT).
        2. Normalise the complex u,v plane by the stored ``base_uv`` for this
           fit (see :meth:`initialise_params`).
        3. Downsample the u,v plane to the DISCO sampling using
           :func:`dlu.downsample`.
        4. Flatten the 2D u,v array and return the first half of the vector —
           for a real-valued image the Fourier transform is Hermitian symmetric
           and only half the plane is needed.

        Parameters
        ----------
        model : object
            Model object providing UV/OTF parameters and access to
            ``model.params['base_uv']`` for normalisation.
        distribution : array-like
            2D image array (npixels x npixels) describing the resolved source
            brightness distribution.

        Returns
        -------
        numpy.ndarray
            1-D complex array containing the flattened (half) complex
            visibilities suitable for DISCO-style modelling.

        Notes
        -----
        The returned vector contains only the first half of the flattened
        u,v array because of u/v symmetry; callers expecting a full u,v
        representation should reconstruct it using Hermitian symmetry.
        """

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

    def initialise_params(
        self, optics, source_id=None, vis_model=None, one_on_fs_order=1, normalise_flux=True
    ):
        params = {}

        im = np.where(self.badpix, np.nan, self.slopes[0])
        psf = np.where(np.isnan(im), 0.0, im)

        # Position
        pos = find_position(psf, optics.psf_pixel_scale)
        pos += np.array([-optics.psf_pixel_scale / 4, 0])  # apply small shift, seems to help

        # Log flux (1.6 is the ~ gain)
        if normalise_flux:
            log_flux = np.log10((80**2) * 1.61 * np.nanmean(im) / self.n_subexps)
        else:
            log_flux = np.log10((80**2) * 1.61 * np.nanmean(im))

        # Initialise flat WF
        abb = np.zeros_like(optics.pupil_mask.abb_coeffs)

        # positions
        params["positions"] = (self.get_key("positions", source_id), pos)
        params["fluxes"] = (self.get_key("fluxes", source_id), log_flux)
        params["aberrations"] = (self.get_key("aberrations", source_id), abb)
        params["spectra"] = (self.get_key("spectra", source_id), np.array(0.0))
        params["defocus"] = (self.get_key("defocus", source_id), np.array(0.01))

        # Reflectivity
        if self.fit_reflectivity:
            params["reflectivity"] = (
                self.get_key("reflectivity", source_id),
                np.zeros_like(optics.pupil_mask.amp_coeffs),
            )

        # One on fs
        if self.fit_one_on_fs:
            params["one_on_fs"] = (
                self.get_key("one_on_fs", source_id),
                np.zeros((self.ngroups, 80, one_on_fs_order + 1)),
            )

        # Biases
        if self.fit_bias:
            params["biases"] = (self.get_key("biases", source_id), np.zeros((80, 80)))

        return params

    @property
    def n_subexps(self):
        return len(self.exposures)

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
            if isinstance(exp, _BaseResolvedFit):
                image = exp.model_interferogram(psf, model, source_id=source_id, **kwargs)
            else:
                image = psf
            if isinstance(model, ResolvedAmigoModel):
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

    def rotate(self, distribution, clip=True, interp_method="linear", source_id=None):
        if source_id is None:
            print("Warning: source_id is None in rotate, taking the first exposure.")
            source_id = list(self.exposures.keys())[0]

        exposure = self.exposures[source_id]

        return exposure.rotate(distribution, clip, interp_method)

    # def __getattr__(self, name):
    #     # called only if attribute not found normally
    #     print(f"Delegating {name} to inner B")
    #     return getattr(list(self.exposures.values())[0], name)

    def __call__(self, model, return_slopes=True):
        return self.simulate(model, return_slopes=return_slopes).data
