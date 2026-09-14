"""Fitting helpers and OIModelFit subclasses for OI resolved sources.

This module provides OIModelFit classes for fitting resolved sources using
amigo, including utilities to handle the log distribution parameter,
rotation by parallactic angle, and simulation of resolved source interferograms.
This model uses classes from model_fits. 
"""

from amigo.model_fits import ModelFit
from amigo.vis_models import vis_to_im
from amigo.vis_analysis import AmigoOIData
from amigo.misc import interp
from jax import numpy as np
import dLux.utils as dlu
from .model_fits import _BaseResolvedFit

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