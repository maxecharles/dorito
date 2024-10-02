from . import misc
from . import models
from . import model_fits

import jax
from jax import numpy as np, random as jr, Array
import amigo
from zodiax.experimental import deserialise
import equinox as eqx
import dLux as dl
import dLux.utils as dlu
from astropy import units as u


def build_resolved_model(
    cal_files,
    sci_files,
    final_state,
    width: int = None,
    depth: int = None,
    source_size: int = 100,  # in oversampled pixels
    log_dist_prior: Array = None,
    spectral_coeffs_prior=np.array([1.0, 0.0]),
    separate_exposures: bool = False,
    ramp_model=None,
    cal_fit=None,
    sci_fit=None,
    optics=None,
):
    """
    Constructing the model.
    """

    if log_dist_prior is not None:
        if log_dist_prior.shape[0] is not source_size:
            raise ValueError("log_dist_prior must have the same shape as source_size")
    else:
        log_dist_prior = np.log10(np.ones((source_size, source_size)) / source_size**2)

    if ramp_model is None:
        if width is None or depth is None:
            raise ValueError("If ramp_model is not provided, width and depth must be specified")

    if ramp_model is not None and (width is not None or depth is not None):
        raise ValueError("If ramp_model is provided, width and depth must not be specified")

    # Ramp
    if ramp_model is None:
        layers, pooling_layer = amigo.ramp_models.build_pooled_layers(
            width=width,
            depth=depth,
            pooling="avg",
        )
        ramp_model = amigo.ramp_models.MinimalConv(layers, pooling_layer)

    if cal_fit is None:
        cal_fit = amigo.model_fits.PointFit()

    if sci_fit is None:
        sci_fit = model_fits.DynamicResolvedFit()

    if optics is None:
        optics = amigo.core_models.AMIOptics()

    # Setting up calibrator model
    cal_fit = amigo.model_fits.PointFit()
    cal_exposures = [amigo.core_models.Exposure(file, cal_fit) for file in cal_files]
    cal_params = amigo.files.initialise_params(cal_exposures, optics)
    cal_params["Teffs"] = amigo.search_Teffs.get_Teffs(cal_files)

    # Setting up science model
    sci_exposures = [amigo.core_models.Exposure(file, sci_fit) for file in sci_files]
    sci_params = amigo.files.initialise_params(sci_exposures, optics)

    # Setting priors for log distribution and spectral coefficients
    log_distributions = {}
    spectral_coeffs = {}
    for exp in sci_exposures:
        dist_key = exp.get_key("log_distribution")
        log_distributions[dist_key] = log_dist_prior

        spec_key = exp.get_key("spectral_coeffs")
        spectral_coeffs[spec_key] = spectral_coeffs_prior

    sci_params["log_distribution"] = log_distributions
    sci_params["spectral_coeffs"] = spectral_coeffs

    # combining calibrator and science
    params = misc.combine_param_dicts(cal_params, sci_params)

    # setting up filters
    filters = {}
    for filt in list(set([exp.filter for exp in [*sci_exposures, *cal_exposures]])):
        filters[filt] = amigo.misc.calc_throughput(filt, nwavels=9)

    model = models.ResolvedAmigoModel(
        params,
        optics=optics,
        ramp=ramp_model,
        detector=amigo.core_models.LinearDetectorModel(),
        read=amigo.read_models.ReadModel(),
        filters=filters,
    )
    model = amigo.misc.populate_from_state(model, final_state)

    if separate_exposures:
        return model, params, cal_exposures, sci_exposures
    else:
        exposures = [*cal_exposures, *sci_exposures]
        return model, params, exposures


def gaussian_prior(source_size=100, scale=5):
    """
    A gaussian array to initialise the source distribution.
    NOTE: this is not logged.
    """
    mean = np.array([0, 0])
    cov = np.array([[1, 0], [0, 1]])  # covariance matrix of the distribution
    x = scale * np.linspace(-5, 5, source_size)
    y = scale * np.linspace(-5, 5, source_size)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    gaussian = jax.scipy.stats.multivariate_normal.pdf(pos, mean=mean, cov=cov)
    return gaussian / gaussian.sum()


def multi_gaussian_prior(source_size=100, scale1=20.0, scale2=0.5, contrast=0.1):
    """
    A gaussian array to initialise the source distribution.
    NOTE: this is not logged.
    """
    mean = np.array([0, 0])
    cov = np.array([[1, 0], [0, 1]])  # covariance matrix of the distribution

    # first gaussian
    x1 = scale1 * np.linspace(-1, 1, source_size)
    X1, Y1 = np.meshgrid(x1, x1)
    pos1 = np.dstack((X1, Y1))
    gaussian1 = jax.scipy.stats.multivariate_normal.pdf(pos1, mean=mean, cov=cov)

    # second gaussian
    x2 = scale2 * np.linspace(-1, 1, source_size)
    X2, Y2 = np.meshgrid(x2, x2)
    pos2 = np.dstack((X2, Y2))
    gaussian2 = jax.scipy.stats.multivariate_normal.pdf(pos2, mean=mean, cov=cov)

    # combining
    gaussian = gaussian1 + contrast * gaussian2

    # normalising
    return gaussian / gaussian.sum()


def softcirc_prior(source_size=100, radius=0.65, clip_dist=0.2):
    """
    A soft circle array to initialise the source distribution.
    NOTE: this is not logged.
    """
    circ = dlu.soft_circle(dlu.pixel_coords(source_size, 2), radius, clip_dist)
    circ = np.maximum(circ, 1e-6)
    return circ / circ.sum()


def TD_prior(source_size, star_flux):
    """
    Prior log distribution for a transition disk.
    """

    shape = (source_size, source_size)
    star_ind = int(source_size // 2)

    # components
    star = np.zeros(shape).at[star_ind, star_ind].set(star_flux)
    field = (
        np.ones(shape).at[star_ind, star_ind].set(0.0) * (1 - star_flux) / (source_size**2 - 1)
    )

    # summing components
    distribution = star + field

    return np.log10(distribution / distribution.sum())


# def initialise_disk(
#     pixel_scale=0.065524085, oversample=4, normalise=True, npix=100, return_psf=True
# ):
#     io_initial_distance = 4.36097781166671 * u.AU
#     io_final_distance = 4.36088492436330 * u.AU
#     io_diameter = 3643.2 * u.km  # from wikipedia

#     io_mean_distance = (io_initial_distance + io_final_distance).to(u.km) / 2
#     angular_size = dlu.rad2arcsec(
#         io_diameter / io_mean_distance
#     )  # angular size in arcseconds

#     if npix is None:
#         npix = oversample * np.ceil(angular_size / pixel_scale).astype(int)
#     coords = dlu.pixel_coords(npixels=npix, diameter=npix * pixel_scale / oversample)

#     circle = dlu.soft_circle(coords, radius=angular_size / 2, clip_dist=2e-3)

#     if not normalise:
#         return circle

#     if return_psf:
#         return dl.PSF(circle / circle.sum(), pixel_scale / oversample)
#     else:
#         return circle / circle.sum()


# def generate_volc_coords(offset=None, radius=1e-1):
#     if offset is None:
#         disk = initialise_disk()
#         offset = disk.pixel_scale / 4 / 2

#     thetas = np.linspace(0, 2 * np.pi, 8, endpoint=False)
#     polars = dlu.polar2cart((np.ones_like(thetas), thetas))
#     coords = np.vstack((np.array([0, 0]), polars.T))

#     volc_coords = radius * coords + np.array([0, offset])

#     return volc_coords


# def generate_dotted_disk(
#     weight=4e-2,
#     offset=None,
#     radius=1e-1,
#     eps=1e-16,
# ):
#     disk = initialise_disk()
#     one_pix = disk.pixel_scale / 4
#     if offset is None:
#         offset = disk.pixel_scale / 4 / 2

#     base_coords = dlu.translate_coords(
#         dlu.pixel_coords(
#             npixels=disk.data.shape[0],
#             diameter=disk.data.shape[0] * one_pix,
#         ),
#         np.array(2 * [one_pix]),
#     )

#     distribution = disk.data
#     npix = 3
#     volc_coords = generate_volc_coords(offset=offset, radius=radius)

#     brightnesses = weight * np.linspace(1, 1e-2, npix**2)

#     coords = [dlu.translate_coords(base_coords, np.array(vc)) for vc in volc_coords]
#     coords = jr.permutation(jr.PRNGKey(0), np.array(coords))

#     volcanoes = np.array([dlu.square(coord, width=one_pix) for coord in coords])
#     distribution += np.dot(volcanoes.T, brightnesses)

#     return distribution + eps


# def generate_bloated_disk(
#     weight=2e-2,
#     offset=0,
#     radius=1e-1,
#     eps=1e-16,
# ):
#     disk = initialise_disk()
#     base_coords = dlu.pixel_coords(
#         npixels=disk.data.shape[0],
#         diameter=disk.data.shape[0] * disk.pixel_scale / 4,
#     )
#     distribution = disk.data
#     npix = 3
#     volc_coords = generate_volc_coords(offset=offset, radius=radius)

#     radii = 2e-2 * np.linspace(1e-5, 1, npix**2)
#     brightness = weight

#     coords = [dlu.translate_coords(base_coords, np.array(vc)) for vc in volc_coords]
#     coords = jr.permutation(jr.PRNGKey(0), np.array(coords))

#     volcanoes = []
#     for coord, r in zip(coords, radii):
#         volcano = dlu.soft_circle(coord, radius=r, clip_dist=3e-3)
#         volcano *= brightness / volcano.sum()
#         volcanoes.append(volcano)

#     distribution += np.array(volcanoes).sum(0)

#     return distribution + eps


# def generate_ringed_disk(
#     weight=4e-1,
#     offset=0,
#     radius=1e-1,
#     eps=1e-16,
# ):
#     disk = initialise_disk()
#     base_coords = dlu.pixel_coords(
#         npixels=disk.data.shape[0],
#         diameter=disk.data.shape[0] * disk.pixel_scale / 4,
#     )
#     distribution = disk.data
#     volc_coords = generate_volc_coords(offset=offset, radius=radius)
#     npix = 3
#     radii = 3e-2 * np.linspace(1e-5, 1, npix**2)
#     brightness = weight

#     coords = [dlu.translate_coords(base_coords, np.array(vc)) for vc in volc_coords]
#     coords = jr.permutation(jr.PRNGKey(0), np.array(coords))

#     volcanoes = []
#     for coord, r in zip(coords, radii):
#         volcano_outer = dlu.soft_circle(coord, radius=r, clip_dist=3e-3)
#         volcano_inner = dlu.soft_circle(coord, radius=0.9 * r, clip_dist=1e-3)
#         volcano = np.clip(volcano_outer - volcano_inner, 0)
#         volcano *= brightness / volcano.sum()
#         volcanoes.append(volcano)

#     volcanoes = np.maximum(distribution, np.array(volcanoes))
#     distribution += np.array(volcanoes).sum(0)

#     return distribution + eps


# def build_simulated_models(files: list, model_cache: str, n_ints: int = 1):
#     # simulated source distributions
#     A = generate_dotted_disk()
#     B = generate_bloated_disk()
#     C = generate_ringed_disk()
#     source_distributions = [A, B, C]

#     # building model
#     sim_files = files[:3]  # only want 3 files
#     temp_model, temp_exposures = build_io_model(
#         sim_files, model_cache, initial_distributions=source_distributions
#     )

#     # simulating uncertainties
#     for idx, exp in enumerate(temp_exposures):
#         # simulating slope data
#         clean_slope = temp_model.model(exp)

#         # defining variance from photon and read noise processes
#         photon_var = clean_slope / n_ints  # bc poisson
#         read_noise_var = 100.0 / n_ints
#         var = photon_var + read_noise_var  # variances add, how good

#         # drawing from a normal distribution to get the data
#         std = np.sqrt(var)
#         data = jr.normal(jr.PRNGKey(0), shape=var.shape) * std + clean_slope

#         # setting
#         sim_files[idx]["SCI"].data = data
#         sim_files[idx]["SCI_VAR"].data = var

#     # creating initial model with initial distribution of ones
#     initial_model, _ = build_io_model(
#         sim_files, model_cache, initial_distributions=None
#     )

#     # true model with the simulated distributions
#     true_model, exposures = build_io_model(
#         sim_files, model_cache, initial_distributions=source_distributions
#     )

#     return initial_model, true_model, exposures
