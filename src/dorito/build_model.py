import webbpsf
from jax import numpy as np, random as jr
import amigo
from amgio import models, model_fits
from zodiax.experimental import deserialise
import equinox as eqx
import dLux as dl
import dLux.utils as dlu
from astropy import units as u


def build_io_model(
    files: list, model_cache: str, initial_distributions: list | None = None
):
    ramp_weights = np.load(model_cache + "Max_w8s.npy")

    # Get webbpsf optical system for OPD
    inst = webbpsf.NIRISS()
    inst.load_wss_opd_by_date(files[0][0].header["DATE-BEG"], verbose=False)
    opd = np.array(inst.get_optical_system().planes[0].opd)

    ### OPTICS
    # Get dLux optical system
    optics = amigo.core_models.AMIOptics(
        opd=opd,
        normalise=True,
    )

    ### DETECTOR
    conv_layers, pooled_layer = amigo.ramp_models.build_pooled_layers(width=32, depth=9)
    ramp_model = amigo.ramp_models.MinimalConv(conv_layers, pooled_layer)
    ramp_model = eqx.tree_at(lambda ramp: ramp.conv.values, ramp_model, ramp_weights)

    # Setting up science model
    fit = model_fits.ResolvedFit()
    exposures = [amigo.core_models.Exposure(file, fit) for file in files]
    params = amigo.files.initialise_params(exposures, optics)

    source_size = 100  # in oversampled pixels

    # Setting initial parameters
    if initial_distributions is None:
        initial_distributions = [np.ones((source_size, source_size))] * len(exposures)

    if len(initial_distributions) != len(exposures):
        raise ValueError(
            "Initial distributions must be the same length as exposures"
        )

    log_distributions = {}
    spectral_coeffs = {}
    for dist, exp in zip(initial_distributions, exposures):

        # log distribution
        dist_key = exp.get_key("log_distribution")
        log_distributions[dist_key] = np.log10(dist)

        # spectral coefficients
        spec_key = exp.get_key("spectral_coeffs")
        spectral_coeffs[spec_key] = np.array([1.0, 0.0])

    params["log_distribution"] = log_distributions
    params["spectral_coeffs"] = spectral_coeffs
    print(spectral_coeffs)


    filters = {}
    for filt in list(set([exp.filter for exp in exposures])):
        filters[filt] = amigo.misc.calc_throughput(filt, nwavels=9)

    print(filters)

    initial_model = models.ResolvedAmigoModel(
        params,
        optics=optics,
        ramp=ramp_model,
        detector=amigo.core_models.LinearDetectorModel(),
        read=amigo.read_models.ReadModel(),
        filters=filters,
        )

    ab_dict = deserialise(model_cache + "aberrations.zdx")

    # Setting io aberrations to those recovered from calibrator
    model = initial_model.set("aberrations", ab_dict)

    return model, exposures


def initialise_disk(
    pixel_scale=0.065524085, oversample=4, normalise=True, npix=100, return_psf=True
):
    io_initial_distance = 4.36097781166671 * u.AU
    io_final_distance = 4.36088492436330 * u.AU
    io_diameter = 3643.2 * u.km  # from wikipedia

    io_mean_distance = (io_initial_distance + io_final_distance).to(u.km) / 2
    angular_size = dlu.rad2arcsec(
        io_diameter / io_mean_distance
    )  # angular size in arcseconds

    if npix is None:
        npix = oversample * np.ceil(angular_size / pixel_scale).astype(int)
    coords = dlu.pixel_coords(npixels=npix, diameter=npix * pixel_scale / oversample)

    circle = dlu.soft_circle(coords, radius=angular_size / 2, clip_dist=2e-3)

    if not normalise:
        return circle

    if return_psf:
        return dl.PSF(circle / circle.sum(), pixel_scale / oversample)
    else:
        return circle / circle.sum()


def generate_volc_coords(offset=None, radius=1e-1):
    if offset is None:
        disk = initialise_disk()
        offset = disk.pixel_scale / 4 / 2

    thetas = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    polars = dlu.polar2cart((np.ones_like(thetas), thetas))
    coords = np.vstack((np.array([0, 0]), polars.T))

    volc_coords = radius * coords + np.array([0, offset])

    return volc_coords


def generate_dotted_disk(
    weight=4e-2,
    offset=None,
    radius=1e-1,
    eps=1e-16,
):
    disk = initialise_disk()
    one_pix = disk.pixel_scale / 4
    if offset is None:
        offset = disk.pixel_scale / 4 / 2

    base_coords = dlu.translate_coords(
        dlu.pixel_coords(
            npixels=disk.data.shape[0],
            diameter=disk.data.shape[0] * one_pix,
        ),
        np.array(2 * [one_pix]),
    )

    distribution = disk.data
    npix = 3
    volc_coords = generate_volc_coords(offset=offset, radius=radius)

    brightnesses = weight * np.linspace(1, 1e-2, npix**2)

    coords = [dlu.translate_coords(base_coords, np.array(vc)) for vc in volc_coords]
    coords = jr.permutation(jr.PRNGKey(0), np.array(coords))

    volcanoes = np.array([dlu.square(coord, width=one_pix) for coord in coords])
    distribution += np.dot(volcanoes.T, brightnesses)

    return distribution + eps


def generate_bloated_disk(
    weight=2e-2,
    offset=0,
    radius=1e-1,
    eps=1e-16,
):
    disk = initialise_disk()
    base_coords = dlu.pixel_coords(
        npixels=disk.data.shape[0],
        diameter=disk.data.shape[0] * disk.pixel_scale / 4,
    )
    distribution = disk.data
    npix = 3
    volc_coords = generate_volc_coords(offset=offset, radius=radius)

    radii = 2e-2 * np.linspace(1e-5, 1, npix**2)
    brightness = weight

    coords = [dlu.translate_coords(base_coords, np.array(vc)) for vc in volc_coords]
    coords = jr.permutation(jr.PRNGKey(0), np.array(coords))

    volcanoes = []
    for coord, r in zip(coords, radii):
        volcano = dlu.soft_circle(coord, radius=r, clip_dist=3e-3)
        volcano *= brightness / volcano.sum()
        volcanoes.append(volcano)

    distribution += np.array(volcanoes).sum(0)

    return distribution + eps


def generate_ringed_disk(
    weight=4e-1,
    offset=0,
    radius=1e-1,
    eps=1e-16,
):
    disk = initialise_disk()
    base_coords = dlu.pixel_coords(
        npixels=disk.data.shape[0],
        diameter=disk.data.shape[0] * disk.pixel_scale / 4,
    )
    distribution = disk.data
    volc_coords = generate_volc_coords(offset=offset, radius=radius)
    npix = 3
    radii = 3e-2 * np.linspace(1e-5, 1, npix**2)
    brightness = weight

    coords = [dlu.translate_coords(base_coords, np.array(vc)) for vc in volc_coords]
    coords = jr.permutation(jr.PRNGKey(0), np.array(coords))

    volcanoes = []
    for coord, r in zip(coords, radii):
        volcano_outer = dlu.soft_circle(coord, radius=r, clip_dist=3e-3)
        volcano_inner = dlu.soft_circle(coord, radius=0.9 * r, clip_dist=1e-3)
        volcano = np.clip(volcano_outer - volcano_inner, 0)
        volcano *= brightness / volcano.sum()
        volcanoes.append(volcano)

    volcanoes = np.maximum(distribution, np.array(volcanoes))
    distribution += np.array(volcanoes).sum(0)

    return distribution + eps


def build_simulated_models(files: list, model_cache: str, n_ints: int = 1):
    # simulated source distributions
    A = generate_dotted_disk()
    B = generate_bloated_disk()
    C = generate_ringed_disk()
    source_distributions = [A, B, C]

    # building model
    sim_files = files[:3]  # only want 3 files
    temp_model, temp_exposures = build_io_model(
        sim_files, model_cache, initial_distributions=source_distributions
    )

    # simulating uncertainties
    for idx, exp in enumerate(temp_exposures):
        # simulating slope data
        clean_slope = temp_model.model(exp)

        # defining variance from photon and read noise processes
        photon_var = clean_slope / n_ints  # bc poisson
        read_noise_var = 100.0 / n_ints
        var = photon_var + read_noise_var  # variances add, how good

        # drawing from a normal distribution to get the data
        std = np.sqrt(var)
        data = jr.normal(jr.PRNGKey(0), shape=var.shape) * std + clean_slope

        # setting
        sim_files[idx]["SCI"].data = data
        sim_files[idx]["SCI_VAR"].data = var

    # creating initial model with initial distribution of ones
    initial_model, _ = build_io_model(
        sim_files, model_cache, initial_distributions=None
    )

    # true model with the simulated distributions
    true_model, exposures = build_io_model(
        sim_files, model_cache, initial_distributions=source_distributions
    )

    return initial_model, true_model, exposures
