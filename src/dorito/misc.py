from jax import numpy as np, scipy as jsp, random as jr
from dLux import utils as dlu


def truncate_files(files, ngroups):
    """
    Truncate the ramp of files to only have ngroups.
    """

    for file in files:

        top_group = file["RAMP"].data.shape[0]
        up_to = top_group - ngroups

        # files are mutable, so they will change in place
        for attr in ["RAMP", "SLOPE", "RAMP_SUP", "SLOPE_SUP"]:
            file[attr].data = file[attr].data[:-up_to, ...]

        for attr in ["RAMP_COV", "SLOPE_COV"]:
            file[attr].data = file[attr].data[:-up_to, :-up_to, ...]


def combine_param_dicts(cal_params, sci_params):
    """
    Combining the calibration and science parameter dictionaries.
    """
    params = {**cal_params}

    # to avoid doubling up
    for key in sci_params:
        if key in cal_params:
            params[key] = cal_params[key] | sci_params[key]
        else:
            params[key] = sci_params[key]

    return params


def fwhm_to_sigma(fwhm):
    """
    Convert FWHM to sigma for a Gaussian distribution.

    Args:
        fwhm (float): Full Width at Half Maximum.

    Returns:
        float: Corresponding standard deviation (sigma).
    """
    return fwhm / (2 * np.sqrt(2 * np.log(2)))


def difflim_to_cov(model, exposure, factor):
    # getting effective wavelength
    wavels, weights = model.filters[exposure.filter]
    lambd = np.dot(wavels, weights)

    # calculating lambda over D
    lambda_on_D = dlu.rad2arcsec(lambd / model.optics.diameter)

    # calculating sigma, with some factor in front of lambda on D
    sigma = fwhm_to_sigma(factor * lambda_on_D)

    # return covariance matrix
    return [[sigma**2, 0], [0, sigma**2]]


def get_pscale(model):
    return model.optics.psf_pixel_scale / model.optics.oversample / model.source_oversample


def blur_distribution(array, model, exposure, extent=0.25, factor=1.0):
    cov = difflim_to_cov(model, exposure, factor)

    x = np.arange(-extent, extent, get_pscale(model))
    X, Y = np.meshgrid(x, x)
    pos = np.dstack((X, Y))

    kernel = jsp.stats.multivariate_normal.pdf(jr.PRNGKey(0), pos, np.array(cov))

    distribution = jsp.signal.convolve2d(array, kernel, mode="same", method="fft")
    return distribution / distribution.sum()
