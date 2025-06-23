from jax import numpy as np


def truncate_files(files, ngroups):
    """
    Truncate the ramp of files to only have ngroups. This is use for saturated or highly nonlinear data where you want to discard the top of the ramp.

    Args:
        files (list): List of calslope files to truncate.
        ngroups (int): Number of groups to keep in the ramp.
    """

    for file in files:

        # grabbing the total number of groups in the ramp
        top_group = file["RAMP"].data.shape[0]
        up_to = top_group - ngroups

        # files are mutable, so they will change in place
        for attr in ["RAMP", "SLOPE", "RAMP_SUP", "SLOPE_SUP"]:
            file[attr].data = file[attr].data[:-up_to, ...]
        for attr in ["RAMP_COV", "SLOPE_COV"]:
            file[attr].data = file[attr].data[:-up_to, :-up_to, ...]


def fwhm_to_sigma(fwhm):
    """
    Convert FWHM to sigma for a Gaussian distribution.

    Args:
        fwhm (float): Full Width at Half Maximum.

    Returns:
        float: Corresponding standard deviation (sigma).
    """
    return fwhm / (2 * np.sqrt(2 * np.log(2)))


def calc_parang(file):
    """
    Calculate the parallactic angle for a given file.
    """
    return np.array(file["PRIMARY"].header["ROLL_REF"])
