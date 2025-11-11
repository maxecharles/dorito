"""Small miscellaneous helpers used in examples and notebooks.

The utilities here are lightweight helpers that operate on the simple file
structures used across the dorito examples (for example calslope-like objects)
and a couple of geometry helpers used when building model inputs.
"""

from jax import numpy as np

__all__ = [
    "truncate_files",
    "fwhm_to_sigma",
    "calc_parang",
]


def truncate_files(files, ngroups):
    """Truncate the ramp of files to only have ngroups. This is use for
    saturated or highly nonlinear data where you want to discard the top of
    the ramp.

    This modifies the list of file-like objects in-place by trimming the top
    groups from the arrays stored under keys like ``RAMP`` / ``SLOPE`` and
    their covariance counterparts.

    Parameters
    ----------
    files : list
        Sequence of file-like objects where attributes such as ``RAMP`` and
        ``SLOPE`` provide ``.data`` arrays whose first axis indexes groups.
    ngroups : int
        Number of groups to preserve at the start of the ramp.
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

    Parameters
    ----------
    fwhm : float
        Full width at half maximum.

    Returns:
        float: Corresponding standard deviation (sigma).
    """
    return fwhm / (2 * np.sqrt(2 * np.log(2)))


def calc_parang(file):
    """Return the parallactic/roll angle extracted from a file header.

    Parameters
    ----------
    file : mapping-like
        File-like object with a ``PRIMARY`` header containing ``ROLL_REF``.

    Returns
    -------
    ndarray
        Numeric parallactic/roll angle.
    """
    return np.array(file["PRIMARY"].header["ROLL_REF"])
