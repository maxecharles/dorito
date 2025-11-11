"""Statistical loss functions and regularisers used in model fitting.

This module collects likelihood wrappers and a set of regulariser loss
functions (TV, TSV, L1, L2, Maximum Entropy) used across the fitting
utilities. Regularisation functions of the for "REG_loss" accept raw
arrays, and those of the form "REG" accept the common
``model, exposure`` pair used by `amigo`.
"""

from jax import numpy as np

__all__ = [
    "TV",
    "TSV",
    "ME",
    "L1",
    "L2",
    "ramp_regularised_loss_fn",
    "apply_regularisers",
    # "ramp_posterior_balance",
    "ramp_posterior_balances",
    "disco_regularised_loss_fn",
    "oi_log_likelihood",
    # "L1_loss",
    # "L2_loss",
    # "tikhinov",
    # "TV_loss",
    # "TSV_loss",
    # "ME_loss",
]


# Regularisers
def L1_loss(arr):
    """L1 norm loss for array-like inputs."""
    return np.nansum(np.abs(arr))


def L2_loss(arr):
    """L2 (quadratic) loss for array-like inputs."""
    return np.nansum(arr**2)


def tikhinov(arr):
    """Finite-difference approximation used by several regularisers."""
    pad_arr = np.pad(arr, 2)  # padding
    dx = np.diff(pad_arr[0:-1, :], axis=1)
    dy = np.diff(pad_arr[:, 0:-1], axis=0)
    return dx**2 + dy**2


def TV_loss(arr, eps=1e-16):
    """Total variation (approx.) loss computed from finite differences."""
    return np.sqrt(tikhinov(arr) + eps**2).sum()


def TSV_loss(arr):
    """Total squared variation (quadratic) loss."""
    return tikhinov(arr).sum()


def ME_loss(arr, eps=1e-16):
    """Maximum-entropy inspired loss (negative entropy of distribution)."""
    P = arr / np.nansum(arr)
    S = np.nansum(-P * np.log(P + eps))
    return -S


# Wrapper functions for the regularisers to accept model and exposure
def TV(model, exposure):
    return TV_loss(model.get_distribution(exposure))


def TSV(model, exposure):
    return TSV_loss(model.get_distribution(exposure))


def ME(model, exposure):
    return ME_loss(model.get_distribution(exposure))


def L1(model, exposure):
    return L1_loss(model.get_distribution(exposure))


def L2(model, exposure):
    return L2_loss(model.get_distribution(exposure))


def ramp_regularised_loss_fn(model, exp, args={"reg_dict": {}}):
    """Compute a regularised negative-log-likelihood for ramp data.

    Returns the scalar loss for a single exposure and a placeholder tuple
    (kept for amigo API compatibility).
    """

    # regular likelihood term
    likelihood = -np.nanmean(exp.mv_zscore(model))
    prior = apply_regularisers(model, exp, args) if not exp.calibrator else 0.0

    return likelihood + prior, ()


def apply_regularisers(model, exposure, args):
    """Apply registered regularisers stored in ``args['reg_dict']``.

    The expected format of ``args['reg_dict']`` is a mapping to pairs
    ``(coeff, fun)`` where ``fun(model, exposure)`` returns a scalar regulariser
    value.
    """

    if "reg_dict" not in args.keys():
        return 0.0

    # evaluating the regularisation term with each for each regulariser
    priors = [coeff * fun(model, exposure) for coeff, fun in args["reg_dict"].values()]

    # summing the different regularisers
    return np.array(priors).sum()


def ramp_posterior_balance(model, exp, args={"reg_dict": {}}):
    """Return likelihood and prior separately for an exposure.

    This is useful for diagnostics and for balancing regularisation strengths
    (e.g. L-curve construction).

    !!! danger "Note"
        This may not work for multiple simultaneous regularisers.
    """

    # regular likelihood term
    likelihood = -np.nanmean(exp.mv_zscore(model))

    # evaluating the regularisation term with each for each regulariser
    priors = [fun(model, exp) for _, fun in args["reg_dict"].values()]
    prior = np.array(priors).sum()

    return likelihood, prior


def ramp_posterior_balances(model, exposures, args={"reg_dict": {}}):
    """Compute likelihood/prior balances for a collection of exposures.

    Returns a dict with arrays of likelihoods and priors and the exposure
    keys for convenience in diagnostics.
    """

    balances = np.array([ramp_posterior_balance(model, exp, args) for exp in exposures]).T

    return {
        "likelihoods": balances[0],
        "priors": balances[1],
        "exp_keys": [exp.key for exp in exposures],
        "args": args,
    }


def disco_regularised_loss_fn(model, exposure, args={"reg_dict": {}}):
    """Compute a regularised loss for interferometric (DISCO) data.

    The returned value mirrors other loss wrappers and returns a scalar plus
    an empty tuple for compatibility with calling code.
    """

    # regular likelihood term
    likelihood = oi_log_likelihood(model, exposure)

    # grabbing and exponentiating log distributions
    prior = apply_regularisers(model, exposure, args)

    return likelihood + prior, ()


def oi_log_likelihood(model, oi):
    """Compute a Gaussian negative log-likelihood for OI data.

    Parameters
    ----------
    model : object
        Model object callable as ``oi(model)`` to return model predictions.
    oi : object
        Object exposing ``vis``, ``phi``, ``d_vis`` and ``d_phi`` arrays.
    """
    data = np.concatenate([oi.vis, oi.phi])
    err = np.concatenate([oi.d_vis, oi.d_phi])
    model_vis = oi(model)

    residual = data - model_vis
    nll = np.sum(0.5 * (residual / err) ** 2 + np.log(err * np.sqrt(2 * np.pi)))

    return nll
