from jax import numpy as np
from .models import MCAModel, MCADiscoModel


def oi_log_likelihood(model, oi):
    data = np.concatenate([oi.vis, oi.phi])
    err = np.concatenate([oi.d_vis, oi.d_phi])
    model_vis = oi(model)

    residual = data - model_vis
    nll = np.sum(0.5 * (residual / err) ** 2 + np.log(err * np.sqrt(2 * np.pi)))

    return nll


def apply_regularisers(model, exposure, args):

    if "reg_dict" not in args.keys():
        return 0.0

    # evaluating the regularisation term with each for each regulariser
    priors = [coeff * fun(model, exposure) for coeff, fun in args["reg_dict"].values()]

    # summing the different regularisers
    return np.array(priors).sum()


def disco_regularised_loss_fn(model, exposure, args={"reg_dict": {}}):
    # this is per exposure

    # regular likelihood term
    likelihood = oi_log_likelihood(model, exposure)

    # grabbing and exponentiating log distributions
    prior = apply_regularisers(model, exposure, args)

    return likelihood + prior, ()


def ramp_regularised_loss_fn(model, exp, args={"reg_dict": {}}):
    # this is per exposure

    # regular likelihood term
    likelihood = -np.nanmean(exp.mv_zscore(model))
    prior = apply_regularisers(model, exp, args) if not exp.calibrator else 0.0

    return likelihood + prior, ()


def ramp_posterior_balance(model, exp, args={"reg_dict": {}}):
    # this is per exposure
    # NOTE this might not work for multiple regularisers

    # regular likelihood term
    likelihood = -np.nanmean(exp.mv_zscore(model))

    # evaluating the regularisation term with each for each regulariser
    priors = [fun(model, exp) for _, fun in args["reg_dict"].values()]
    prior = np.array(priors).sum()

    return likelihood, prior


def ramp_posterior_balances(model, exposures, args={"reg_dict": {}}):

    balances = np.array([ramp_posterior_balance(model, exp, args) for exp in exposures]).T

    return {
        "likelihoods": balances[0],
        "priors": balances[1],
        "exp_keys": [exp.key for exp in exposures],
        "args": args,
    }


def L1_loss(arr):
    """
    L1 Norm loss function.
    """
    return np.nansum(np.abs(arr))


def L2_loss(arr):
    """
    L2 Norm loss function.
    """
    return np.nansum(arr**2)


def tikhinov(arr):
    """
    https://www-users.cse.umn.edu/~jwcalder/5467/lec_tv_denoising.pdf
    """
    pad_arr = np.pad(arr, 2)  # padding
    dx = np.diff(pad_arr[0:-1, :], axis=1)
    dy = np.diff(pad_arr[:, 0:-1], axis=0)
    return dx**2 + dy**2


def TV_loss(arr, eps=1e-16):
    """
    Approximation of the L1 norm of the gradient of the image.
    """
    return np.sqrt(tikhinov(arr) + eps**2).sum()


def TSV_loss(arr):
    """
    Quadratic variation loss function.
    """
    return tikhinov(arr).sum()


def ME_loss(arr, eps=1e-16):
    """
    Maximum Entropy loss function.
    """
    P = arr / np.nansum(arr)
    S = np.nansum(-P * np.log(P + eps))
    return -S


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
