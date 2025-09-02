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

    # grabbing and exponentiating log distributions
    prior = apply_regularisers(model, exp, args) if not exp.calibrator else 0.0

    return likelihood + prior, ()


def ramp_posterior_balance(model, exp, args={"reg_dict": {}}):
    # this is per exposure
    # NOTE this might not work for multiple regularisers

    # regular likelihood term
    likelihood = -np.nanmean(exp.mv_zscore(model))

    # evaluating the regularisation term with each for each regulariser
    priors = [fun(model, exp) for _, fun in args["reg_dict"].values()]
    prior = np.array(priors).sum() if not exp.calibrator else 0.0

    return {"likelihood": likelihood, "prior": prior, "args": args, "exp_key": exp.key}


# def L1_loss(arr):
#     """
#     L1 Norm loss function.
#     """
#     return np.nansum(np.abs(arr))


# def L2_loss(arr):
#     """
#     L2 Norm loss function.
#     """
#     # TODO - check if this is correct
#     return np.nansum(arr**2)


def TV_loss(arr):
    """
    Total variation loss function.
    """
    pad_arr = np.pad(arr, 2)  # padding
    dx = np.diff(pad_arr[0:-1, :])
    dy = np.diff(pad_arr[:, 0:-1])
    return dx.sum() + dy.sum()
    # return np.sqrt(dx[:, :-1] ** 2 + dy[:-1, :] ** 2).sum()


def TSV_loss(arr):
    """
    Quadratic variation loss function.
    """
    pad_arr = np.pad(arr, 2)  # padding
    dx = np.diff(pad_arr[0:-1, :], axis=1)
    dy = np.diff(pad_arr[:, 0:-1], axis=0)
    return np.sum(dx**2 + dy**2)


def ME_loss(arr, eps=1e-16):
    """
    Maximum Entropy loss function.
    """
    P = arr / np.nansum(arr)
    S = np.nansum(-P * np.log(P + eps))
    return -S


def get_distribution(model, exposure):

    if isinstance(model, MCAModel) or isinstance(model, MCADiscoModel):
        return model.get_distribution(exposure, with_star=False)
    else:
        return model.get_distribution(exposure)


# def L1(model, exposure):
#     flux = 10 ** model.fluxes[exposure.get_key("fluxes")]
#     distribution = model.get_distribution(exposure)
#     source = flux * distribution

#     return L1_loss(source)


# def L1_on_wavelets(model, exposure):
#     details = model.details[exposure.get_key("details")]
#     return L1_loss(details)


# def L2(model, exposure):
#     flux = 10 ** model.fluxes[exposure.get_key("fluxes")]
#     distribution = model.get_distribution(exposure)
#     source = flux * distribution

#     return L2_loss(source)


def TV(model, exposure):
    return TV_loss(get_distribution(model, exposure))


def TSV(model, exposure):
    return TSV_loss(get_distribution(model, exposure))


def ME(model, exposure):
    return ME_loss(get_distribution(model, exposure))


# def normalise_wavelets(model_params, args):
#     params = model_params.params

#     if "wavelets" not in params.keys():
#         return model_params, args

#     for k, wavelets in params["wavelets"].items():
#         wavelets = wavelets.normalise()
#         params["wavelets"][k] = wavelets

#     return model_params.set("params", params), args


# def lcurve_sweep(
#     regulariser: str,  # e.g. "L1", "L2", "TV", "QV", "ME", "SF"
#     coeffs,
#     model,
#     exposures: list,
#     args: dict,  # must contain reg_dict and reg_func_dict
#     optimisers: dict,
#     fishers: dict,
#     **optimise_kwargs,
# ):
#     """
#     Function to find an optimal regularisation hyperparameter using the l-curve method.

#     Parameters
#     ----------
#     regulariser : str
#         Name of the regulariser to be optimised.
#     coeffs : array_like
#         Regularisation hyperparameters to be tested.
#     model : amigo.Model
#         Model to be optimised.
#     exposures : list
#         List of exposures to be used in the optimisation.
#     args : dict
#         Dictionary containing the regularisation dictionary and regularisation function dictionary.
#     optimisers : dict
#         Dictionary containing the optimisers to be used in the optimisation.
#     fishers : dict
#         Dictionary containing the fishers to be used in the optimisation.
#     optimise_kwargs : dict
#         Additional keyword arguments to be passed to the optimisation function.

#     Returns
#     -------
#     tuple
#         Tuple containing the balance, regularisation hyperparameters and log distributions.
#     """

#     @zdx.filter_jit
#     def calc_balance(model, exposures, args, coeff):
#         return np.array([prior_data_balance(model, exp, args, coeff) for exp in exposures]).sum(0)

#     balances = []
#     dists = []

#     for coeff in coeffs:

#         args["reg_dict"][regulariser] = coeff

#         optim_model, _, _, _ = amigo.fitting.optimise(
#             model,
#             exposures,
#             optimisers,
#             fishers,
#             args=args,
#             **optimise_kwargs,
#         )

#         balance = calc_balance(optim_model, exposures, args, coeff)
#         balances.append(balance)

#         if "log_distribution" in optim_model.params.keys():
#             dists.append(optim_model.log_distribution)
#         else:
#             dists.append(optim_model.get_distribution(exposures[0]))

#     return np.array(balances).T, coeffs, dists
