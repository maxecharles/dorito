from jax import numpy as np
import amigo
import zodiax as zdx


def get_star_idx(arr):
    i = arr.shape[0] // 2
    return (i, i)


def star_reg_on_array(arr, args):
    """
    Regulariser to penalise a central pixel from straying from a given prior value."
    """

    prior = args["prior"]
    star_idx = args["star_idx"]

    # grabbing index of star
    if star_idx is None:
        star_idx = get_star_idx(arr)

    arr /= arr.sum()  # normalising
    star_flux = arr[star_idx]  # grabbing star pixel flux

    return (star_flux - prior) ** 2


def star_reg(model, exposure, args={"prior": 0.948, "star_idx": None}):
    """
    Regulariser to penalise a central pixel from straying from a given prior value."
    """
    # grabbing index of star
    arr = model.get_distribution(exposure)
    return star_reg_on_array(arr, args)


def regfunc_with_star(reg_func):
    """
    Takes in a regularisation function and returns a new regularisation
    function that interpolates the star pixel.
    """
    return lambda model, exposure, args: reg_func(interp_star_pixel(model, exposure, args))


def interp_star_pixel_on_array(arr, star_idx=None):
    # grabbing index of star
    if star_idx is None:
        star_idx = get_star_idx(arr)

    # setting star pixel to NaN
    nanpix_arr = arr.at[star_idx].set(np.nan)

    # unpacking indices
    a, b = star_idx

    # linear interpolation of surrounding pixels
    interp_value = np.array(
        [nanpix_arr[idx] for idx in [(a - 1, b), (a + 1, b), (a, b - 1), (a, b + 1)]]
    ).mean()

    # setting NaN value to interpolated value
    return np.nan_to_num(nanpix_arr, nan=interp_value)


def interp_star_pixel(model, exposure, args):
    return interp_star_pixel_on_array(model.get_distribution(exposure), star_idx=args["star_idx"])


def L1_loss(arr):
    """
    L1 Norm loss function.
    """
    return np.nansum(np.abs(arr))


def L2_loss(arr):
    """
    L2 Norm loss function.
    """
    # TODO - check if this is correct
    return np.nansum(arr**2)


def TV_loss(arr):
    """
    Total variation loss function.
    """
    pad_arr = np.pad(arr, 2)  # padding
    diff_y = np.abs(pad_arr[1:, :] - pad_arr[:-1, :]).sum()
    diff_x = np.abs(pad_arr[:, 1:] - pad_arr[:, :-1]).sum()
    return np.hypot(diff_x, diff_y)
    # return diff_x + diff_y


def TSV_loss(arr):
    """
    Quadratic variation loss function.
    """
    pad_arr = np.pad(arr, 2)  # padding
    diff_y = np.square(pad_arr[1:, :] - pad_arr[:-1, :]).sum()
    diff_x = np.square(pad_arr[:, 1:] - pad_arr[:, :-1]).sum()
    return diff_x + diff_y


def ME_loss(arr, eps=1e-16):
    """
    Maximum Entropy loss function.
    """
    P = arr / np.nansum(arr)
    S = np.nansum(-P * np.log(P + eps))
    return -S


def L1(model, exposure, args):
    flux = 10 ** model.fluxes[exposure.get_key("fluxes")]
    distribution = model.get_distribution(exposure)
    source = flux * distribution

    return L1_loss(source)


def L1_on_wavelets(model, exposure, args):
    details = model.details[exposure.get_key("details")]
    return L1_loss(details)


def L2(model, exposure, args):
    flux = 10 ** model.fluxes[exposure.get_key("fluxes")]
    distribution = model.get_distribution(exposure)
    source = flux * distribution

    return L2_loss(source)


def TV(model, exposure, args):
    return TV_loss(model.get_distribution(exposure))


def TSV(model, exposure, args):
    return TSV_loss(model.get_distribution(exposure))


def ME(model, exposure, args):
    return ME_loss(model.get_distribution(exposure))


def TSV_with_star(arr, star_idx=None):
    # grabbing index of star
    if star_idx is None:
        star_idx = get_star_idx(arr)

    # setting star pixel to NaN
    nanpix_arr = arr.at[star_idx].set(np.nan)

    return TSV_loss(interp_star_pixel(nanpix_arr, star_idx))


reg_func_dict = {
    "L1": L1_on_wavelets,
    "L2": L2,
    "TV": TV,
    "TSV": TSV,
    "ME": ME,
}


def apply_regularisers(model, exposure, args):
    # creating a list of regularisation functions and regularisation hyperparameters
    fn_list = [args["reg_func_dict"][reg] for reg in args["reg_dict"].keys()]
    coeff_list = list(args["reg_dict"].values())

    # evaluating the regularisation term with each for each regulariser
    priors = [coeff * fn(model, exposure, args) for coeff, fn in zip(coeff_list, fn_list)]
    prior = np.array(priors).sum()  # summing the different regularisers

    return prior


def regularised_loss_fn(model, exposure, args):
    # this is per exposure

    # regular likelihood term
    likelihood = amigo.stats.reg_loss_fn(model, exposure, args)

    # grabbing and exponentiating log distributions
    if not exposure.calibrator:
        prior = apply_regularisers(model, exposure, args)

    else:
        prior = 0.0

    return np.hypot(likelihood, prior)


def prior_data_balance(model, exposure, args, coeff=1.0):
    # regular likelihood term
    likelihood = amigo.stats.reg_loss_fn(model, exposure, args)

    # grabbing and exponentiating log distributions
    if not exposure.calibrator:
        prior = apply_regularisers(model, exposure, args)

    else:
        prior = 0.0

    return likelihood, prior / coeff


def normalise_distribution(model, model_params, args, key):
    params = model_params.params
    if "log_distribution" in params.keys():
        for k, log_dist in params["log_distribution"].items():
            distribution = 10**log_dist
            params["log_distribution"][k] = np.log10(distribution / distribution.sum())

    return model_params.set("params", params), key


def normalise_wavelets(model, model_params, args, key):
    params = model_params.params

    if "wavelets" not in params.keys():
        return model_params, key

    for k, wavelets in params["wavelets"].items():
        wavelets = wavelets.normalise()
        params["wavelets"][k] = wavelets

    return model_params.set("params", params), key


def lcurve_sweep(
    regulariser: str,  # e.g. "L1", "L2", "TV", "QV", "ME", "SF"
    coeffs,
    model,
    exposures: list,
    args: dict,  # must contain reg_dict and reg_func_dict
    optimisers: dict,
    fishers: dict,
    **optimise_kwargs,
):
    """
    Function to find an optimal regularisation hyperparameter using the l-curve method.

    Parameters
    ----------
    regulariser : str
        Name of the regulariser to be optimised.
    coeffs : array_like
        Regularisation hyperparameters to be tested.
    model : amigo.Model
        Model to be optimised.
    exposures : list
        List of exposures to be used in the optimisation.
    args : dict
        Dictionary containing the regularisation dictionary and regularisation function dictionary.
    optimisers : dict
        Dictionary containing the optimisers to be used in the optimisation.
    fishers : dict
        Dictionary containing the fishers to be used in the optimisation.
    optimise_kwargs : dict
        Additional keyword arguments to be passed to the optimisation function.

    Returns
    -------
    tuple
        Tuple containing the balance, regularisation hyperparameters and log distributions.
    """

    @zdx.filter_jit
    def calc_balance(model, exposures, args, coeff):
        return np.array([prior_data_balance(model, exp, args, coeff) for exp in exposures]).sum(0)

    balances = []
    log_dists = []

    for coeff in coeffs:

        args["reg_dict"][regulariser] = coeff

        optim_model, _, _, _ = amigo.fitting.optimise(
            model,
            exposures,
            optimisers,
            fishers,
            args=args,
            **optimise_kwargs,
        )

        balance = calc_balance(optim_model, exposures, args, coeff)
        balances.append(balance)
        log_dists.append(optim_model.log_distribution)

    return np.array(balances).T, coeffs, log_dists
