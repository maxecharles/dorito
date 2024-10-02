from jax import numpy as np
import amigo


def get_star_idx(arr):
    i = arr.shape[0] // 2
    return (i, i)


def star_regulariser(arr, prior=0.948, star_idx=None):
    """
    Regulariser to penalise a central pixel from straying from a given prior value."
    """
    # grabbing index of star
    if star_idx is None:
        star_idx = get_star_idx(arr)

    arr /= arr.sum()  # normalising
    star_flux = arr[star_idx]  # grabbing star pixel flux

    return (star_flux - prior) ** 2


# def L1_loss(model):
#     # only applied to the volcano array
#     return np.nansum(model.source.volc_frac * np.abs(10**model.source.log_volcanoes))


def L2_loss(arr):
    """
    L2 Norm loss function.
    """
    return np.nansum((arr - arr.mean()) ** 2)


def TV_loss(arr):
    """
    Total variation loss function.
    """
    pad_arr = np.pad(arr, 2)  # padding
    diff_y = np.abs(pad_arr[1:, :] - pad_arr[:-1, :]).sum()
    diff_x = np.abs(pad_arr[:, 1:] - pad_arr[:, :-1]).sum()
    # return np.hypot(diff_x, diff_y)
    return diff_x + diff_y


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


def interp_star_pixel(arr, star_idx):
    a, b = star_idx

    # linear interpolation of surrounding pixels
    interp_value = np.array(
        [arr[idx] for idx in [(a - 1, b), (a + 1, b), (a, b - 1), (a, b + 1)]]
    ).mean()

    return np.nan_to_num(arr, nan=interp_value)


def TSV_with_star(arr, star_idx=None):
    # grabbing index of star
    if star_idx is None:
        star_idx = get_star_idx(arr)

    # setting star pixel to NaN
    nanpix_arr = arr.at[star_idx].set(np.nan)

    return TSV_loss(interp_star_pixel(nanpix_arr, star_idx))


reg_func_dict = {
    # "L1": L1_loss,
    "L2": L2_loss,
    "TV": TV_loss,
    "TSV": TSV_loss,
    "ME": ME_loss,
}


def regularised_loss_fn(model, exposure, args):
    # this is per exposure

    # regular likelihood term
    likelihood = amigo.stats.reg_loss_fn(model, exposure, args)

    # grabbing and exponentiating log distributions
    if not exposure.calibrator:
        distribution = 10 ** model.params["log_distribution"][exposure.get_key("log_distribution")]

        # creating a list of regularisation functions and regularisation hyperparameters
        fn_list = [args["reg_func_dict"][reg] for reg in args["reg_dict"].keys()]
        coeff_list = list(args["reg_dict"].values())

        # evaluating the regularisation term with each for each regulariser
        priors = [coeff * fn(distribution) for coeff, fn in zip(coeff_list, fn_list)]
        prior = np.array(priors).sum()  # summing the different regularisers

    else:
        prior = 0.0

    return likelihood + prior


def normalise_distribution(model, model_params, args, key):
    params = model_params.params
    if "log_distribution" in params.keys():
        for k, log_dist in params["log_distribution"].items():
            distribution = 10**log_dist
            params["log_distribution"][k] = np.log10(distribution / distribution.sum())

    return model_params.set("params", params), key
