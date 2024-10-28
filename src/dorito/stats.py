from jax import numpy as np
import amigo


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
    return lambda arr, star_idx=None: reg_func(interp_star_pixel(arr, star_idx))


def interp_star_pixel(arr, star_idx=None):
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


# def L1_loss(model):
#     # only applied to the volcano array
#     return np.nansum(model.source.volc_frac * np.abs(10**model.source.log_volcanoes))


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
    return np.nansum((arr - arr.mean()) ** 2)


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
    flux = 10 ** model.fluxes[exposure.get_key("fluxes")]
    wavelets = model.wavelets[exposure.get_key("wavelets")]

    return L1_loss(flux * wavelets)


def L2(model, exposure, args):
    return L2_loss(model.get_distribution(exposure))


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
    # "L1": L1_loss,
    "L2": L2_loss,
    "TV": TV_loss,
    "TSV": TSV_loss,
    "ME": ME_loss,
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

    return likelihood + prior


def prior_data_loss(model, exposure, args):
    # regular likelihood term
    likelihood = amigo.stats.reg_loss_fn(model, exposure, args)

    # grabbing and exponentiating log distributions
    if not exposure.calibrator:
        prior = apply_regularisers(model, exposure, args)

    else:
        prior = 0.0

    return likelihood, prior


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

    # this is only to use methods! Does not update outside this function
    model = model.set("params", params)

    for k in params["wavelets"].keys():
        # reconstructing, normalising
        distribution = model._get_distribution_from_key(k)
        distribution = np.clip(distribution, 0.0)
        distribution = distribution / distribution.sum()

        # converting to wavelet coefficients
        norm_wavelets = model.wavelet_transform(distribution)
        norm_coeffs = model.flatten_wavelets(norm_wavelets)

        # re-assigning
        params["wavelets"][k] = norm_coeffs

    return model_params.set("params", params), key
