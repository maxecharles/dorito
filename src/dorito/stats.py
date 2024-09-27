from jax import numpy as np
import amigo


def star_regulariser(arr, prior=0.948, star_idx=(49, 49)):

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


def QV_loss(arr):   
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


reg_func_dict = {
    # "L1": L1_loss,
    "L2": L2_loss,
    "TV": TV_loss,
    "QV": QV_loss,
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