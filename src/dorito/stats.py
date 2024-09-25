from amigo.fitting import reg_loss_fn
from jax import numpy as np


def L1_loss(model):
    # only applied to the volcano array
    return np.nansum(model.source.volc_frac * np.abs(10**model.source.log_volcanoes))


def L2_loss(model, exposure):
    distribution = model.distribution(exposure)
    return np.nansum((distribution - distribution.mean()) ** 2)


def TV_loss(model, exposure):
    array = np.pad(model.distribution(exposure), 2)
    diff_y = np.abs(array[1:, :] - array[:-1, :]).sum()
    diff_x = np.abs(array[:, 1:] - array[:, :-1]).sum()
    # return np.hypot(diff_x, diff_y)
    return diff_x + diff_y


def QV_loss(model, exposure):
    array = np.pad(model.distribution(exposure), 2)
    diff_y = np.square(array[1:, :] - array[:-1, :]).sum()
    diff_x = np.square(array[:, 1:] - array[:, :-1]).sum()
    return diff_x + diff_y


def ME_loss(model, exposure, eps=1e-16):
    """
    Maximum Entropy loss function.
    """
    P = model.distribution(exposure) / np.nansum(model.distribution(exposure))
    S = np.nansum(-P * np.log(P + eps))
    return -S


reg_func_dict = {
    "L1": L1_loss,
    "L2": L2_loss,
    "TV": TV_loss,
    "QV": QV_loss,
    "ME": ME_loss,
}


def loss_fn(model, exposure, args):
    loss = reg_loss_fn(model, exposure, args)

    # regularisation
    for reg in args["reg_dict"].keys():
        coefficient = args["reg_dict"][reg]
        reg_function = args["reg_func_dict"][reg]
        loss += coefficient * reg_function(model, exposure)

    return loss
