from jax import numpy as np
from dLux import utils as dlu
from matplotlib import pyplot as plt
import matplotlib as mpl

# for setting NaNs to grey
seismic = mpl.colormaps["seismic"]
seismic.set_bad("k", 0.5)


def plot_diffraction_limit(model, ax=None, OOP=False):
    effective_wl = np.dot(model.source_spectrum.wavelengths, model.source_spectrum.weights)
    diff_lim = dlu.rad2arcsec(effective_wl / model.optics.diameter)
    scale_length = diff_lim

    scale_bar_x = -0.7
    scale_bar_y = scale_bar_x
    fontdict = {
        "fontstyle": "normal",
        "color": "hotpink",
        "weight": "demi",
        "size": 7,
    }

    if OOP and ax is not None:
        ax.plot(
            [scale_bar_x, scale_bar_x + scale_length],
            [scale_bar_y, scale_bar_y],
            color="hotpink",
            linewidth=2,
        )
        ax.text(
            scale_bar_x + scale_length / 2 - 0.075,
            scale_bar_y + 0.03,
            r"$\lambda / D$",
            **fontdict,
        )
        return ax

    else:
        plt.plot(
            [scale_bar_x, scale_bar_x + scale_length],
            [scale_bar_y, scale_bar_y],
            color="hotpink",
            linewidth=2,
        )
        plt.text(
            scale_bar_x + scale_length / 2 - 0.046,
            scale_bar_y + 0.02,
            r"$\lambda / D$",
            **fontdict,
        )


def get_arcsec_extents(pixel_scale, shape):
    """
    Get the arcsec extents of an image given the pixel scale and shape.
    """
    return np.array([0.5, -0.5, -0.5, 0.5]) * pixel_scale * shape[0]


def plot_result(
    ax,
    array,
    pixel_scale,
    roll_angle_degrees: float = None,
    cmap: str = "afmhot_10u",
    bg_color: str = "k",
    axis_labels: dict = {
        "xlabel": r"$\Delta$RA [arcsec]",
        "ylabel": r"$\Delta$DEC [arcsec]",
    },
    norm=mpl.colors.PowerNorm(1, vmin=0, vmax=None),
    diff_lim: float = None,
    scale=1.0,
    translate=(0.0, 0.0),
    ticks=[0.5, 0, -0.5],
):

    ax.set_facecolor(bg_color)  # Set the background colour
    ax.tick_params(direction="out")
    ax.set(
        xticks=ticks,
        yticks=ticks[::-1],
        **axis_labels,
    )  # Set the axis labels

    kwargs = {
        "cmap": cmap,
        "extent": get_arcsec_extents(pixel_scale / scale, array.shape),
        "norm": norm,
        "aspect": "equal",
    }

    im = ax.imshow(
        array,
        **kwargs,
    )

    if diff_lim is not None:

        centre = 0.95 * np.array(kwargs["extent"][0:2]) + np.array([-diff_lim, diff_lim])

        beam = mpl.patches.Circle(
            centre,
            radius=diff_lim,
            facecolor="white",
            edgecolor="black",
            alpha=0.7,
            zorder=10,
        )
        ax.add_patch(beam)

    if roll_angle_degrees is not None or scale is not None:

        if scale is None:
            scale = 1.0
        if roll_angle_degrees is None:
            roll_angle_degrees = 0.0

        rotation_transform = (
            mpl.transforms.Affine2D()
            .rotate_deg(roll_angle_degrees)
            .scale(scale)
            .translate(*translate)
        )  # Create a rotation transformation
        trans_data = rotation_transform + ax.transData  # creating transformation
        im.set_transform(trans_data)  # applying transformation to image

    return im


# def tree_plot(coeffs):

#     # approximation
#     A = coeffs[0].squeeze()
#     levels = [A]

#     # details
#     for level in coeffs[1:]:
#         H = level[0].squeeze()
#         V = level[1].squeeze()
#         D = level[2].squeeze()

#         levels += [H, V, D]

#     lvl = len(coeffs[1:])

#     # generating subplot axes kwargs
#     axes = [
#         (2**lvl, 2**lvl, 1),
#         *[(2**j, 2**j, i) for j in range(lvl, 0, -1) for i in [2, 2**j + 1, 2**j + 2]],
#     ]

#     # plotting
#     plt.figure(figsize=2 * [6 + (2**lvl)])

#     # looping over subplots
#     for i, ax_kwargs in enumerate(axes):

#         arr = levels[i]

#         # approximatino
#         if i == 0:
#             cmap = "viridis"
#             vmin, vmax = None, None

#         # details
#         else:
#             cmap = "cmr.wildfire"
#             v = np.nanmax(np.abs(arr))
#             vmin, vmax = -v, v

#         imshow_kwargs = {"cmap": cmap, "vmin": vmin, "vmax": vmax}

#         ax = plt.subplot(*ax_kwargs)
#         ax.axis("off")
#         ax.imshow(arr, **imshow_kwargs)

#     plt.tight_layout()
#     plt.show()
