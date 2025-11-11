"""Plotting helpers used by notebooks and examples.

This module provides small convenience functions for common plotting tasks
used across the dorito examples, such as plotting the diffraction-limit scale
bar, computing axis extents in arcseconds and a standard image plotting
function that handles normalization, rotation and optional scale bars.
"""

from jax import numpy as np
from dLux import utils as dlu
from matplotlib import pyplot as plt
import matplotlib as mpl

# for setting NaNs to grey
seismic = mpl.colormaps["seismic"]
seismic.set_bad("k", 0.5)

__all__ = [
    "plot_diffraction_limit",
    "get_arcsec_extents",
    "plot_result",
]


def plot_diffraction_limit(model, ax=None, OOP=False):
    """Plot a scale bar showing the diffraction limit (lambda / D).

    Parameters
    ----------
    model
        Object exposing ``source_spectrum`` and ``optics.diameter`` used to
        compute an effective wavelength and the diffraction-limited scale.
    ax : matplotlib.axes.Axes, optional
        Axis to draw onto. If not provided, the function draws to the current
        pyplot axes.
    OOP : bool, optional
        If True and ``ax`` is provided, draw the bar on the supplied axis and
        return the axis.
    """
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
    """Return axis extents in arcseconds for use with ``imshow``.

    Parameters
    ----------
    pixel_scale : float
        Pixel scale in arcseconds per pixel.
    shape : tuple
        Shape of the image (ny, nx) or (n, n). The function uses the first
        axis length to compute the extent.
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
    """Convenience wrapper to display an image with sensible defaults.

    This helper sets axis labels, background colour, computes an extent in
    arcseconds from the provided `pixel_scale` and applies optional rotation
    and scaling to the resulting image artist.
    """

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
