import jax
from jax import numpy as np
from dLux import utils as dlu
from matplotlib import pyplot as plt
import matplotlib as mpl
from PIL import Image
import os

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
    roll_angle_degrees: float = 0.0,
    model=None,
    show_diff_lim: bool = True,
    cmap: str = "afmhot_10u",
    bg_color: str = "k",
    axis_labels: dict = {
        "xlabel": r"$\Delta$RA [arcsec]",
        "ylabel": r"$\Delta$DEC [arcsec]",
    },
    vmin: float = 0.0,
    vmax: float = None,
    power=0.5,
    contour=False,
    scale=1.0,
    translate=(0.0, 0.0),
):
    rotation_transform = (
        mpl.transforms.Affine2D().translate(*translate).rotate_deg(roll_angle_degrees).scale(scale)
    )  # Create a rotation transformation

    ax.set_facecolor(bg_color)  # Set the background colour
    ax.tick_params(direction="out")
    ax.set(
        xticks=[0.5, 0, -0.5],
        yticks=[-0.5, 0, 0.5],
        **axis_labels,
    )  # Set the axis labels
    # if model is not None:
    #     pixel_scale = model.psf_pixel_scale / model.optics.oversample / scale
    #     if show_diff_lim:
    #         ax = dorito.plotting.plot_diffraction_limit(model, ax, OOP=True)

    kwargs = {
        "cmap": cmap,
        "extent": get_arcsec_extents(pixel_scale / scale, array.shape),
        "norm": mpl.colors.PowerNorm(power, vmin=vmin, vmax=vmax),
        "aspect": "equal",
    }
    if contour:
        im = ax.contour(
            array,
            linewidths=0.5,
            levels=array.max() * np.linspace(0, 1, 10) ** 2,
            **kwargs,
        )
    else:
        im = ax.imshow(
            array,
            **kwargs,
        )

    # ax.axis("equal")
    trans_data = rotation_transform + ax.transData  # creating transformation
    im.set_transform(trans_data)  # applying transformation to image

    return im


def create_gif_from_dir(png_dir, name, **kwargs):
    """
    Create an animated GIF from all PNG files in the specified directory.

    Args:
        png_dir (str): Path to the directory containing PNG files.
        name (str): Name of the output GIF file.
    """
    # Get a sorted list of all PNG files in the directory
    png_files = sorted([f for f in os.listdir(png_dir) if f.endswith(".png")])

    # Ensure there are PNG files to process
    if not png_files:
        raise ValueError("No PNG files found in the directory!")

    # Load the images into a list
    images = [Image.open(os.path.join(png_dir, file)) for file in png_files]

    # Save the images as an animated GIF
    images[0].save(
        png_dir + name, save_all=True, append_images=images[1:], optimize=False, **kwargs
    )


def tree_plot(coeffs):

    # approximation
    A = coeffs[0].squeeze()
    levels = [A]

    # details
    for level in coeffs[1:]:
        H = level[0].squeeze()
        V = level[1].squeeze()
        D = level[2].squeeze()

        levels += [H, V, D]

    lvl = len(coeffs[1:])

    # generating subplot axes kwargs
    axes = [
        (2**lvl, 2**lvl, 1),
        *[(2**j, 2**j, i) for j in range(lvl, 0, -1) for i in [2, 2**j + 1, 2**j + 2]],
    ]

    # plotting
    plt.figure(figsize=2 * [6 + (2**lvl)])

    # looping over subplots
    for i, ax_kwargs in enumerate(axes):

        arr = levels[i]

        # approximatino
        if i == 0:
            cmap = "viridis"
            vmin, vmax = None, None

        # details
        else:
            cmap = "cmr.wildfire"
            v = np.nanmax(np.abs(arr))
            vmin, vmax = -v, v

        imshow_kwargs = {"cmap": cmap, "vmin": vmin, "vmax": vmax}

        ax = plt.subplot(*ax_kwargs)
        ax.axis("off")
        ax.imshow(arr, **imshow_kwargs)

    plt.tight_layout()
    plt.show()
