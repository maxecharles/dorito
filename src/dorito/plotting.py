import jax
from jax import numpy as np, Array
from dLux import utils as dlu
from matplotlib import pyplot as plt
import matplotlib as mpl
import planetmapper
from PIL import Image
import os

# for setting NaNs to grey
seismic = mpl.colormaps["seismic"]
seismic.set_bad("k", 0.5)


def calc_parang(file):
    """
    Calculate the parallactic angle for a given file.
    """
    return np.array(file["PRIMARY"].header["ROLL_REF"])


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
    pixel_scale: float,
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
    ticks=[0.5, 0, -0.5],
):
    rotation_transform = mpl.transforms.Affine2D().rotate_deg(
        roll_angle_degrees
    )  # Create a rotation transformation

    ax.set_facecolor(bg_color)  # Set the background colour
    ax.tick_params(direction="out")
    ax.set(
        xticks=ticks,
        yticks=ticks[::-1],
        **axis_labels,
    )  # Set the axis labels
    if model is not None:
        pixel_scale = model.psf_pixel_scale / model.optics.oversample
        if show_diff_lim:
            ax = plot_diffraction_limit(model, ax, OOP=True)
    im = ax.imshow(
        array,
        cmap=cmap,
        extent=get_arcsec_extents(pixel_scale, array.shape),
        norm=mpl.colors.PowerNorm(power, vmin=vmin, vmax=vmax),
        aspect="equal",
    )

    trans_data = rotation_transform + ax.transData  # creating transformation
    im.set_transform(trans_data)  # applying transformation to image

    return im


def plot_io_with_ephemeris(
    ax, array, date, roll_angle_degrees=246.80584209034947, legend=False, **kwargs
):
    body = planetmapper.Body("io", date, observer="jwst")

    plot_result(ax, array, roll_angle_degrees, show_diff_lim=True, **kwargs)

    body.plot_wireframe_angular(
        ax,
        add_title=False,
        label_poles=True,
        indicate_equator=True,
        indicate_prime_meridian=False,
        grid_interval=15,
        grid_lat_limit=75,
        aspect_adjustable="box",
        formatting={
            "limb": {
                "linestyle": "--",
                "linewidth": 0.8,
                "alpha": 0.8,
                "color": "white",
            },
            "grid": {
                "linestyle": "--",
                "linewidth": 0.5,
                "alpha": 0.8,
                "color": "white",
            },
            "equator": {"linewidth": 1, "color": "r", "label": "equator"},
            "terminator": {
                "linewidth": 1,
                "linestyle": "-",
                "color": "aqua",
                "alpha": 0.7,
                "label": "terminator",
            },
            "coordinate_of_interest_lonlat": {
                "color": "g",
                "marker": "^",
                "s": 50,
                "label": "volcano",
            },
            # 'limb_illuminated': {'color': 'b'},
        },
    )

    if legend:
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(
            by_label.values(),
            by_label.keys(),
            loc="upper left",
        )


def get_residuals(
    arr1: Array,
    arr2: Array,
    return_bounds: bool = False,
    halfrange: float = None,  # passed to CenteredNorm
):
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)
    residuals = arr1 - arr2

    if return_bounds:
        norm = mpl.colors.CenteredNorm(halfrange=halfrange)
        bound_dict = {"norm": norm, "cmap": seismic}
        return residuals, bound_dict

    return residuals


# def get_loglike_maps(true_model, final_model, exposures, std_min: int = 100):
#     flux = 10 ** true_model.params["fluxes"]["IO_F430M"]

#     maps = []

#     for exp in exposures:
#         truth = flux * true_model.distribution(exp)
#         recovered = flux * final_model.distribution(exp)
#         std = np.maximum(np.sqrt(truth), std_min)

#         maps.append(-jax.scipy.stats.norm.logpdf(truth, recovered, std))

#     return maps


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
