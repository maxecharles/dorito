from jaxtyping import Array
from zodiax import Base
import equinox as eqx
import jax.numpy as np
from photutils.psf.matching import TopHatWindow
from scipy.ndimage import binary_dilation
import numpy as onp

__all__ = [
    "inscribed_circ_basis",
    "inscribed_annulus_basis",
    "ImageBasis",
]


def inscribed_circ_basis(size: int, return_window=True) -> Array:
    """Create an image basis using pixels inside an inscribed circular window.

    The function builds a linear basis matrix `M` that maps basis
    coefficients to the flattened image pixels for the pixels that lie
    inside a circular top-hat window inscribed in a square of side ``size``.

    Parameters
    ----------
    size : int
        Number of pixels on a side of the (square) image. The function
        constructs a ``size x size`` window and selects pixels inside the
        inscribed top-hat.
    return_window : bool, optional
        If True (default) return a tuple ``(ImageBasis, window_arr)`` where
        ``window_arr`` is the boolean mask of selected pixels. Otherwise
        return only the ``ImageBasis`` instance.

    Returns
    -------
    ImageBasis or (ImageBasis, ndarray)
        The basis mapping (and optionally the boolean window mask).
    """

    window_arr = TopHatWindow(1)((size, size))
    mask = onp.where(window_arr.flatten())[0]
    M = onp.eye(size**2)[:, mask]
    if return_window:
        return ImageBasis(np.array(M), ortho=True), window_arr
    return ImageBasis(np.array(M))


def inscribed_annulus_basis(size: int, iterations=2, return_window=True) -> Array:
    """Create an image basis for an annulus (inscribed ring) window.

    Constructs a basis matrix selecting pixels that lie in an annulus
    defined by subtracting an inner dilated mask from an outer top-hat
    window. The inner radius is controlled by ``iterations`` passed to
    ``scipy.ndimage.binary_dilation`` so larger ``iterations`` produce a
    wider inner hole.

    Parameters
    ----------
    size : int
        Number of pixels on a side of the (square) image.
    iterations : int, optional
        Number of binary dilation iterations used to build the inner hole
        (default: 2).
    return_window : bool, optional
        If True (default) return a tuple ``(ImageBasis, window_arr)`` where
        ``window_arr`` is the boolean mask of selected annulus pixels.

    Returns
    -------
    ImageBasis or (ImageBasis, ndarray)
        The basis mapping (and optionally the boolean window mask).
    """

    outer = TopHatWindow(1)((size, size))
    inner = binary_dilation(
        np.zeros_like(outer).at[size // 2, size // 2].set(1), iterations=iterations
    )
    window_arr = outer - inner

    mask = onp.where(window_arr.flatten())[0]
    M = onp.eye(size**2)[:, mask]
    if return_window:
        return ImageBasis(np.array(M), ortho=True), window_arr
    return ImageBasis(np.array(M))


class ImageBasis(Base):
    """Linear image basis wrapper.

    Parameters
    ----------
    transform_matrix
        Matrix that maps basis coefficients to flattened image pixels. The
        class stores the forward matrix `M` and a pseudo-inverse `M_inv` used
        to move between pixel and basis representations.

    Notes
    -----
    The class intentionally accepts non-orthogonal bases and computes the
    pseudo-inverse when `ortho=False`.
    """

    M: Array
    M_inv: Array
    n_basis: int = eqx.field(static=True)
    size: int = eqx.field(static=True)

    def __init__(self, transform_matrix: Array, n_basis: int = None, ortho=False):
        if n_basis is None:
            n_basis = transform_matrix.shape[1]
        self.n_basis = n_basis
        self.M = transform_matrix[:, :n_basis]
        if ortho:
            self.M_inv = self.M.T
        else:
            self.M_inv = np.linalg.pinv(self.M)
        self.size = int(np.sqrt(self.M.shape[0]))

    def to_basis(self, img: Array) -> Array:
        """Project a 2D image into the basis and return coefficients.

        Parameters
        ----------
        img : Array
            2D image array with shape (size, size).

        Returns
        -------
        Array
            Basis coefficients (flattened).
        """
        return np.dot(self.M_inv, img.flatten())

    def from_basis(self, coeffs: Array) -> Array:
        """Reconstruct an image from basis coefficients.

        Parameters
        ----------
        coeffs : Array
            Basis coefficients.

        Returns
        -------
        Array
            2D image reconstructed from the provided coefficients.
        """
        return np.dot(self.M, coeffs).reshape((self.size, self.size))
