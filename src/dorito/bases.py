from jaxtyping import Array
from zodiax import Base
import equinox as eqx
import jax.numpy as np
from photutils.psf.matching import TopHatWindow
from scipy.ndimage import binary_dilation
import numpy as onp


def inscribed_circ_basis(size: int, return_window=True) -> Array:

    window_arr = TopHatWindow(1)((size, size))
    mask = onp.where(window_arr.flatten())[0]
    M = onp.eye(size**2)[:, mask]
    if return_window:
        return ImageBasis(np.array(M)), window_arr
    return ImageBasis(np.array(M))


def inscribed_annulus_basis(size: int, iterations=2, return_window=True) -> Array:

    outer = TopHatWindow(1)((size, size))
    inner = binary_dilation(
        np.zeros_like(outer).at[size // 2, size // 2].set(1), iterations=iterations
    )
    window_arr = outer - inner

    mask = onp.where(window_arr.flatten())[0]
    M = onp.eye(size**2)[:, mask]
    if return_window:
        return ImageBasis(np.array(M)), window_arr
    return ImageBasis(np.array(M))


class ImageBasis(Base):

    M: Array
    M_inv: Array
    n_basis: int = eqx.field(static=True)
    size: int = eqx.field(static=True)

    def __init__(self, transform_matrix: Array, n_basis: int = None):
        if n_basis is None:
            n_basis = transform_matrix.shape[1]
        self.n_basis = n_basis
        self.M = transform_matrix[:, :n_basis]
        self.M_inv = np.linalg.pinv(self.M)
        self.size = int(np.sqrt(self.M.shape[0]))

    def to_basis(self, img: Array) -> Array:
        return np.dot(self.M_inv, img.flatten())

    def from_basis(self, coeffs: Array) -> Array:
        return np.dot(self.M, coeffs).reshape((self.size, self.size))
