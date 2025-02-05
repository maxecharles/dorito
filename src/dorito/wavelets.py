from zodiax import Base
import jax
from jax import numpy as np, tree as jtu, Array
import jaxwt


class Wavelets(Base):
    level: int
    wavelet: str
    approx: Array
    details: Array

    shapes: list
    sizes: list
    starts: list
    tree_def: None

    def __init__(
        self,
        distribution,
        level=2,
        wavelet="db2",
    ):
        # setting the wavelet parameters
        self.level = level
        self.wavelet = wavelet

        # converting to wavelet coefficients
        distribution = distribution / distribution.sum()
        approx, detail_tree = self.wavelet_transform(distribution)

        # getting shapes, sizes and starts for the wavelet coefficients
        detail_leaves, tree_def = jtu.flatten(detail_tree)
        self.shapes = [v.shape for v in detail_leaves]
        self.sizes = [int(v.size) for v in detail_leaves]
        self.starts = [int(i) for i in np.cumsum(np.array([0] + self.sizes))]
        self.tree_def = tree_def

        # setting the approx and details
        self.approx = approx
        self.details = self.flatten_detail_tree(detail_tree)

    def flatten_detail_tree(self, detail_tree):
        detail_leaves, _ = jtu.flatten(detail_tree)
        return np.concatenate([val.flatten() for val in detail_leaves])

    @property
    def distribution(self) -> Array:
        """
        Returns the normalised intensity distribution of the source.
        """
        detail_leaves = [
            jax.lax.dynamic_slice(self.details, (start,), (size,)).reshape(shape)
            for start, size, shape in zip(self.starts, self.sizes, self.shapes)
        ]
        detail_tree = jtu.unflatten(self.tree_def, detail_leaves)

        coeffs = [self.approx[None, ...], *detail_tree]
        return jaxwt.waverec2(coeffs, self.wavelet)[0]

    def wavelet_transform(self, array) -> Array:
        # converting to wavelet coefficients
        coeffs = jaxwt.conv_fwt_2d.wavedec2(
            array,
            self.wavelet,
            level=self.level,
        )
        approx = coeffs[0][0]
        detail_tree = coeffs[1:]

        return approx, detail_tree

    def normalise(self):
        """
        Normalises the wavelet coefficients such that the
        corresponding distribution sums to 1.
        """
        dist = self.distribution
        norm_dist = dist / dist.sum()
        approx, detail_tree = self.wavelet_transform(norm_dist)
        details = self.flatten_detail_tree(detail_tree)

        return self.set(["approx", "details"], [approx, details])
