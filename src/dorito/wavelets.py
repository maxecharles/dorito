import zodiax as zdx
import jax
from jax import numpy as np, tree as jtu, Array
import jaxwt


class Wavelets(zdx.WrapperHolder):
    level: int
    wavelet: str
    approx: Array

    def __init__(
        self,
        distribution,
        level=2,
        wavelet="db2",
    ):

        # Overriden later, just to avoid recursion error
        self.structure = None

        # setting the wavelet parameters
        self.level = level
        self.wavelet = wavelet

        # converting to wavelet coefficients
        distribution = distribution / distribution.sum()
        approx, detail_tree = self.wavelet_transform(distribution)
        self.approx = approx

        # storing the details in a EquinoxWrapper
        details, structure = zdx.build_wrapper(detail_tree)
        self.values = details
        self.structure = structure

    @property
    def distribution(self) -> Array:
        """
        Returns the normalised intensity distribution of the source.
        """
        # building the detail tree from the values and structure
        detail_tree = self.build

        # converting to wavelet coefficients with approx
        coeffs = [self.approx[None, ...], *detail_tree]

        # reconstructing the distribution
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

        # clipping and normalising the distribution
        dist = np.clip(dist, 0)
        dist = dist / dist.sum()

        # converting to wavelet coefficients
        approx, detail_tree = self.wavelet_transform(dist)
        detail_leaves, _ = jtu.flatten(detail_tree)
        details = np.concatenate([val.flatten() for val in detail_leaves])

        return self.set(["approx", "values"], [approx, details])
