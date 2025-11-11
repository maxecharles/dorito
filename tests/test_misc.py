import pytest


def test_fwhm_to_sigma():
    # Skip if jax isn't available in the environment (some CI setups may not install it)
    pytest.importorskip("jax")

    from dorito import misc
    import math

    fwhm = 2 * math.sqrt(2 * math.log(2))  # corresponds to sigma == 1
    sigma = misc.fwhm_to_sigma(fwhm)

    # Convert to Python float for comparison (jax array -> float)
    assert abs(float(sigma) - 1.0) < 1e-6


def test_calc_parang():
    pytest.importorskip("jax")

    from dorito import misc

    class FakePrimary:
        def __init__(self, val):
            self.header = {"ROLL_REF": val}

    fake_file = {"PRIMARY": FakePrimary(42.0)}

    parang = misc.calc_parang(fake_file)
    assert float(parang) == 42.0
