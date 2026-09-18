import jax.numpy as np
import jax.random as jr
import pytest

import dorito.model_fits as mf
from dorito.model_fits import (
    DynamicResolvedFit,
    PointResolvedFit,
    ResolvedFit,
    TransformedResolvedFit,
    _BaseResolvedFit,
)


def _stub(cls, **attrs):
    """Build an instance without running __init__.

    The key-routing methods only read `self.filter` / `self.key`. `key` is a
    read-only property on ModelFit, and a data descriptor on the class wins
    over anything in the instance dict, so it can't be set or shadowed from
    the instance side. Instead we make a throwaway subclass whose class-level
    attributes shadow the properties earlier in the MRO, then allocate it
    without __init__ (which would want a real file and optics model).

    The stub is still a genuine subclass, so the `super()` chains in `get_key`
    and `map_param` resolve exactly as they do in production.
    """
    stub_cls = type(f"_Stub{cls.__name__}", (cls,), dict(attrs))
    return object.__new__(stub_cls)


# ------------------------------------------------------------ module surface


def test_all_exported_names_exist():
    missing = [name for name in mf.__all__ if not hasattr(mf, name)]
    assert missing == [], f"__all__ lists names that do not exist: {missing}"


# --------------------------------------------------------------- key routing


def test_resolved_fit_keys_on_filter():
    fit = _stub(ResolvedFit, filter="F430M")
    assert fit.get_key("log_dist") == "F430M"
    assert fit.map_param("log_dist") == "log_dist.F430M"


def test_dynamic_fit_keys_per_exposure():
    fit = _stub(DynamicResolvedFit, key="exp_001", filter="F430M")
    assert fit.get_key("log_dist") == "exp_001_F430M"
    assert fit.map_param("log_dist") == "log_dist.exp_001_F430M"


def test_dynamic_keys_are_unique_across_exposures():
    a = _stub(DynamicResolvedFit, key="exp_001", filter="F430M")
    b = _stub(DynamicResolvedFit, key="exp_002", filter="F430M")
    assert a.get_key("log_dist") != b.get_key("log_dist")


def test_transformed_fit_inherits_resolved_keying():
    fit = _stub(TransformedResolvedFit, filter="F380M")
    assert fit.get_key("log_dist") == "F380M"
    assert fit.map_param("log_dist") == "log_dist.F380M"


def test_point_fit_keys_contrast_and_log_dist():
    fit = _stub(PointResolvedFit, filter="F480M")

    assert fit.get_key("contrast") == "F480M"
    assert fit.map_param("contrast") == "contrast.F480M"

    # log_dist keying falls through to ResolvedFit
    assert fit.get_key("log_dist") == "F480M"
    assert fit.map_param("log_dist") == "log_dist.F480M"


# -------------------------------------------------------------------- rotate


class _RotStub(_BaseResolvedFit):
    """Minimal carrier for `rotate` — it only needs `self.parang`."""

    def __init__(self, parang):
        self.parang = parang


@pytest.fixture
def compact_image():
    """Asymmetric image with a two-pixel border of zeros.

    `rotate` samples through `interp`, which zero-fills out-of-domain samples.
    Float32 rounding in the rotation matrix pushes boundary samples ~1 ULP
    outside the grid, so edge pixels are silently dropped — see
    `test_rotation_drops_boundary_pixels`. Keeping the support away from the
    border isolates the geometry from that fill behaviour. Odd size so the
    rotation centre lands on a pixel.
    """
    noise = jr.uniform(jr.PRNGKey(0), (11, 11))
    return np.zeros((11, 11)).at[2:-2, 2:-2].set(noise[2:-2, 2:-2])


def test_zero_rotation_is_identity(compact_image):
    out = _RotStub(0.0).rotate(compact_image)
    assert np.allclose(out, compact_image, atol=1e-5)


def test_180_degree_rotation_reverses_both_axes(compact_image):
    # 180 degrees maps the pixel grid exactly onto itself, so interpolation is
    # exact and the direction convention doesn't matter.
    out = _RotStub(180.0).rotate(compact_image)
    assert np.allclose(out, compact_image[::-1, ::-1], atol=1e-5)


def test_full_turn_is_identity(compact_image):
    out = _RotStub(360.0).rotate(compact_image)
    assert np.allclose(out, compact_image, atol=1e-5)


def test_90_degree_rotation_is_a_lattice_rotation(compact_image):
    out = _RotStub(90.0).rotate(compact_image)
    matches_ccw = np.allclose(out, np.rot90(compact_image, 1), atol=1e-5)
    matches_cw = np.allclose(out, np.rot90(compact_image, -1), atol=1e-5)
    # NOTE: once the parang sign convention is confirmed by hand, pin this to
    # the single correct direction so a sign flip upstream fails the suite.
    assert matches_ccw or matches_cw


def test_symmetric_image_is_rotation_invariant():
    # Bilinear interpolation error scales as h^2 * f'' / 8, so the test image
    # has to be broad to keep it small: a sigma-2 Gaussian costs ~0.03 at the
    # peak, which swamps any real error. A wide raised-cosine bump is smooth
    # enough (~1e-3) and compactly supported, so the boundary zero-fill can't
    # contribute either.
    coords = np.arange(55) - 27.0
    x, y = np.meshgrid(coords, coords)
    r = np.hypot(x, y)
    img = np.where(r < 24.0, 0.5 * (1 + np.cos(np.pi * r / 24.0)), 0.0)

    out = _RotStub(37.0).rotate(img)
    assert np.allclose(out, img, atol=5e-3)  # peak is 1.0, so 0.5% of peak


def test_rotation_drops_boundary_pixels():
    """Documents current behaviour, not desired behaviour.

    Float32 rounding puts boundary samples just outside the interpolation
    domain and `interp` zero-fills them, so rotation loses flux from the
    outermost ring in an angle-dependent way. If `rotate` is fixed to pad or
    to clamp at the boundary, this test should start failing.
    """
    img = np.ones((7, 7))
    out = _RotStub(180.0).rotate(img)

    assert np.allclose(out[2:-2, 2:-2], 1.0)  # interior is untouched
    assert np.min(out) == 0.0  # but some edge pixels are gone
    assert np.sum(out) < np.sum(img)


def test_clip_enforces_positivity():
    img = jr.normal(jr.PRNGKey(1), (7, 7))  # contains negatives
    assert np.min(img) < 0

    clipped = _RotStub(30.0).rotate(img, clip=True)
    unclipped = _RotStub(30.0).rotate(img, clip=False)

    assert np.min(clipped) >= 0.0
    assert np.min(unclipped) < 0.0


# ------------------------------------------------------------------ simulate


class _FakeImage:
    def __init__(self, tag="interferogram", downsampled_by=None):
        self.tag = tag
        self.downsampled_by = downsampled_by

    def downsample(self, factor):
        return _FakeImage(tag="downsampled", downsampled_by=factor)


class _FakeRamp:
    def __init__(self, data):
        self.data = data

    def set(self, name, value):
        assert name == "data"
        return _FakeRamp(value)


class _FakeModel:
    source_oversample = 3


class _SimStub(_BaseResolvedFit):
    """Records the pipeline order so `simulate` wiring can be checked."""

    def __init__(self, ramp_data):
        self.calls = []
        self.downsample_factor = None
        self._ramp_data = ramp_data

    def model_psf(self, model):
        self.calls.append("psf")
        return "psf"

    def model_interferogram(self, psf, model, **kwargs):
        self.calls.append("interferogram")
        assert psf == "psf"
        return _FakeImage()

    def model_illuminance(self, image, model):
        self.calls.append("illuminance")
        self.downsample_factor = image.downsampled_by
        return "illuminance"

    def model_ramp(self, illuminance, model):
        self.calls.append("ramp")
        assert illuminance == "illuminance"
        return _FakeRamp(self._ramp_data)

    def model_read(self, ramp, model):
        self.calls.append("read")
        return ramp


@pytest.fixture
def ramp_data():
    # (n_groups, y, x) — cumulative counts up the ramp.
    return np.cumsum(np.ones((4, 3, 3)), axis=0)


def test_simulate_calls_pipeline_in_order(ramp_data):
    fit = _SimStub(ramp_data)
    fit.simulate(_FakeModel())

    assert fit.calls == [
        "psf",
        "interferogram",
        "illuminance",
        "ramp",
        "read",
    ]


def test_simulate_downsamples_to_source_oversample(ramp_data):
    fit = _SimStub(ramp_data)
    fit.simulate(_FakeModel())
    assert fit.downsample_factor == _FakeModel.source_oversample


def test_simulate_returns_group_differences_by_default(ramp_data):
    out = _SimStub(ramp_data).simulate(_FakeModel())

    assert out.data.shape == (ramp_data.shape[0] - 1, 3, 3)
    assert np.allclose(out.data, np.diff(ramp_data, axis=0))


def test_simulate_returns_raw_ramp_when_asked(ramp_data):
    out = _SimStub(ramp_data).simulate(_FakeModel(), return_slopes=False)

    assert out.data.shape == ramp_data.shape
    assert np.allclose(out.data, ramp_data)


def test_base_model_interferogram_is_not_implemented():
    # Currently returns None silently; this documents that a subclass which
    # forgets to override it produces a None image rather than an error.
    assert _BaseResolvedFit().model_interferogram(None, None) is None