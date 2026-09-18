# import importlib


# def test_model_fits_basic_smoke():
#     """Basic smoke test for the `dorito.model_fits` module.

#     This test only asserts the presence of the main ModelFit-derived
#     classes. We intentionally avoid exercising `initialise_params` here
#     (it requires more detailed model state) — the previous deeper tests
#     that called `initialise_params` have been removed and replaced by
#     this lightweight smoke check.
#     """
#     if "dorito.model_fits" in importlib.sys.modules:
#         importlib.reload(importlib.sys.modules["dorito.model_fits"])

#     mod = importlib.import_module("dorito.model_fits")

#     assert hasattr(mod, "ResolvedFit")
#     assert hasattr(mod, "ResolvedOIFit")
#     assert hasattr(mod, "TransformedResolvedFit")
#     assert hasattr(mod, "PointResolvedFit")
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

    The key-routing methods only read `self.filter` / `self.key`, both of
    which are read-only properties on ModelFit. We make a throwaway subclass
    whose class-level attributes shadow those properties, then allocate it
    without __init__ (which would want a real file and optics model).
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
def asymmetric_image():
    # Odd size so the rotation centre lands on a pixel.
    return jr.uniform(jr.PRNGKey(0), (7, 7))


def test_zero_rotation_is_identity(asymmetric_image):
    out = _RotStub(0.0).rotate(asymmetric_image)
    assert np.allclose(out, asymmetric_image, atol=1e-5)


def test_180_degree_rotation_reverses_both_axes(asymmetric_image):
    # 180 degrees maps the pixel grid exactly onto itself, so interpolation is
    # exact and the direction convention doesn't matter.
    out = _RotStub(180.0).rotate(asymmetric_image)
    assert np.allclose(out, asymmetric_image[::-1, ::-1], atol=1e-5)


def test_full_turn_is_identity(asymmetric_image):
    out = _RotStub(360.0).rotate(asymmetric_image)
    assert np.allclose(out, asymmetric_image, atol=1e-5)


def test_symmetric_image_is_rotation_invariant():
    coords = np.arange(9) - 4.0
    x, y = np.meshgrid(coords, coords)
    img = np.exp(-(x**2 + y**2) / 8.0)

    out = _RotStub(37.0).rotate(img)
    assert np.allclose(out, img, atol=1e-3)


def test_90_degree_rotation_is_a_lattice_rotation(asymmetric_image):
    out = _RotStub(90.0).rotate(asymmetric_image)
    matches_ccw = np.allclose(out, np.rot90(asymmetric_image, 1), atol=1e-5)
    matches_cw = np.allclose(out, np.rot90(asymmetric_image, -1), atol=1e-5)
    # NOTE: once the parang sign convention is confirmed by hand, pin this to
    # the single correct direction so a sign flip upstream fails the suite.
    assert matches_ccw or matches_cw


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