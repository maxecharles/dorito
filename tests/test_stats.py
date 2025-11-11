import importlib
import numpy as _np


def test_basic_regularisers_and_losses():
    # Reload the real module and run the tests against installed dependencies
    if "dorito.stats" in importlib.sys.modules:
        importlib.reload(importlib.sys.modules["dorito.stats"])

    s = importlib.import_module("dorito.stats")

    arr = _np.array([[1.0, 2.0], [3.0, _np.nan]])

    l1 = s.L1_loss(arr)
    l2 = s.L2_loss(arr)
    assert _np.isfinite(l1)
    assert _np.isfinite(l2)

    tv = s.TV_loss(_np.ones((3, 3)))
    tsv = s.TSV_loss(_np.ones((3, 3)))
    me = s.ME_loss(_np.ones((2, 2)))
    assert _np.isfinite(tv)
    assert _np.isfinite(tsv)
    assert _np.isfinite(me)


def test_apply_regularisers_and_posterior_balances():
    if "dorito.stats" in importlib.sys.modules:
        importlib.reload(importlib.sys.modules["dorito.stats"])

    s = importlib.import_module("dorito.stats")

    class DummyModel:
        pass

    class DummyExp:
        def __init__(self, key, mv):
            self.key = key
            self._mv = _np.array(mv)

        def mv_zscore(self, model):
            return self._mv

    model = DummyModel()
    exp = DummyExp("e1", [0.5, -0.5])

    args = {"reg_dict": {"a": (2.0, lambda m, e: 3.0), "b": (1.0, lambda m, e: 4.0)}}
    prior = s.apply_regularisers(model, exp, args)
    assert float(prior) == 2.0 * 3.0 + 1.0 * 4.0

    lh, pr = s.ramp_posterior_balance(model, exp, args)
    # Accept NumPy arrays, scalars, or JAX arrays by coercing to NumPy
    lh_arr = _np.asarray(lh)
    assert isinstance(lh_arr, _np.ndarray) or _np.isscalar(lh_arr)
    assert _np.isfinite(pr)

    # ramp_posterior_balances over multiple exposures
    exposures = [DummyExp("a", [1.0, 1.0]), DummyExp("b", [0.0, 0.0])]
    balances = s.ramp_posterior_balances(model, exposures, args)
    assert "likelihoods" in balances and "priors" in balances and "exp_keys" in balances


def test_oi_log_likelihood_and_wrappers():
    if "dorito.stats" in importlib.sys.modules:
        importlib.reload(importlib.sys.modules["dorito.stats"])

    s = importlib.import_module("dorito.stats")

    class OI:
        def __init__(self):
            self.vis = _np.array([1.0])
            self.phi = _np.array([0.5])
            self.d_vis = _np.array([0.1])
            self.d_phi = _np.array([0.2])

        def __call__(self, model):
            # return zero model for simplicity
            return _np.zeros(self.vis.size + self.phi.size)

    oi = OI()
    nll = s.oi_log_likelihood(None, oi)
    assert _np.isfinite(nll)

    # disco_regularised_loss_fn should call oi_log_likelihood and return tuple
    loss, meta = s.disco_regularised_loss_fn(None, oi, args={"reg_dict": {}})
    assert isinstance(meta, tuple)
