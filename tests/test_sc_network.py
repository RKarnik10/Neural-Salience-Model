"""Tests for the v3 architectural SC network. Run with: python -m pytest tests/"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from src.synthetic import generate_dataset
from src.baseline import NaiveTTCModel
from src.evaluation import evaluate
from src.sc_network import (
    CHANNEL_NAMES,
    IN_DIM,
    SCConfig,
    SCNetwork,
    _NumpySC,
    interaction_features,
    sc_channel_importances,
    superficial_channels,
    torch_available,
    train_sc_network,
)


# ---------- Feature plumbing ----------

def test_superficial_channels_shape_and_range():
    df = generate_dataset(n_scenes=10, seed=1)
    C = superficial_channels(df)
    assert C.shape == (len(df), 4)
    # Each SC transform is bounded to [0, 1].
    assert C.min() >= 0.0 and C.max() <= 1.0 + 1e-9


def test_interaction_features_dim():
    C = np.random.default_rng(0).random((7, 4))
    X = interaction_features(C)
    assert X.shape == (7, IN_DIM)          # 4 channels + 6 pairwise products
    # First 4 cols are the channels unchanged.
    assert np.allclose(X[:, :4], C)


# ---------- Gradient correctness (the heart of the NumPy backend) ----------

def test_numpy_backprop_matches_finite_differences():
    rng = np.random.default_rng(0)
    N = 12
    X = rng.normal(size=(N, IN_DIM))
    y = (rng.random(N) > 0.5).astype(float)
    t = np.zeros(N)
    g0 = np.array([0, 1, 2]); t[g0] = np.array([0.5, 0.3, 0.2])
    g1 = np.array([5, 6]);    t[g1] = np.array([0.7, 0.3])
    groups = [g0, g1]

    net = _NumpySC(SCConfig(hidden=6, lam=1.0, weight_decay=1e-3, learn_tau=True))
    _, grads = net._loss_and_grads(X, y, t, groups)

    eps = 1e-6
    max_rel = 0.0
    for name in ["W1", "b1", "w2", "b2", "tau_raw"]:
        P = getattr(net, name)
        ga = grads[name]
        if np.ndim(P) == 0:
            base = float(P)
            setattr(net, name, base + eps); lp, _ = net._loss_and_grads(X, y, t, groups)
            setattr(net, name, base - eps); lm, _ = net._loss_and_grads(X, y, t, groups)
            setattr(net, name, base)
            gn = (lp - lm) / (2 * eps)
            max_rel = max(max_rel, abs(gn - ga) / (abs(gn) + abs(ga) + 1e-9))
        else:
            Pf, gaf = P.ravel(), np.asarray(ga).ravel()
            for i in range(len(Pf)):
                old = Pf[i]
                Pf[i] = old + eps; lp, _ = net._loss_and_grads(X, y, t, groups)
                Pf[i] = old - eps; lm, _ = net._loss_and_grads(X, y, t, groups)
                Pf[i] = old
                gn = (lp - lm) / (2 * eps)
                max_rel = max(max_rel, abs(gn - gaf[i]) / (abs(gn) + abs(gaf[i]) + 1e-9))
    assert max_rel < 1e-4, f"gradient check failed, max rel err = {max_rel:.2e}"


# ---------- Training behavior ----------

def test_fit_reduces_loss():
    df = generate_dataset(n_scenes=60, seed=2)
    model = SCNetwork(SCConfig(epochs=200, seed=2), backend="numpy").fit(df)
    hist = model._impl.loss_history
    assert hist[-1] < hist[0], "training loss should decrease"


def test_predict_returns_probabilities():
    df = generate_dataset(n_scenes=40, seed=3)
    model = SCNetwork(SCConfig(epochs=150, seed=3), backend="numpy").fit(df)
    p = model.predict(df)
    assert p.shape == (len(df),)
    assert np.all(np.isfinite(p))
    assert (p >= 0).all() and (p <= 1).all()


def test_sc_network_beats_naive_floor():
    """The whole point of adding architecture: it must clear the zero-shot floor."""
    df = generate_dataset(n_scenes=200, seed=0)
    res = train_sc_network(df, backend="numpy", seed=0)
    test_ids = set(res.test_scene_ids.tolist())
    test_df = df[df["scene_id"].isin(test_ids)].reset_index(drop=True)

    sc_ndcg = evaluate(test_df, res.model.predict(test_df)).ndcg_at_3
    naive_ndcg = evaluate(test_df, NaiveTTCModel().predict(test_df)).ndcg_at_3
    assert sc_ndcg > naive_ndcg, f"SC-Net NDCG {sc_ndcg:.3f} should beat naive {naive_ndcg:.3f}"


def test_channel_importances_normalized():
    df = generate_dataset(n_scenes=80, seed=4)
    res = train_sc_network(df, backend="numpy", seed=4)
    imp = sc_channel_importances(res.model, df)
    assert set(imp) == set(CHANNEL_NAMES)
    assert abs(sum(imp.values()) - 1.0) < 1e-6


def test_learned_tau_is_positive():
    df = generate_dataset(n_scenes=50, seed=5)
    res = train_sc_network(df, backend="numpy", seed=5)
    assert res.model.tau > 0.0


# ---------- Backend parity (only runs if torch is installed) ----------

@pytest.mark.skipif(not torch_available(), reason="PyTorch not installed")
def test_numpy_and_torch_backends_agree():
    """Same architecture + config + seed: the two backends should land close.

    They will not be bit-identical (different optimizers/init RNG), but a well-
    trained pair should rank obstacles very similarly."""
    df = generate_dataset(n_scenes=150, seed=6)
    cfg = lambda: SCConfig(epochs=400, seed=6)
    np_model = SCNetwork(cfg(), backend="numpy").fit(df)
    pt_model = SCNetwork(cfg(), backend="torch").fit(df)
    from scipy.stats import spearmanr
    rho, _ = spearmanr(np_model.predict(df), pt_model.predict(df))
    assert rho > 0.9, f"backends disagree: score rank correlation only {rho:.3f}"


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception:
            print(f"  FAIL  {t.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
