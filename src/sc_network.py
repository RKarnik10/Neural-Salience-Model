"""
v3 — Architectural Superior-Colliculus (SC) network.

v2 established that the SC's *feature weighting* alone (four fixed numbers in one
weighted sum) is a competitive zero-shot linear baseline. The v2 analysis then
made a specific prediction: the gap to the tree models is almost entirely
**feature interactions** ("this AND that" conditions), and the one biological
signature no model in the ladder has yet is **within-scene competition**
(lateral inhibition / winner-take-all).

This module tests that prediction with a small network whose architecture maps
onto SC anatomy rather than a generic MLP:

    Superficial sensory layer   (fixed)
        The four SC-derived receptive-field transforms from `SalienceModel`
        (loom, proximity, ttc, forward). These are *structural* in biology, so
        we keep them fixed — the learning happens downstream. This also isolates
        the thing we are testing: does adding integration + competition on top of
        the SAME bio features close the gap?

    Intermediate integration layer   (learned)
        Takes the four channels PLUS their pairwise products (explicit
        coincidence-detection / multiplicative gain terms) and passes them
        through a small tanh hidden layer. This is where interactions live.

    Winner-take-all output   (learned)
        A within-scene softmax at a learnable temperature. Expressed as a
        listwise training objective (compete to attend the most-urgent obstacle),
        so competition shapes the learned representation — not just an
        inference-time cosmetic.

Two interchangeable backends are provided and share one architecture, config,
loss, and public interface:

    * NumPy  — forward pass + hand-derived backprop + a small Adam optimizer,
               no heavy dependency. This is the default and always available.
    * PyTorch — the same network as an ``nn.Module`` with autograd. Used
                automatically when ``torch`` is importable, or on request.

Pick with ``SCNetwork(backend="auto" | "numpy" | "torch")``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from src.baseline import scene_level_split
from src.salience import SalienceModel, SalienceWeights

# Channel order used everywhere below. Matches the SalienceModel weight names.
CHANNEL_NAMES: list[str] = ["loom", "proximity", "ttc", "forward"]

# Pairwise interaction terms among the four channels (i < j): 6 products.
_PAIRS: list[tuple[int, int]] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

# ---- optional torch backend -------------------------------------------------
try:  # pragma: no cover - exercised only when torch is installed
    import torch

    _HAS_TORCH = True
except Exception:  # torch is an optional dependency
    _HAS_TORCH = False


def torch_available() -> bool:
    """True if the PyTorch backend can be used in this environment."""
    return _HAS_TORCH


# ---------------------------------------------------------------------------
# Superficial layer + interaction features (shared by both backends)
# ---------------------------------------------------------------------------

def superficial_channels(df: pd.DataFrame, weights: SalienceWeights | None = None) -> np.ndarray:
    """Fixed SC receptive-field transforms → 4 channels, ordered as CHANNEL_NAMES.

    Reuses `SalienceModel`'s per-feature transforms so the superficial layer is
    literally the same biology the v2 bio model uses.
    """
    comp = SalienceModel(weights).raw_scores(df)
    return np.stack(
        [comp["loom"].values, comp["proximity"].values,
         comp["ttc"].values, comp["forward"].values],
        axis=1,
    ).astype(float)


def interaction_features(channels: np.ndarray) -> np.ndarray:
    """[4 channels | 6 pairwise products] → (N, 10) integration-layer input."""
    prods = np.stack([channels[:, i] * channels[:, j] for (i, j) in _PAIRS], axis=1)
    return np.concatenate([channels, prods], axis=1)


IN_DIM = 4 + len(_PAIRS)  # 10


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SCConfig:
    hidden: int = 8            # intermediate integration layer width
    lam: float = 1.0           # weight on the winner-take-all (listwise) loss
    epochs: int = 500
    lr: float = 0.05
    weight_decay: float = 1e-4
    tau_init: float = 0.5      # WTA softmax temperature (lower = sharper competition)
    learn_tau: bool = True
    seed: int = 0


# ---------------------------------------------------------------------------
# NumPy backend — forward, hand-derived backprop, Adam
# ---------------------------------------------------------------------------

class _NumpySC:
    """Two-layer integration network with a within-scene WTA loss.

    Params: W1 (H, D), b1 (H,), w2 (H,), b2 (scalar), tau_raw (scalar; tau = exp(raw)).
    """

    def __init__(self, config: SCConfig):
        self.cfg = config
        rng = np.random.default_rng(config.seed)
        H, D = config.hidden, IN_DIM
        # Small random init; He-ish scale for tanh.
        self.W1 = rng.normal(0.0, np.sqrt(1.0 / D), size=(H, D))
        self.b1 = np.zeros(H)
        self.w2 = rng.normal(0.0, np.sqrt(1.0 / H), size=H)
        self.b2 = 0.0
        self.tau_raw = float(np.log(max(config.tau_init, 1e-3)))
        self._tau_floor = np.log(0.05)

    # ---- forward ----

    def _forward(self, X: np.ndarray):
        A1 = X @ self.W1.T + self.b1        # (N, H)
        Hh = np.tanh(A1)
        z = Hh @ self.w2 + self.b2          # (N,)
        return z, Hh

    def predict_z(self, X: np.ndarray) -> np.ndarray:
        return self._forward(X)[0]

    def predict_p(self, X: np.ndarray) -> np.ndarray:
        return expit(self.predict_z(X))

    # ---- loss + gradients ----

    def _loss_and_grads(self, X, y, t, groups):
        """Return (total_loss, grads dict). `t` is per-row normalized relevance
        (0 outside threat scenes); `groups` is a list of row-index arrays for
        scenes that contain at least one threat."""
        cfg = self.cfg
        N = len(y)
        eps = 1e-7
        z, Hh = self._forward(X)

        # --- BCE on is_threat (obstacle-level threat discrimination) ---
        p = np.clip(expit(z), eps, 1 - eps)
        L_bce = float(np.mean(-y * np.log(p) - (1 - y) * np.log(1 - p)))
        dz = (p - y) / N

        # --- Winner-take-all listwise loss (within-scene competition) ---
        tau = float(np.exp(self.tau_raw))
        L_list = 0.0
        dtau = 0.0
        G = max(len(groups), 1)
        for idx in groups:
            zi = z[idx]
            ti = t[idx]
            u = zi / tau
            u = u - u.max()
            e = np.exp(u)
            q = e / e.sum()
            L_list += float(-np.sum(ti * np.log(q + eps)))
            g = (q - ti) / tau
            dz[idx] += cfg.lam * g / G
            dtau += float(-np.sum((q - ti) * zi) / (tau * tau) / G)
        L_list = L_list / G

        # --- backprop from dz to params ---
        dw2 = Hh.T @ dz
        db2 = float(dz.sum())
        dHh = np.outer(dz, self.w2)
        dA1 = dHh * (1.0 - Hh ** 2)
        dW1 = dA1.T @ X
        db1 = dA1.sum(axis=0)

        # weight decay (on the weight matrices only)
        dW1 += cfg.weight_decay * self.W1
        dw2 += cfg.weight_decay * self.w2

        dtau_raw = cfg.lam * dtau * tau if cfg.learn_tau else 0.0

        reg = 0.5 * cfg.weight_decay * (float(np.sum(self.W1 ** 2)) + float(np.sum(self.w2 ** 2)))
        total = L_bce + cfg.lam * L_list + reg
        grads = {"W1": dW1, "b1": db1, "w2": dw2, "b2": db2, "tau_raw": dtau_raw}
        return total, grads

    # ---- training ----

    def fit(self, X, y, t, groups):
        cfg = self.cfg
        params = {"W1": self.W1, "b1": self.b1, "w2": self.w2, "b2": self.b2, "tau_raw": self.tau_raw}
        m = {k: np.zeros_like(np.asarray(v, dtype=float)) for k, v in params.items()}
        v = {k: np.zeros_like(np.asarray(vv, dtype=float)) for k, vv in params.items()}
        b1_, b2_, epsA = 0.9, 0.999, 1e-8
        self.loss_history: list[float] = []
        for step in range(1, cfg.epochs + 1):
            loss, grads = self._loss_and_grads(X, y, t, groups)
            self.loss_history.append(loss)
            for k in params:
                g = grads[k]
                m[k] = b1_ * m[k] + (1 - b1_) * g
                v[k] = b2_ * v[k] + (1 - b2_) * (np.asarray(g) ** 2)
                mhat = m[k] / (1 - b1_ ** step)
                vhat = v[k] / (1 - b2_ ** step)
                upd = cfg.lr * mhat / (np.sqrt(vhat) + epsA)
                if k == "b2":
                    self.b2 = float(self.b2 - float(upd))
                elif k == "tau_raw":
                    self.tau_raw = float(np.clip(self.tau_raw - float(upd), self._tau_floor, 5.0))
                else:
                    getattr(self, k)[...] = getattr(self, k) - upd
        return self


# ---------------------------------------------------------------------------
# PyTorch backend — same architecture and loss via autograd
# ---------------------------------------------------------------------------

if _HAS_TORCH:  # pragma: no cover - exercised only when torch is installed

    class _TorchSC:
        def __init__(self, config: SCConfig):
            self.cfg = config
            g = torch.Generator().manual_seed(config.seed)
            H, D = config.hidden, IN_DIM
            self.W1 = (torch.randn(H, D, generator=g) * (1.0 / D) ** 0.5).requires_grad_(True)
            self.b1 = torch.zeros(H, requires_grad=True)
            self.w2 = (torch.randn(H, generator=g) * (1.0 / H) ** 0.5).requires_grad_(True)
            self.b2 = torch.zeros(1, requires_grad=True)
            self.tau_raw = torch.tensor(
                float(np.log(max(config.tau_init, 1e-3))), requires_grad=config.learn_tau
            )

        def _params(self):
            ps = [self.W1, self.b1, self.w2, self.b2]
            if self.cfg.learn_tau:
                ps.append(self.tau_raw)
            return ps

        def _z(self, X):
            return torch.tanh(X @ self.W1.T + self.b1) @ self.w2 + self.b2

        def fit(self, X, y, t, groups):
            cfg = self.cfg
            Xt = torch.as_tensor(X, dtype=torch.float32)
            yt = torch.as_tensor(y, dtype=torch.float32)
            tt = torch.as_tensor(t, dtype=torch.float32)
            opt = torch.optim.Adam(self._params(), lr=cfg.lr, weight_decay=0.0)
            bce = torch.nn.BCEWithLogitsLoss()
            self.loss_history = []
            for _ in range(cfg.epochs):
                opt.zero_grad()
                z = self._z(Xt)
                loss = bce(z, yt)
                tau = torch.exp(self.tau_raw.clamp(min=float(np.log(0.05)), max=5.0))
                if groups:
                    l_list = z.new_zeros(())
                    for idx in groups:
                        zi, ti = z[idx], tt[idx]
                        logq = torch.log_softmax(zi / tau, dim=0)
                        l_list = l_list - (ti * logq).sum()
                    l_list = l_list / len(groups)
                    loss = loss + cfg.lam * l_list
                reg = 0.5 * cfg.weight_decay * (self.W1.pow(2).sum() + self.w2.pow(2).sum())
                (loss + reg).backward()
                opt.step()
                self.loss_history.append(float(loss.detach()))
            return self

        def predict_z(self, X):
            with torch.no_grad():
                return self._z(torch.as_tensor(X, dtype=torch.float32)).numpy()

        def predict_p(self, X):
            with torch.no_grad():
                return torch.sigmoid(self._z(torch.as_tensor(X, dtype=torch.float32))).numpy()


# ---------------------------------------------------------------------------
# Unified wrapper
# ---------------------------------------------------------------------------

def _resolve_backend(backend: str) -> str:
    if backend == "numpy":
        return "numpy"
    if backend == "torch":
        if not _HAS_TORCH:
            raise RuntimeError("backend='torch' requested but PyTorch is not installed.")
        return "torch"
    if backend == "auto":
        return "torch" if _HAS_TORCH else "numpy"
    raise ValueError(f"unknown backend {backend!r}; use 'auto', 'numpy', or 'torch'")


class SCNetwork:
    """Architectural SC network with pluggable NumPy / PyTorch backend.

    Interface deliberately takes a DataFrame (it needs `scene_id` for the WTA
    competition and the raw features for the superficial transforms), so it does
    NOT share sklearn's `predict_proba(X)` signature. Call `.fit(df)` / `.predict(df)`.
    """

    def __init__(self, config: SCConfig | None = None, backend: str = "auto",
                 weights: SalienceWeights | None = None):
        self.config = config or SCConfig()
        self.backend = _resolve_backend(backend)
        self._weights = weights or SalienceWeights()
        self._impl: Any = None
        self._x_mean: np.ndarray | None = None
        self._x_std: np.ndarray | None = None

    # ---- feature plumbing ----

    def _featurize(self, df: pd.DataFrame) -> np.ndarray:
        return interaction_features(superficial_channels(df, self._weights))

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self._x_mean) / self._x_std

    @staticmethod
    def _targets(df: pd.DataFrame) -> tuple[np.ndarray, list[np.ndarray]]:
        """Per-row normalized relevance (0 outside threat scenes) + the list of
        row-index arrays for scenes that contain a threat."""
        df = df.reset_index(drop=True)
        rank = df["threat_rank"].values
        t = np.zeros(len(df), dtype=float)
        groups: list[np.ndarray] = []
        for _, idx in df.groupby("scene_id").groups.items():
            idx = np.asarray(list(idx))
            r = rank[idx]
            mask = r > 0
            if not mask.any():
                continue
            rel = np.where(mask, r.max() - r + 1, 0).astype(float)
            s = rel.sum()
            if s > 0:
                t[idx] = rel / s
                groups.append(idx)
        return t, groups

    # ---- public API ----

    def fit(self, df: pd.DataFrame) -> "SCNetwork":
        df = df.reset_index(drop=True)
        X = self._featurize(df)
        self._x_mean = X.mean(axis=0)
        self._x_std = X.std(axis=0) + 1e-8
        Xs = self._standardize(X)
        y = df["is_threat"].astype(int).values.astype(float)
        t, groups = self._targets(df)
        impl = _TorchSC(self.config) if self.backend == "torch" else _NumpySC(self.config)
        impl.fit(Xs, y, t, groups)
        self._impl = impl
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """P(is_threat) per obstacle — the salience score used for evaluation."""
        return self._impl.predict_p(self._standardize(self._featurize(df)))

    def predict_z(self, df: pd.DataFrame) -> np.ndarray:
        """Pre-competition salience logit per obstacle."""
        return self._impl.predict_z(self._standardize(self._featurize(df)))

    @property
    def tau(self) -> float:
        """Learned winner-take-all temperature (lower = sharper competition)."""
        return float(np.exp(float(self._impl.tau_raw)))


# ---------------------------------------------------------------------------
# Training helper + introspection (mirrors baseline.SklearnBaselineResult)
# ---------------------------------------------------------------------------

@dataclass
class SCNetworkResult:
    model: SCNetwork
    train_auc: float
    test_auc: float
    train_scene_ids: np.ndarray
    test_scene_ids: np.ndarray
    backend: str


def train_sc_network(
    df: pd.DataFrame,
    backend: str = "auto",
    config: SCConfig | None = None,
    test_frac: float = 0.2,
    seed: int = 0,
) -> SCNetworkResult:
    """Fit the SC network on the same scene-level split the other models use.

    Uses `scene_level_split` with the same seed, so when called with the seed the
    baselines use, the SC network trains and tests on identical scene sets.
    """
    cfg = config or SCConfig(seed=seed)
    train_df, test_df = scene_level_split(df, test_frac=test_frac, seed=seed)
    model = SCNetwork(cfg, backend=backend).fit(train_df)

    def _auc(d):
        y = d["is_threat"].astype(int).values
        return roc_auc_score(y, model.predict(d)) if len(np.unique(y)) > 1 else float("nan")

    return SCNetworkResult(
        model=model,
        train_auc=float(_auc(train_df)),
        test_auc=float(_auc(test_df)),
        train_scene_ids=np.asarray(sorted(train_df["scene_id"].unique())),
        test_scene_ids=np.asarray(sorted(test_df["scene_id"].unique())),
        backend=model.backend,
    )


def sc_channel_importances(model: SCNetwork, df: pd.DataFrame, seed: int = 0) -> dict[str, float]:
    """Permutation importance over the four SC channels, normalized to sum=1.

    Shuffles each channel (rebuilding its interaction terms) and measures the
    mean absolute change in predicted salience. Directly comparable to the fixed
    bio-inspired channel weights.
    """
    base = model.predict(df)
    C = superficial_channels(df, model._weights)
    rng = np.random.default_rng(seed)
    imps = []
    for k in range(len(CHANNEL_NAMES)):
        Cp = C.copy()
        Cp[:, k] = Cp[rng.permutation(len(Cp)), k]
        Xp = model._standardize(interaction_features(Cp))
        imps.append(float(np.mean(np.abs(model._impl.predict_p(Xp) - base))))
    imp = np.asarray(imps)
    if imp.sum() > 0:
        imp = imp / imp.sum()
    return dict(zip(CHANNEL_NAMES, imp.tolist()))


if __name__ == "__main__":
    import sys, os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.synthetic import generate_dataset

    df = generate_dataset(n_scenes=200, seed=0)
    res = train_sc_network(df, backend="auto", seed=0)
    print(f"backend   : {res.backend}")
    print(f"train AUC : {res.train_auc:.3f}")
    print(f"test  AUC : {res.test_auc:.3f}")
    print(f"tau       : {res.model.tau:.3f}")
    print("channel importances:", {k: round(v, 3) for k, v in sc_channel_importances(res.model, df).items()})
