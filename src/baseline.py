"""
XGBoost baseline for threat detection.

Trained on the same per-obstacle feature set the bio-inspired model sees,
so the comparison is apples-to-apples. The baseline learns whatever
weighting best fits the ground-truth labels — without any biological prior.

Train / test split is done at the *scene* level: all obstacles in a given
scene go to either train or test, never both. This prevents the model from
cheating by memorizing scene-specific patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


FEATURES: list[str] = [
    "distance",
    "angle",
    "radial_velocity",
    "tangential_velocity",
    "size",
    "ttc",
    "loom_rate",
]


@dataclass
class BaselineResult:
    model: XGBClassifier
    train_auc: float
    test_auc: float
    train_scene_ids: np.ndarray
    test_scene_ids: np.ndarray


def scene_level_split(
    df: pd.DataFrame,
    test_frac: float = 0.2,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split scene IDs (not rows) into train/test so no scene crosses the boundary."""
    rng = np.random.default_rng(seed)
    scene_ids = df["scene_id"].unique()
    rng.shuffle(scene_ids)
    n_test = max(1, int(len(scene_ids) * test_frac))
    test_ids = set(scene_ids[:n_test])
    test_mask = df["scene_id"].isin(test_ids)
    return df.loc[~test_mask].copy(), df.loc[test_mask].copy()


def train_baseline(
    df: pd.DataFrame,
    features: Iterable[str] = FEATURES,
    test_frac: float = 0.2,
    seed: int = 0,
    **xgb_kwargs,
) -> BaselineResult:
    train_df, test_df = scene_level_split(df, test_frac=test_frac, seed=seed)

    X_train = train_df[list(features)].values
    y_train = train_df["is_threat"].astype(int).values
    X_test = test_df[list(features)].values
    y_test = test_df["is_threat"].astype(int).values

    default_params = dict(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        eval_metric="logloss",
    )
    default_params.update(xgb_kwargs)
    model = XGBClassifier(**default_params)
    model.fit(X_train, y_train)

    train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]) \
        if len(np.unique(y_train)) > 1 else float("nan")
    test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) \
        if len(np.unique(y_test)) > 1 else float("nan")

    return BaselineResult(
        model=model,
        train_auc=train_auc,
        test_auc=test_auc,
        train_scene_ids=np.asarray(sorted(train_df["scene_id"].unique())),
        test_scene_ids=np.asarray(sorted(test_df["scene_id"].unique())),
    )


def predict_salience(model: XGBClassifier, df: pd.DataFrame, features: Iterable[str] = FEATURES) -> np.ndarray:
    """Use the predicted probability of is_threat as the 'salience' signal
    for comparison with the bio-inspired model."""
    return model.predict_proba(df[list(features)].values)[:, 1]


def rank_within_scene(scores: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    """Rank within each scene; rank 1 = highest score."""
    ranks = np.zeros(len(df), dtype=int)
    for sid, idx in df.groupby("scene_id").groups.items():
        idx = np.asarray(list(idx))
        order = np.argsort(-scores[idx])
        for r, j in enumerate(order, start=1):
            ranks[idx[j]] = r
    return ranks


# ---------------------------------------------------------------------------
# Zero-shot naive baseline
# ---------------------------------------------------------------------------

class NaiveTTCModel:
    """Rank obstacles by TTC alone — no training, no features besides TTC.

    This is the floor: if the bio-inspired model can't beat it, the bio model
    isn't doing real work. Uses the same saturating-inverse transform as
    SalienceModel so scores are on a comparable [0,1] scale.
    """

    _ttc_tau: float = 2.0  # seconds — same as SalienceModel default

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return 1.0 / (1.0 + df["ttc"].values / self._ttc_tau)


# ---------------------------------------------------------------------------
# Generic sklearn baseline infrastructure
# ---------------------------------------------------------------------------

@dataclass
class SklearnBaselineResult:
    model: Any                  # fitted sklearn estimator or Pipeline
    train_auc: float
    test_auc: float
    train_scene_ids: np.ndarray
    test_scene_ids: np.ndarray


def train_sklearn_baseline(
    df: pd.DataFrame,
    estimator: Any,
    features: Iterable[str] = FEATURES,
    test_frac: float = 0.2,
    seed: int = 0,
) -> SklearnBaselineResult:
    """Fit any sklearn-compatible estimator on a scene-level split.

    The split uses the same RNG seed as `train_baseline` so all models
    see identical train/test scene sets when called with the same seed.
    """
    train_df, test_df = scene_level_split(df, test_frac=test_frac, seed=seed)

    X_train = train_df[list(features)].values
    y_train = train_df["is_threat"].astype(int).values
    X_test = test_df[list(features)].values
    y_test = test_df["is_threat"].astype(int).values

    estimator.fit(X_train, y_train)

    train_auc = roc_auc_score(y_train, estimator.predict_proba(X_train)[:, 1]) \
        if len(np.unique(y_train)) > 1 else float("nan")
    test_auc = roc_auc_score(y_test, estimator.predict_proba(X_test)[:, 1]) \
        if len(np.unique(y_test)) > 1 else float("nan")

    return SklearnBaselineResult(
        model=estimator,
        train_auc=train_auc,
        test_auc=test_auc,
        train_scene_ids=np.asarray(sorted(train_df["scene_id"].unique())),
        test_scene_ids=np.asarray(sorted(test_df["scene_id"].unique())),
    )


def predict_proba_salience(
    model: Any,
    df: pd.DataFrame,
    features: Iterable[str] = FEATURES,
) -> np.ndarray:
    """P(is_threat) from any model with predict_proba — works for sklearn and XGBoost."""
    return model.predict_proba(df[list(features)].values)[:, 1]


# ---------------------------------------------------------------------------
# Model factories (pre-configured estimators ready for train_sklearn_baseline)
# ---------------------------------------------------------------------------

def make_logistic_regression(seed: int = 0) -> Pipeline:
    """Scaled logistic regression — the key comparison to bio-inspired.

    Same structural form as the bio model (linear weighted sum of features),
    but weights learned from data rather than derived from neuroscience.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
    ])


def make_random_forest(seed: int = 0) -> RandomForestClassifier:
    """Random forest — sanity check that XGBoost isn't winning purely on tuning."""
    return RandomForestClassifier(n_estimators=200, random_state=seed)


def make_mlp(seed: int = 0) -> Pipeline:
    """Small MLP — shows that 'neural network' ≠ bio-like; it performs like XGBoost."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=seed)),
    ])


def lr_feature_importances(result: SklearnBaselineResult, features: list[str] = list(FEATURES)) -> dict[str, float]:
    """Return LR absolute coefficients normalized to sum=1 (comparable to bio weights)."""
    clf = result.model.named_steps["clf"]
    abs_coefs = np.abs(clf.coef_[0])
    normalized = abs_coefs / abs_coefs.sum()
    return dict(zip(features, normalized.tolist()))


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.synthetic import generate_dataset

    df = generate_dataset(n_scenes=200, seed=0)
    res = train_baseline(df)
    print(f"Train AUC: {res.train_auc:.3f}")
    print(f"Test  AUC: {res.test_auc:.3f}")
    print(f"Feature importances:")
    for feat, imp in sorted(zip(FEATURES, res.model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {feat:22s} {imp:.3f}")
