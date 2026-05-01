"""
Bio-inspired spatial salience model.

Motivated by properties of the mammalian superior colliculus (SC), which
integrates spatial signals to orient attention toward the most threatening
stimulus. Key SC-derived properties we mimic:

    1. Looming sensitivity. Deep SC neurons respond selectively to objects
       on collision course (rapidly expanding in angular size).
    2. Proximity priority. Closer objects dominate attention, approximately
       following an inverse-distance law (cf. spatial receptive-field density).
    3. Forward / heading bias. SC is retinotopic with foveal over-representation;
       for a navigator, this maps onto "things in your path matter most".
    4. Winner-take-all dynamics. Lateral inhibition in the intermediate layers
       collapses many candidate signals onto a single dominant one; we model
       this via a temperature-scaled softmax.

The weighting coefficients are intentionally exposed and hand-set. That is
the whole point — this model is interpretable and falsifiable, unlike the
XGBoost baseline it will be compared against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SalienceWeights:
    """Component weights in the linear combination of bio-inspired scores.

    Defaults were chosen so looming dominates (it's the SC's defining feature)
    with secondary contributions from proximity and forward-bias. Tune freely.
    """
    w_loom: float = 0.45
    w_proximity: float = 0.25
    w_ttc: float = 0.20
    w_forward: float = 0.10

    # Non-linear shaping parameters
    proximity_scale: float = 2.0      # meters — below this, proximity saturates high
    forward_sigma: float = 0.9        # radians — Gaussian bandwidth of forward bias
    ttc_tau: float = 2.0              # seconds — TTC score half-max
    loom_cap: float = 5.0             # saturation cap on raw loom_rate
    wta_temperature: float = 0.35     # softmax temperature for winner-take-all
                                      # lower = sharper / more WTA


class SalienceModel:
    """Hand-crafted biologically-inspired salience scorer."""

    def __init__(self, weights: SalienceWeights | None = None):
        self.weights = weights or SalienceWeights()

    # ---- Individual feature transforms ----

    def _proximity_score(self, distance: np.ndarray) -> np.ndarray:
        # Saturating inverse: 1 / (1 + d / scale). Close obstacles -> ~1, far -> ~0.
        return 1.0 / (1.0 + distance / self.weights.proximity_scale)

    def _forward_score(self, angle: np.ndarray) -> np.ndarray:
        # Gaussian centered on heading direction (angle = 0).
        return np.exp(-0.5 * (angle / self.weights.forward_sigma) ** 2)

    def _ttc_score(self, ttc: np.ndarray) -> np.ndarray:
        # Urgency: 1 / (1 + ttc / tau). Imminent -> ~1, far-future -> ~0.
        # Non-approaching obstacles are given ttc = large sentinel upstream, so score -> 0.
        return 1.0 / (1.0 + ttc / self.weights.ttc_tau)

    def _loom_score(self, loom_rate: np.ndarray) -> np.ndarray:
        # Normalize and saturate. Loom rate is positive only for approaching obstacles.
        capped = np.clip(loom_rate, 0.0, self.weights.loom_cap)
        return capped / self.weights.loom_cap

    # ---- Public API ----

    def raw_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute each per-obstacle salience component separately, for
        interpretability and inspection."""
        w = self.weights
        components = pd.DataFrame({
            "proximity": self._proximity_score(df["distance"].values),
            "forward":   self._forward_score(df["angle"].values),
            "ttc":       self._ttc_score(df["ttc"].values),
            "loom":      self._loom_score(df["loom_rate"].values),
        })
        components["salience"] = (
            w.w_loom * components["loom"]
            + w.w_proximity * components["proximity"]
            + w.w_ttc * components["ttc"]
            + w.w_forward * components["forward"]
        )
        return components

    def predict(self, df: pd.DataFrame, apply_wta: bool = False) -> np.ndarray:
        """Return a salience score per row. If `apply_wta`, apply winner-take-all
        softmax *within each scene* (requires a 'scene_id' column)."""
        raw = self.raw_scores(df)["salience"].values
        if not apply_wta:
            return raw
        if "scene_id" not in df.columns:
            raise ValueError("apply_wta=True requires a 'scene_id' column in df")
        out = np.empty_like(raw)
        for sid, idx in df.groupby("scene_id").groups.items():
            idx = np.asarray(list(idx))
            s = raw[idx]
            s = s / self.weights.wta_temperature
            s = s - s.max()                        # numerical stability
            e = np.exp(s)
            out[idx] = e / e.sum()
        return out

    def rank_within_scene(self, df: pd.DataFrame) -> np.ndarray:
        """Return rank per row (1 = most salient) within each scene."""
        scores = self.predict(df)
        ranks = np.zeros(len(df), dtype=int)
        for sid, idx in df.groupby("scene_id").groups.items():
            idx = np.asarray(list(idx))
            # argsort descending -> rank 1 is largest score
            order = np.argsort(-scores[idx])
            for r, j in enumerate(order, start=1):
                ranks[idx[j]] = r
        return ranks


if __name__ == "__main__":
    # Smoke test
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.synthetic import generate_dataset

    df = generate_dataset(n_scenes=3, seed=7)
    model = SalienceModel()
    components = model.raw_scores(df)
    print("Component scores (first 10 rows):")
    print(components.head(10).round(3))
    print(f"\nSalience range: [{components['salience'].min():.3f}, {components['salience'].max():.3f}]")
