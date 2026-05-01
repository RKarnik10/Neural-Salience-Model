"""
Evaluation metrics for comparing salience models.

Two models producing per-obstacle scores can be compared along several axes:

    1. Threat discrimination (AUC on is_threat) — can it tell a threat from a non-threat?
    2. Ranking quality (NDCG@k, top-1 accuracy) — does it put the RIGHT threat first?
    3. Calibration with ground-truth urgency (Spearman correlation with -threat_time,
       restricted to actual threats) — does it agree on how urgent each threat is?

Ranking quality matters most for the SC-inspired framing: the superior colliculus
doesn't classify threats, it orients attention. Getting #1 right is the task.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr


@dataclass
class ScoreReport:
    auc: float
    top1_accuracy: float        # fraction of scenes whose top-ranked obstacle matches GT top-1
    ndcg_at_3: float
    urgency_spearman: float     # correlation with true urgency on actual threats

    def pretty(self, label: str = "") -> str:
        header = f"{label}" if label else ""
        return (
            f"{header:<20s}  AUC={self.auc:.3f}   top-1={self.top1_accuracy:.3f}   "
            f"NDCG@3={self.ndcg_at_3:.3f}   urgency-ρ={self.urgency_spearman:+.3f}"
        )


def _ndcg_at_k(true_ranks: np.ndarray, pred_scores: np.ndarray, k: int = 3) -> float:
    """Compute NDCG@k for a single scene.

    true_ranks: integer ground-truth ranks, 1 = most urgent threat; 0 = non-threat.
    pred_scores: model scores (higher = more salient).

    Relevance is defined as (N - rank + 1) for threats (rank >= 1), 0 for non-threats.
    """
    n_obs = len(true_ranks)
    # Relevance grades
    relevance = np.where(true_ranks > 0, (true_ranks.max() - true_ranks + 1), 0).astype(float)
    if relevance.sum() == 0:
        return float("nan")  # no threats in scene, no ranking to evaluate

    # DCG of the model's ordering
    order = np.argsort(-pred_scores)[:k]
    dcg = np.sum(relevance[order] / np.log2(np.arange(2, len(order) + 2)))
    # Ideal DCG
    ideal_order = np.argsort(-relevance)[:k]
    idcg = np.sum(relevance[ideal_order] / np.log2(np.arange(2, len(ideal_order) + 2)))
    return dcg / idcg if idcg > 0 else float("nan")


def evaluate(df: pd.DataFrame, scores: np.ndarray) -> ScoreReport:
    """Evaluate per-obstacle scores against ground truth.

    df must include: is_threat, threat_rank, threat_time, scene_id.
    scores: array of length len(df), same order as df rows.
    """
    df = df.reset_index(drop=True).copy()
    df["_score"] = scores

    # 1. Threat discrimination AUC (obstacle-level, across whole dataset)
    y = df["is_threat"].astype(int).values
    auc = roc_auc_score(y, scores) if len(np.unique(y)) > 1 else float("nan")

    # 2. Top-1 accuracy (scene-level): in scenes that have >=1 threat, does the
    #    top-scored obstacle match the ground-truth rank-1 threat?
    top1_hits, top1_total = 0, 0
    ndcgs: list[float] = []
    for sid, scene_df in df.groupby("scene_id"):
        if not scene_df["is_threat"].any():
            continue
        top1_total += 1
        gt_top = scene_df.loc[scene_df["threat_rank"] == 1].index[0]
        pred_top = scene_df["_score"].idxmax()
        if pred_top == gt_top:
            top1_hits += 1
        ndcg = _ndcg_at_k(
            scene_df["threat_rank"].values,
            scene_df["_score"].values,
            k=3,
        )
        if not np.isnan(ndcg):
            ndcgs.append(ndcg)
    top1_accuracy = top1_hits / top1_total if top1_total else float("nan")
    ndcg_at_3 = float(np.mean(ndcgs)) if ndcgs else float("nan")

    # 3. Urgency correlation: among threats, does score correlate with -threat_time?
    threat_df = df[df["is_threat"]]
    if len(threat_df) >= 3:
        rho, _ = spearmanr(threat_df["_score"].values, -threat_df["threat_time"].values)
        urgency_spearman = float(rho) if np.isfinite(rho) else float("nan")
    else:
        urgency_spearman = float("nan")

    return ScoreReport(
        auc=float(auc),
        top1_accuracy=float(top1_accuracy),
        ndcg_at_3=ndcg_at_3,
        urgency_spearman=urgency_spearman,
    )
