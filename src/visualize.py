"""
Visualization utilities.

The main plot is a side-by-side comparison for a single scene, showing obstacles
in user-centric polar coordinates (angle vs distance), colored by each model's
salience score. Ground-truth threats are marked so you can eyeball where each
model agrees or disagrees with physical reality.
"""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_scene_comparison(
    scene_df: pd.DataFrame,
    bio_scores: np.ndarray,
    base_scores: np.ndarray,
    title: str = "",
    save_path: str | None = None,
) -> plt.Figure:
    """Plot a single scene with both models' salience scores side by side.

    Bottom of each polar plot = 'in front of user' (angle = 0).
    Obstacles are circles sized by their physical size; color = salience score.
    Ground-truth rank-1 threat is outlined in red; other threats outlined in orange.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={"projection": "polar"})
    for ax, scores, label in zip(axes, [bio_scores, base_scores], ["Bio-Inspired (SC)", "XGBoost Baseline"]):
        ax.set_theta_zero_location("N")   # 0 degrees at the top = "forward"
        ax.set_theta_direction(-1)        # clockwise = right side positive
        ax.set_rmax(11)
        ax.set_rticks([2, 5, 8, 11])
        ax.set_title(label, pad=20, fontsize=12, fontweight="bold")

        # Normalize scores for consistent color scaling within the scene
        s_min, s_max = float(np.min(scores)), float(np.max(scores))
        s_range = s_max - s_min if s_max > s_min else 1.0
        norm_scores = (scores - s_min) / s_range

        # Obstacle markers (angles in the dataframe are radians from heading)
        angles = scene_df["angle"].values
        dists = scene_df["distance"].values
        sizes = scene_df["size"].values * 800  # scatter size for matplotlib

        sc = ax.scatter(
            angles, dists,
            s=sizes, c=norm_scores, cmap="plasma",
            vmin=0, vmax=1, alpha=0.9, edgecolors="white", linewidths=0.5,
        )

        # Overlay ground-truth threat outlines
        for _, row in scene_df.iterrows():
            if row["threat_rank"] == 1:
                ax.scatter(row["angle"], row["distance"],
                           s=row["size"] * 800 * 2.2, facecolors="none",
                           edgecolors="red", linewidths=2.5, zorder=5)
            elif row["is_threat"]:
                ax.scatter(row["angle"], row["distance"],
                           s=row["size"] * 800 * 1.6, facecolors="none",
                           edgecolors="orange", linewidths=1.5, zorder=4)

        # User position at origin
        ax.scatter(0, 0.05, s=120, c="black", marker="^", zorder=10)

    cbar = fig.colorbar(sc, ax=axes, orientation="horizontal", shrink=0.6, pad=0.1, aspect=40)
    cbar.set_label("Normalized salience score")
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_feature_importance_comparison(
    bio_weights: dict[str, float],
    xgb_importances: dict[str, float],
    save_path: str | None = None,
) -> plt.Figure:
    """Side-by-side bar chart of where each model puts its weight.

    Shows the interpretability contrast: bio-inspired weights are fixed and
    hand-set from the SC literature; XGBoost importances are learned from data.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    bio_items = sorted(bio_weights.items(), key=lambda x: -x[1])
    axes[0].barh([k for k, _ in bio_items], [v for _, v in bio_items], color="#5E81AC")
    axes[0].set_title("Bio-Inspired (fixed SC-derived weights)", fontweight="bold")
    axes[0].set_xlabel("Weight")
    axes[0].invert_yaxis()

    xgb_items = sorted(xgb_importances.items(), key=lambda x: -x[1])
    axes[1].barh([k for k, _ in xgb_items], [v for _, v in xgb_items], color="#BF616A")
    axes[1].set_title("XGBoost (learned importances)", fontweight="bold")
    axes[1].set_xlabel("Feature importance")
    axes[1].invert_yaxis()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_model_ladder(
    model_importances: list[tuple[str, dict[str, float], str]],
    save_path: str | None = None,
) -> plt.Figure:
    """Multi-panel bar chart comparing feature weights across the full model ladder.

    model_importances: list of (label, {feature: weight}, bar_color) tuples,
    ordered from simplest to most complex model.
    """
    n = len(model_importances)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, (label, weights, color) in zip(axes, model_importances):
        items = sorted(weights.items(), key=lambda x: -x[1])
        ax.barh([k for k, _ in items], [v for _, v in items], color=color)
        ax.set_title(label, fontweight="bold", fontsize=10)
        ax.set_xlabel("Weight / importance")
        ax.invert_yaxis()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
