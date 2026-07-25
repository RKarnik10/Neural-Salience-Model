"""
End-to-end comparison: generate scenes -> train all baselines -> evaluate the
full model ladder -> visualize representative scenes.

Model ladder (simplest to most complex):
  Naive TTC  ->  Logistic Regression  ->  Bio-Inspired  ->  Random Forest  ->  XGBoost  ->  MLP

Run:
    python scripts/run_comparison.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

# Make `src` importable when running as a script from anywhere
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.synthetic import generate_dataset
from src.salience import SalienceModel
from src.baseline import (
    FEATURES,
    NaiveTTCModel,
    lr_feature_importances,
    make_logistic_regression,
    make_mlp,
    make_random_forest,
    predict_proba_salience,
    predict_salience,
    train_baseline,
    train_sklearn_baseline,
)
from src.sc_network import sc_channel_importances, torch_available, train_sc_network
from src.evaluation import evaluate
from src.visualize import plot_feature_importance_comparison, plot_model_ladder, plot_scene_comparison


N_SCENES = 600
SEED = 0
OUT_DIR = ROOT / "data"
OUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    print("=== Neural-Inspired Obstacle Salience Model — Full Model Ladder ===\n")

    # 1. Generate dataset (with sensor noise)
    print(f"Generating {N_SCENES} synthetic scenes with sensor noise...")
    df = generate_dataset(n_scenes=N_SCENES, seed=SEED, sensor_noise=True)
    print(f"  -> {len(df)} total obstacles, threat rate = {df['is_threat'].mean():.2%}\n")

    # 2. Train all learned models on the same scene-level 80/20 split (same seed)
    print("Training baselines on scene-level 80/20 split...")
    baseline    = train_baseline(df, seed=SEED)
    lr_result   = train_sklearn_baseline(df, make_logistic_regression(SEED), seed=SEED)
    rf_result   = train_sklearn_baseline(df, make_random_forest(SEED), seed=SEED)
    mlp_result  = train_sklearn_baseline(df, make_mlp(SEED), seed=SEED)

    # v3: architectural SC network (superficial -> integration -> winner-take-all).
    # backend="auto" uses PyTorch if installed, else the from-scratch NumPy backend.
    sc_result   = train_sc_network(df, backend="auto", seed=SEED)

    # All models share the same test scenes (deterministic split from seed)
    test_scene_ids = set(baseline.test_scene_ids.tolist())
    test_df = df[df["scene_id"].isin(test_scene_ids)].reset_index(drop=True)
    print(f"  XGBoost  train AUC={baseline.train_auc:.3f}  test AUC={baseline.test_auc:.3f}")
    print(f"  LogReg   train AUC={lr_result.train_auc:.3f}  test AUC={lr_result.test_auc:.3f}")
    print(f"  RandForest train AUC={rf_result.train_auc:.3f}  test AUC={rf_result.test_auc:.3f}")
    print(f"  MLP      train AUC={mlp_result.train_auc:.3f}  test AUC={mlp_result.test_auc:.3f}")
    print(f"  SC-Net   train AUC={sc_result.train_auc:.3f}  test AUC={sc_result.test_auc:.3f}"
          f"  [backend={sc_result.backend}, tau={sc_result.model.tau:.2f}]"
          f"  (torch available: {torch_available()})\n")

    # 3. Score every model on the shared test set
    bio_model    = SalienceModel()
    naive_model  = NaiveTTCModel()

    naive_scores = naive_model.predict(test_df)
    bio_scores   = bio_model.predict(test_df)
    lr_scores    = predict_proba_salience(lr_result.model, test_df)
    sc_scores    = sc_result.model.predict(test_df)
    rf_scores    = predict_proba_salience(rf_result.model, test_df)
    xgb_scores   = predict_salience(baseline.model, test_df, FEATURES)
    mlp_scores   = predict_proba_salience(mlp_result.model, test_df)

    # 4. Evaluate all — print in ladder order
    naive_report = evaluate(test_df, naive_scores)
    bio_report   = evaluate(test_df, bio_scores)
    lr_report    = evaluate(test_df, lr_scores)
    sc_report    = evaluate(test_df, sc_scores)
    rf_report    = evaluate(test_df, rf_scores)
    xgb_report   = evaluate(test_df, xgb_scores)
    mlp_report   = evaluate(test_df, mlp_scores)

    print("Evaluation on held-out test scenes (ladder order):")
    print(f"  {naive_report.pretty('Naive TTC:   ')}")
    print(f"  {lr_report.pretty(  'Log. Reg.:   ')}")
    print(f"  {bio_report.pretty( 'Bio-Inspired:')}")
    print(f"  {sc_report.pretty(  'SC-Net (v3): ')}")
    print(f"  {rf_report.pretty(  'Rand. Forest:')}")
    print(f"  {xgb_report.pretty( 'XGBoost:     ')}")
    print(f"  {mlp_report.pretty( 'MLP:         ')}\n")

    # 5. Visualize agree/disagree scenes (bio vs XGBoost — the key comparison)
    unique_scenes = test_df["scene_id"].unique()
    threat_scenes = [
        sid for sid in unique_scenes
        if test_df[test_df["scene_id"] == sid]["is_threat"].any()
    ]
    agree_scene, disagree_scene = None, None
    for sid in threat_scenes:
        sdf = test_df[test_df["scene_id"] == sid].reset_index(drop=True)
        if not sdf["is_threat"].any():
            continue
        b_idx = np.argmax(bio_scores[test_df["scene_id"].values == sid])
        x_idx = np.argmax(xgb_scores[test_df["scene_id"].values == sid])
        gt_idx = sdf[sdf["threat_rank"] == 1].index[0]
        if b_idx == x_idx == gt_idx and agree_scene is None:
            agree_scene = sid
        elif b_idx != x_idx and disagree_scene is None:
            disagree_scene = sid
        if agree_scene is not None and disagree_scene is not None:
            break

    for label, sid in [("agree", agree_scene), ("disagree", disagree_scene)]:
        if sid is None:
            continue
        mask = test_df["scene_id"].values == sid
        sdf = test_df[mask].reset_index(drop=True)
        fig = plot_scene_comparison(
            sdf,
            bio_scores=bio_scores[mask],
            base_scores=xgb_scores[mask],
            title=f"Scene #{sid}  —  models {label} on top threat",
            save_path=str(OUT_DIR / f"scene_{label}_{sid}.png"),
        )
        print(f"  -> saved {OUT_DIR / f'scene_{label}_{sid}.png'}")

    # 6. Feature importance ladder: Bio | SC-Net | Log. Reg. | XGBoost
    #    Bio and SC-Net both weight the 4 SC channels (fixed vs. learned);
    #    LR and XGBoost use the 7 raw features.
    bio_weights   = {
        "loom":      bio_model.weights.w_loom,
        "proximity": bio_model.weights.w_proximity,
        "ttc":       bio_model.weights.w_ttc,
        "forward":   bio_model.weights.w_forward,
    }
    sc_weights  = sc_channel_importances(sc_result.model, test_df)
    lr_weights  = lr_feature_importances(lr_result, list(FEATURES))
    xgb_weights = dict(zip(FEATURES, baseline.model.feature_importances_.tolist()))

    plot_model_ladder(
        [
            ("Bio-Inspired\n(fixed SC weights)",       bio_weights, "#5E81AC"),
            ("SC-Net v3\n(learned channel salience)",  sc_weights,  "#B48EAD"),
            ("Log. Regression\n(|coef|, learned)",     lr_weights,  "#A3BE8C"),
            ("XGBoost\n(learned importances)",         xgb_weights, "#BF616A"),
        ],
        save_path=str(OUT_DIR / "feature_importance.png"),
    )
    print(f"  -> saved {OUT_DIR / 'feature_importance.png'}")

    # 7. Save full summary for reproducibility
    summary_path = OUT_DIR / "results.txt"
    with summary_path.open("w") as f:
        f.write("Neural-Inspired Obstacle Salience Model — Results\n")
        f.write(f"Seed: {SEED}  |  Scenes: {N_SCENES}  |  Test obstacles: {len(test_df)}\n\n")
        f.write(f"SC-Net backend: {sc_result.backend}  |  learned WTA tau: {sc_result.model.tau:.3f}\n\n")
        f.write("Model ladder (held-out test scenes):\n")
        f.write(naive_report.pretty("Naive TTC:   ") + "\n")
        f.write(lr_report.pretty(  "Log. Reg.:   ") + "\n")
        f.write(bio_report.pretty( "Bio-Inspired:") + "\n")
        f.write(sc_report.pretty(  "SC-Net (v3): ") + "\n")
        f.write(rf_report.pretty(  "Rand. Forest:") + "\n")
        f.write(xgb_report.pretty( "XGBoost:     ") + "\n")
        f.write(mlp_report.pretty( "MLP:         ") + "\n")
    print(f"  -> saved {summary_path}\n")

    print("Done.")


if __name__ == "__main__":
    main()
