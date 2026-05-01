"""Basic pipeline sanity tests. Run with: python -m pytest tests/"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.synthetic import generate_dataset, Obstacle, Scene, label_scene
from src.salience import SalienceModel, SalienceWeights
from src.baseline import train_baseline, predict_salience, FEATURES
from src.evaluation import evaluate


# ---------- Synthetic data ----------

def test_dataset_generation_shape():
    df = generate_dataset(n_scenes=10, seed=1)
    assert len(df) > 0
    assert "is_threat" in df.columns
    assert "threat_rank" in df.columns
    assert "scene_id" in df.columns
    assert df["scene_id"].nunique() == 10
    for f in FEATURES:
        assert f in df.columns, f"missing feature {f}"


def test_ground_truth_head_on_collision():
    """A stationary obstacle directly ahead should always be labeled a threat."""
    obs = Obstacle(x=0.0, y=2.5, vx=0.0, vy=0.0, size=0.3)
    scene = Scene(obstacles=[obs])
    df = label_scene(scene)
    assert df["is_threat"].iloc[0] == True
    assert df["threat_rank"].iloc[0] == 1


def test_ground_truth_behind_user_no_threat():
    """An obstacle directly behind with no approach velocity is not a threat."""
    obs = Obstacle(x=0.0, y=-3.0, vx=0.0, vy=0.0, size=0.3)
    scene = Scene(obstacles=[obs])
    df = label_scene(scene)
    assert df["is_threat"].iloc[0] == False


# ---------- Bio-inspired model ----------

def test_bio_salience_scores_in_range():
    df = generate_dataset(n_scenes=20, seed=2)
    model = SalienceModel()
    scores = model.predict(df)
    assert scores.shape == (len(df),)
    assert np.all(np.isfinite(scores))
    # Scores should be non-negative given all component scores are in [0,1]
    assert (scores >= 0).all()


def test_bio_looming_dominates_for_closer_approach():
    """Two otherwise identical obstacles, one approaching fast, one not:
    the approaching one should have higher salience."""
    approaching = Obstacle(x=0.0, y=3.0, vx=0.0, vy=-2.0, size=0.4)
    passive = Obstacle(x=0.0, y=3.0, vx=0.0, vy=0.0, size=0.4)
    scene = Scene(obstacles=[approaching, passive])
    df = label_scene(scene)
    df["scene_id"] = 0
    model = SalienceModel()
    scores = model.predict(df)
    assert scores[0] > scores[1], "approaching obstacle should score higher"


def test_bio_wta_scores_sum_to_one_per_scene():
    df = generate_dataset(n_scenes=5, seed=3)
    model = SalienceModel()
    scores = model.predict(df, apply_wta=True)
    for sid, idx in df.groupby("scene_id").groups.items():
        assert abs(scores[list(idx)].sum() - 1.0) < 1e-6


# ---------- Baseline ----------

def test_baseline_trains_and_scores():
    df = generate_dataset(n_scenes=60, seed=4)
    res = train_baseline(df, seed=4)
    assert 0.5 < res.test_auc <= 1.0
    probs = predict_salience(res.model, df)
    assert probs.shape == (len(df),)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_scene_split_is_disjoint():
    df = generate_dataset(n_scenes=50, seed=5)
    res = train_baseline(df, seed=5)
    assert set(res.train_scene_ids).isdisjoint(set(res.test_scene_ids))


# ---------- Evaluation ----------

def test_evaluation_perfect_scores_on_ground_truth():
    """If we use ground-truth threat_rank as 'scores' (inverted), every metric
    should be essentially perfect."""
    df = generate_dataset(n_scenes=30, seed=6)
    # Give rank-1 highest score, non-threats lowest
    scores = np.where(df["is_threat"], 1.0 / df["threat_rank"].clip(lower=1), 0.0)
    report = evaluate(df, scores)
    assert report.top1_accuracy > 0.99
    assert report.ndcg_at_3 > 0.99


if __name__ == "__main__":
    # Allow running without pytest
    import traceback
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    passed, failed = 0, 0
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
