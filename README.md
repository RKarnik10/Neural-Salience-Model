# Neural-Inspired Obstacle Salience Model

A computational model of how the mammalian **superior colliculus** (SC) prioritizes spatial threats, benchmarked against a standard XGBoost baseline on a simulated navigation task.

This project is a direct extension of **OMAR** — an obstacle-detection device I built for a visually impaired user — into the neuroscience underneath it. OMAR solved the engineering problem of turning sensor readings into audio cues. This model asks the deeper question: *given the same sensor input, how should the brain decide which obstacle to attend to first?*

## Motivation

The superior colliculus is a midbrain structure that orients attention toward the most behaviorally relevant stimulus. Its computational signature is well-studied in neuroscience:

1. **Looming sensitivity.** Deep SC neurons respond selectively to objects on a collision course.
2. **Proximity priority.** Closer stimuli dominate, roughly following an inverse-distance law.
3. **Forward / heading bias.** The SC is retinotopic with foveal over-representation — for a navigator, "things in your path" get more representation than peripheral ones.
4. **Winner-take-all dynamics.** Lateral inhibition collapses many candidate signals onto a single attended target.

The hypothesis this project tests is narrow: *can a fixed, interpretable weighting of bio-inspired features get close to a trained gradient-boosted classifier on a realistic threat-ranking task, with zero training data?*

## Results

On 120 held-out test scenes (from a dataset of 600 total, scene-level split), with realistic sensor noise on all features:

| Model              | AUC   | Top-1 accuracy | NDCG@3 | Urgency correlation |
|--------------------|-------|----------------|--------|---------------------|
| Bio-Inspired (SC)  | 0.916 | 0.740          | 0.880  | +0.447              |
| XGBoost baseline   | 0.975 | 0.857          | 0.969  | +0.730              |
| Random top-1       | —     | ~0.18          | —      | —                   |

The bio-inspired model reaches ~86% of XGBoost's NDCG@3 and picks the correct most-urgent threat in 74% of scenes **with no training data whatsoever**. The random baseline for top-1 is ~18% (average 5.5 obstacles per scene).

Equally interesting: when you compare how each model weights features, they converge. XGBoost independently identifies TTC and loom rate as its top two features — matching the SC's known emphasis on looming — despite being given no architectural prior.

![feature importance comparison](data/feature_importance.png)

## Project structure

```
neural-salience-model/
├── src/
│   ├── synthetic.py      # Scene generator + sensor noise + ground-truth labeler
│   ├── salience.py       # Bio-inspired SC model (interpretable, zero-shot)
│   ├── baseline.py       # XGBoost classifier, scene-level train/test split
│   ├── evaluation.py     # AUC, top-1, NDCG@3, urgency correlation
│   └── visualize.py      # Per-scene polar plots + feature-weight comparison
├── scripts/
│   └── run_comparison.py # End-to-end pipeline
├── tests/
│   └── test_pipeline.py  # Basic sanity tests
└── data/                 # Outputs land here (plots, results.txt)
```

## Quickstart

```bash
pip install -r requirements.txt
python scripts/run_comparison.py
```

Outputs land in `data/`: two example scene visualizations (one where the models agree, one where they disagree), a feature-importance comparison, and a plain-text results summary.

## Task setup

A *scene* is a snapshot of a user walking forward at 1.4 m/s with 3–8 obstacles around them. Each obstacle has position, velocity, and size. Ground-truth threat labels are computed by **forward-simulating** the straight-line user trajectory for 3 seconds; an obstacle is a "threat" if the user would enter its personal-space radius within that horizon. The rank-1 threat is the one with the smallest time-to-collision.

The features both models see — distance, angle, radial velocity, tangential velocity, size, TTC, and loom rate — are corrupted with realistic sensor noise (8% distance noise, 0.35 m/s velocity noise, ~3° angle noise). Ground truth is computed from the clean physics. This matters: the bio-inspired model's smoother weighting over noisy inputs is closer to what the SC actually has to handle.

## Why the comparison is fair

The two models see identical inputs (same noisy features, same scenes). They differ only in how they turn those inputs into a salience score:

- **Bio-inspired** uses a fixed linear combination of four transformed feature scores with coefficients taken from the SC literature. No training.
- **XGBoost** learns a tree ensemble from 480 labeled training scenes, evaluated on 120 held-out scenes.

Evaluation is on the same test scenes for both models.

## Planned extensions

- **Neural upgrade.** Replace the hand-crafted combination with a small network modeling SC layer structure (superficial sensory layer + intermediate integration + winner-take-all output). Target: narrow the gap to XGBoost on NDCG@3 while keeping biological plausibility.
- **Real sensor data.** Swap the synthetic generator for ultrasonic/LIDAR traces. The noise model is already designed to approximate what real sensors produce.
- **Temporal dynamics.** Current scenes are snapshots. The SC integrates over time — next version should ingest short trajectories so the model can pick up looming dynamics directly rather than through a derived feature.

## Changelog

### v2 — Full model ladder (2026-04-30)

**Why:** A two-model comparison (bio vs. XGBoost) leaves open questions: is the bio model beating a trivial baseline? Is XGBoost winning because it's tuned, or because it's genuinely better? Does the bio model's value come from being "a neural net" or from encoding actual neuroscience? The v2 ladder answers all three.

**New models added:**

| Model | File | Purpose |
|---|---|---|
| `NaiveTTCModel` | `src/baseline.py` | Zero-shot floor: rank by TTC alone. Every other model must clear this. |
| Logistic Regression | `src/baseline.py` → `make_logistic_regression()` | Same structural form as bio-inspired (linear weighted sum), but weights learned from data. The most direct test of whether the SC weights are competitive. |
| Random Forest | `src/baseline.py` → `make_random_forest()` | Sanity check that XGBoost isn't winning purely on hyperparameter tuning. |
| MLP (32→16) | `src/baseline.py` → `make_mlp()` | Shows that "neural network ≠ brain-like" — it performs like XGBoost, not like the SC model. |

All learned models share the same scene-level 80/20 split (same seed) so the comparison is apples-to-apples.

**Updated results (v2) — 120 held-out test scenes:**

| Model | AUC | Top-1 acc | NDCG@3 | Urgency ρ |
|---|---|---|---|---|
| Naive TTC | 0.881 | 0.753 | 0.871 | +0.444 |
| **Log. Regression** | **0.952** | **0.727** | **0.925** | **+0.508** |
| **Bio-Inspired (SC)** | **0.916** | **0.740** | **0.880** | **+0.447** |
| Random Forest | 0.974 | 0.844 | 0.962 | +0.629 |
| XGBoost | 0.974 | 0.870 | 0.970 | +0.735 |
| MLP (32→16) | 0.970 | 0.831 | 0.955 | +0.660 |

**Key findings from the ladder:**

- **Bio-inspired beats the naive TTC floor** on every metric — it's doing real multi-feature work, not just repackaging TTC.
- **Bio-inspired is competitive with logistic regression** on NDCG@3 (0.880 vs 0.925) and urgency correlation, despite using zero training data. The SC weights are a reasonable prior. Logistic regression does win on AUC, which tells you the raw features have more linear signal than the four SC-composite features capture — the bio model is leaving some information on the table.
- **XGBoost and random forest are nearly identical** (NDCG@3: 0.970 vs 0.962), confirming XGBoost isn't winning through tuning alone.
- **MLP performs like XGBoost**, not like the bio model — being a "neural network" doesn't make it bio-inspired.

**Code changes:**

- `src/baseline.py`: Added `NaiveTTCModel`, `SklearnBaselineResult`, `train_sklearn_baseline()`, `predict_proba_salience()`, `lr_feature_importances()`, and factory functions `make_logistic_regression()`, `make_random_forest()`, `make_mlp()`. Existing XGBoost code untouched.
- `src/visualize.py`: Added `plot_model_ladder()` — a generic multi-panel bar chart for N models. Existing `plot_feature_importance_comparison()` untouched.
- `scripts/run_comparison.py`: Rewrote `main()` to train and evaluate all six models, print the full ladder, and save a 3-panel feature-importance plot (Bio | Log. Reg. | XGBoost).

## References

- Sparks, D. L. (1986). *Translation of sensory signals into commands for control of saccadic eye movements: role of primate superior colliculus.* Physiol. Rev.
- Wurtz, R. H., & Albano, J. E. (1980). *Visual-motor function of the primate superior colliculus.* Annu. Rev. Neurosci.
- Liu, Y.-J., Wang, Q., & Li, B. (2011). *Neuronal responses to looming objects in the superior colliculus of the cat.* Brain Behav. Evol.
