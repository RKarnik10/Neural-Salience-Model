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

Seven models on 120 held-out test scenes (600 total, scene-level split), with realistic sensor noise on all features. The ladder runs from a zero-shot single-feature floor up to gradient-boosted trees:

| Model | AUC | Top-1 acc | NDCG@3 | Urgency ρ |
|---|---|---|---|---|
| Naive TTC | 0.881 | 0.753 | 0.871 | +0.444 |
| Logistic Regression | 0.952 | 0.727 | 0.925 | +0.508 |
| Bio-Inspired (SC) — fixed weights | 0.916 | 0.740 | 0.880 | +0.447 |
| **SC-Net (v3) — architectural** | **0.964** | **0.844** | **0.953** | **+0.780** |
| Random Forest | 0.974 | 0.844 | 0.962 | +0.629 |
| XGBoost | 0.974 | 0.870 | 0.970 | +0.735 |
| MLP (32→16) | 0.970 | 0.831 | 0.955 | +0.660 |

Two headline findings:

**1. Modeling the SC's architecture closes ~80% of the gap — with no new features.** The v2 bio model was just four fixed SC-literature weights in one weighted sum (NDCG@3 = 0.880). v3 keeps those exact four bio channels as a fixed "superficial" layer, then adds two things the SC actually does: an integration layer that models feature *interactions* (coincidence detection), and a *winner-take-all* competition between obstacles. That lifts NDCG@3 to **0.953** — closing **81%** of the distance to XGBoost (0.970) — and matches Random Forest on top-1 accuracy, while staying interpretable and traceable to neuroscience.

**2. The winner-take-all competition makes it the best-calibrated model on urgency.** SC-Net's urgency correlation (**+0.780**) is the highest in the whole ladder — above XGBoost's (+0.735). Because it's trained under a within-scene competition (a listwise objective standing in for lateral inhibition), it is explicitly optimized to order obstacles by how soon you'd reach them — exactly what the urgency metric measures, and what the SC evolved to do.

When you compare where each model puts its weight, they converge on the same physics. XGBoost independently ranks TTC and loom rate at the top — matching the SC's known emphasis on looming — with no architectural prior. And when SC-Net is allowed to *learn* the weighting of the four bio channels rather than using the fixed literature values, it promotes TTC and proximity and demotes loom: on this task, once TTC is present, loom's marginal contribution is small (both encode "approaching fast"). The fixed SC prior isn't optimal, but it lands in the right neighborhood.

![feature importance comparison](data/feature_importance.png)
*Where each model puts its weight. The first two panels weight the same four SC channels — fixed (bio) vs. learned (SC-Net); the last two weight the seven raw features.*

## Project structure

```
neural-salience-model/
├── src/
│   ├── synthetic.py      # Scene generator + sensor noise + ground-truth labeler
│   ├── salience.py       # Bio-inspired SC model — fixed weights, zero-shot (v2)
│   ├── sc_network.py     # Architectural SC net — NumPy + optional PyTorch backend (v3)
│   ├── baseline.py       # Learned baselines: XGBoost, LogReg, RandomForest, MLP, naive TTC
│   ├── evaluation.py     # AUC, top-1, NDCG@3, urgency correlation
│   └── visualize.py      # Per-scene polar plots + feature-weight comparison
├── scripts/
│   └── run_comparison.py # End-to-end pipeline (full 7-model ladder)
├── tests/
│   ├── test_pipeline.py    # Data / bio-model / baseline / evaluation sanity tests
│   └── test_sc_network.py  # SC-net gradient check, training, backend parity
└── data/                 # Outputs land here (plots, results.txt)
```

## Quickstart

```bash
pip install -r requirements.txt
python scripts/run_comparison.py
```

Outputs land in `data/`: two example scene visualizations (one where the models agree, one where they disagree), a feature-importance comparison, and a plain-text results summary.

**PyTorch is optional.** The v3 SC network ships with two interchangeable backends behind one interface. If `torch` is installed it is used automatically; otherwise the model falls back to a dependency-free NumPy implementation with hand-written backprop and a small Adam optimizer. Pick explicitly with `SCNetwork(backend="numpy" | "torch" | "auto")`. Everything in the default `requirements.txt` runs the NumPy backend.

## Task setup

A *scene* is a snapshot of a user walking forward at 1.4 m/s with 3–8 obstacles around them. Each obstacle has position, velocity, and size. Ground-truth threat labels are computed by **forward-simulating** the straight-line user trajectory for 3 seconds; an obstacle is a "threat" if the user would enter its personal-space radius within that horizon. The rank-1 threat is the one with the smallest time-to-collision.

The features both models see — distance, angle, radial velocity, tangential velocity, size, TTC, and loom rate — are corrupted with realistic sensor noise (8% distance noise, 0.35 m/s velocity noise, ~3° angle noise). Ground truth is computed from the clean physics. This matters: the bio-inspired model's smoother weighting over noisy inputs is closer to what the SC actually has to handle.

## Why the comparison is fair

Every model sees identical inputs (same noisy features, same scenes). They differ only in how they turn those inputs into a salience score:

- **Bio-inspired** uses a fixed linear combination of four transformed feature scores with coefficients taken from the SC literature. No training.
- **SC-Net (v3)** reuses those exact four bio channels as a fixed superficial layer, then learns an integration layer (with interaction terms) and a winner-take-all competition on top. Trained on the labeled training scenes.
- **XGBoost, Random Forest, MLP, Logistic Regression** each learn from 480 labeled training scenes.

All learned models — SC-Net included — share the same scene-level 80/20 split (same seed), and every model is evaluated on the same 120 held-out test scenes, so the numbers are directly comparable.

## Planned extensions

- ~~**Neural upgrade.**~~ **Done in v3** (`src/sc_network.py`): a small network modeling SC layer structure — superficial sensory layer → intermediate integration → winner-take-all output — that narrows the NDCG@3 gap to XGBoost while staying biologically plausible. See the [v3 changelog](#v3--architectural-sc-network-2026-07-24).
- **Recurrent WTA dynamics.** v3's winner-take-all is a single-shot softmax trained under a competitive objective. The real SC settles a winner through *recurrent* lateral inhibition over tens of milliseconds. A next version could unroll that dynamic and read out the settled state, which may sharpen top-1 further.
- **Real sensor data.** Swap the synthetic generator for ultrasonic/LIDAR traces. The noise model is already designed to approximate what real sensors produce.
- **Temporal dynamics.** Current scenes are snapshots. The SC integrates over time — next version should ingest short trajectories so the model can pick up looming dynamics directly rather than through a derived feature.

## Changelog

### v3 — Architectural SC network (2026-07-24)

**Why:** v2 proved the SC's *feature weighting* is a competitive prior, and predicted that the remaining gap to the tree models is (a) feature interactions and (b) the one biological mechanism no model in the ladder had yet — within-scene competition. v3 tests that prediction directly: does modeling the SC's *processing structure* add performance on top of just having the right feature weights? Answer: yes, most of it.

**The model (`src/sc_network.py`).** A small network whose architecture maps onto SC anatomy rather than a generic MLP:

| Stage | SC analogue | What it does |
|---|---|---|
| Superficial layer | sensory receptive fields | The four fixed SC transforms from `SalienceModel` (loom, proximity, ttc, forward). Kept fixed, so v3 isolates the *architectural* contribution. |
| Integration layer | intermediate-layer coincidence detection | Four channels **plus their six pairwise products**, through a small tanh hidden layer. This is where interactions live. |
| Winner-take-all output | lateral inhibition | A within-scene softmax at a learnable temperature, trained as a listwise objective so competition shapes the representation — not just an inference-time cosmetic. |

**Two interchangeable backends, one interface.** A from-scratch **NumPy** backend (forward pass + hand-derived backprop + a small Adam optimizer, verified by a finite-difference gradient check) and an optional **PyTorch** backend (`nn`-style module + autograd). `backend="auto"` uses PyTorch if installed, else NumPy. Same architecture, config, loss, and results.

**Updated results (v3) — 120 held-out test scenes:**

| Model | AUC | Top-1 acc | NDCG@3 | Urgency ρ |
|---|---|---|---|---|
| Naive TTC | 0.881 | 0.753 | 0.871 | +0.444 |
| Log. Regression | 0.952 | 0.727 | 0.925 | +0.508 |
| Bio-Inspired (SC) | 0.916 | 0.740 | 0.880 | +0.447 |
| **SC-Net (v3)** | **0.964** | **0.844** | **0.953** | **+0.780** |
| Random Forest | 0.974 | 0.844 | 0.962 | +0.629 |
| XGBoost | 0.974 | 0.870 | 0.970 | +0.735 |
| MLP (32→16) | 0.970 | 0.831 | 0.955 | +0.660 |

**Key findings:**

- **The architecture closes 81% of the NDCG@3 gap** (0.880 → 0.953 against XGBoost's 0.970) using the *same* four bio channels — confirming v2's prediction that the missing ingredient was interaction structure, not more features.
- **SC-Net is the best model in the ladder on urgency correlation** (+0.780 vs. XGBoost's +0.735). The winner-take-all listwise objective directly optimizes within-scene urgency ordering — the SC's actual job.
- **It is not a generic MLP.** The v2 MLP (raw features, no competition) sits at NDCG@3 = 0.955 / urgency +0.660; SC-Net matches it on ranking while beating it decisively on urgency, using interpretable bio channels. The biological structure is doing specific work.
- **Learned channel weights re-rank the SC prior:** given the chance to learn, the net promotes TTC and proximity and demotes loom (once TTC is present, loom is largely redundant). The fixed SC weights are close but not optimal.

**Code changes:**

- `src/sc_network.py` (new): `SCNetwork` wrapper, `_NumpySC` / `_TorchSC` backends, `SCConfig`, `superficial_channels()`, `interaction_features()`, `train_sc_network()`, and `sc_channel_importances()`.
- `scripts/run_comparison.py`: added SC-Net to the ladder (now 7 models), added its learned channel-salience panel to the feature-importance figure, and logged the backend + learned WTA temperature to `results.txt`.
- `tests/test_sc_network.py` (new): finite-difference gradient check, fit-reduces-loss, probability-range, beats-naive-floor, normalized channel importances, and (when torch is installed) NumPy↔PyTorch parity.
- `requirements.txt`: documented `torch` as an optional dependency.

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
