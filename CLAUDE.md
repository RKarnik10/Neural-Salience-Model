# CLAUDE.md — working context

Orientation file for Claude (web, Code, or any assistant with repo access). The
README is the public-facing story; this file is the *working state*: what's
built, what's true right now, what the conventions are, and what comes next.

---

## What this is

A **research project**, not a product. It asks one falsifiable question:

> Given noisy sensor input about obstacles around a walking user, does a model
> built from the **superior colliculus**'s known computational signature match a
> trained gradient-boosted classifier at deciding *which obstacle to attend to
> first*?

It is the neuroscience follow-up to **OMAR**, an obstacle-detection device the
author built for a visually impaired user. OMAR answered the engineering
question (sensor → audio cue). This answers the prioritization question.

The ML models are **not the point** — they are the measuring stick. XGBoost
doesn't tell you anything about the brain; it tells you how much signal the bio
model is leaving on the table, and what *kind* of signal it is (linear vs.
interaction vs. temporal).

**The task is ranking, not classification.** Any model can learn "threat /
not-threat." The interesting question is: given 5 obstacles, which one do you
orient to? That's why `top-1` and `NDCG@3` matter more than `AUC` here.

---

## Status — as of 2026-08-28

| | |
|---|---|
| **Current version** | **v3 — architectural SC network. Complete and merged.** |
| Last code change | 2026-07-24 (`058b860`, PR #2) |
| Branch | `main`, in sync with `origin/main` (0 ahead, 0 behind) |
| Working tree | Clean except two cosmetic README capitalization edits, uncommitted |
| Tests | **17 passed, 1 skipped** (`pytest -q`, Python 3.12). The skip is the NumPy↔PyTorch parity test — it skips by design when `torch` isn't installed |
| Results reproduced | Yes — `data/results.txt` matches the tables in the README |
| Next milestone | **v4 — not started.** See "What's next" below |

**Nothing is in flight.** There is no half-finished branch, no failing test, no
TODO left dangling in the source. v3 landed complete. Any new work starts from a
clean base.

### Version history at a glance

- **v1** — bio model vs. XGBoost, two models only. Superseded, not tagged.
- **v2 (2026-04-30)** — full 6-model ladder (naive TTC → LogReg → bio → RF →
  XGBoost → MLP). Proved the SC's *feature weighting* is a competitive zero-shot
  prior, and predicted the remaining gap was **feature interactions**.
- **v3 (2026-07-24)** — `src/sc_network.py`. Tested that prediction by modeling
  the SC's *architecture* (integration layer + winner-take-all) on top of the
  same four bio channels. **Prediction confirmed:** closed 81% of the NDCG@3 gap
  and produced the best urgency correlation of any model in the ladder,
  XGBoost included.

### Current numbers (120 held-out test scenes, seed 0)

| Model | AUC | Top-1 | NDCG@3 | Urgency ρ |
|---|---|---|---|---|
| Naive TTC | 0.881 | 0.753 | 0.871 | +0.444 |
| Logistic Regression | 0.952 | 0.727 | 0.925 | +0.508 |
| Bio-Inspired (SC), fixed weights | 0.916 | 0.740 | 0.880 | +0.447 |
| **SC-Net (v3), architectural** | **0.964** | **0.844** | **0.953** | **+0.780** |
| Random Forest | 0.974 | 0.844 | 0.962 | +0.629 |
| XGBoost *(ceiling)* | 0.974 | **0.870** | **0.970** | +0.735 |
| MLP (32→16) | 0.970 | 0.831 | 0.955 | +0.660 |

The two headline claims, stated so they can be checked:

1. **Architecture, not more features, closed the gap.** SC-Net uses the *exact
   same* four bio channels as the v2 bio model. Adding pairwise interaction
   terms and within-scene competition took NDCG@3 from 0.880 → 0.953 — 81% of
   the distance to XGBoost.
2. **Winner-take-all bought urgency calibration.** SC-Net's +0.780 urgency
   correlation beats every learned model. The listwise objective *is* a
   competition to attend the soonest collision, which is what the SC evolved to
   do and what the metric measures.

---

## Repo map

```
Neural-Salience-Model/
├── CLAUDE.md                  # this file — working state
├── README.md                  # public narrative + changelog
├── src/
│   ├── synthetic.py           # scene generator, forward-sim labeler, sensor noise
│   ├── salience.py            # v2 bio model: 4 fixed SC weights, zero training
│   ├── sc_network.py          # v3 SC-Net: superficial → integration → WTA
│   ├── baseline.py            # naive TTC + XGBoost/LogReg/RF/MLP, scene-level split
│   ├── evaluation.py          # AUC, top-1, NDCG@3, urgency Spearman
│   └── visualize.py           # polar scene plots + N-model weight comparison
├── scripts/run_comparison.py  # the whole pipeline; the only entry point
├── tests/
│   ├── test_pipeline.py       # data / bio / baseline / eval sanity
│   └── test_sc_network.py     # finite-diff gradient check, training, backend parity
├── assets/feature_importance.png   # tracked — the README embeds it
└── data/
    ├── model_analysis.md      # tracked — the long-form reasoning log
    └── *.png, results.txt     # gitignored run artifacts
```

**`data/model_analysis.md` is the most valuable file for context.** It's the
running analysis log: per-model pros/cons, why XGBoost is strong on this task,
what each version actually proved, and what the next version should test. Read
it before proposing a v4 direction.

---

## How to run

```bash
pip install -r requirements.txt
python scripts/run_comparison.py     # full 7-model ladder, ~seconds
python -m pytest -q                  # 17 passed, 1 skipped
```

`run_comparison.py` writes scene plots + `results.txt` to `data/` (gitignored)
and `feature_importance.png` to `assets/` (tracked, because the README embeds
it). Run from the repo root — the script inserts the root on `sys.path` itself,
and `src/sc_network.py` imports as `from src.baseline import ...`.

**PyTorch is optional.** `SCNetwork(backend="auto")` uses torch if importable,
else a from-scratch NumPy backend (hand-derived backprop + small Adam, verified
by finite-difference gradient check). Both backends share one architecture,
config, loss, and interface. The default `requirements.txt` runs NumPy; `torch`
is commented out there.

---

## Data contract

One long-format DataFrame, **one row per obstacle**, threaded through every
model. Same object everywhere — models differ only in how they score it.

**Inputs the models see (corrupted with sensor noise):**
`distance`, `angle`, `radial_velocity`, `tangential_velocity`, `size`, `ttc`,
`loom_rate` — this is `baseline.FEATURES`, in that order.

**Ground truth (computed from clean physics, never noised):**
`is_threat`, `threat_rank` (1 = most urgent, 0 = non-threat), `threat_time`,
`min_forward_distance`.

**Grouping / plotting:** `scene_id`, plus un-noised `x, y, vx, vy` for
visualization only.

### Invariants — violating these silently invalidates the comparison

- **Noise on features, clean physics on labels.** `add_sensor_noise` corrupts
  observables (8% distance, 0.35 m/s velocity, ~3° angle); labels come from
  forward-simulating the *clean* trajectory. This asymmetry is the point — it's
  the situation the SC actually faces.
- **Splits are scene-level, never row-level.** `baseline.scene_level_split`,
  seed 0, 80/20. Obstacles from one scene must never straddle the split.
- **Every model is evaluated on the identical 120 test scenes.** Any new model
  must reuse `scene_level_split` with the same seed or the ladder stops being
  comparable.
- **Non-approaching obstacles get `ttc = 1e6`**, not `inf` — infinities break
  downstream sklearn/XGBoost. Set in `generate_dataset`.
- **Scenes with zero threats yield `NDCG = nan`** and are excluded from that
  metric rather than scored as 0.

---

## Architecture of the two bio models

**v2 — `SalienceModel` (`src/salience.py`).** Four saturating transforms of raw
features, combined in one fixed weighted sum, zero training:

```
salience = 0.45·loom + 0.25·proximity + 0.20·ttc + 0.10·forward
```

The weights come from SC literature (loom dominates — it's the SC's defining
response). Shaping params live in `SalienceWeights`.

**v3 — `SCNetwork` (`src/sc_network.py`).** Each stage maps to SC anatomy:

| Stage | SC analogue | Implementation |
|---|---|---|
| Superficial | sensory receptive fields | The *same four* `SalienceModel` transforms — **kept fixed**, so v3 isolates the architectural contribution |
| Integration | intermediate-layer coincidence detection | 4 channels + their 6 pairwise products → tanh hidden layer (width 8). Interactions live here |
| Output | lateral inhibition / WTA | Within-scene softmax at learnable temperature `tau`, trained as a **listwise** objective so competition shapes the representation — not an inference-time cosmetic |

**Why the superficial layer stays fixed:** it's the experimental control. If you
let it learn, you can no longer say the gain came from *architecture* rather
than from better features. `sc_channel_importances()` measures what the net
*would* prefer (it promotes TTC and proximity, demotes loom — once TTC is
present, loom is largely redundant since both encode "approaching fast"), but
that's diagnostic output, not the trained model. **Don't unfix it without
renaming the experiment.**

---

## What's next — v4, not started

From `data/model_analysis.md`, in priority order:

1. **Recurrent WTA dynamics.** v3's winner-take-all is a single-shot softmax.
   The real SC settles a winner through *recurrent* lateral inhibition over tens
   of milliseconds. Unroll that dynamic and read out the settled state — the
   most likely source of the remaining top-1 gap (0.844 vs 0.870).
2. **Temporal input.** Scenes are currently snapshots. The SC integrates over
   time; ingesting short trajectories would let the model pick up looming
   directly instead of through a derived feature.
3. **Real sensor data.** Swap the synthetic generator for ultrasonic/LIDAR
   traces. The noise model was written to approximate real sensor behavior, so
   this is the intended path. Prediction on record: naive TTC degrades more than
   the bio model under worse noise, because loom rate is more forgiving of noisy
   distance than TTC (a ratio that explodes at small denominators).

The remaining ~1.7 NDCG@3 points to XGBoost are deep interaction structure plus,
plausibly, the single-shot-vs-recurrent difference. **Closing that gap is not
itself the goal** — the goal is testing which biological mechanisms are
computationally justified. A v4 that closed the gap with a mechanism the SC
doesn't have would be a *worse* result, not a better one.

---

## Conventions

- **Every version makes a falsifiable prediction, then the next version tests
  it.** v2 predicted interactions were the missing ingredient; v3 confirmed it.
  Whatever v4 is, state its prediction up front in `data/model_analysis.md`.
- **Negative results are kept, not buried.** The MLP matching XGBoost while
  looking nothing like the bio model is load-bearing evidence: "neural network"
  ≠ "brain-like." Don't quietly drop a model that makes the story less tidy.
- **Interpretability is a hard constraint, not a nice-to-have.** Every number in
  the bio models traces to a specific neuroscience claim. A change that improves
  a metric while severing that traceability defeats the project.
- **Numbers in `README.md`, `CLAUDE.md`, and `data/model_analysis.md` must agree
  with `data/results.txt`.** If you change the model, the seed, or the noise
  parameters, re-run `scripts/run_comparison.py` and update all four together.
- Style: type hints throughout, `from __future__ import annotations`,
  dataclasses for config, NumPy-style docstrings that explain the *biology*
  behind a transform rather than restating the code.
