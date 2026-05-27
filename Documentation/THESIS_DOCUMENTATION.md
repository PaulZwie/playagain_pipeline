# The PlayAgain EMG Gesture Pipeline — Technical Reference

> A reference document intended to support the methods, results and discussion
> chapters of the master thesis. It covers the architecture, data flow, data
> schema, algorithms, validation methodology and extension points of the
> `playagain_pipeline` package in sufficient detail to be reproducible by a
> reader who has the code but no prior context.

---

## Table of contents

1. [Goals and scope](#1-goals-and-scope)
2. [System architecture](#2-system-architecture)
3. [Data model and on-disk schema](#3-data-model-and-on-disk-schema)
4. [Signal acquisition](#4-signal-acquisition)
5. [Calibration](#5-calibration)
6. [Recording protocol engine](#6-recording-protocol-engine)
7. [Dataset construction](#7-dataset-construction)
8. [Feature pipeline](#8-feature-pipeline)
9. [Model catalogue](#9-model-catalogue)
10. [Training procedures](#10-training-procedures)
11. [Validation harness](#11-validation-harness)
12. [Real-time prediction and smoothing](#12-real-time-prediction-and-smoothing)
13. [Unity integration and game recording](#13-unity-integration-and-game-recording)
14. [Configuration system](#14-configuration-system)
15. [Reproducibility and determinism](#15-reproducibility-and-determinism)
16. [Extension points](#16-extension-points)
17. [Known limitations and caveats](#17-known-limitations-and-caveats)
18. [Glossary](#18-glossary)

---

## 1. Goals and scope

The pipeline is an end-to-end research tool for hand-gesture recognition from
surface electromyography (sEMG). It supports four workflows that can be
combined in any order:

- **Record** — acquire EMG while the participant performs a prompted
  gesture protocol, save raw samples and trial-level annotations to disk.
- **Train** — build a windowed dataset from one or more sessions, fit a
  classifier, persist the model for later use.
- **Use live** — run a trained model on incoming EMG and stream predictions
  to a Unity client over TCP, optionally with a synchronised game-recording
  log.
- **Validate** — evaluate feature sets, models and cross-validation strategies
  reproducibly, dumping enough metadata on each run to reproduce it exactly.

Design goals that shape many of the details in this document:

- **Reproducibility over convenience.** Every model and every validation run
  persists the exact configuration, the git SHA of the code that produced it,
  and the list of session paths it consumed. Reruns on the same commit and
  data produce bit-identical numbers.
- **Honest cross-validation.** Splits happen at session granularity, never at
  window granularity — windows from the same recording never appear in both
  train and test.
- **Two recorders, one corpus.** Sessions from the Python recording path and
  the Unity C# recording path share the same on-disk layout so cross-domain
  experiments are a configuration change, not a glue-code project.
- **Pluggable models and features.** New classifiers and new feature
  extractors register themselves into the respective registries so the GUI
  and validation tooling pick them up automatically.

---

## 2. System architecture

The package is organised into loosely-coupled subpackages that each expose a
small public surface and hide their implementation.

```text
                            ┌───────────────────────┐
                            │  GUI (PySide6)        │
                            │  main_window + tabs   │
                            └────────┬──────────────┘
                                     │ calls
            ┌────────────────────────┼────────────────────────┐
            │                        │                        │
            ▼                        ▼                        ▼
    devices/emg_device.py     core/data_manager.py     models/classifier.py
    + calibration/                  + session.py           + feature_pipeline.py
            │                        │                        │
            └────── writes ──────────┴────── writes ──────────┘
                                     │
                                     ▼
                              data/ directory
                                     ▲
                                     │ read-only
                            validation/runner.py
                            validation/corpus.py
                            validation/cv_strategies.py
```

Subpackage responsibilities:

| Subpackage                 | Role                                                                                                                |
|----------------------------|---------------------------------------------------------------------------------------------------------------------|
| `devices/`                 | Abstract device interface plus concrete wrappers (Muovi, Muovi Plus, Synthetic, Quattrocento replay).               |
| `core/`                    | Domain model: `Gesture`, `GestureSet`, `RecordingSession`, `DataManager`.                                           |
| `calibration/`             | Electrode-rotation detection; writes per-session calibration JSON and a reference calibration.                      |
| `protocols/`               | State machine that sequences cues, holds, rests, and the new calibration-sync phase.                                |
| `models/`                  | Classifier abstractions + concrete models (LDA, SVM, RF, CatBoost, MLP, CNN, AttentionNet, MSTNet) and features.    |
| `validation/`              | Reproducible validation harness: corpus discovery, CV strategies, runner.                                           |
| `gui/`                     | PySide6 UI: main window, tab factories, dialogs, live plots, bracelet viewer.                                       |
| `performance_assessment/`  | Multi-feature / multi-model comparison scaffolding and plot generators (pre-dates the validation harness).          |
| `prediction_server.py`     | TCP server for Unity; also usable standalone.                                                                       |
| `game_recorder.py`         | Synchronised EMG + prediction + ground-truth logger.                                                                |

Threading model:

- The GUI runs on the main Qt thread.
- Devices emit samples on their own thread; they reach the UI through Qt
  signals connected with `Qt.QueuedConnection`, which queues the call onto
  the UI thread.
- Training, dataset creation, and validation each run on a `QThread` worker
  so the UI remains responsive.
- The prediction server has a socket accept loop on one thread, a per-client
  receive loop on another, and a prediction worker on a third; the GUI polls
  results via signals.

Code lives under `playagain_pipeline/`; on-disk data lives under
`playagain_pipeline/data/` by default. The default `data_dir` is
configurable, and none of the code assumes its location.

---

## 3. Data model and on-disk schema

### 3.1 Sessions

A recording session is a single, continuous EMG stream plus metadata. The
canonical session layout is

```text
data/sessions/<subject_id>/<session_id>/
    data.npy             # (N_samples, N_channels) float32
    data.csv             # same data, CSV-formatted for inspection
    metadata.json        # session + trials + channel-mapping + rotation info
    gesture_set.json     # labels + display metadata
```

`session_id` is derived from the recording timestamp plus the protocol name,
e.g. `2026-03-20_12:54:45_10rep`.

`metadata.json` has two top-level keys: `metadata` (describing the whole
session) and `trials` (an ordered list of per-trial annotations).

The session-level `metadata` includes everything needed to interpret the raw
samples: `device_name`, `num_channels`, `sampling_rate`, `gesture_set_name`,
`protocol_name`, `calibration_applied`, `channel_mapping`, `rotation_offset`,
`rotation_confidence`, `bad_channels`, and a free `custom_metadata` bag.

Each trial entry records the gesture performed and the sample index window
the participant held the gesture during:

```json
{
  "trial_id": 3,
  "gesture_name": "pinch",
  "gesture_label": 2,
  "start_sample": 24000,
  "end_sample": 34000,
  "start_time": 12.0,
  "end_time": 17.0,
  "is_valid": true,
  "notes": "",
  "trial_type": "gesture"
}
```

`trial_type` is either `"gesture"` (the default) or `"calibration_sync"`.
Calibration-sync trials exist at the start of each session and record the
wave-out gesture used by the calibrator for electrode-rotation detection;
they are never included in model training (they carry `gesture_label = -1`).

Older sessions predate the `trial_type` field — `RecordingTrial.from_dict`
fills in the default `"gesture"` so they load transparently.

### 3.2 Unity sessions

Sessions recorded from within the Unity game follow the same four-file
layout but live under `data/sessions/unity_sessions/...`. The validation
corpus detects this path and tags the session with
`source_domain = "unity"`, enabling cross-domain (pipeline ↔ unity)
experiments without any custom code.

### 3.3 Datasets

A dataset is a windowed, feature-extracted (or raw) numpy array derived from
one or more sessions:

```text
data/datasets/<name>/
    X.npy            # features or raw windows
    y.npy            # int labels
    trial_ids.npy    # per-window trial id for leakage-aware splits
    metadata.json    # window size, stride, feature config, label names
```

`X.shape` is `(n_windows, n_features)` when features are extracted and
`(n_windows, window_size_samples, n_channels)` when raw windows are
requested. The `metadata.json` records exactly what was done to produce it,
so a dataset is reproducible from its source sessions.

### 3.4 Models

```text
data/models/<model_name>/
    <backend-specific binaries>   # e.g. model.pt + scaler.pkl for MLP/CNN
    metadata.json                 # ModelMetadata — class names, features,
                                  # bad-channel mode, feature config, etc.
    params.json                   # only for deep models — architecture
                                  # shapes needed to rebuild the nn.Module
```

`ModelMetadata` carries the information a caller needs to run the model
without consulting the original training code — input channels, expected
sampling rate, the feature configuration used, the bad-channel strategy
used, and the mapping from integer labels to gesture names.

### 3.5 Calibrations

Calibration files record per-gesture activation patterns and the detected
rotation offset:

```text
data/calibrations/
    reference_calibration.json           # monopolar reference
    reference_calibration_monopolar.json
    calibration_<session_id>.json        # one per session
    plots/                               # optional PNG overlays
```

The reference calibration acts as the target that later calibrations align
to; `rotation_offset` is the integer number of channels the bracelet is
rotated relative to the reference.

### 3.6 Validation runs

```text
data/validation_runs/<ts>__<name>/
    experiment.json          # frozen ExperimentConfig
    environment.json         # git SHA, Python/numpy/sklearn/torch versions
    session_index.json       # every session path used across all folds
    results.json             # per-fold metrics + aggregates + confusion
    results.csv              # per-fold flat table
    per_class_f1.csv         # per-class F1 per fold
    plots/                   # optional post-hoc plot output
```

Combined, these files are sufficient to reconstruct the run bit-for-bit
given the same git commit and the same `data/` directory.

### 3.7 Game recordings

```text
data/game_recordings/<subject_or_event>/game_<ts>/
    recording.csv   # EMG + prediction + Unity ground-truth rows
    config.json     # model used, class names, participant snapshot
```

Rows interleave raw EMG samples with prediction updates and Unity state
changes, all timestamped from a single clock so alignment is trivial.

---

## 4. Signal acquisition

### 4.1 Device abstraction

`devices/emg_device.py` exposes `DeviceType` (currently `SYNTHETIC`,
`MUOVI`, `MUOVI_PLUS`) and a `DeviceManager`-like interface. Concrete
backends are chosen by `device_type` in `DeviceConfig`. All devices emit
EMG samples as a numpy array shaped `(n_samples, n_channels)` and typed
`float32`.

The Muovi wrapper delegates to the `device_interfaces` package maintained
separately from this codebase. The synthetic backend generates band-limited
Gaussian noise modulated by the active gesture; it's sufficient for
pipeline-level testing without hardware.

Quattrocento replay is implemented through a GUI workflow in
`quattrocento_loader.py` + `quattrocento_training_dialog.py` rather than a
`Device` subclass, because the Quattrocento files arrive as `.pkl` /
`.otb+` blobs that need label reconstruction before they can be treated
like a stream.

### 4.2 Sampling rate and channel count

Canonical setup is **32 channels at 2 kHz** (Muovi monopolar). The
pipeline also handles 16-channel bipolar (Muovi Plus), 64-channel
(Quattrocento), and custom channel counts on the synthetic device.
Sampling rate is carried in session metadata; nothing in the
pipeline hardcodes it — *with one historical exception that was
fixed during the rework*: the validation runner formerly read
`cfg.windowing.sampling_rate` via a `getattr` with a default of
2000 Hz, silently defaulting for every run because `WindowingConfig`
has no such field. The current runner reads the rate from session
metadata via `_sampling_rate_for_fold`.

### 4.3 Bad-channel handling

The GUI's live EMG plot  lets
the user flag channels as bad per-session by toggling a checkbox.
Bad-channel indices are persisted to `metadata.bad_channels` and the
selected strategy is persisted in
`metadata.custom_metadata["bad_channel_mode"]`.

Two strategies are available, implemented in
`classifier.apply_bad_channel_strategy`:

- `"interpolate"` — replace a bad channel's samples with the average of its
  two nearest non-bad channels on the bracelet. Uses a cyclic nearest-neighbour
  search so the electrode at index 0 falls back to indices 31 and 1 rather than
  a random neighbour.
- `"zero"` — zero the bad channel's samples. Cheaper and loses less data,
  but forces downstream models to learn that "no signal" is a valid state for
  a channel that sometimes carries real signal.

The strategy is applied once when the dataset is constructed and once again
at prediction time if the live stream's channel count matches the model's
training-time channel count.

---

## 5. Calibration

EMG bracelets rotate on the forearm between sessions. A model trained with
the bracelet at orientation A produces garbled predictions if the user wears
it at orientation B. The calibrator solves this by measuring per-electrode
activation patterns for a known gesture (wave-out) and finding the circular
shift that best aligns the current pattern to a reference.

### 5.1 The reference calibration

`reference_calibration.json` stores per-gesture activation vectors collected
during a reference session. Each vector has length `num_channels` and
records, for each electrode, a robust estimate of the RMS energy while the
participant performed that gesture.

The calibrator uses either a single sync gesture (wave-out, recommended per
Barona López et al., 2020) or a combination across several gestures. The
wave-out gesture was chosen because independent testing found it produces
the sharpest azimuthal energy peak across subjects, which makes the cross-
correlation against the reference the most selective.

### 5.2 Rotation detection

Given a current pattern `P` and a reference pattern `R`, both of length
`N = num_channels`, the rotation offset is

```
offset = argmax_{k ∈ [0, N)} Σ_i P[(i + k) mod N] · R[i]
```

and the confidence is the peak of this cross-correlation, normalised by the
total energy.

The channel mapping is then `mapping[i] = (i + offset) mod N`; applying
`signal_aligned[:, i] = signal_raw[:, mapping[i]]` yields a stream whose
electrodes match the orientation the reference was recorded at. The Muovi
bracelet has two rows (outer/inner) that don't rotate independently, so a
single scalar offset is sufficient.

### 5.3 Reference incompatibility flags

A calibration can be declared `reference_incompatible` when the channel
count or signal mode (monopolar vs bipolar) differs from the reference.
In that case the UI falls back to manual entry of the rotation offset.

### 5.4 Calibration-sync trials

To avoid asking users to run a separate calibration session, every training
session starts with a brief sync phase (wave-out) recorded as trials with
`trial_type = "calibration_sync"`. Model training explicitly excludes these
via `RecordingSession.get_valid_trials` so they never become training
examples. The calibrator consumes them via `get_calibration_trials`.

---

## 6. Recording protocol engine

`protocols/protocol.py` is a deterministic state machine that sequences the
visual prompts shown to the participant during a session. Phases:

- `PREPARATION` — one-off intro shown at the very start.
- `CALIBRATION_SYNC` — wave-out sync gesture recorded once at the start.
- `REST` — 2–8 s of neutral hand position inserted between active cues.
- `CUE` — short notification that a gesture is coming.
- `HOLD` — the active gesture itself; this window is labelled as a trial.
- `RELEASE` — brief settling phase after a hold.
- `FEEDBACK` — optional per-trial feedback.
- `COMPLETE` — end-of-protocol marker.

A `ProtocolConfig` carries the timings (cue, hold, release, rest, feedback)
and the number of repetitions per gesture. The protocol enumerates all
steps up front so the total duration is known before recording starts —
handy for displaying an ETA in the GUI.

The engine is UI-agnostic: the GUI's `ProtocolWidget` subscribes to
`step_started` and `step_completed` signals and updates its display
accordingly. The session itself calls `start_trial` on a `HOLD` step and
`end_trial` on the next non-`HOLD` step.

---

## 7. Dataset construction

`DataManager.create_dataset` is the single entry point for turning one or
more sessions into `(X, y)`. It performs four steps:

1. **Per-session preparation.** For each session, load raw samples, apply
   the bad-channel strategy, and optionally apply the calibration channel
   mapping. If `use_per_session_rotation=True`, each session's own
   `rotation_offset` is applied before windowing — this is the recommended
   setting when combining sessions from different days.
2. **Windowing.** For each session, extract overlapping windows of
   `window_size_ms` with stride `window_stride_ms` from each valid trial.
   Windows crossing trial boundaries are discarded. This is where the
   leakage discipline lives: a window always belongs to exactly one trial,
   and trials belong to exactly one session.
3. **Labelling.** Windows inherit the gesture label of the trial they were
   drawn from. The dataset metadata carries a `label_names` mapping from
   integer → gesture name so downstream code can display readable class
   names.
4. **Feature extraction.** If a `feature_config` is supplied, windows are
   reduced to a feature vector per window via the feature pipeline
   (Section 8). Otherwise `X` stays as raw windows; only CNN-family models
   accept this shape.

Dataset metadata includes the exact window size, stride, feature config
snapshot, per-session rotation flag, bad-channel strategy, source session
paths, and the sampling rate — sufficient to regenerate the dataset from
scratch.

### 7.1 Mixed-channel-count handling

When combining sessions recorded with different hardware (16-ch bipolar vs
32-ch monopolar), `create_dataset` raises. The validation runner handles
this by pre-filtering each fold to a single dominant channel count and
logging dropped sessions at `WARNING` level; see
`ValidationRunner._materialise_fold._filter_by_channels`.

---

## 8. Feature pipeline

`models/feature_pipeline.py` defines a registry-driven feature system. A
feature extractor subclasses `BaseFeatureExtractor` and registers itself via
the `@register_feature("<name>")` decorator. Each feature produces one or
more scalars per channel per window.

### 8.1 Built-in time-domain features

| Key     | Name                     | Formula (per channel, per window of size T)                   |
|---------|--------------------------|---------------------------------------------------------------|
| `rms`   | Root Mean Square         | `sqrt(mean(x²))`                                              |
| `mav`   | Mean Absolute Value      | `mean(|x|)`                                                   |
| `var`   | Variance                 | `var(x)`                                                      |
| `wl`    | Waveform Length          | `sum(|diff(x)|)`                                              |
| `zc`    | Zero Crossings           | count of sign flips in x with |diff| above a threshold (def. 0.01) |
| `ssc`   | Slope Sign Changes       | count of sign flips in diff(x) with consecutive diff magnitude above threshold |
| `iemg`  | Integrated EMG           | `sum(|x|)`                                                    |
| `ssi`   | Simple Square Integral   | `sum(x²)`                                                     |

These eight features cover the classical Hudgins time-domain feature set
(mav, zc, ssc, wl) plus four additional descriptors that are widely used in
the sEMG literature. All features are purely time-domain; no spectral
features are implemented yet — adding one (e.g. median frequency) is a
matter of adding a `BaseFeatureExtractor` subclass.

### 8.2 Pipeline semantics

`FeaturePipeline.compute(data)` accepts either a 3-D batch
`(n_windows, T, n_channels)` or a 2-D single-window `(T, n_channels)`. It
runs every enabled feature on the batch and horizontally stacks the result,
so the output shape is `(n_windows, n_channels * n_features)`.

Feature order in the output is the order features were added to the
pipeline, grouped by feature. For N channels and K features, column layout
is `[feat0_ch0, feat0_ch1, ..., feat0_chN-1, feat1_ch0, ...]`. This matters
when aligning features to physical channels for bad-channel handling and
for debugging feature-attribution plots.

### 8.3 Raw-window mode

Deep models (`cnn`, `attention_net`, `mstnet`) don't accept features;
they operate on raw `(n_windows, T, n_channels)` tensors. The dataset
builder returns raw windows when `feature_config` is `None` or
`mode="raw"`. The validation runner auto-detects CNN-family models per
fold and materialises the fold twice — once as features for classical
models and once as raw windows for deep models — so a single mixed-model
run is possible without configuring anything.

---

## 9. Model catalogue

All models inherit from `BaseClassifier` in `models/classifier.py` and
expose the same API:

```python
model.train(X_train, y_train, X_val=None, y_val=None, **kwargs)
model.predict(X) -> np.ndarray            # (n_windows,)
model.predict_proba(X) -> np.ndarray      # (n_windows, n_classes)
model.save(path); model.load(path)
```

The `**kwargs` channel forwards hyperparameters set on the constructor plus
runtime configuration (sampling rate, window size, class weights, and since
the audit-patch round, a `random_state` seed).

### 9.1 Classical models

| Key             | Class                    | What it is                           | When to use                          |
|-----------------|--------------------------|--------------------------------------|--------------------------------------|
| `lda`           | `LDAClassifier`          | Linear Discriminant Analysis         | Fast, strong EMG baseline (~ms inference) |
| `svm`           | `SVMClassifier`          | Linear/RBF support vector machine    | Small datasets, when decision boundary is likely smooth |
| `random_forest` | `RandomForestClassifier` | sklearn random forest                | Noise-robust, handles outliers well, no scaling needed |
| `catboost`      | `CatBoostClassifier`     | Gradient-boosted trees               | Often best on tabular time-domain features |

All classical models accept a 2-D feature matrix `(n_windows, n_features)`
or a 3-D raw tensor (which they flatten internally). They standardise
features with an `sklearn.StandardScaler` that's fit on the training set and
persisted with the model.

### 9.2 Deep models

| Key             | Class                      | Architecture                                            |
|-----------------|----------------------------|---------------------------------------------------------|
| `mlp`           | `MLPClassifier`            | Fully-connected net on feature vectors                  |
| `cnn`           | `CNNClassifier`            | 1-D convolutional net on raw windows `(N, C, T)`        |
| `attention_net` | `AttentionNetClassifier`   | CNN front-end + self-attention; subclasses `CNNClassifier` |
| `mstnet`        | `MSTNetClassifier`         | Multi-scale temporal CNN; subclasses `CNNClassifier`    |

Deep models use PyTorch. Device selection honours Apple MPS, CUDA, and CPU
via `utils.platform_utils.resolve_device`.

The MLP accepts either 2-D features or 3-D raw windows (it flattens). The
CNN family requires 3-D raw windows — a 2-D feature tensor is coerced to
`(N, 1, F)` at training time. **Historically, `CNNClassifier.predict` did
not have a matching coercion, which caused a runtime `ValueError` when a
CNN trained on features was asked to predict on features. Patch 4 of the
audit fixes this asymmetry by sharing a `_coerce_to_3d` helper between
`train`, `predict`, and `predict_proba`.**

### 9.3 Shared infrastructure

- `ModelMetadata` records class names, feature config, bad-channel mode,
  and training-time channel count. Loading a model restores this metadata
  so the caller doesn't need to remember how the model was trained.
- `EMGFeatureExtractor` is a legacy helper that pre-dates the modular
  feature pipeline. It remains in use inside individual classifier
  implementations for backwards compatibility but is superseded by
  `feature_pipeline.FeaturePipeline` for new code.
- `ModelManager` is the factory that wraps the `AVAILABLE_MODELS` dict.
  `ModelManager.create_model(key, name=..., **hyperparams)` returns a
  trainable instance; `ModelManager.train_model(model, dataset, ...)` runs
  a stratified train/val split and calls the underlying `model.train`.

---

## 10. Training procedures

### 10.1 Standard GUI training (`training_dialog.py`)

The Train tab does four things in order:

1. Builds the dataset — windowing + feature extraction per the current
   config — via `DataManager.create_dataset`.
2. Splits the dataset into train/val with
   `sklearn.model_selection.train_test_split(test_size=0.2,
   stratify=y, random_state=cfg.seed)`. Before the audit patch this was
   hardcoded to `random_state=42`; after the patch it honours the seed
   from `PipelineConfig`.
3. Launches a `TrainingWorker(QThread)` that calls `model.train` with a
   progress callback for per-epoch reporting. Deep models emit
   `iteration_update` signals; classical models emit simulated progress.
4. On completion, saves the model to `data/models/<name>/`.

### 10.2 Advanced training (`AdvancedTrainingDialog`)

The advanced variant exposes:

- Hyperparameter editing for the chosen model.
- Auto-LR finder for deep models.
- Feature configuration editing before training.
- Bad-channel strategy override.
- Per-subject / per-session data selection.

It shares `TrainingWorker` with the standard dialog.

### 10.3 Quattrocento training (`quattrocento_training_dialog.py`)

This dialog targets the 64-channel Quattrocento hardware and its
`.pkl`/`.otb+` file formats. It supports k-fold within-subject CV and a
LOSO variant for cross-subject generalisation. Channel reduction via
variance ranking or PCA is available; PCA was hardcoded to
`random_state=42` before the audit patch and now honours the worker's
seed (`self._seed`).

Cross-validation here is **within-subject k-fold by default**, which gives
an optimistic accuracy estimate. The dialog surfaces this in its own
documentation and the runtime log.

---

## 11. Validation harness

The validation harness (`playagain_pipeline/validation/`) is the primary
means of producing results suitable for a thesis or paper. It runs
experiments end-to-end from a YAML configuration and persists everything
needed to reproduce them.

### 11.1 The corpus

`SessionCorpus.discover()` walks `data/sessions/` once and caches every
session it finds. Each session is wrapped in a `SessionRecord` with
`subject_id`, `session_id`, `path`, `source_domain ∈ {pipeline, unity}`,
`sampling_rate`, `num_channels`, `label_names`, and `notes`.

The `source_domain` is derived from the path — anything under
`unity_sessions/` is `unity`, everything else is `pipeline`. `SessionRecord`
is intentionally lazy: it doesn't load the signal or labels until asked.

### 11.2 CV strategies

Five cross-validation strategies are registered in
`cv_strategies.STRATEGIES` and selectable from YAML:

| Strategy           | Unit of split                | Typical use                                                               |
|--------------------|------------------------------|---------------------------------------------------------------------------|
| `within_session`   | Temporal tail of each session | Optimistic ceiling; don't use as a headline metric.                       |
| `loso_session`     | One session                  | Session-to-session generalisation within and across subjects.             |
| `loso_subject`     | All sessions of one subject  | **The honest single number.** Canonical headline for a paper.             |
| `k_fold_subjects`  | k subject groups             | Use when LOSO is too expensive (>20 subjects).                            |
| `cross_domain`     | `source_domain`              | Does a model trained on pipeline data still work when played in Unity?    |
| `holdout_split`    | User-specified train/val/test| Explicit ratios; best for deep-model tuning with real early-stopping val. |

Every strategy operates at **session granularity**. Windows from the same
recording cannot end up in both train and test. This is the single most
important methodological choice in the harness — it's what makes the
numbers it produces credible.

### 11.3 The runner

`ValidationRunner.run(cfg, progress=None)` executes a validated experiment
end-to-end:

1. Materialise the output directory and write `experiment.json`,
   `environment.json`, `session_index.json` upfront so an interrupted run
   still leaves a diagnostic trail.
2. Select sessions per `cfg.data` — either subject / domain filters, or
   an explicit session list.
3. Instantiate the CV strategy, enumerate folds, and compute
   `n_folds × n_models` total evaluations.
4. For each `(fold, model)` pair:
   - Seed numpy, stdlib random, and torch deterministically via a
     per-fold hash of `(cfg.seed, fold_idx, model_type)`. This is what
     makes results reproducible across runs with the same seed.
   - Materialise the fold as features or raw windows depending on the
     model type.
   - Fit the model, measure training time.
   - Evaluate on the test split; capture accuracy, macro-F1, per-class
     F1, inference latency, and the confusion matrix.
   - If the strategy provides a val split (only `holdout_split`), also
     evaluate there.
5. Persist per-fold results to `results.csv` and `per_class_f1.csv`, and
   the full `results.json` with the aggregate summary and the summed
   confusion matrices per model.
6. Between folds, check `progress.should_cancel()` — the GUI cancel button
   routes here.

The progress reporter is a `Protocol`, not a base class. The GUI provides a
Qt-bridge implementation; headless scripts can pass `None` and get the
default `NoopProgress` (silent logging).

### 11.4 Reproducibility metadata

`environment.json` captures:

- ISO timestamp
- `platform.platform()`
- `sys.version`, `sys.executable`
- Git SHA and dirty flag (if running inside a repo)
- Versions of `numpy`, `scipy`, `scikit-learn`, `catboost`, `torch`,
  `pandas`, `PySide6`.

`session_index.json` is the ground truth of which sessions went into the
run. A reviewer can use it to re-verify the split; differ from it and
the numbers will not match.

### 11.5 Known limitations

- The runner's error handling for a failing fold is *log and continue* —
  one crashed model doesn't kill the whole run, but it does leave a gap
  in the aggregate table.
- Mixed channel counts in a single fold are resolved by dropping sessions
  that don't match the dominant channel count (with a WARNING). An
  alternative — training separate models per channel count — is out of
  scope.

---

## 12. Real-time prediction and smoothing

`prediction_server.py` runs the prediction loop. Two deployment modes:

- Standalone CLI: `python -m playagain_pipeline.prediction_server --model <name>`.
- Embedded in the GUI: the Predict tab spawns the server in the app process.

### 12.1 Inference loop

A ring buffer collects incoming EMG samples. Once the buffer holds at least
`window_size_ms * sampling_rate / 1000` samples, the server extracts the
latest window, applies the model's feature pipeline (if configured), and
calls `model.predict_proba(X)`. The window stride is the same as at
training time, so the inference rate is `sampling_rate / stride_samples`.

### 12.2 Smoothing

Raw gesture decisions flicker — a single low-confidence frame can jump to
a different class. `PredictionSmoother` combines two mechanisms:

- **Exponential moving average** on the predicted class probabilities
  with `alpha` (default `0.3`). Reduces high-frequency jitter.
- **Stability gate** — the output class only changes when the new class
  has been the argmax for at least `min_stable_ms` (default `150 ms`).
  Prevents the bistability problem at the boundary between two classes of
  similar probability.

Both parameters are exposed in the GUI under a "Smoothing" group.

### 12.3 Ground truth feedback

Unity can send `{"type": "game_state", ...}` messages back to the server.
These carry the gesture the game *requested* at that moment. The server
passes these to `GameRecorder` so each recording has time-aligned ground
truth without needing post-hoc alignment.

---

## 13. Unity integration and game recording

### 13.1 Wire format

Newline-delimited JSON over TCP. Default `127.0.0.1:5555`.

**Outgoing (Python → Unity):**

Handshake on connect:
```json
{"type": "handshake", "model_name": "...", "class_names": {"0": "rest", ...},
 "sampling_rate": 2000, "num_channels": 32}
```

Per-prediction update:
```json
{"gesture": "fist", "gesture_id": 1, "confidence": 0.93,
 "probabilities": {"rest": 0.02, "fist": 0.93, "pinch": 0.03, "tripod": 0.02},
 "timestamp": 1712234567.123}
```

**Incoming (Unity → Python):**

```json
{"type": "game_state", "gesture_requested": "fist", "ground_truth": true,
 "camera_blocking": true, "timestamp": 1712234567.456}
```

The server is agnostic to additional fields in either direction — it
forwards everything to the registered callback unchanged.

### 13.2 Game recorder

`GameRecorder` consumes three parallel data streams:

- `on_emg_data(data: np.ndarray)` from the device.
- `on_prediction(gesture, gesture_id, confidence, probabilities)` from the
  server.
- `on_game_state(state: dict)` from Unity.

A background writer thread (deque + `threading.Event`) flushes rows to
`recording.csv` at its own pace so the live pipeline isn't blocked by disk
I/O. Each row carries a monotonic `time_s` relative to the start of the
recording, plus the most recent prediction and game state at that sample
index. The recorder also writes a `config.json` snapshot (model name,
class names, participant info) so a reviewer can re-interpret the CSV
without external context.

---

## 14. Configuration system

`config/config.py` defines five dataclasses that together form a
`PipelineConfig`:

- `DeviceConfig` — type, channel count, sampling rate, network settings.
- `RecordingConfig` — default window / stride settings.
- `CalibrationConfig` — calibration durations, confidence thresholds.
- `ProtocolSettings` — quick/standard/long/calibration timing presets.
- `ModelConfig` — default hyperparameters per model type, plus
  `bad_channel_mode`.

The top-level `PipelineConfig` aggregates these, adds a `seed` field, and
provides `to_json` / `from_json` round-tripping. `config.json` lives at
the package root by default and is loaded at startup. The GUI surfaces a
"Preferences" dialog (`config_dialog.py`) that edits the same dataclass.

The validation harness has its own `ExperimentConfig` (in
`validation/config.py`) which is a narrower, reproducibility-focused
dataclass. It encodes a single experiment and is deliberately decoupled
from `PipelineConfig` so a collaborator can send a YAML file without
including personal defaults.

---

## 15. Reproducibility and determinism

Reproducibility in this codebase is a matter of four things being
deterministic together:

1. **Seed discipline.** Every randomised operation takes a seed. After
   the audit patches, the seed flows from `ExperimentConfig.seed` or
   `PipelineConfig.seed` all the way down to numpy, stdlib `random`,
   torch, `sklearn.train_test_split`, `StratifiedKFold`, and PCA. The
   previous hardcoded `random_state=42` cases in `classifier.py`,
   `training_dialog.py`, and `quattrocento_training_dialog.py` are
   identified and patched in `AUDIT_PATCHES.md`.
2. **Per-fold seeding in the runner.** Each `(fold, model)` pair gets a
   SHA-1-hashed seed of `(cfg.seed, fold_idx, model_type)` so fold order
   and model order don't alter the results.
3. **Environment capture.** `environment.json` records the exact Python
   and library versions used. Point this out when a number differs by
   more than noise — usually it's a numpy or sklearn version bump.
4. **Frozen input.** `session_index.json` enumerates every session path
   consumed. If a session is added or removed, re-running won't match.

What is **not** deterministic:

- CUDA kernels. `torch.backends.cudnn.deterministic` is deliberately
  *not* forced to `True` because it measurably slows training and is
  unnecessary for within-version bit-exactness on CPU and MPS. CUDA users
  should expect per-run drift of ≤ 1e-4 in macro-F1.
- Wall-clock `train_seconds`. Don't treat this as a seeded metric.
- File-system enumeration order on unusual filesystems. `SessionCorpus`
  sorts its results before returning to defend against this, but any code
  that walks `data/sessions/` with `os.listdir` without sorting is vulnerable.

---

## 16. Extension points

### 16.1 Add a new model

1. Create a `Classifier` subclass in `models/classifier.py` implementing
   `train`, `predict`, `predict_proba`, `save`, `load`.
2. Add it to `ModelManager.AVAILABLE_MODELS`.
3. Add hyperparameter defaults to `ModelConfig` in `config/config.py`.
4. If it's a CNN-family model (wants raw windows, not features), add its
   key to `validation/runner._RAW_WINDOW_MODELS` so the validation runner
   materialises raw windows for it automatically.

### 16.2 Add a new feature

1. Subclass `BaseFeatureExtractor` in `models/feature_pipeline.py`.
2. Decorate with `@register_feature("your_name")`.
3. Implement `name`, `description`, and `compute(data)`. The compute
   function must accept both 2-D `(T, C)` and 3-D `(N, T, C)` inputs.

No registration in config is required — the GUI and validation tooling
discover features via `get_registered_features`.

### 16.3 Add a new CV strategy

1. Write a callable in `validation/cv_strategies.py` that takes
   `records: List[SessionRecord]` and `**kwargs` and yields fold dicts
   with keys `id`, `idx`, `train`, `test`, and optionally `val`,
   `split_kind`, `test_fraction` for temporal tails.
2. Register it in `STRATEGIES`.
3. (Optional) Add a friendly label and tooltip to `validation_tab.py`.

### 16.4 Add a new device backend

1. Implement an `EMGDevice` subclass in `devices/emg_device.py`. Emit
   `(n_samples, n_channels)` `float32` via whatever signal hook the rest
   of the code connects to (`data_ready` Qt signal or a callback).
2. Add a new entry to `DeviceType`.
3. Wire device selection into the GUI's connection controls.

---

## 17. Known limitations and caveats

- **Inflated within-session accuracy.** The `within_session` CV strategy
  uses a temporal tail, which is optimistic because the user and hardware
  don't change across the split. Never report this as a generalisation
  number.
- **Temporal tail with trial-ordered windows.** Because windows are stored
  in trial order and some sessions hold gestures back to back without
  shuffling, the "last 20%" of a session's windows can be concentrated on
  the last few classes. The runner logs a WARNING when this happens, but
  the user still sees the number. Treat with care.
- **Class imbalance handling.** Default class weights are uniform. The
  Quattrocento dialog exposes "balanced" class weights; the standard
  Train tab does not yet.
- **MPS is the supported GPU.** CUDA works but is less tested. Deep-model
  reproducibility is best on CPU or MPS.
- **CNN trained on 2-D features.** Before Patch 4 of the audit this
  combination trained successfully but crashed at prediction time. Now
  fixed; historical CNN-on-features models remain loadable but the
  prediction path is only correct after the patch is applied.
- **Bracelet rotation on 16-channel bipolar.** The calibrator's circular
  shift assumes all channels rotate together. Bipolar recordings halve
  the channel count but the same assumption holds — just with half the
  resolution.
- **No cross-subject calibration transfer.** A reference calibration
  works within a subject/bracelet-pair. Moving to a new subject or a
  different bracelet requires recording a new reference.

---

## 18. Glossary

| Term                | Definition                                                                                 |
|---------------------|--------------------------------------------------------------------------------------------|
| sEMG                | Surface electromyography — non-invasive muscle-electrical recording via skin electrodes.   |
| Session             | One continuous recording for one subject, ~1–10 min, one gesture set.                      |
| Trial               | A single gesture-hold within a session; labelled with a gesture id.                        |
| Window              | A fixed-length slice of a trial, typically 200 ms, with 50 ms stride.                      |
| Dataset             | A windowed, optionally feature-extracted `(X, y)` built from one or more sessions.         |
| Model               | A trained classifier plus its metadata, persisted under `data/models/`.                    |
| Domain              | `pipeline` (Python recorder) or `unity` (C# recorder) — same on-disk layout.               |
| LOSO                | Leave-One-Subject-Out — train on everyone but one, test on the one. Canonical headline.    |
| LOSO-session        | Same idea at session granularity — hold out one session, train on the others.              |
| Holdout             | A single train / val / test split with explicit ratios. Used for deep-model tuning.        |
| Calibration-sync    | A wave-out gesture recorded at session start, used only for rotation detection.            |
| Rotation offset     | Integer number of channels the bracelet is rotated relative to the reference.              |
| Bad channel         | An electrode declared unreliable for this session; handled by interpolation or zeroing.    |
| Smoothing           | EMA + stability gate applied to live predictions to prevent class flicker.                 |

---

## Appendix A — Default values at a glance

| Parameter                    | Default | Where                             |
|------------------------------|---------|-----------------------------------|
| Sampling rate                | 2 kHz   | `DeviceConfig.sampling_rate`      |
| Channel count                | 32      | `DeviceConfig.num_channels`       |
| Window size                  | 200 ms  | `RecordingConfig.window_size_ms`  |
| Window stride                | 50 ms   | `RecordingConfig.window_stride_ms`|
| EMA alpha                    | 0.3     | `PredictionSmoother`              |
| Min stable time              | 150 ms  | `PredictionSmoother`              |
| Train/val split (non-val CV) | 80/20   | `ModelManager.train_model`        |
| Default seed                 | 42      | `PipelineConfig.seed`             |
| Default CV strategy          | LOSO-subject | `validation_tab` combo       |
| TCP host                     | 127.0.0.1 | `prediction_server.main`        |
| TCP port                     | 5555    | `prediction_server.main`          |

## Appendix B — Command cheatsheet

```bash
# Launch the GUI
python -m playagain_pipeline.run_gui

# Inspect the corpus
python -m playagain_pipeline.validation summary
python -m playagain_pipeline.validation list

# Run a YAML experiment
python -m playagain_pipeline.validation run \
    playagain_pipeline/validation/configurations/experiments_example.yaml

# Start the prediction server standalone (model in data/models/<name>/)
python -m playagain_pipeline.prediction_server --model <name>
python -m playagain_pipeline.prediction_server --model <name> \
    --host 127.0.0.1 --port 5555 --device muovi
```

## Appendix C — File-size hints for a typical run

- A 5-minute 32-channel 2 kHz session: ~76 MB as `data.npy`, ~230 MB as
  `data.csv`.
- A dataset built from 12 such sessions with 200 ms windows / 50 ms stride
  and 8 time-domain features: `X.npy` ~ 45 MB.
- A LOSO validation run over 12 subjects × 4 models: `results.json` < 1 MB
  after the `_slim_train_meta` patch (was ~30–60 MB before).
