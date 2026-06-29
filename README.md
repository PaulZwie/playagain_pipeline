# The PlayAgain EMG Gesture Pipeline

## Start here

From a clean checkout to a validated model in five steps (each links to the full
details below):

1. **Install** — build the environment from the lockfile:
   ```bash
   cd playagain_pipeline && uv sync
   ```
   ↳ [Setup and running](#setup-and-running)
2. **Prepare data** — restore the archived study data into
   `playagain_pipeline/data/` (or record your own sessions in the GUI). It is
   archived in `_thesis_archive_REMOVED/data/pipeline_data/`.
   ↳ [Data model and on-disk schema](#3-data-model-and-on-disk-schema)
3. **Run the GUI**:
   ```bash
   python run_gui.py
   ```
   ↳ [Graphical user interface](#14-graphical-user-interface)
4. **Train** — in the GUI: **Record new data** → **Train a model** (builds a
   windowed dataset from the sessions and fits a classifier).
   ↳ [Training procedures](#10-training-procedures)
5. **Validate** — run reproducible cross-validation via **Tools ▸ Validate
   models…**, or headless:
   ```bash
   python -m playagain_pipeline.validation run \
       playagain_pipeline/validation/configurations/experiments_example.yaml
   ```
   ↳ [Validation harness](#11-validation-harness)

---

## Table of contents

- [Start here](#start-here)
- [Thesis submission — repository contents](#thesis-submission--repository-contents)
- [Setup and running](#setup-and-running)
- [Project structure](#project-structure)
- [1. Goals and scope](#1-goals-and-scope)
- [2. System architecture](#2-system-architecture)
- [3. Data model and on-disk schema](#3-data-model-and-on-disk-schema)
- [4. Signal acquisition](#4-signal-acquisition)
- [5. Calibration](#5-calibration)
- [6. Recording protocol engine](#6-recording-protocol-engine)
- [7. Dataset construction](#7-dataset-construction)
- [8. Feature pipeline](#8-feature-pipeline)
- [9. Model catalogue](#9-model-catalogue)
- [10. Training procedures](#10-training-procedures)
- [11. Validation harness](#11-validation-harness)
- [12. Real-time prediction and smoothing](#12-real-time-prediction-and-smoothing)
- [13. Unity integration and game recording](#13-unity-integration-and-game-recording)
- [14. Graphical user interface](#14-graphical-user-interface)
- [15. Configuration system](#15-configuration-system)
- [16. Reproducibility and determinism](#16-reproducibility-and-determinism)
- [17. End-to-end workflow](#17-end-to-end-workflow)
- [18. Utility and analysis scripts](#18-utility-and-analysis-scripts)
- [19. Scripting API](#19-scripting-api)
- [20. Extension points](#20-extension-points)
- [21. Known limitations and caveats](#21-known-limitations-and-caveats)
- [22. Glossary](#22-glossary)
- [Appendix A — Default values at a glance](#appendix-a--default-values-at-a-glance)
- [Appendix B — Command cheatsheet](#appendix-b--command-cheatsheet)
- [Appendix C — File-size hints for a typical run](#appendix-c--file-size-hints-for-a-typical-run)

---

## Thesis submission — repository contents

This is the **cleaned source submission** of the `playagain_pipeline` package.
It contains all Python source, configuration (`pyproject.toml`, `uv.lock`,
`config.json`) and documentation needed to install and run the pipeline, but
**not** the regenerable environment or the recorded study data.

| Not included | Why | How to obtain |
|---|---|---|
| `.venv/` | Virtual environment, reproducible from `uv.lock` | `uv sync` (see [Setup](#setup-and-running)) |
| `__pycache__/`, `catboost_info/` | Python / CatBoost caches | Regenerated automatically on run |
| `data/` (`sessions/`, `datasets/`, `models/`, `game_recordings/`, `calibrations/`, `Participant_Info/`, `validation_runs/`, …) | Recorded EMG study data and derived artifacts — data, not code | Archived separately (`_thesis_archive_REMOVED/`); restore under `playagain_pipeline/data/` using the layout in [§3](#3-data-model-and-on-disk-schema) to reprocess real recordings |

The pipeline installs and runs from this clean checkout after `uv sync`. The
`data/` directory is (re)created as you record sessions, build datasets and
train models; to reproduce the thesis results, restore the archived `data/`
into `playagain_pipeline/data/` first. The recorded study data is archived in
`_thesis_archive_REMOVED/data/pipeline_data/` (a sibling of the project,
gitignored); from inside `playagain_pipeline/data/` it can be restored with:

```bash
mv ../../../../_thesis_archive_REMOVED/data/pipeline_data/* .
```

---

## Setup and running

### Requirements

From `pyproject.toml`:

- Python `>=3.11,<3.13`
- Package name: `playagain-pipeline`

The project references local development dependencies via `tool.uv.sources` for
`device-interfaces` and `gui-custom-elements`. These sibling packages are
located automatically at startup by `utils/platform_utils.py` (no hard-coded
paths required).

### Install — Option A: `uv` (matches `pyproject.toml` source configuration)

```bash
cd /Users/paul/Coding_Projects/Master/Dataprocessing/playagain_pipeline
uv sync
```

### Install — Option B: `pip` editable install (if local dependency paths resolve)

```bash
cd /Users/paul/Coding_Projects/Master/Dataprocessing
python -m venv .venv
source .venv/bin/activate
pip install -e ./playagain_pipeline
```

### Run the GUI

```bash
cd /Users/paul/Coding_Projects/Master/Dataprocessing
python run_gui.py
```

`run_gui.py` (at the repo root) delegates to
`playagain_pipeline.gui.main_window.main()`, which instantiates `MainWindowV2`
(the current top-level window).

### Run the headless prediction server

```bash
cd /Users/paul/Coding_Projects/Master/Dataprocessing
python -m playagain_pipeline.prediction_server --model <trained_model_name>
```

Optional arguments (from `prediction_server.py`):

```bash
python -m playagain_pipeline.prediction_server --model <trained_model_name> \
    --host 127.0.0.1 --port 5555 --device muovi
```

Supported `--device` values: `muovi`, `muovi_plus`, `synthetic`.

---

## Project structure

```text
playagain_pipeline/
├── __init__.py
├── prediction_server.py
├── game_recorder.py
├── training_game_coordinator.py
├── unity_launcher.py
├── config.json
├── pyproject.toml
├── README.md
│
├── calibration/
│   ├── __init__.py
│   ├── calibrator.py
│   ├── calibration_stability.py
│   └── calibration_validation.py
├── config/
│   ├── __init__.py
│   └── config.py
├── core/
│   ├── __init__.py
│   ├── data_manager.py
│   ├── gesture.py
│   └── session.py
├── devices/
│   ├── __init__.py
│   └── emg_device.py
├── evaluation/
│   ├── __init__.py
│   ├── core.py
│   ├── cross_domain_eval.py
│   ├── game_eval.py
│   ├── intra_session_eval.py
│   ├── loaders.py
│   ├── metrics.py
│   ├── session_eval.py
│   ├── threshold_eval.py
│   └── unity_eval.py
├── gui/
│   ├── __init__.py
│   ├── gui_style.py
│   ├── main_window.py
│   └── widgets/
│       ├── main_window_v2.py
│       ├── home_tab.py
│       ├── workflow_stepper.py
│       ├── status_strip.py
│       ├── emg_plot.py
│       ├── emg_plot_panel.py
│       ├── evaluation_tab.py
│       ├── thesis_report_dialog.py
│       ├── protocol_popup.py
│       ├── protocol_widget.py
│       ├── game_protocol_popup.py
│       ├── bracelet_graphic.py
│       ├── busy_overlay.py
│       ├── calibration_dialog.py
│       ├── config_dialog.py
│       ├── feature_selection.py
│       ├── participant_groups_dialog.py
│       ├── quickstart_wizard.py
│       ├── quattrocento_loader.py
│       ├── quattrocento_training_dialog.py
│       └── training_dialog.py
├── models/
│   ├── __init__.py
│   ├── classifier.py
│   └── feature_pipeline.py
├── performance_assessment/
│   ├── __init__.py
│   ├── _generate_plots.py
│   ├── model_comparison.py
│   ├── performance_assessment.ipynb
│   ├── session_picker_ui.py
│   └── results/                       # runtime-generated, gitignored (not shipped)
├── protocols/
│   ├── __init__.py
│   └── protocol.py
├── utils/
│   ├── __init__.py
│   ├── platform_utils.py
│   ├── rest_gap_filler.py
│   └── migrate_requested_gesture.py
├── validation/
│   ├── __init__.py
│   ├── __main__.py
│   ├── config.py
│   ├── corpus.py
│   ├── corpus_report.py
│   ├── cv_strategies.py
│   ├── feature_cache.py
│   ├── game_corpus.py
│   ├── game_report.py
│   ├── intra_session_compare.py
│   ├── calibration_report.py
│   ├── generate_thesis_outputs.py
│   ├── participant_groups.py
│   ├── plots_thesis.py
│   ├── recompute_calibration_metrics.py
│   ├── runner.py
│   ├── thesis_reports.py
│   ├── threshold_plots.py
│   ├── threshold_report.py
│   ├── configurations/
│   │   └── experiments_example.yaml
│   ├── README.md
│   └── README_report.md
└── data/                          # runtime-generated; recorded study data archived separately
    └── README.md                   # only this ships — see "Thesis Submission" table + §3
```

> The `data/` subtree (`sessions/`, `datasets/`, `models/`, `calibrations/`,
> `game_recordings/`, `Participant_Info/`, `unity/`, `thesis_outputs/`,
> `validation_runs/`) is created at runtime and is archived separately for the
> submission — see [§3](#3-data-model-and-on-disk-schema) for the full layout.

The GUI entry point `run_gui.py` lives one level up
(`/Users/paul/Coding_Projects/Master/Dataprocessing/run_gui.py`) and just calls
`playagain_pipeline.gui.main_window.main()`.

---

## 1. Goals and scope

The pipeline is an end-to-end research tool for hand-gesture recognition from
surface electromyography (sEMG). It supports four workflows that can be combined
in any order:

- **Record** — acquire EMG while the participant performs a prompted gesture
  protocol, save raw samples and trial-level annotations to disk. Includes an
  optional Unity-game-driven "easy mode" so first-time users (e.g. children) can
  collect their first dataset by playing.
- **Train** — build a windowed dataset from one or more sessions, fit a
  classifier, persist the model for later use.
- **Use live** — run a trained model on incoming EMG and stream predictions to a
  Unity client over TCP, optionally with a synchronised game-recording log.
- **Validate** — evaluate feature sets, models and cross-validation strategies
  reproducibly, dumping enough metadata on each run to reproduce it exactly.

Design goals that shape many of the details in this document:

- **Reproducibility over convenience.** Every model and every validation run
  persists the exact configuration, the git SHA of the code that produced it,
  and the list of session paths it consumed. Reruns on the same commit and data
  produce bit-identical numbers.
- **Honest cross-validation.** Splits happen at session granularity, never at
  window granularity — windows from the same recording never appear in both
  train and test.
- **Two recorders, one corpus.** Sessions from the Python recording path and the
  Unity C# recording path (`DataManager.cs` + `DeviceManager.cs`) share the same
  on-disk layout, so cross-domain experiments are a configuration change, not a
  glue-code project.
- **Pluggable models and features.** Feature extractors register themselves into a
  decorator-based registry (`@register_feature`) so the GUI and validation tooling
  pick them up automatically; classifiers are listed in a central
  `ModelManager.AVAILABLE_MODELS` table that a new model is added to, with the GUI
  and evaluation model lists kept in sync with it.

---

## 2. System architecture

The package is organised into loosely-coupled subpackages that each expose a
small public surface and hide their implementation.

```text
EMG Source (Muovi / Muovi Plus / Synthetic / Quattrocento replay)
    -> Recording + Dataset + Model Pipeline
    -> Real-time Prediction + Smoothing
    -> Unity TCP Bridge (JSON over newline-delimited TCP)
    -> Optional synchronized game recording (CSV + config.json metadata)
    -> Optional Training-Game coordinator (Unity-driven dataset collection)
```

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

| Subpackage                 | Role                                                                                                             |
|----------------------------|------------------------------------------------------------------------------------------------------------------|
| `devices/`                 | Abstract device interface plus concrete wrappers (Muovi, Muovi Plus, Synthetic, Quattrocento replay).            |
| `core/`                    | Domain model: `Gesture`, `GestureSet`, `RecordingSession`, `DataManager`.                                        |
| `calibration/`             | Electrode-rotation detection; writes per-session calibration JSON and a reference calibration.                   |
| `protocols/`               | State machine that sequences cues, holds, rests, and the calibration-sync phase.                                 |
| `models/`                  | Classifier abstractions + concrete models (LDA, SVM, RF, CatBoost, MLP, CNN, AttentionNet, MSTNet) and features. |
| `evaluation/`              | Single-run evaluation primitives (one model, one set of recordings, one set of metrics).                        |
| `validation/`              | Reproducible validation harness: corpus discovery, CV strategies, runner, thesis-report generators.             |
| `gui/`                     | PySide6 UI: main window, tab factories, dialogs, live plots, bracelet viewer.                                    |
| `performance_assessment/`  | Multi-feature / multi-model comparison scaffolding and plot generators (pre-dates the validation harness).       |
| `prediction_server.py`     | TCP server for Unity; also usable standalone.                                                                    |
| `game_recorder.py`         | Synchronised EMG + prediction + ground-truth logger.                                                             |
| `training_game_coordinator.py` | Unity-driven "easy mode" first-dataset collection.                                                          |

Primary data flows:

1. **Recording Flow** — device stream → `RecordingSession` → session files.
2. **Easy-mode Recording Flow** — device stream → `TrainingGameCoordinator` →
   Unity feedback loop → `RecordingSession` files (Unity drives the trial
   schedule; the coordinator broadcasts synthetic predictions so children can
   record their first dataset by playing).
3. **Training Flow** — sessions → windowed dataset → trained model directory.
4. **Prediction Flow** — live EMG → model inference → smoother → GUI/TCP output.
5. **Game Logging Flow** — EMG + predictions + Unity state → synchronized CSV
   rows + per-recording `config.json`.
6. **Validation Flow** — configured experiment YAML → rigorous CV across
   sessions → timestamped artifact folder under `data/validation_runs/`.

Threading model:

- The GUI runs on the main Qt thread.
- Devices emit samples on their own thread; they reach the UI through Qt signals
  connected with `Qt.QueuedConnection`, which queues the call onto the UI thread.
- Training, dataset creation, and validation each run on a `QThread` worker so
  the UI remains responsive.
- The prediction server has a socket accept loop on one thread, a per-client
  receive loop on another, and a prediction worker on a third; the GUI polls
  results via signals.

By default the GUI and standalone prediction server use the in-repo
`playagain_pipeline/data/` folder, while the validation CLI auto-detects a
`data/` folder near the package (or falls back to `cwd/data`).
`platform_utils.get_default_data_dir()` returns the cross-platform
`~/Documents/PlayAgain/data` location (`%USERPROFILE%/Documents/PlayAgain/data`
on Windows) for external tooling or future overrides.

---

## 3. Data model and on-disk schema

The pipeline reads and writes all study data under a `data/` root, created and
populated automatically as you record sessions, build datasets and train models.
A clean checkout starts with only `data/README.md`. Expected top-level layout:

```text
data/
├── sessions/            # Raw recording sessions (data.npy, metadata.json, gesture_set.json)
├── datasets/            # Windowed datasets built from sessions (X.npy, y.npy, trial_ids.npy)
├── models/              # Trained models (.pkl/.pt/.cbm) + metadata.json
├── calibrations/        # Reference + per-session rotation calibrations
├── game_recordings/     # Synchronized EMG + predictions + ground truth from gameplay
├── Participant_Info/    # Per-participant metadata (<subject_id>.json)
├── validation_runs/     # Reproducible validation outputs (config, env, metrics)
└── thesis_outputs/      # Generated thesis tables/figures
```

### 3.1 Sessions

A recording session is a single, continuous EMG stream plus metadata. The
canonical layout is:

```text
data/sessions/<subject_id>/<session_id>/
    data.npy             # (N_samples, N_channels) float32 (raw EMG)
    data.csv             # same data, CSV-formatted for inspection
    metadata.json        # session + trials + channel-mapping + rotation info
    gesture_set.json     # labels + display metadata
```

`session_id` is derived from the recording timestamp plus a suffix that encodes
the repetition count or recording mode (e.g. `2026-03-20_12-54-45_10rep` or
`2026-03-20_12-54-45_manual`).

`metadata.json` has two top-level keys: `metadata` (describing the whole
session) and `trials` (an ordered list of per-trial annotations). The
session-level `metadata` includes everything needed to interpret the raw
samples: `device_name`, `num_channels`, `sampling_rate`, `gesture_set_name`,
`protocol_name`, `calibration_applied`, `channel_mapping`, `rotation_offset`,
`rotation_confidence`, `bad_channels`, and a free `custom_metadata` bag. The
signal mode is persisted in `custom_metadata` under both `signal_mode` and the
legacy boolean `bipolar_mode`.

`RecordingMetadata` (in `core/session.py`) additionally carries:

- `rotation_offset` and `rotation_confidence` — the bracelet rotation detected
  for this session relative to the reference (filled by the calibrator).
- `bad_channels` — 0-based channel indices marked as bad, applied at dataset
  build time via `apply_bad_channel_strategy`.
- `channel_mapping` — the resolved channel order after rotation correction.

Each trial entry records the gesture performed and the sample window the
participant held the gesture during:

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

`RecordingTrial.trial_type ∈ {"gesture", "calibration_sync"}`:

- `"gesture"` (the default) — a normal training trial; included in
  `RecordingSession.get_valid_trials()`.
- `"calibration_sync"` — the wave-out sync gesture optionally recorded at the
  start of each session for rotation detection. Stored with `gesture_label = -1`
  and **excluded from model training**; retrievable via
  `get_calibration_trials()`.

Older sessions predate the `trial_type` field — `RecordingTrial.from_dict` fills
in the default `"gesture"` so they load transparently. Loaded sessions
memory-map `data.npy` (read-only) so only accessed pages are faulted into RAM —
important for large multi-session datasets.

### 3.2 Unity sessions

Sessions recorded from within the Unity game follow the same four-file layout
but live under `data/sessions/unity_sessions/<subject>/<session>/`. The
validation corpus detects this path and tags the session with
`source_domain = "unity"`, enabling cross-domain (pipeline ↔ unity) experiments
without any custom code. `DataManager.list_sessions(...)` lists them
transparently.

### 3.3 Datasets

A dataset is a windowed, feature-extracted (or raw) numpy array derived from one
or more sessions:

```text
data/datasets/<name>/
    X.npy            # features or raw windows
    y.npy            # int labels
    trial_ids.npy    # per-window trial id for leakage-aware splits
    metadata.json    # window size, stride, feature config, label names
```

`X.shape` is `(n_windows, n_features)` when features are extracted and
`(n_windows, window_size_samples, n_channels)` when raw windows are requested.
`metadata.json` records exactly what was done to produce the dataset
(`bad_channel_mode`, `signal_modes_used`, `features_extracted` +
`feature_config`, `per_session_rotation`, `session_rotation_offsets`, the source
session IDs (`sessions_used`), sampling rate), so a dataset records exactly which
sessions it was built from.

### 3.4 Models

```text
data/models/<model_name>/
    <backend-specific binaries>   # e.g. model.pt + scaler.pkl for MLP/CNN;
                                  # .pkl for sklearn models; .cbm for CatBoost
    metadata.json                 # ModelMetadata — class names, features,
                                  # bad-channel mode, feature config, etc.
    params.json                   # only for deep models — architecture
                                  # shapes needed to rebuild the nn.Module
```

`ModelMetadata` carries the information a caller needs to run the model without
consulting the original training code — input/training-time channel count,
expected sampling rate, the feature configuration used, the bad-channel strategy
used, and the mapping from integer labels to gesture names.

### 3.5 Calibrations

Calibration files record per-gesture activation patterns and the detected
rotation offset, kept **per signal mode**:

```text
data/calibrations/
    reference_calibration_monopolar.json  # current reference for monopolar acquisition
    reference_calibration_bipolar.json    # current reference for bipolar acquisition
    reference_calibration.json            # legacy, honoured for backward compatibility
    calibration_<session_id>[_<mode>].json # per-session calibration snapshots
    plots/                                # optional PNG overlays
```

The reference calibration acts as the target that later calibrations align to;
`rotation_offset` is the integer number of channels the bracelet is rotated
relative to the reference.

### 3.6 Game recordings

```text
data/game_recordings/<subject_id>/game_[<name>_]<timestamp>/
    recording.csv   # EMG + prediction + Unity ground-truth rows
    config.json     # model used, class names, calibration + participant snapshot
```

Rows interleave raw EMG samples with prediction updates and Unity state changes,
all timestamped from a single clock so alignment is trivial. See
[§13.2](#132-game-recorder) for column details.

### 3.7 Participant info

```text
data/Participant_Info/<subject_id>.json
```

Persistent participant record (name fields, demographics, free-form metadata)
saved/loaded by `DataManager.save_participant_info` / `load_participant_info`.

### 3.8 Validation runs

```text
data/validation_runs/<timestamp>__<name>/
    experiment.json          # frozen ExperimentConfig
    environment.json         # git SHA, Python/numpy/scipy/sklearn/catboost/torch/pandas/PySide6 versions
    session_index.json       # every session path used across all folds (subject, domain, sr)
    results.json             # per-fold metrics + aggregates + summed confusion matrices
    results.csv              # per-fold flat table, pandas/Excel-friendly
    per_class_f1.csv         # per-class F1 per fold
    plots/                   # optional post-hoc plot output
```

The combination of `experiment.json` + `environment.json` + `session_index.json`
is sufficient to fully reconstruct a run; a collaborator with the same git
commit and the same `data/` directory gets bit-identical numbers.

---

## 4. Signal acquisition

### 4.1 Gestures and labeling

`core/gesture.py` defines `Gesture`, `GestureSet`, and `GestureCategory`.

- Default gesture set (`create_default_gesture_set()`):
  - `rest` (label `0`), `fist` (label `1`), `pinch` (label `2`), `tripod`
    (label `3`).
- Single-gesture sets (`create_single_gesture_set("fist" | "pinch" | "tripod")`)
  pair rest with one target gesture — used by the per-gesture protocols.
- Gestures include display metadata (`display_name`, `emoji`, `duration_hint`).
- Gesture sets serialize to JSON and are stored with each session
  (`gesture_set.json`).

### 4.2 Signal modes (monopolar / bipolar)

The pipeline supports two EMG signal-acquisition modes, selected on the Record
tab via the **Bipolar Mode** checkbox:

- **Monopolar** (default): raw per-channel signal from each electrode.
- **Bipolar**: top-minus-bottom differential across the bracelet's two rings.
  Each pair becomes a single channel, so the channel count is halved.

Downstream consumers (calibration, dataset creation) read the persisted mode and:

- Maintain **separate reference calibrations** for monopolar vs. bipolar
  (`reference_calibration_monopolar.json` / `reference_calibration_bipolar.json`).
- **Refuse to mix signal modes** in a single dataset — `DataManager.create_dataset`
  raises if sessions have inconsistent channel counts.

### 4.3 Device abstraction

`devices/emg_device.py` exposes `DeviceType` (`SYNTHETIC`, `MUOVI`,
`MUOVI_PLUS`, `QUATTROCENTO`, `CUSTOM`) and a `DeviceManager`-like interface.
Concrete backends are chosen by `device_type` in `DeviceConfig`. All devices
emit EMG samples as a numpy array shaped `(n_samples, n_channels)` and typed
`float32`.

- The Muovi wrapper delegates to the `device_interfaces` package maintained
  separately from this codebase.
- The synthetic backend (`SyntheticEMGDevice`) generates band-limited Gaussian
  noise modulated by the active gesture; it's sufficient for pipeline-level
  testing without hardware.
- Quattrocento replay is implemented through a GUI workflow in
  `gui/widgets/quattrocento_loader.py` + `gui/widgets/quattrocento_training_dialog.py`
  rather than a device subclass, because Quattrocento files arrive as `.pkl` /
  `.otb+` blobs that need label reconstruction before they can be treated like a
  stream.

### 4.4 Sampling rate and channel count

Canonical setup is **32 channels at 2 kHz** (Muovi monopolar). The pipeline also
handles 16-channel bipolar (Muovi Plus), 64-channel (Quattrocento), and custom
channel counts on the synthetic device. Sampling rate is carried in session
metadata; nothing in the pipeline hardcodes it — *with one historical exception
that was fixed during the rework*: the validation runner formerly read
`cfg.windowing.sampling_rate` via a `getattr` with a default of 2000 Hz,
silently defaulting for every run because `WindowingConfig` has no such field.
The current runner reads the rate from session metadata via
`_sampling_rate_for_fold`.

### 4.5 Bad-channel handling

The GUI's live EMG plot lets the user flag channels as bad per-session by
toggling a checkbox. Bad-channel indices are persisted to
`metadata.bad_channels` and the selected strategy is persisted in
`metadata.custom_metadata["bad_channel_mode"]`.

Two strategies are available, implemented in
`classifier.apply_bad_channel_strategy(data, bad_channels, mode=...)`:

- `"interpolate"` — replace a bad channel's samples with the average of its two
  nearest non-bad channels on the bracelet. Uses a cyclic nearest-neighbour
  search so the electrode at index 0 falls back to indices 31 and 1 rather than a
  random neighbour.
- `"zero"` — zero the bad channel's samples. Cheaper and loses less data, but
  forces downstream models to learn that "no signal" is a valid state for a
  channel that sometimes carries real signal.

The strategy is applied once when the dataset is constructed and again at
prediction time if the live stream's channel count matches the model's
training-time channel count.

---

## 5. Calibration

EMG bracelets rotate on the forearm between sessions. A model trained with the
bracelet at orientation A produces garbled predictions if the user wears it at
orientation B. The calibrator solves this by measuring per-electrode activation
patterns for a known gesture (wave-out) and finding the circular shift that best
aligns the current pattern to a reference. Calibration is performed directly from
normal recording sessions — no separate calibration recording is required.

Calibration is split across three modules.

### 5.1 `calibrator.py` — rotation detection

`CalibrationProcessor` and `AutoCalibrator` implement rotation-aware channel
alignment via the Maximum Energy Channel (MEC) method from Barona López et al.,
*Sensors* 2020. The processor computes per-channel energy using

```
E_ch = Σ |x_i·|x_i| − x_{i-1}·|x_{i-1}||
```

folds the 32-channel pattern into 16 angular sectors, and detects the rotation
offset via circular cross-correlation against the reference. Given a current
pattern `P` and a reference pattern `R`, both of length `N = num_channels`:

```
offset = argmax_{k ∈ [0, N)} Σ_i P[(i + k) mod N] · R[i]
```

and the confidence is the z-score of the cross-correlation peak — `(peak − mean) /
std` taken over the cross-correlation values — scaled into `[0, 1]` via
`clip(z / 5, 0, 1)`. The wave-out gesture was chosen because independent testing found it
produces the sharpest azimuthal energy peak across subjects, which makes the
cross-correlation against the reference the most selective.

Applying `signal_aligned[:, i] = signal_raw[:, mapping[i]]` yields a stream whose
electrodes match the orientation the reference was recorded at. For the
32-channel Muovi bracelet, `calibrator.create_channel_mapping` builds the mapping
on the physical 2×16 (inner/outer ring) topology rather than as a flat circular
shift: the scalar offset is split into a column shift (`offset mod 16`) and a row
shift (`offset // 16`), and each electrode `(row, col)` is remapped to
`((row + offset // 16) mod 2, (col + offset mod 16) mod 16)`. For an in-plane
rotation around the wrist the offset stays in `[0, 16)`, so only columns shift;
the row term lets a larger offset swap the two rings. Devices with other channel
counts fall back to a plain circular shift `mapping[i] = (i + offset) mod N`.

References are kept per signal mode
(`reference_calibration_monopolar.json` / `reference_calibration_bipolar.json`);
each per-gesture activation vector has length `num_channels` and records the
waveform-length (Barona-López) energy per electrode, using the same `E_ch` formula
shown above. A calibration can be declared
`reference_incompatible` when the channel count or signal mode differs from the
reference, in which case the UI falls back to manual entry of the rotation
offset. `backfill_session_rotations(...)` is a one-shot migration utility that
re-detects and writes rotation offsets for all historical sessions.

### 5.2 `calibration_stability.py` — stability metrics

`compute_stability_metrics(...)` returns a `StabilityResult` whose primary
`stability` value answers "do my trials agree on the same offset?" rather than
the legacy peak-prominence z-score (which measured the sharpness of the user's
activation pattern, not the correctness of the offset). The legacy
`peak_prominence` is still reported as a diagnostic, alongside `top2_ratio`, the
per-trial offsets, and an optional bootstrap distribution.

### 5.3 `calibration_validation.py` — acceptance checks

Runs a battery of independent acceptance checks on a fitted calibration and
produces a `CalibrationReport`:

- **`self_consistency`** (error): the calibrator must agree with itself within
  `SELF_CONSISTENCY_MAX_DRIFT` channels when re-run on the same data.
- **`symmetry`** (warning): the primary energy peak is expected to dominate its
  half-ring (180°) twin by at least `SYMMETRY_MIN_RATIO` — a failure flags that the
  bracelet may be on backwards. It is emitted as a warning, so it does not by
  itself block acceptance.
- **`held_out_accuracy`** (error, when a held-out session is supplied): top-1
  classification accuracy must clear `HELD_OUT_MIN_ACCURACY`.
- **`confidence`** (warning by default; the underlying method is
  `_check_confidence_floor`): the confidence number is compared against a floor;
  low values are informational unless explicitly upgraded to an error.

`CalibrationReport.is_acceptable` passes when every `error`-severity check passes;
`warning`-severity checks can fail without blocking acceptance.

`BatchValidator` runs `CalibrationValidator` across many sessions with
save/restore around the calibrator state (so the user's current calibration is
not mutated by the batch). `RotationDetectionStudy` aggregates per-session
offsets and drift relative to a reference session for corpus-wide "is rotation
detection stable across my recordings?" analyses; per-check pass-rates are computed
separately by `BatchReport.check_pass_rates`.

### 5.4 Calibration-sync trials

To avoid asking users to run a separate calibration session, every training
session can start with a brief sync phase (wave-out) recorded as trials with
`trial_type = "calibration_sync"`. Model training explicitly excludes these via
`RecordingSession.get_valid_trials` so they never become training examples; the
calibrator consumes them via `get_calibration_trials`.

---

## 6. Recording protocol engine

`protocols/protocol.py` is a deterministic state machine that sequences the
visual prompts shown to the participant during a session. `ProtocolPhase`
includes:

- `PREPARATION` — one-off intro shown at the very start.
- `CALIBRATION_SYNC` — wave-out sync gesture recorded once at the start.
- `REST` — neutral hand position inserted between active steps (typically 2–8 s;
  1.5 s for the calibration preset).
- `HOLD` — the active gesture itself; this window is labelled as a trial.
- `COMPLETE` — end-of-protocol marker.

`ProtocolPhase` also defines `CUE`, `RELEASE`, and `FEEDBACK` members, but the
current `_build_protocol` sequence does not emit them — the "next gesture" notice
is folded into the preceding `REST` step.

`RecordingProtocol` composes step sequences (`ProtocolStep`) from a `GestureSet`
and a `ProtocolConfig`. The `ProtocolConfig` carries the timings (preparation,
cue, hold, release, rest) and the number of repetitions per gesture;
`ProtocolConfig.include_calibration_sync` toggles whether each session opens with
a wave-out sync trial. The protocol enumerates all steps up front so the total
duration is known before recording starts — handy for displaying an ETA.

The engine is UI-agnostic: the GUI's `ProtocolWidget` subscribes to
`step_started` and `step_completed` signals and updates its display accordingly.
Each recording step bounds its own trial: the session starts a trial when a
`HOLD`, `REST`, or `CALIBRATION_SYNC` step begins and ends it when that same step
completes.

Built-in factories: `create_quick_protocol`, `create_standard_protocol`,
`create_extended_protocol`, `create_calibration_protocol`, plus per-gesture
single-target protocols (`create_pinch_protocol`, `create_tripod_protocol`,
`create_fist_protocol`).

---

## 7. Dataset construction

`DataManager.create_dataset` (in `core/data_manager.py`) is the single entry
point for turning one or more sessions into `(X, y)`. It performs four steps:

1. **Per-session preparation.** For each session, load raw samples, apply the
   bad-channel strategy, and optionally apply the calibration channel mapping. If
   `use_per_session_rotation=True`, each session's own `rotation_offset` is
   applied before windowing — recommended when combining sessions from different
   days. Accepts an optional `preprocessing_fn` and an explicit `calibration`.
2. **Windowing.** For each session, extract overlapping windows of
   `window_size_ms` with stride `window_stride_ms` from each valid trial. Windows
   crossing trial boundaries are discarded. This is where the leakage discipline
   lives: a window always belongs to exactly one trial, and trials belong to
   exactly one session.
3. **Labelling.** Windows inherit the gesture label of the trial they were drawn
   from. The dataset metadata carries a `label_names` mapping from integer →
   gesture name so downstream code can display readable class names.
4. **Feature extraction.** If a `feature_config` is supplied, windows are reduced
   to a feature vector per window via the feature pipeline ([§8](#8-feature-pipeline)).
   Otherwise `X` stays as raw windows; only CNN-family models accept this shape.

Per-session `bad_channels` plus a `bad_channel_mode` (`"interpolate"` or
`"zero"`) are honoured. Dataset metadata includes the exact window size, stride,
feature config snapshot, per-session rotation flag, bad-channel strategy, the
source session IDs (`sessions_used`), and the sampling rate — a record of how the
dataset was built. Optional memory-mapped loading is available via
`DataManager.load_dataset(name, mmap=True)`.

`DataManager` also handles: session save/load and subject/session listing with
sorted natural ordering (`VP_00 < VP_01 < VP_10`); legacy-safe path resolution
for sanitized/raw folder names (including `sessions/unity_sessions/<subject>/`);
automatic `VP_NN` subject-ID allocation via `get_next_subject_id()`; participant
metadata save/load; and train/validation/test splitting via sklearn helpers.

### 7.1 Mixed-channel-count handling

When combining sessions recorded with different hardware (16-ch bipolar vs 32-ch
monopolar), `create_dataset` raises. The validation runner handles this by
pre-filtering each fold to a single dominant channel count and logging dropped
sessions at `WARNING` level; see
`ValidationRunner._materialise_fold._filter_by_channels`.

---

## 8. Feature pipeline

`models/feature_pipeline.py` defines a registry-driven feature system. A feature
extractor subclasses `BaseFeatureExtractor` and registers itself via the
`@register_feature("<name>")` decorator. Each feature produces one or more
scalars per channel per window.

### 8.1 Built-in time-domain features

| Key     | Name                     | Formula (per channel, per window of size T)                 |
|---------|--------------------------|-------------------------------------------------------------|
| `rms`   | Root Mean Square         | `sqrt(mean(x²))`                                            |
| `mav`   | Mean Absolute Value      | `mean(\|x\|)`                                               |
| `var`   | Variance                 | `var(x)`                                                    |
| `wl`    | Waveform Length          | `sum(\|diff(x)\|)`                                          |
| `zc`    | Zero Crossings           | count of sign flips in x with `\|diff\|` above a threshold (def. 0.01) |
| `ssc`   | Slope Sign Changes       | count of sign flips in `diff(x)` with consecutive diff magnitude above threshold |
| `iemg`  | Integrated EMG           | `sum(\|x\|)`                                                |
| `ssi`   | Simple Square Integral   | `sum(x²)`                                                   |                                       |

These eight features cover the classical Hudgins time-domain feature set (mav,
zc, ssc, wl) plus four additional descriptors widely used in the sEMG
literature. All features are purely time-domain; no spectral features are
implemented yet — adding one (e.g. median frequency) is a matter of adding a
`BaseFeatureExtractor` subclass.

### 8.2 Pipeline semantics

`FeaturePipeline.compute(data)` accepts either a 3-D batch
`(n_windows, T, n_channels)` or a 2-D single-window `(T, n_channels)`. It runs
every enabled feature on the batch and horizontally stacks the result, so the
output shape is `(n_windows, n_channels * n_features)`.

Feature order in the output is the order features were added to the pipeline,
grouped by feature. For N channels and K features, column layout is
`[feat0_ch0, feat0_ch1, ..., feat0_chN-1, feat1_ch0, ...]`. This matters when
aligning features to physical channels for bad-channel handling and for debugging
feature-attribution plots.

### 8.3 Raw-window mode

Deep models (`cnn`, `attention_net`, `mstnet`) don't accept features; they
operate on raw `(n_windows, T, n_channels)` tensors. The dataset builder returns
raw windows when `feature_config` is `None` or `mode="raw"`. The validation
runner auto-detects CNN-family models per fold and materialises the fold twice —
once as features for classical models and once as raw windows for deep models —
so a single mixed-model run is possible without configuring anything.

---

## 9. Model catalogue

All models inherit from `BaseClassifier` in `models/classifier.py` and expose the
same API:

```text
model.train(X_train, y_train, X_val=None, y_val=None, **kwargs)
model.predict(X) -> np.ndarray            # (n_windows,)
model.predict_proba(X) -> np.ndarray      # (n_windows, n_classes)
model.save(path); model.load(path)
```

The `**kwargs` channel forwards hyperparameters set on the constructor plus
runtime configuration (sampling rate, window size, class weights, and a
`random_state` seed when relevant). `ModelManager.AVAILABLE_MODELS` exposes eight
keys: `svm`, `random_forest`, `lda`, `catboost`, `mlp`, `cnn`, `attention_net`,
`mstnet`.

### 9.1 Classical models

| Key             | Class                    | What it is                           | When to use                          |
|-----------------|--------------------------|--------------------------------------|--------------------------------------|
| `lda`           | `LDAClassifier`          | Linear Discriminant Analysis         | Fast, strong EMG baseline (~ms inference) |
| `svm`           | `SVMClassifier`          | Linear/RBF support vector machine    | Small datasets, when decision boundary is likely smooth |
| `random_forest` | `RandomForestClassifier` | sklearn random forest                | Noise-robust, handles outliers well, no scaling needed |
| `catboost`      | `CatBoostClassifier`     | Gradient-boosted trees               | Often best on tabular time-domain features |

All classical models accept a 2-D feature matrix `(n_windows, n_features)` or a
3-D raw tensor (which they flatten internally). `lda`, `svm`, and `catboost`
standardise features with an `sklearn.StandardScaler` that's fit on the training
set and persisted with the model; `random_forest` is scale-invariant and skips
standardisation (as noted in the table above).

### 9.2 Deep models

| Key             | Class                      | Architecture                                            |
|-----------------|----------------------------|---------------------------------------------------------|
| `mlp`           | `MLPClassifier`            | Fully-connected net on feature vectors                  |
| `cnn`           | `CNNClassifier`            | 1-D convolutional net on raw windows `(N, C, T)`        |
| `attention_net` | `AttentionNetClassifier`   | CNN front-end + self-attention; subclasses `CNNClassifier` |
| `mstnet`        | `MSTNetClassifier`         | Multi-scale temporal CNN; subclasses `CNNClassifier`    |

Deep models use PyTorch. Device selection honours Apple MPS, CUDA, and CPU via
`models.classifier.resolve_device`. The MLP accepts either 2-D features or 3-D
raw windows (it flattens). The CNN family requires 3-D raw windows — a 2-D
feature tensor is coerced to `(N, 1, F)` at training time, and the same coercion
is applied during `predict()` and `predict_proba()` so CNN-on-features models
remain usable.

### 9.3 Shared infrastructure

- `ModelMetadata` records class names, feature config, bad-channel mode, and
  training-time channel count. Loading a model restores this metadata so the
  caller doesn't need to remember how the model was trained.
- `EMGFeatureExtractor` is a legacy helper that pre-dates the modular feature
  pipeline. It remains in use inside individual classifier implementations for
  backwards compatibility but is superseded by `feature_pipeline.FeaturePipeline`
  for new code.
- `ModelManager` is the factory that wraps the `AVAILABLE_MODELS` dict.
  `ModelManager.create_model(key, name=..., **hyperparams)` returns a trainable
  instance; `ModelManager.train_model(model, dataset, ...)` runs a stratified
  train/val split and calls the underlying `model.train`.
- `apply_bad_channel_strategy(data, bad_channels, mode=...)` implements the
  shared zero / linear-interpolate logic used at both training and inference time.

---

## 10. Training procedures

### 10.1 Standard GUI training (`gui/widgets/training_dialog.py`)

The Train tab does four things in order:

1. Builds the dataset — windowing + feature extraction per the current config —
   via `DataManager.create_dataset`.
2. Splits the dataset into train/val with
   `sklearn.model_selection.train_test_split(test_size=0.2, stratify=y,
   random_state=seed)`. The GUI uses `seed` when present in the loaded config
   object, otherwise defaults to `42` for reproducible splits.
3. Launches a `TrainingWorker(QThread)` that calls `model.train` with a progress
   callback for per-epoch reporting. Deep models emit `iteration_update` signals;
   classical models emit simulated progress.
4. On completion, saves the model to `data/models/<name>/`.

### 10.2 Advanced training (`TrainingProgressDialog` in advanced mode)

"Advanced Training…" opens the same `TrainingProgressDialog` with `model_type=None`
(window title "Advanced Model Training"). It lets you pick any model and a
pre-built dataset, and exposes hyperparameter editing for the chosen model, an
auto-LR finder for deep models, and feature-configuration editing before training.
The bad-channel strategy is read from the selected dataset's metadata rather than
overridden here, and data selection is by pre-built dataset — per-subject /
per-session selection lives in the Quattrocento dialog (§10.3). It shares
`TrainingWorker` with the standard dialog.

### 10.3 Quattrocento training (`gui/widgets/quattrocento_training_dialog.py`)

This dialog targets the 64-channel Quattrocento hardware. It consumes
pre-converted `.npy` recordings (`VHI_Recording_*.npy`); the loader accepts `.npy`
only, so upstream `.otb+`/`.pkl` acquisition formats must be converted first. It
supports k-fold within-subject CV and a LOSO variant for cross-subject
generalisation. Channel reduction via variance ranking or PCA is available; PCA
uses the dialog seed (default 42) for reproducibility. Cross-validation defaults to
**None** (use the split assignments); when within-subject k-fold is selected
instead it gives an optimistic accuracy estimate, surfaced in the dialog's own
documentation and the runtime log.

---

## 11. Validation harness

The validation harness (`playagain_pipeline/validation/`) is a reproducible,
config-driven means of producing results suitable for a thesis or paper. It runs
experiments end-to-end from a YAML configuration and persists everything needed
to reproduce them. It solves three problems:

1. **Two recorders, one corpus.** Sessions are written by two stacks — the
   Python pipeline (`data/sessions/<subject>/...`) and the Unity C# game
   (`data/sessions/unity_sessions/...`, `DataManager.cs` + `DeviceManager.cs`).
   Both write the same on-disk layout, so `SessionCorpus` discovers them
   uniformly and tags each with a `source_domain ∈ {"pipeline", "unity"}`.
   Cross-domain experiments become a one-line config change.
2. **Honest cross-validation.** Splitting random *windows* from one session into
   train/test is the single biggest source of inflated accuracy in EMG papers.
   All splitters operate at session granularity: no two windows from the same
   physical recording can ever be in both train and test.
3. **Reproducibility.** Every run dumps the exact config, git SHA, package
   versions, and the list of consumed session paths into a timestamped folder.
   Re-running the same YAML on the same git commit produces the same numbers.

### 11.1 CLI

```bash
# See what's on disk (counts per domain, subjects)
python -m playagain_pipeline.validation summary

# List individual sessions and their source domain
python -m playagain_pipeline.validation list

# Run an experiment from a YAML/JSON config
python -m playagain_pipeline.validation run \
    playagain_pipeline/validation/configurations/experiments_example.yaml
```

Example `summary` / `run` output:

```text
SessionCorpus @ /…/playagain_pipeline/data
  total sessions : 137
  pipeline       :  92
  unity          :  45
  subjects       :  11

Wrote: data/validation_runs/2026-04-14_113022__loso_baseline/
RandomForest        acc=0.812±0.041  f1=0.804±0.046  (n=11)
CatBoost            acc=0.838±0.038  f1=0.829±0.043  (n=11)
LDA                 acc=0.751±0.052  f1=0.738±0.057  (n=11)
```

### 11.2 The corpus

`SessionCorpus.discover()` walks `data/sessions/` once and caches every session
it finds. Each session is wrapped in a `SessionRecord` with `subject_id`,
`session_id`, `path`, `source_domain ∈ {pipeline, unity}`, `sampling_rate`,
`num_channels`, `label_names`, and `notes`. `source_domain` is derived from the
path — anything under `unity_sessions/` is `unity`, everything else is
`pipeline`. `SessionRecord` is intentionally lazy: it doesn't load the signal or
labels until asked. `SessionRecord.load_labels` tries both `labels.npy` and a
`label`/`gesture_id`/`ground_truth` column in `data.csv` to absorb Unity-recorder
variants. `game_corpus.py` discovers game recordings as `RecordingDescriptor`
objects (kind `GAME`) for game-CSV evaluation, and `participant_groups.py` provides
healthy/clinical group lookup.

### 11.3 CV strategies

Nine cross-validation strategies are registered in `cv_strategies.STRATEGIES` and
selectable from YAML. **Every strategy operates at session granularity** —
windows from the same recording cannot end up in both train and test. This is the
single most important methodological choice in the harness; it is what makes the
numbers credible.

| Strategy                     | Unit of split                 | Typical use                                                                                                  |
|------------------------------|-------------------------------|--------------------------------------------------------------------------------------------------------------|
| `within_session`             | Temporal tail of each session | Optimistic ceiling / sanity check; never a headline metric.                                                  |
| `loso_session`               | One session                   | Session-to-session generalisation within and across subjects.                                                |
| `intra_subject_loso_session` | One session (within subject)  | Same-subject generalisation; closer to real deployment.                                                      |
| `loso_subject`               | All sessions of one subject   | Subject to subject generalisation; goal for future deployment.                                               |
| `k_fold_sessions`            | k session groups              | Faster pooled-session baselines.                                                                             |
| `k_fold_subjects`            | k subject groups              | Use when LOSO is too expensive (>20 subjects).                                                               |
| `cross_domain`               | `source_domain`               | Does a model trained on pipeline data still work when played in Unity?                                       |
| `session_to_game`            | pipeline → game recordings    | Train on sessions, test on game-recorded CSVs.                                                               |
| `holdout_split`              | User-specified train/val/test | Explicit ratios (stratified by subject by default); best for deep-model tuning with real early-stopping val. |

`holdout_split` is the only strategy that produces a separate **validation set**.
When present, the runner passes it to the model's `train()` method as
`X_val` / `y_val`, so deep-learning models (`mlp`, `cnn`, `attention_net`,
`mstnet`) can use it for early stopping and live training curves. Classical
models ignore it gracefully and report test-set numbers only.

`cross_domain` requires both domains to be non-empty after the `data` filter:
asking for `domains: [unity]` *and* `cv.strategy: cross_domain` yields zero folds.

### 11.4 The runner

`ValidationRunner.run(cfg, progress=None)` executes a validated experiment
end-to-end:

1. Materialise the output directory and write `experiment.json`,
   `environment.json`, `session_index.json` upfront so an interrupted run still
   leaves a diagnostic trail.
2. Select sessions per `cfg.data` — either subject / domain filters, or an
   explicit session list.
3. Instantiate the CV strategy, enumerate folds, and compute
   `n_folds × n_models` total evaluations.
4. For each `(fold, model)` pair:
   - Seed numpy, stdlib `random`, and torch deterministically via a per-fold
     hash (SHA-1) of `(cfg.seed, fold_idx, model_type)`. This is what makes
     results reproducible across runs with the same seed.
   - Materialise the fold as features or raw windows depending on the model
     type (CNN-family keys live in `validation/runner._RAW_WINDOW_MODELS`).
   - Fit the model, measure training time.
   - Evaluate on the test split; capture accuracy, macro-F1, per-class F1,
     inference latency, and the confusion matrix.
   - If the strategy provides a val split (only `holdout_split`), also evaluate
     there.
5. Persist per-fold results to `results.csv` and `per_class_f1.csv`, and the
   full `results.json` with the aggregate summary and summed confusion matrices
   per model.
6. Between folds, check `progress.should_cancel()` — the GUI cancel button routes
   here.

The runner reuses the GUI's own code paths so behaviour matches byte-for-byte:

| Method                | Reuses…                                                                 |
|-----------------------|-------------------------------------------------------------------------|
| `_materialise_fold`   | `core.data_manager.DataManager` for windowing + `models.feature_pipeline.FeaturePipeline` for features (and `FeatureCache`). |
| `_fit_model`          | `models.classifier.ModelManager.create_model` + `train_model`.          |
| `_evaluate`           | `sklearn.metrics.accuracy_score` + `f1_score` (+ per-class F1, latency). |

`runner.py` deliberately avoids importing heavyweight ML deps at the top level;
the integration points all live inside `ValidationRunner`. The progress reporter
is a `Protocol`, not a base class: the GUI provides a Qt-bridge implementation,
headless scripts pass `None` and get the default `NoopProgress` reporter, whose
`on_*` hooks are no-ops while its `log()` still emits through the standard logger
at INFO level.

### 11.5 Reproducibility metadata

`environment.json` captures the ISO timestamp, `platform.platform()`,
`sys.version`, `sys.executable`, the git SHA and dirty flag (if inside a repo),
and versions of `numpy`, `scipy`, `scikit-learn`, `catboost`, `torch`, `pandas`
and `PySide6`. `session_index.json` is the ground truth of which sessions went
into the run — a reviewer can use it to re-verify the split; differ from it and
the numbers will not match.

### 11.6 Thesis-report generation layer

A reporting layer sits **on top of** `runner.py` (it does not replace it) so
every numerical placeholder and figure reference in the thesis chapters is
produced automatically from saved runner outputs. These modules read what is
already on disk — `RunResult` / `FoldResult` from `runner.py`, `SessionCorpus`
from `corpus.py`, and per-session `metadata.json` — and emit thesis-shaped
outputs without touching the recording or training paths. They depend only on
`numpy` and `matplotlib` (and optionally `scipy` for correlation p-values) — no
PySide6, torch, or sklearn at import time, so they run headless in CI.

Reporting modules (all alongside `runner.py`):

| Module                            | Role                                                        |
|-----------------------------------|-------------------------------------------------------------|
| `corpus_report.py`                | Participant + corpus tables.                                |
| `calibration_report.py`           | Calibration confidence tables/plots + flags.                |
| `thesis_reports.py`               | LOSO aggregation (mean ± SD), latency, cross-domain tables. |
| `plots_thesis.py`                 | matplotlib helpers for figures.                             |
| `threshold_report.py` / `threshold_plots.py` | Threshold-sweep summary tables and plots.                   |
| `game_report.py`                  | Game-recording evaluation tables.                           |
| `intra_session_compare.py`        | Intra-session model-comparison analysis.                    |
| `recompute_calibration_metrics.py`| CLI: recompute / rewrite calibration stats in place.        |
| `generate_thesis_outputs.py`      | One-shot orchestrator that builds all Thesis artefacts.     |

Artefacts produced (every table and figure referenced from Chapters 6–8):

| Thesis reference | Artefact |
|---|---|
| Table 6.1 (per-participant corpus) | `table_6_1_participants.csv` |
| Table 6.2 (class distribution) | `table_6_2_class_distribution.csv` |
| §6.1.3 (recording origins) | `section_6_1_3_recording_origins.json` |
| Fig 6.1 (calibration confidence) | `fig_6_1_calibration_confidence.{pdf,png}` |
| §6.2 (median, IQR, flagged counts) | `calibration_report.json` |
| Table 6.3 (LOSO-session, mean ± SD) | `table_6_3_loso_session.csv` |
| Fig 6.3 (per-class F1 with error bars) | `fig_6_3_per_class_f1.{pdf,png}` |
| Fig 6.4 (normalised confusion matrices) | `fig_6_4_confusion_matrices.{pdf,png}` |
| Fig 6.5 (per-session variability) | `fig_6_5_per_session_variability.{pdf,png}` |
| Table 6.4 (LOSO-subject) | `table_6_4_loso_subject.csv` |
| Table 6.5 (feature ablation) | `table_6_5_feature_ablation.csv` |
| Fig 6.6 (feature ablation bar chart) | `fig_6_6_feature_ablation.{pdf,png}` |
| Table 6.6 (cross-domain) | `table_6_6_cross_domain.csv` |
| Table 6.7 (latency + 150 ms gate) | `table_6_7_latency.csv` |
| §7.4 (calibration ↔ F1 correlation) | `correlation_calibration_f1.json`, `fig_7_4_calibration_vs_f1.{pdf,png}` |

Chapter 8 reuses numbers from the chapter-6 tables — no separate artefact.

Generate everything at once after your runs are done:

```bash
python -m playagain_pipeline.validation.generate_thesis_outputs \
    --data-dir         data/ \
    --primary          data/validation_runs/2026-04-12__loso_sess \
    --loso-subj        data/validation_runs/2026-04-13__loso_subj \
    --ablation         "rms=runs/abl_rms,mav=runs/abl_mav,var=runs/abl_var,wl=runs/abl_wl,zc=runs/abl_zc,ssc=runs/abl_ssc,iemg=runs/abl_iemg,ssi=runs/abl_ssi,combined=runs/abl_all" \
    --xdomain          "pipeline_to_game=runs/p2g" \
    --primary-model    catboost \
    --out              thesis_outputs/
```

Anything left off is skipped — partial generations work as soon as the matching
runs finish. Each module is also usable as a library, e.g.:

```python
from playagain_pipeline.validation.corpus import SessionCorpus
from playagain_pipeline.validation.corpus_report import participant_summary
from playagain_pipeline.validation.thesis_reports import (
    load_run_result, summarise_run, latency_table,
)

corpus = SessionCorpus("data/"); corpus.discover()
print(participant_summary(corpus))

run = load_run_result("data/validation_runs/2026-04-12__loso_sess")
for model, s in summarise_run(run).items():
    print(model, s.fmt_mean_sd("macro_f1"))

for row in latency_table(run, gate_ms=150.0):
    print(row.model_type, row.inference_ms_mean, "gate:", row.passes_gate)
```

The reporting modules read several optional `custom_metadata` fields the recorder
already writes (`source = "training_game"`, `outreach = true`,
`participant_info.group`, `exclude_from_eval`, `calibration_manual`); missing
fields fall back to `"?"` / a default session type rather than crashing.

### 11.7 Adding to the harness

- **New experiment.** Copy `configurations/experiments_example.yaml`, edit the
  `data`, `features`, `models`, and `cv` sections, and run it. Commit the YAML
  alongside any code change so the `environment.json` records the SHA.
- **New CV strategy.** Drop a function in `cv_strategies.py` taking
  `List[SessionRecord]` and yielding fold dicts of the form
  `{"id", "idx", "train", "test", ...}` (optionally `val`, `split_kind`,
  `test_fraction`), then register it in `STRATEGIES`. The runner picks it up by
  name from any YAML.
- **New metric.** `_evaluate` reports accuracy + macro-F1 + per-class F1 +
  per-window inference latency; extend it and the matching `FoldResult` fields.

### 11.8 Validation harness — known limitations

- The runner's error handling for a failing fold is *log and continue* — one
  crashed model doesn't kill the whole run, but it does leave a gap in the
  aggregate table.
- Mixed channel counts in a single fold are resolved by dropping sessions that
  don't match the dominant channel count (with a `WARNING`). Training separate
  models per channel count is out of scope.

---

## 12. Real-time prediction and smoothing

`prediction_server.py` runs the prediction loop in two deployment modes:

- Standalone CLI: `python -m playagain_pipeline.prediction_server --model <name>`.
- Embedded in the GUI: the **Use a model live** tab spawns the server in the app
  process.

Responsibilities:

- TCP server startup (`host` default `127.0.0.1`, `port` default `5555`).
- Model loading and per-buffer inference, with a dedicated prediction worker
  thread and a separate sender thread so network hiccups can't stall inference.
- `pause()` / `resume()` to stop predictions without tearing down the TCP server
  (used during calibration, training-game setup, etc.).
- Smoothed prediction output via `PredictionSmoother`.
- Bidirectional messaging with Unity clients.
- `enqueue_json_message(msg)` — public helper used by the
  `TrainingGameCoordinator` to inject synthetic predictions / target-gesture cues
  through the same sender thread that handles model predictions.
- Throttled `error_callback` so the GUI surfaces "your model can't predict on
  this stream" instead of a frozen `rest` label, without flooding the log.

### 12.1 Inference loop

A buffer of `200 ms * sampling_rate / 1000` samples is pre-allocated and
zero-filled — the inference window is fixed at `_prediction_window_ms = 200`,
independent of the model's trained `window_size_ms`. On each incoming EMG chunk the
server updates the buffer, extracts the latest window, applies the model's feature
pipeline (if configured), and calls `model.predict_proba(X)`. There is no warm-up
gate, so the earliest predictions run on a partly zero-padded window. The
prediction cadence is driven by incoming EMG chunks, so the inference rate depends
on the device frame size rather than a fixed training-time stride.

### 12.2 Smoothing

Raw gesture decisions flicker — a single low-confidence frame can jump to a
different class. `PredictionSmoother` combines two mechanisms:

- **Exponential moving average** on the predicted class probabilities with
  `alpha` (default `0.3`). Reduces high-frequency jitter.
- **Stability gate** — the output class only changes when the new class has been
  the argmax for at least `min_stable_ms` (`PredictionSmoother` default `150 ms`;
  the GUI spinbox ships `300 ms`). Prevents bistability at the boundary between two
  classes of similar probability.

Both parameters are exposed in the GUI under a "Prediction Smoothing" group.

---

## 13. Unity integration and game recording

### 13.1 Wire format

Newline-delimited JSON over TCP. Default `127.0.0.1:5555`. The server is agnostic
to additional fields in either direction — it forwards everything to registered
callbacks unchanged.

**Outgoing (Python → Unity):**

Handshake on connect:
```json
{"type": "handshake", "model_name": "...", "model_type": "...",
 "class_names": {"0": "rest", "1": "..."}, "num_classes": 4,
 "smoothing_enabled": true, "timestamp": 1712234567.123}
```

Per-prediction update (the default payload). `gesture_id` is the model's class
index (the argmax of `probabilities`), so its value for a given gesture depends on
the trained model's class ordering — the number below is illustrative:
```json
{"gesture": "fist", "gesture_id": 1, "confidence": 0.93,
 "probabilities": {"rest": 0.02, "fist": 0.93, "pinch": 0.03, "tripod": 0.02},
 "timestamp": 1712234567.123}
```

`target_gesture` — from the `TrainingGameCoordinator`, tells a patched Unity
build which animal to spawn next.

**Incoming (Unity → Python):**

```json
{"type": "game_state", "gesture_requested": "fist", "ground_truth": true,
 "camera_blocking": true, "timestamp": 1712234567.456}
```

- `game_state` — ground-truth update; fans out to registered game-state callbacks
  (e.g. `GameRecorder`).
- `game_level_started` — fired once per session when Level 1 finishes loading and
  the child has pressed Start; the coordinator uses this to defer CSV recording
  until gameplay begins.
- `session_config` — Unity's settings panel forwarding balanced/sequential mode,
  repetitions, gesture list, hold/pause durations so the Python coordinator can
  auto-configure itself.

### 13.2 Game recorder

`GameRecorder` records synchronized gameplay data into a per-recording folder
under `data/game_recordings/<subject_id>/game_[<name>_]<timestamp>/`. It consumes
three parallel streams: `on_emg_data` from the device, `on_prediction` from the
server, and `on_game_state` from Unity.

`recording.csv` — one row per EMG sample with:

- `Timestamp`, `PredictedGesture`, `PredictedGestureId`, `Confidence`
- Per-class probability columns (`Prob_<class>`)
- Ground-truth columns: `GroundTruthActive`, `RawGroundTruth`, `RequestedGesture`,
  `CameraBlocking`
- Raw EMG channels (`EMG_Ch0 … EMG_ChN`)

`config.json` — session-level metadata: model info, calibration (rotation offset,
confidence, channel mapping), participant info, recording duration / sample
count. Calibration / participant metadata is attached via `set_calibration_info(...)`
and `set_participant_info(...)` *before* `start_recording(...)`.

Behaviour notes:

- **Ground-truth state machine.** `GroundTruthActive` is True only while a
  gesture is requested AND (Unity's raw flag is true OR the camera is blocking OR
  we are inside a short pre-block grace window). The grace window keeps labels
  robust when request and camera updates arrive slightly out of sync.
- **`RequestedGesture` column** writes `"rest"` (the `REST_LABEL` sentinel)
  whenever Unity is between requests — the user is at rest during those
  intervals, so the column is directly usable as ground truth. Older recordings
  that wrote `"none"` migrate in place via
  `python -m playagain_pipeline.utils.migrate_requested_gesture <data_dir>`.
- **Background writer thread + bounded queue** (`queue.Queue(maxsize=256)`)
  protects RAM when the disk stalls; under extreme load the oldest batch is
  dropped rather than the recording UI freezing. Flush cadence is tuned per OS
  (slower on Windows for stable file I/O).

### 13.3 Unity launcher

`unity_launcher.py` locates and launches the PlayAgain Unity executable
cross-platform. The user picks the build once via **Locate Unity…** on the Record
or Use-a-model-live tab; the path is persisted via `QSettings` and remembered
across sessions. Sub-process lifecycle (`terminate`, signal handling) is
centralised here. `UnityNotFoundError` is raised when the configured path no
longer points to a valid executable.

### 13.4 Training game coordinator

`training_game_coordinator.py` bridges dataset recording with the Unity PlayAgain
game so children can collect their **first** EMG dataset by playing instead of by
sitting through a bare "perform gesture now" prompt. Collecting the initial
dataset is the single biggest friction point in the pipeline — no trained model
exists yet, so the game can't recognise what the child is doing.

The coordinator runs the game in **"easy mode"**:

1. The child opens PlayAgain. An animal walks in on cue from the coordinator
   (`target_gesture` message over the existing Python↔Unity TCP channel).
2. The coordinator watches the EMG stream and computes RMS per chunk. The moment
   RMS crosses a per-subject threshold while the target trial is active, the
   coordinator broadcasts a synthetic prediction
   (`{"gesture": <current_target>, "confidence": 1.0}`) so Unity's existing
   `PipelineGestureClient` fires `OnGestureActiveChanged(true)` and the animal
   gets fed.
3. Unity signals back via its normal `game_state` callback when the animal walks
   away. The coordinator closes the current trial in the `RecordingSession`,
   queues the next gesture, and repeats.
4. When every trial is done the coordinator emits `all_complete` and the GUI
   tears everything down.

Key components:

- **`_RmsDetector`** — per-chunk RMS with a **frozen** rest baseline. Phases per
  session: `IDLE` → `CALIBRATING` (collects ~2 s of rest RMS during a "REST" cue)
  → `READY` (fires when `rms > baseline × trigger_ratio` and the caller has set
  `armed=True`). The baseline uses the **median** of calibration chunks for
  robustness against the occasional twitch a child sneaks in during "rest". The
  default `trigger_ratio` is 1.3 — forgiving by design; real classification
  happens later from the recorded data.
- **`TrialSpec`** — a single trial in the easy-mode schedule (gesture_name,
  hold_seconds, rest_seconds).
- **`TrainingGameCoordinator`** (a `QObject`) — owns the trial schedule, the
  detector, and the Unity callbacks. Two scheduling modes:
  - `set_schedule([TrialSpec(...), ...])` — strict order.
  - `set_balanced_mode(gestures, reps_per_gesture)` — re-cues the most
    under-represented gesture on each advance, with a safety cap of
    `max_oversample × total_expected` trials. Guarantees per-gesture completion
    counts even when Unity spawns animals in its own order.
- **`build_default_schedule(session, ...)`** — convenience builder for a balanced
  shuffled schedule from a session's gesture set.

Cross-thread safety: the prediction server's network thread can't directly touch
QTimers; the coordinator funnels all network-thread callbacks through private Qt
signals so slots execute on the GUI thread.

---

## 14. Graphical user interface

The entry point `gui.main_window.main()` constructs `MainWindowV2` (defined in
`gui/widgets/main_window_v2.py`). Across the top of the window sits a
**`WorkflowStepper`** banner — three primary stages, clickable to jump between
tabs: **Record → Train & Evaluate → Predict**.

The window has **four tabs** (the redesigned v2 navigation); advanced and
optional actions live in a **Tools ▾ menu** at the top-right corner of the tab
bar rather than in their own tabs:

1. **Home** — landing page with three numbered task cards (1 Record new data, 2
   Train a model, 3 Use a model live), a recent-activity list, and the quickstart
   wizard.
2. **Record new data** — Connect a device and record gesture sessions for one
   subject. Includes the **Training Game (Unity) — easy mode** panel for
   first-time dataset collection ([§13.4](#134-training-game-coordinator)).
3. **Train a model** — Build datasets from recordings, then train and validate
   models.
4. **Use a model live** — Run a trained model live on the device or stream
   predictions to Unity, with smoothing controls and game-recording start/stop.

**Tools ▾ menu** (this replaces the former *Calibration* and *Validation* side
tabs — calibration and validation are **not** side tabs):

- **Validate models…** — Reproducible cross-validation across feature sets,
  models, and CV strategies. Opens the merged evaluation window backed by
  `gui/widgets/evaluation_tab.py` (replaces the old Performance Review tab); every
  run is saved to `data/validation_runs/` with full config and environment.
- **Build thesis report…** — Generate the tables and figures referenced from
  thesis chapters 6–8 (`gui/widgets/thesis_report_dialog.py`).
- **Calibrate bracelet…** — Open the calibration dialog
  (`gui/widgets/calibration_dialog.py`). Only needed when reusing a pretrained
  model with a rotated bracelet, or to set a new reference orientation.
- **Quattrocento training…**, **Feature selection…**, **Bracelet visualisation…**,
  **Edit participant info…**, **Open configuration…**, **Show quickstart again**,
  **About**.

The GUI orchestrates: device connection/stream handling and the
monopolar/bipolar mode toggle; protocol-driven recording, optionally driven by
the Unity training game; dataset creation + model training dialogs; real-time
prediction and the optional TCP server; game recording controls; and Quattrocento
import/training flows.

Notable supporting widgets:

- `widgets/workflow_stepper.py` — the three-step banner across the top.
- `widgets/emg_plot_panel.py` — the live multi-channel EMG view (replaces the
  older single-file `emg_plot.py`).
- `widgets/protocol_popup.py` — full-screen recording cue popup; embeds the
  legacy `ProtocolWidget`.
- `widgets/game_protocol_popup.py` — dedicated overlay for the training game
  showing per-gesture balance progress while the child plays.
- `widgets/evaluation_tab.py` — the body of the Validation / cross-validation view
  (opened from **Tools ▸ Validate models…**); class `EvaluationTab`.
- `widgets/bracelet_graphic.py` — visual feedback on bracelet rotation, aware of
  monopolar / bipolar mode.

### 14.1 Validation view (`gui/widgets/evaluation_tab.py`)

The Validation view (opened from **Tools ▸ Validate models…**) supports
comparison experiments with:

- Session-role assignment UI (train/val/test).
- Holdout and cross-validation oriented workflows, with fold-level progress.
- A model selection covering `lda`, `random_forest`, `catboost`, `svm`, `mlp`, and
  `attention_net` (per-epoch live training curves are a feature of the training
  dialogs in §10, not of this view).
- Aggregated metrics and confusion/report outputs.
- A direct hook into the `validation/` harness so a configured experiment can be
  triggered from the GUI and its artifacts surface under `data/validation_runs/`.

---

## 15. Configuration system

`config/config.py` defines five dataclasses that together form a
`PipelineConfig`:

- `DeviceConfig` — type, channel count, sampling rate, network settings.
- `RecordingConfig` — default window / stride settings.
- `CalibrationConfig` — calibration durations, confidence thresholds.
- `ProtocolSettings` — quick/standard/long/calibration timing presets.
- `ModelConfig` — default hyperparameters per model type, plus `bad_channel_mode`.

The top-level `PipelineConfig` aggregates these and provides `to_dict` /
`from_dict` plus `save` / `load` (JSON) round-tripping. `config.json` lives at the
package root by default and is loaded at startup. The GUI surfaces a preferences
dialog (`gui/widgets/config_dialog.py`) that edits the same dataclass.

The validation harness has its own `ExperimentConfig` (in `validation/config.py`)
— a narrower, reproducibility-focused dataclass that encodes a single experiment
and is deliberately decoupled from `PipelineConfig` so a collaborator can send a
YAML file without including personal defaults.

---

## 16. Reproducibility and determinism

Reproducibility in this codebase is a matter of four things being deterministic
together:

1. **Seed discipline.** Validation runs seed numpy, stdlib `random`, and torch
   from `ExperimentConfig.seed`. GUI training uses a fixed split seed of 42 unless
   a custom `seed` is injected into the loaded config object.
2. **Per-fold seeding in the runner.** Each `(fold, model)` pair gets a
   SHA-1-hashed seed of `(cfg.seed, fold_idx, model_type)` so fold order and model
   order don't alter the results.
3. **Environment capture.** `environment.json` records the exact Python and
   library versions used. Point this out when a number differs by more than noise
   — usually it's a numpy or sklearn version bump.
4. **Frozen input.** `session_index.json` enumerates every session path consumed.
   If a session is added or removed, re-running won't match.

What is **not** deterministic:

- CUDA kernels. `torch.backends.cudnn.deterministic` is deliberately *not* forced
  to `True` because it measurably slows training and is unnecessary for
  within-version bit-exactness on CPU and MPS. CUDA users should expect per-run
  drift of ≤ 1e-4 in macro-F1.
- Wall-clock `train_seconds`. Don't treat this as a seeded metric.
- File-system enumeration order on unusual filesystems. `SessionCorpus` sorts its
  results before returning to defend against this, but any code that walks
  `data/sessions/` with `os.listdir` without sorting is vulnerable.

---

## 17. End-to-end workflow

### 1) Record sessions

Two flavours, depending on whether you already have a trained model.

**Standard protocol-driven recording:**

1. Open the GUI and select device type; toggle bipolar mode if relevant.
2. Configure subject/session/protocol settings.
3. Start recording and complete prompted protocol steps.
4. Save session artifacts under `data/sessions/<subject>/<session>/`.

**Easy-mode recording via the Unity training game (recommended for first
sessions with children):**

1. On the Record tab, click **Locate Unity…** and pick the PlayAgain build
   (one-time setup, persisted via `QSettings`).
2. Set easy-mode sensitivity (default 1.3× resting RMS baseline) and reps per
   gesture (default 3).
3. Click **▶ Launch Training Game**. The pipeline starts a `PredictionServer`,
   launches Unity, builds a balanced trial schedule, and constructs a
   `TrainingGameCoordinator`.
4. The child plays; animals appear, the coordinator broadcasts synthetic
   predictions on each muscle contraction, and Unity feeds the animals. The
   coordinator opens/closes trials in the `RecordingSession` automatically.
5. When complete, the session is saved under `data/sessions/<subject>/<session>/`
   just like a protocol-driven recording.

### 2) Build dataset

1. Select sessions (subject-level or explicit selection).
2. Set `window_size_ms` and `window_stride_ms`.
3. Optionally enable **per-session rotation** to align channels from different
   bracelet placements, mark **bad channels** with an interpolate or zero
   strategy, or **pre-extract features** at dataset build time.
4. Generate dataset arrays and metadata in `data/datasets/<name>/`.

### 3) Train model

1. Choose model type.
2. Train from dataset.
3. Save artifact to `data/models/<model_name>/` with `metadata.json`.

### 4) Run real-time prediction

1. Load a trained model in the **Use a model live** tab.
2. Start the prediction worker.
3. Optionally enable/start the Unity TCP server.
4. Observe smoothed gesture output in the GUI and/or Unity client.

### 5) Record gameplay

1. Start game recording in the **Use a model live** tab.
2. Stream EMG and Unity game-state messages concurrently.
3. Stop recording to finalize `recording.csv` and `config.json`.

### 6) Run automated validation

1. Define an experiment YAML config (e.g. based on
   `validation/configurations/experiments_example.yaml`).
2. Run `python -m playagain_pipeline.validation run <config.yaml>`, or trigger the
   same workflow from the Validation view (**Tools ▸ Validate models…**).
3. Review results in `data/validation_runs/<timestamp>__<name>/`.

A good first sanity run, given a typical corpus:

```yaml
name: vp01_within_subject_sanity
seed: 42
data:
  subjects: [VP_01]
  domains: [pipeline]
windowing:  { window_ms: 200, stride_ms: 50, drop_rest: false }
features:
  - { name: mav }
  - { name: rms }
  - { name: wl }
  - { name: zc, params: { threshold: 0.01 } }
models:
  - { type: RandomForest, params: { n_estimators: 200 } }
cv:
  strategy: loso_session
```

With ~25 sessions for VP_01 this produces ~25 folds and finishes in a couple of
minutes on a laptop — the most defensible single number you can put next to
"VP_01 Random Forest" in a thesis.

---

## 18. Utility and analysis scripts

- `utils/platform_utils.py` — OS detection, default data/config directories
  (`get_default_data_dir()` returns `~/Documents/PlayAgain/data` on macOS/Linux
  and `%USERPROFILE%/Documents/PlayAgain/data` on Windows), and automatic
  resolution of the sibling `device_interfaces` and `gui_custom_elements`
  packages. The GUI and standalone prediction server default to the in-repo
  `playagain_pipeline/data` unless `config.json` or the UI overrides the data
  directory.
- `utils/rest_gap_filler.py` — fills implicit-rest gaps between trials with
  synthetic rest trials, used to reconcile Unity-recorded sessions with the
  pipeline's expected label coverage.
- `utils/migrate_requested_gesture.py` — one-shot migration that rewrites the
  `RequestedGesture` column in old game recordings from `"none"` to `"rest"`;
  runnable as `python -m playagain_pipeline.utils.migrate_requested_gesture <data_dir>`.
- `performance_assessment/` — comparison tooling (`model_comparison.py`,
  `session_picker_ui.py`, `_generate_plots.py`) and the
  `performance_assessment.ipynb` notebook for post-hoc analyses (pre-dates the
  validation harness).
- `evaluation/` — single-run evaluation primitives (`core.py`, `metrics.py`,
  `session_eval.py`, `cross_domain_eval.py`, `game_eval.py`,
  `intra_session_eval.py`, `threshold_eval.py`, `unity_eval.py`, `loaders.py`)
  describing one model / one set of recordings / one set of metrics. The thesis
  tables aggregate *across* folds, models and runs, which is why the thesis-report
  layer ([§11.6](#116-thesis-report-generation-layer)) lives in `validation/`
  rather than bloating `EvaluationResult`.
- `validation/generate_thesis_outputs.py` and
  `validation/recompute_calibration_metrics.py` are user-invokable `python -m …`
  tools that build thesis-shaped artefacts from existing validation runs and
  recompute calibration metrics on disk respectively.

---

## 19. Scripting API

The package exports core symbols in `playagain_pipeline/__init__.py`, including:

- Gesture helpers (`Gesture`, `GestureSet`, `GestureCategory`,
  `create_default_gesture_set`, `create_calibration_gesture_set`).
- Recording session and data manager classes (`RecordingSession`,
  `RecordingTrial`, `DataManager`).
- Protocol helpers (`RecordingProtocol`, `ProtocolConfig`, `ProtocolPhase`,
  `create_quick_protocol`, `create_standard_protocol`, `create_extended_protocol`).
- Device abstractions (`DeviceType`, `DeviceManager`, `SyntheticEMGDevice`).
- Model manager and classifier entry points (`ModelManager`, `SVMClassifier`,
  `RandomForestClassifier`, `LDAClassifier`, `EMGFeatureExtractor`).
- Calibration and configuration classes.

---

## 20. Extension points

### 20.1 Add a new model

1. Create a `BaseClassifier` subclass in `models/classifier.py` implementing
   `train`, `predict`, `predict_proba`, `save`, `load`.
2. Register it in `ModelManager.AVAILABLE_MODELS`.
3. Add hyperparameter defaults to `ModelConfig` in `config/config.py`.
4. Expose/edit parameters in GUI dialogs when needed.
5. If it's a CNN-family model (wants raw windows, not features), add its key to
   `validation/runner._RAW_WINDOW_MODELS` so the validation runner materialises
   raw windows for it automatically.

### 20.2 Add a new feature

1. Subclass `BaseFeatureExtractor` in `models/feature_pipeline.py`.
2. Decorate with `@register_feature("your_name")`.
3. Implement `name`, `description`, and `compute(data)` — must accept both 2-D
   `(T, C)` and 3-D `(N, T, C)` inputs.

No registration in config is required — the GUI and validation tooling discover
features via `get_registered_features`.

### 20.3 Add a new CV strategy

1. Write a callable in `validation/cv_strategies.py` that takes
   `records: List[SessionRecord]` and `**kwargs` and yields fold dicts with keys
   `id`, `idx`, `train`, `test`, and optionally `val`, `split_kind`,
   `test_fraction`.
2. Register it in `STRATEGIES`.
3. (Optional) Add a friendly label and tooltip in `gui/widgets/evaluation_tab.py`
   (the `EvaluationTab` "Validation" tab).

### 20.4 Add a new device backend

1. Implement a `BaseEMGDevice` subclass in `devices/emg_device.py`. Emit
   `(n_samples, n_channels)` `float32` via whatever signal hook the rest of the
   code connects to (`data_ready` Qt signal or a callback).
2. Add a new entry to `DeviceType`.
3. Wire device selection into the GUI's connection controls.

### 20.5 Add a new calibration check

1. Implement the check as a method in
   `calibration/calibration_validation.py:CalibrationValidator` returning a
   `CheckResult` with an appropriate `severity` (`error` blocks acceptance,
   `warning` is informational).
2. Add the call to `run_all(...)`.
3. Update `CalibrationReport.interpret()` if the new check warrants a
   plain-English explanation in the user-facing summary.

---

## 21. Known limitations and caveats

- **Inflated within-session accuracy.** The `within_session` CV strategy uses a
  temporal tail, which is optimistic because the user and hardware don't change
  across the split.
- **Temporal tail with trial-ordered windows.** Because windows are stored in
  trial order and some sessions hold gestures back to back without shuffling, the
  "last 20%" of a session's windows can be concentrated on the last few classes.
  The runner logs a `WARNING` when this happens, but the user still sees the
  number. Treat with care.
- **Class imbalance handling.** Default class weights are uniform. The
  Quattrocento dialog exposes "balanced" class weights; the standard Train tab
  does not yet.
- **MPS is the supported GPU.** CUDA works but is less tested. Deep-model
  reproducibility is best on CPU or MPS.
- **CNN trained on 2-D features.** CNN-family models coerce 2-D feature tensors to
  a single-channel 3-D shape for both training and inference.
- **Bracelet rotation on 16-channel bipolar.** The calibrator's circular shift
  assumes all channels rotate together. Bipolar recordings halve the channel
  count but the same assumption holds — just with half the resolution.
- **No cross-subject calibration transfer.** A reference calibration works within
  a subject/bracelet-pair. Moving to a new subject or a different bracelet
  requires recording a new reference.

> Defaults and behavior evolve with code; treat this document as a maintained
> overview and confirm details against module docstrings when changing internals.
> If you update pipeline behavior, update this README and any related GUI help
> text in the same change.

---

## 22. Glossary

| Term                | Definition                                                                                 |
|---------------------|--------------------------------------------------------------------------------------------|
| sEMG                | Surface electromyography — non-invasive muscle-electrical recording via skin electrodes.   |
| Session             | One continuous recording for one subject, ~1–10 min, one gesture set.                       |
| Trial               | A single gesture-hold within a session; labelled with a gesture id.                         |
| Window              | A fixed-length slice of a trial, typically 200 ms, with 50 ms stride.                       |
| Dataset             | A windowed, optionally feature-extracted `(X, y)` built from one or more sessions.          |
| Model               | A trained classifier plus its metadata, persisted under `data/models/`.                     |
| Domain              | `pipeline` (Python recorder) or `unity` (C# recorder) — same on-disk layout.                |
| LOSO                | Leave-One-Subject-Out — train on everyone but one, test on the one. Canonical headline.     |
| LOSO-session        | Same idea at session granularity — hold out one session, train on the others.               |
| Holdout             | A single train / val / test split with explicit ratios. Used for deep-model tuning.         |
| Calibration-sync    | A wave-out gesture recorded at session start, used only for rotation detection.             |
| Rotation offset     | Integer number of channels the bracelet is rotated relative to the reference.               |
| Bad channel         | An electrode declared unreliable for this session; handled by interpolation or zeroing.     |
| Smoothing           | EMA + stability gate applied to live predictions to prevent class flicker.                  |

---

## Appendix A — Default values at a glance

| Parameter                    | Default      | Where                              |
|------------------------------|--------------|------------------------------------|
| Sampling rate                | 2 kHz        | `DeviceConfig.sampling_rate`       |
| Channel count                | 32           | `DeviceConfig.num_channels`        |
| Window size                  | 200 ms       | `RecordingConfig.window_size_ms`   |
| Window stride                | 50 ms        | `RecordingConfig.window_stride_ms` |
| EMA alpha                    | 0.3          | `PredictionSmoother`               |
| Min stable time              | 150 ms       | `PredictionSmoother`               |
| Easy-mode trigger ratio      | 1.3×         | `_RmsDetector.trigger_ratio`       |
| Train/val split (non-val CV) | 80/20        | `ModelManager.train_model`         |
| Default CV strategy          | LOSO-subject | `evaluation_tab` combo             |
| TCP host                     | 127.0.0.1    | `prediction_server.main`           |
| TCP port                     | 5555         | `prediction_server.main`           |

## Appendix B — Command cheatsheet

```bash
# Launch the GUI
python run_gui.py

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

# Generate thesis tables/figures from saved runs (after runs complete)
python -m playagain_pipeline.validation.generate_thesis_outputs --help

# Migrate old game recordings ("none" -> "rest" in RequestedGesture)
python -m playagain_pipeline.utils.migrate_requested_gesture <data_dir>
```

## Appendix C — File-size hints for a typical run

- A 5-minute 32-channel 2 kHz session: ~76 MB as `data.npy`, ~230 MB as
  `data.csv`.
- A dataset built from 12 such sessions with 200 ms windows / 50 ms stride and 8
  time-domain features: `X.npy` ~ 45 MB.
- A LOSO validation run over 12 subjects × 4 models: `results.json` < 1 MB after
  the `_slim_train_meta` patch (was ~30–60 MB before).
</content>
</invoke>
