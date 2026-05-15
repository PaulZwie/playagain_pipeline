# PlayAgain Gesture Pipeline — Detailed Documentation

## Overview

`playagain_pipeline` is the Python package used to collect EMG data, train
models, run real-time gesture inference, and integrate predictions with the
Unity PlayAgain game via TCP.

At a high level, the pipeline provides:

- Protocol-driven EMG recording sessions, with optional Unity-game-driven
  "easy mode" dataset collection for first-time users.
- Rotation-aware calibration with stability-based confidence and a battery of
  validation checks.
- Dataset creation from raw sessions, including per-session rotation
  correction, bad-channel handling, and optional pre-extracted features.
- Multiple ML model backends (classical and deep learning).
- Real-time prediction with smoothing.
- Bidirectional Unity integration and synchronized game recording.
- Post-hoc model/session analysis and reproducible cross-validation via the
  Validation tab and the headless validation harness.

## Requirements

From `pyproject.toml`:

- Python `>=3.11,<3.13`
- Package name: `playagain-pipeline`

The project also references local development dependencies via `tool.uv.sources`
for `device-interfaces` and `gui-custom-elements`. These sibling packages are
located automatically at startup by `utils/platform_utils.py` (no hard-coded
paths required).

## Setup

### Option A: `uv` (matches `pyproject.toml` source configuration)

```bash
cd /Users/paul/Coding_Projects/Master/Dataprocessing/playagain_pipeline
uv sync
```

### Option B: `pip` editable install (if local dependency paths are resolvable)

```bash
cd /Users/paul/Coding_Projects/Master/Dataprocessing
python -m venv .venv
source .venv/bin/activate
pip install -e ./playagain_pipeline
```

## Running

### GUI entry point

```bash
cd /Users/paul/Coding_Projects/Master/Dataprocessing
python -m playagain_pipeline.run_gui
```

`run_gui.py` delegates to `playagain_pipeline.gui.main_window.main()`, which in
turn instantiates `MainWindowV2` (the current top-level window).

### Headless prediction server

```bash
cd /Users/paul/Coding_Projects/Master/Dataprocessing
python -m playagain_pipeline.prediction_server --model <trained_model_name>
```

Optional arguments (from `prediction_server.py`):

```bash
python -m playagain_pipeline.prediction_server --model <trained_model_name> --host 127.0.0.1 --port 5555 --device muovi
```

Supported `--device` values: `muovi`, `muovi_plus`, `synthetic`.

## Architecture

```text
EMG Source (Muovi / Muovi Plus / Synthetic / Quattrocento replay)
    -> Recording + Dataset + Model Pipeline
    -> Real-time Prediction + Smoothing
    -> Unity TCP Bridge (JSON over newline-delimited TCP)
    -> Optional synchronized game recording (CSV + config.json metadata)
    -> Optional Training-Game coordinator (Unity-driven dataset collection)
```

Primary flows:

1. **Recording Flow**: device stream -> `RecordingSession` -> session files.
2. **Easy-mode Recording Flow**: device stream -> `TrainingGameCoordinator` ->
   Unity feedback loop -> `RecordingSession` files (Unity drives the trial
   schedule; the coordinator broadcasts synthetic predictions so children can
   record their first dataset by playing).
3. **Training Flow**: sessions -> windowed dataset -> trained model directory.
4. **Prediction Flow**: live EMG -> model inference -> smoother -> GUI/TCP output.
5. **Game Logging Flow**: EMG + predictions + Unity state -> synchronized CSV
   rows + per-recording `config.json`.
6. **Validation Flow**: configured experiment YAML -> rigorous CV across
   sessions -> timestamped artifact folder under `data/validation_runs/`.

## Current Project Structure

```text
playagain_pipeline/
├── __init__.py
├── run_gui.py
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
├── gui/
│   ├── __init__.py
│   ├── gui_style.py
│   ├── main_window.py
│   └── widgets/
│       ├── main_window_v2.py
│       ├── workflow_stepper.py
│       ├── emg_plot_panel.py
│       ├── evaluation_tab.py
│       ├── protocol_popup.py
│       ├── game_protocol_popup.py
│       ├── bracelet_graphic.py
│       ├── busy_overlay.py
│       ├── calibration_dialog.py
│       ├── config_dialog.py
│       ├── feature_selection.py
│       ├── protocol_widget.py
│       ├── quattrocento_loader.py
│       ├── quattrocento_loading_dialog.py
│       ├── quattrocento_training_dialog.py
│       └── training_dialog.py
├── models/
│   ├── __init__.py
│   ├── classifier.py
│   └── feature_pipeline.py
├── performance_assessment/
│   ├── _generate_plots.py
│   ├── model_comparison.py
│   ├── performance_assessment.ipynb
│   └── session_picker_ui.py
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
│   ├── cv_strategies_holdout.py
│   ├── cv_strategies.py
│   ├── README.md
│   ├── runner.py
│   └── configurations/
└── data/
    ├── Participant_Info/
    ├── calibrations/
    ├── datasets/
    ├── game_recordings/
    ├── models/
    ├── quattrocento/
    ├── scripts/
    ├── sessions/
    └── validation_runs/
```

## Core Concepts

### Gestures and Labeling

`core/gesture.py` defines `Gesture`, `GestureSet`, and `GestureCategory`.

- Default gesture set (`create_default_gesture_set()`):
  - `rest` (label `0`)
  - `fist` (label `1`)
  - `pinch` (label `2`)
  - `tripod` (label `3`)
- Single-gesture sets (`create_single_gesture_set("fist" | "pinch" | "tripod")`)
  pair rest with one target gesture — used by the per-gesture protocols.
- Gestures include display metadata (`display_name`, `emoji`, `duration_hint`).
- Gesture sets serialize to JSON and are stored with each session.

### Signal Modes (Monopolar / Bipolar)

The pipeline supports two EMG signal-acquisition modes, selected on the
Record tab via the **Bipolar Mode** checkbox:

- **Monopolar** (default): raw per-channel signal from each electrode.
- **Bipolar**: top-minus-bottom differential across the bracelet's two rings.
  Each pair becomes a single channel, so the channel count is halved.

The signal mode is persisted in each session's `metadata.custom_metadata`
under both `signal_mode` and the legacy boolean `bipolar_mode`. Downstream
consumers (calibration, dataset creation) read this and:

- Maintain **separate reference calibrations** for monopolar vs. bipolar
  (`reference_calibration_monopolar.json` and
  `reference_calibration_bipolar.json` under `data/calibrations/`).
- **Refuse to mix signal modes** in a single dataset — `create_dataset` raises
  if sessions have inconsistent channel counts.

### Recording Sessions

`core/session.py` persists each session as:

- `data.npy` (raw EMG)
- `data.csv` (human-readable raw EMG)
- `metadata.json` (session/trials/device details)
- `gesture_set.json` (gesture definitions used)

`RecordingMetadata` additionally carries:

- `rotation_offset` and `rotation_confidence` — the bracelet rotation detected
  for this session relative to the reference (filled by the calibrator).
- `bad_channels` — 0-based channel indices marked as bad, applied at dataset
  build time via `apply_bad_channel_strategy`.
- `channel_mapping` — the resolved channel order after rotation correction.

`RecordingTrial` carries `trial_type ∈ {"gesture", "calibration_sync"}`:

- `"gesture"` — a normal training trial; included in `get_valid_trials()`.
- `"calibration_sync"` — the waveout sync gesture optionally recorded at the
  start of each session for rotation detection. Stored with
  `gesture_label = -1` and **excluded from model training**; retrievable via
  `get_calibration_trials()`. Older sessions that predate this field load
  transparently with the default `"gesture"`.

Loaded sessions memory-map `data.npy` (read-only) so only accessed pages are
faulted into RAM — important for large multi-session datasets.

### Data Management

`core/data_manager.py` manages data directories and conversion workflows.

Notable responsibilities:

- Session save/load and subject/session listing, with sorted natural ordering
  (`VP_00 < VP_01 < VP_10`).
- Legacy-safe path resolution for sanitized/raw folder names, including
  `sessions/unity_sessions/<subject>/` for Unity-recorded sessions.
- Automatic `VP_NN` subject ID allocation via `get_next_subject_id()`.
- Dataset creation with configurable window size/stride, optional
  `preprocessing_fn`, optional `calibration` or `use_per_session_rotation`,
  per-session `bad_channels` plus `bad_channel_mode` (`"interpolate"` or
  `"zero"`), and optional `feature_config` for pre-extracting features at
  dataset build time.
- Mixed-channel-count guard: rejects mixing monopolar + bipolar sessions in
  one dataset.
- Optional memory-mapped dataset loading (`load_dataset(name, mmap=True)`).
- Participant metadata save/load under `data/Participant_Info/<subject_id>.json`.
- Train/validation/test splitting via sklearn helpers.

## Module Documentation

### Configuration (`config/config.py`)

Main dataclasses:

- `DeviceConfig` (`device_type`, channels, sampling rate, network settings)
- `RecordingConfig` (window and stride defaults)
- `CalibrationConfig` (duration and confidence thresholds)
- `ProtocolSettings` (quick/standard/long/calibration timing presets)
- `ModelConfig` (hyperparameters for all supported model backends)
- `PipelineConfig` (top-level aggregation + JSON serialization)

### Devices (`devices/emg_device.py`)

- `DeviceType` includes `SYNTHETIC`, `MUOVI`, and `MUOVI_PLUS`.
- Muovi device wrapper integrates with `device_interfaces`.
- Synthetic backend supports hardware-free testing and pipeline validation.
- Quattrocento replay support is integrated through GUI workflows in
  `gui/main_window.py` and `gui/widgets/quattrocento_loader.py`.

### Protocols (`protocols/protocol.py`)

- `ProtocolPhase` includes standard phases plus `CALIBRATION_SYNC`.
- `RecordingProtocol` composes step sequences (`ProtocolStep`) from a
  `GestureSet` and `ProtocolConfig`.
- `ProtocolConfig.include_calibration_sync` toggles whether each session opens
  with a waveout sync trial.
- Built-in factories: `create_quick_protocol`, `create_standard_protocol`,
  `create_extended_protocol`, `create_calibration_protocol`, plus
  per-gesture single-target protocols (`create_pinch_protocol`,
  `create_tripod_protocol`, `create_fist_protocol`).

### Calibration (`calibration/`)

Calibration is split across three modules:

- **`calibrator.py`** — `CalibrationProcessor` and `AutoCalibrator`,
  rotation-aware channel alignment via the Maximum Energy Channel (MEC) method
  from Barona López et al., *Sensors* 2020. The processor computes per-channel
  energy using
  `E_ch = Σ |x_i·|x_i| − x_{i-1}·|x_{i-1}||`, folds the 32-channel pattern
  into 16 angular sectors, and detects the rotation offset via circular
  cross-correlation against the reference. Calibration is performed directly
  from normal recording sessions — no separate calibration recording is
  required. References are kept per signal mode
  (`reference_calibration_monopolar.json` / `reference_calibration_bipolar.json`).
  `backfill_session_rotations(...)` is a one-shot migration utility that
  re-detects and writes rotation offsets for all historical sessions.

- **`calibration_stability.py`** — `compute_stability_metrics(...)` returns a
  `StabilityResult` whose primary `stability` value answers "do my trials
  agree on the same offset?" rather than the legacy peak-prominence z-score
  (which measured the sharpness of the user's activation pattern, not the
  correctness of the offset). The legacy `peak_prominence` is still
  reported as a diagnostic, alongside `top2_ratio`, the per-trial offsets,
  and an optional bootstrap distribution.

- **`calibration_validation.py`** — runs a battery of independent acceptance
  checks on a fitted calibration and produces a `CalibrationReport`:
  - **`self_consistency`** (error): the calibrator must agree with itself
    within `SELF_CONSISTENCY_MAX_DRIFT` channels when re-run on the same data.
  - **`symmetry`** (error): the primary energy peak must dominate its
    half-ring (180°) twin by at least `SYMMETRY_MIN_RATIO` — fails when the
    bracelet is on backwards.
  - **`held_out_accuracy`** (error, when a held-out session is supplied):
    top-1 classification accuracy must clear `HELD_OUT_MIN_ACCURACY`.
  - **`confidence_floor`** (warning by default): the legacy confidence
    number is compared against a floor; low values are informational unless
    explicitly upgraded to an error.

  `BatchValidator` runs `CalibrationValidator` across many sessions with
  save/restore around the calibrator state (so the user's current calibration
  is not mutated by the batch). `RotationDetectionStudy` aggregates per-session
  offsets, drift relative to a reference session, and check pass-rates for
  corpus-wide "is rotation detection stable across my recordings?" analyses.

### Models and Features (`models/`)

`ModelManager.AVAILABLE_MODELS` currently exposes:

- `svm`
- `random_forest`
- `lda`
- `catboost`
- `mlp`
- `cnn`
- `attention_net`
- `mstnet`

`apply_bad_channel_strategy(data, bad_channels, mode=...)` implements the
shared zero / linear-interpolate logic used at both training and inference
time.

`feature_pipeline.py` provides a registry-driven feature pipeline with
built-ins: `rms`, `mav`, `var`, `wl`, `zc`, `ssc`, `iemg`, `ssi`.

### Prediction Server (`prediction_server.py`)

Responsibilities:

- TCP server startup (`host` default `127.0.0.1`, `port` default `5555`).
- Model loading and per-buffer inference, with a dedicated prediction worker
  thread and a separate sender thread so network hiccups can't stall
  inference.
- `pause()` / `resume()` to stop predictions without tearing down the TCP
  server (used during calibration, training-game setup, etc.).
- Smoothed prediction output via `PredictionSmoother`.
- Bidirectional messaging with Unity clients.
- `enqueue_json_message(msg)` — public helper used by the
  `TrainingGameCoordinator` to inject synthetic predictions / target-gesture
  cues through the same sender thread that handles model predictions.
- Throttled `error_callback` so the GUI surfaces "your model can't predict
  on this stream" instead of a frozen `rest` label, without flooding the log
  on every chunk.

**Outgoing messages** (Python → Unity, newline-delimited JSON):

- `prediction` (the default payload): `{"gesture", "gesture_id",
  "confidence", "probabilities", "timestamp"}`
- `handshake` on connect: `{"type": "handshake", "model_name", "class_names", ...}`
- `target_gesture` (from the training coordinator): tells a patched Unity
  build which animal to spawn next.

**Incoming messages** (Unity → Python):

- `game_state` — ground-truth update: `{"ground_truth", "gesture_requested",
  "camera_blocking", "timestamp"}`. Fans out to registered game-state
  callbacks (e.g., `GameRecorder`).
- `game_level_started` — fired once per session when Level 1 finishes loading
  and the child has pressed Start. The `TrainingGameCoordinator` uses this to
  defer the start of CSV recording until gameplay actually begins.
- `session_config` — Unity's settings panel forwarding balanced/sequential
  mode, repetitions, gesture list, hold/pause durations so the Python
  coordinator can auto-configure itself.

Prediction smoothing combines:

- Exponential moving average (`alpha`, default `0.3`)
- Stability gate (`min_stable_ms`, default `150`)

### Training Game Coordinator (`training_game_coordinator.py`)

Bridges dataset recording with the Unity PlayAgain game so children can
collect their **first** EMG dataset by playing instead of by sitting through
a bare "perform gesture now" prompt. Collecting the initial dataset is the
single biggest friction point in the pipeline — no trained model exists yet,
so the game can't recognise what the child is doing.

The coordinator runs the game in **"easy mode"**:

1. The child opens PlayAgain. An animal walks in on cue from the coordinator
   (`target_gesture` message over the existing Python↔Unity TCP channel).
2. The coordinator watches the EMG stream and computes RMS per chunk. The
   moment RMS crosses a per-subject threshold while the target trial is
   active, the coordinator broadcasts a synthetic prediction
   (`{"gesture": <current_target>, "confidence": 1.0}`) so Unity's existing
   `PipelineGestureClient` fires `OnGestureActiveChanged(true)` and the
   animal gets fed.
3. Unity signals back via its normal `game_state` callback when the animal
   walks away. The coordinator closes the current trial in the
   `RecordingSession`, queues the next gesture, and repeats.
4. When every trial is done the coordinator emits `all_complete` and the GUI
   tears everything down.

Key components:

- **`_RmsDetector`** — per-chunk RMS with a **frozen** rest baseline. Goes
  through three phases per session: `IDLE` → `CALIBRATING` (collects ~2s of
  rest RMS during a "REST" cue) → `READY` (fires when
  `rms > baseline × trigger_ratio` and the caller has set `armed=True`). The
  baseline uses the **median** of calibration chunks for robustness against
  the occasional twitch a child sneaks in during "rest". The default
  `trigger_ratio` is 1.3 — forgiving by design; real classification happens
  later from the recorded data.
- **`TrialSpec`** — a single trial in the easy-mode schedule
  (gesture_name, hold_seconds, rest_seconds).
- **`TrainingGameCoordinator`** (a `QObject`) — owns the trial schedule,
  the detector, and the Unity callbacks. Two scheduling modes:
  - `set_schedule([TrialSpec(...), ...])` — strict order.
  - `set_balanced_mode(gestures, reps_per_gesture)` — re-cues the most
    under-represented gesture on each advance, with a safety cap of
    `max_oversample × total_expected` trials. Guarantees per-gesture
    completion counts even when Unity spawns animals in its own order.
- **`build_default_schedule(session, ...)`** — convenience builder for a
  balanced shuffled schedule from a session's gesture set.

Cross-thread safety: the prediction server's network thread can't directly
touch QTimers; the coordinator funnels all network-thread callbacks through
private Qt signals so slots execute on the GUI thread.

### Unity Launcher (`unity_launcher.py`)

Locates and launches the PlayAgain Unity executable cross-platform. The user
picks the build once via **Locate Unity…** on the Record or Predict tab; the
path is persisted via `QSettings` and remembered across sessions.
Sub-process lifecycle (`terminate`, signal handling) is centralised here so
the GUI doesn't deal with platform quirks. `UnityNotFoundError` is raised
when the configured path no longer points to a valid executable.

### Game Recorder (`game_recorder.py`)

Records synchronized gameplay data into a per-recording session folder under
`data/game_recordings/<subject_id>/game_[<name>_]<timestamp>/`:

- `recording.csv` — one row per EMG sample with:
  - `Timestamp`, `PredictedGesture`, `PredictedGestureId`, `Confidence`
  - Per-class probability columns (`Prob_<class>`)
  - Ground-truth columns: `GroundTruthActive`, `RawGroundTruth`,
    `RequestedGesture`, `CameraBlocking`
  - Raw EMG channels (`EMG_Ch0 … EMG_ChN`)
- `config.json` — session-level metadata: model info, calibration
  (rotation offset, confidence, channel mapping), participant info,
  recording duration / sample count.

Behavior notes:

- **Ground-truth state machine**: `GroundTruthActive` is True only while a
  gesture is requested AND (Unity's raw flag is true OR the camera is
  blocking OR we are inside a short pre-block grace window). The grace
  window keeps labels robust when request and camera updates arrive
  slightly out of sync.
- **`RequestedGesture` column**: now writes `"rest"` (the `REST_LABEL`
  sentinel) whenever Unity is between requests — the user is at rest
  during those intervals, so the column is directly usable as ground
  truth. Older recordings that wrote `"none"` can be migrated in place via
  `python -m playagain_pipeline.utils.migrate_requested_gesture <data_dir>`.
- **Background writer thread + bounded queue** (`maxsize=256`) protects RAM
  when the disk stalls; under extreme load the oldest batch is dropped
  rather than the recording UI freezing. Flush cadence is tuned per OS
  (slower on Windows for stable file I/O).
- Calibration / participant metadata is attached via
  `set_calibration_info(...)` and `set_participant_info(...)` *before*
  `start_recording(...)`.

### GUI (`gui/main_window.py` + `gui/widgets/main_window_v2.py`)

The entry point `gui.main_window.main()` constructs `MainWindowV2`. Across
the top of the window sits a **`WorkflowStepper`** banner — three primary
stages, clickable to jump between tabs: **Record → Train & Evaluate →
Predict**.

Tabs (new order, side tabs in italics):

1. **①  Record** — Connect device and record gesture sessions for one
   subject. Includes the **Training Game (Unity) — easy mode** panel for
   first-time dataset collection (covered above).
2. **②  Train & Evaluate** — Build datasets from recordings, train and
   validate models. Includes Quattrocento offline training.
3. **③  Predict** — Run a trained model live on the device or stream
   predictions to Unity, with smoothing controls and game-recording
   start/stop.
4. **Calibration  (optional)** — Only needed when reusing a pretrained
   model with a rotated bracelet, or to set a new reference orientation.
5. **Validation** — Reproducible cross-validation across feature sets,
   models, and CV strategies. Replaces the old Performance Review tab;
   every run is saved to `data/validation_runs/` with full config and
   environment. Backed by `gui/widgets/evaluation_tab.py`.

The GUI orchestrates:

- Device connection/stream handling and monopolar/bipolar mode toggle.
- Protocol-driven recording, optionally driven by the Unity training game.
- Dataset creation + model training dialogs.
- Real-time prediction and optional TCP server.
- Game recording controls.
- Quattrocento import/training flows.

Notable supporting widgets:

- `widgets/workflow_stepper.py` — the three-step banner across the top.
- `widgets/emg_plot_panel.py` — the live multi-channel EMG view (replaces
  the older single-file `emg_plot.py`).
- `widgets/protocol_popup.py` — full-screen recording cue popup; embeds the
  legacy `ProtocolWidget`.
- `widgets/game_protocol_popup.py` — dedicated overlay for the training
  game showing per-gesture balance progress while the child plays.
- `widgets/evaluation_tab.py` — the Validation tab body.
- `widgets/bracelet_graphic.py` — visual feedback on bracelet rotation,
  aware of monopolar / bipolar mode.

### Validation Tab (`gui/widgets/evaluation_tab.py`)

The Validation tab supports comparison experiments with:

- Session-role assignment UI (train/val/test).
- Holdout and cross-validation oriented workflows.
- Deep model live training curves (MLP/CNN/AttentionNet/MSTNet).
- Aggregated metrics and confusion/report outputs.
- Direct hook into the `validation/` harness so a configured experiment can
  be triggered from the GUI and its artifacts surface under
  `data/validation_runs/`.

### Validation Harness (`validation/`)

A reproducible, config-driven validation harness for the EMG pipeline. It
allows for rigorous evaluation of features and models, supporting:

- **Two domains, one corpus:** Uniformly reads sessions from both the Python
  `pipeline` and Unity C# `game` recorders, allowing cross-domain experiments
  (e.g., train on pipeline, test on Unity).
- **Honest Cross-Validation (CV):** Validates models using rigorous boundaries
  (`loso_session`, `loso_subject`, `k_fold_subjects`, `cross_domain`,
  `holdout_split`) to prevent train/test data leakage.
- **Reproducibility:** Every run outputs timestamped artifacts including the
  config, git SHA, and results. Rerunning with the same git commit and data
  produces bit-identical numbers.
- **CLI Commands:** Use
  `python -m playagain_pipeline.validation summary | list | run <yaml_config>`.

### Utility modules (`utils/`)

- `platform_utils.py` — OS detection, default data/config directories
  (`~/Documents/PlayAgain/data` cross-platform; per-OS app config dir), and
  automatic resolution of the sibling `device_interfaces` and
  `gui_custom_elements` packages so no hard-coded paths are required.
- `rest_gap_filler.py` — fills implicit-rest gaps between trials with
  synthetic rest trials, used to reconcile Unity-recorded sessions with the
  pipeline's expected label coverage.
- `migrate_requested_gesture.py` — one-shot migration that rewrites the
  `RequestedGesture` column in old game recordings from `"none"` to `"rest"`.

## End-to-End Workflow

### 1) Record Sessions

Two flavours, depending on whether you already have a trained model:

**Standard protocol-driven recording:**

1. Open GUI and select device type; toggle bipolar mode if relevant.
2. Configure subject/session/protocol settings.
3. Start recording and complete prompted protocol steps.
4. Save session artifacts under `data/sessions/<subject>/<session>/`.

**Easy-mode recording via the Unity training game (recommended for first
sessions with children):**

1. On the Record tab, click **Locate Unity…** and pick the PlayAgain build
   (one-time setup, persisted via `QSettings`).
2. Set easy-mode sensitivity (default 1.8× resting RMS baseline) and reps
   per gesture (default 3).
3. Click **▶ Launch Training Game**. The pipeline starts a
   `PredictionServer`, launches Unity, builds a balanced trial schedule, and
   constructs a `TrainingGameCoordinator`.
4. The child plays; animals appear, the coordinator broadcasts synthetic
   predictions on each muscle contraction, and Unity feeds the animals. The
   coordinator opens/closes trials in the `RecordingSession` automatically
   so the recording lines up with what the child actually saw.
5. When complete, session is saved under `data/sessions/<subject>/<session>/`
   just like a protocol-driven recording.

### 2) Build Dataset

1. Select sessions (subject-level or explicit selection).
2. Set `window_size_ms` and `window_stride_ms`.
3. Optionally enable **per-session rotation** to align channels from
   different bracelet placements, mark **bad channels** with an interpolate
   or zero strategy, or **pre-extract features** at dataset build time.
4. Generate dataset arrays and metadata in `data/datasets/<name>/`.

### 3) Train Model

1. Choose model type.
2. Train from dataset.
3. Save artifact to `data/models/<model_name>/` with `metadata.json`.

### 4) Run Real-Time Prediction

1. Load a trained model in Prediction tab.
2. Start prediction worker.
3. Optionally enable/start Unity TCP server.
4. Observe smoothed gesture output in GUI and/or Unity client.

### 5) Record Gameplay

1. Start game recording in Prediction tab.
2. Stream EMG and Unity game-state messages concurrently.
3. Stop recording to finalize `recording.csv` and `config.json`.

### 6) Run Automated Validation

1. Define an experiment YAML config (e.g.
   `experiments/loso_baseline.yaml`).
2. Run
   `python -m playagain_pipeline.validation run experiments/loso_baseline.yaml`,
   or trigger the same workflow from the Validation tab.
3. Review results in `data/validation_runs/<timestamp>_<name>/`.

## Data Artifacts

### Sessions (`data/sessions/<subject>/<session>/`)

- `data.npy`
- `data.csv`
- `metadata.json` (includes `signal_mode`, `rotation_offset`,
  `rotation_confidence`, `bad_channels`, `trial_type` per trial)
- `gesture_set.json`

Unity-recorded sessions land under
`data/sessions/unity_sessions/<subject>/<session>/` and are listed
transparently by `DataManager.list_sessions(...)`.

### Datasets (`data/datasets/<name>/`)

- `X.npy`
- `y.npy`
- `trial_ids.npy`
- `metadata.json` (includes `bad_channel_mode`, `signal_modes_used`,
  `features_extracted` + `feature_config`, `per_session_rotation`,
  `session_rotation_offsets`)

### Models (`data/models/<name>/`)

- Serialized model files (`.pkl`, `.pt`, `.cbm`, depending on backend)
- `metadata.json`

### Calibrations (`data/calibrations/`)

- `reference_calibration_monopolar.json` — current reference for monopolar
  acquisition.
- `reference_calibration_bipolar.json` — current reference for bipolar
  acquisition.
- `calibration_<session>[_<mode>].json` — per-session calibration snapshots.
- Legacy `reference_calibration.json` is honoured for backward compatibility.

### Game Recordings (`data/game_recordings/<subject>/game_[<name>_]<timestamp>/`)

- `recording.csv` — synchronized EMG + predictions + Unity ground truth.
- `config.json` — recording metadata, model info, calibration info,
  participant info, final sample / duration stats.

### Participant Info (`data/Participant_Info/<subject_id>.json`)

- Persistent participant record (name fields, demographics, free-form
  metadata) saved/loaded by `DataManager.save_participant_info` /
  `load_participant_info`.

### Validation Runs (`data/validation_runs/<timestamp>_<name>/`)

- `experiment.json` (frozen configuration used for the run)
- `environment.json` (git SHA, python and key dependency versions)
- `session_index.json` (all session paths used across all folds)
- `results.json` (machine-readable metrics)
- `results.csv` (pandas/Excel-friendly flat table)

## Scripting API

The package exports core symbols in `playagain_pipeline/__init__.py`,
including:

- Gesture and protocol helpers.
- Recording session and data manager classes.
- Device manager abstractions.
- Model manager and feature extractor entry points.
- Calibration and configuration classes.

## Utility and Analysis Scripts

- `data/scripts/` contains conversion/import/plotting helpers for dataset
  work.
- `performance_assessment/` contains comparison tooling and notebook-based
  analysis.

## Extension Points

### Add a New Model

1. Implement a classifier class in `models/classifier.py` with the expected
   API.
2. Register it in `ModelManager.AVAILABLE_MODELS`.
3. Add corresponding `ModelConfig` parameters in `config/config.py`.
4. Expose/edit parameters in GUI dialogs when needed.

### Add a New Feature

1. Implement a `BaseFeatureExtractor` subclass.
2. Register via `@register_feature("feature_name")` in
   `models/feature_pipeline.py`.
3. Enable/configure through feature-selection tooling.

### Add a New Device Source

1. Extend `devices/emg_device.py` with a compatible device class.
2. Wire device selection/controls in `gui/main_window.py`.
3. Ensure emitted data shapes match downstream expectations.

### Add a New Calibration Check

1. Implement the check as a method in
   `calibration/calibration_validation.py:CalibrationValidator` returning a
   `CheckResult` with an appropriate `severity` (`error` blocks acceptance,
   `warning` is informational).
2. Add the call to `run_all(...)`.
3. Update `CalibrationReport.interpret()` if the new check warrants a
   plain-English explanation in the user-facing summary.

## Notes

- Defaults and behavior evolve with code; treat this file as a maintained
  overview and confirm details against module docstrings when changing
  internals.
- If you are updating pipeline behavior, update this README and any related
  GUI help text in the same change.