# PlayAgain Gesture Pipeline — Detailed Documentation

## Overview

`playagain_pipeline` is the Python package used to collect EMG data, train
models, run real-time gesture inference, and integrate predictions with Unity
via TCP.

At a high level, the pipeline provides:

- Protocol-driven EMG recording sessions.
- Rotation-aware calibration support.
- Dataset creation from raw sessions.
- Multiple ML model backends (classical and deep learning).
- Real-time prediction with smoothing.
- Unity integration and synchronized game recording.
- Post-hoc model/session analysis tooling (Performance Review).

## Requirements

From `pyproject.toml`:

- Python `>=3.11,<3.13`
- Package name: `playagain-pipeline`

The project also references local development dependencies via `tool.uv.sources`
for `device-interfaces` and `gui-custom-elements`.

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

`run_gui.py` delegates to `playagain_pipeline.gui.main_window.main()`.

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
    -> Optional synchronized game recording (CSV + config metadata)
```

Primary flows:

1. **Recording Flow**: device stream -> `RecordingSession` -> session files.
2. **Training Flow**: sessions -> windowed dataset -> trained model directory.
3. **Prediction Flow**: live EMG -> model inference -> smoother -> GUI/TCP output.
4. **Game Logging Flow**: EMG + predictions + Unity state -> synchronized CSV rows.

## Current Project Structure

```text
playagain_pipeline/
├── __init__.py
├── run_gui.py
├── prediction_server.py
├── game_recorder.py
├── config.json
├── pyproject.toml
├── README.md
│
├── calibration/
│   ├── __init__.py
│   └── calibrator.py
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
│       ├── bracelet_graphic.py
│       ├── busy_overlay.py
│       ├── calibration_dialog.py
│       ├── config_dialog.py
│       ├── emg_plot.py
│       ├── feature_selection.py
│       ├── performance_tab.py
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
│   └── platform_utils.py
└── data/
    ├── Participant_Info/
    ├── calibrations/
    ├── datasets/
    ├── game_recordings/
    ├── models/
    ├── quattrocento/
    ├── scripts/
    └── sessions/
```

## Core Concepts

### Gestures and Labeling

`core/gesture.py` defines `Gesture`, `GestureSet`, and `GestureCategory`.

- Default gesture set (`create_default_gesture_set()`):
  - `rest` (label `0`)
  - `fist` (label `1`)
  - `pinch` (label `2`)
  - `tripod` (label `3`)
- Gestures include display metadata (`display_name`, `emoji`, `duration_hint`).
- Gesture sets serialize to JSON and are stored with each session.

### Recording Sessions

`core/session.py` persists each session as:

- `data.npy` (raw EMG)
- `data.csv` (human-readable raw EMG)
- `metadata.json` (session/trials/device details)
- `gesture_set.json` (gesture definitions used)

### Data Management

`core/data_manager.py` manages data directories and conversion workflows.

Notable responsibilities:

- Session save/load and subject/session listing.
- Legacy-safe path resolution for sanitized/raw folder names.
- Dataset creation with configurable window size/stride.
- Participant metadata save/load under `data/Participant_Info`.
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
- Built-in behavior includes rest insertion between active gestures and optional
  calibration sync capture at session start.

### Calibration (`calibration/calibrator.py`)

Calibration components provide rotation-aware channel alignment:

- Activation extraction and envelope processing.
- Rotation offset detection.
- Channel remapping and confidence reporting.
- Integration hooks used by GUI calibration workflows.

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

`feature_pipeline.py` provides a registry-driven feature pipeline with built-ins:

- `rms`, `mav`, `var`, `wl`, `zc`, `ssc`, `iemg`, `ssi`

### Prediction Server (`prediction_server.py`)

Responsibilities:

- TCP server startup (`host` default `127.0.0.1`, `port` default `5555`).
- Model loading and per-buffer inference.
- Smoothed prediction output via `PredictionSmoother`.
- Bidirectional messaging with Unity clients.

Prediction smoothing combines:

- Exponential moving average (`alpha`, default `0.3`)
- Stability gate (`min_stable_ms`, default `150`)

### Game Recorder (`game_recorder.py`)

Records synchronized rows containing:

- Time since recording start.
- Current prediction (label, id, confidence, class probabilities).
- Unity-derived ground-truth state fields.
- Raw EMG channel samples.

Also stores session-level metadata (model, calibration, participant fields) and
uses a background writer queue for stable disk throughput.

### GUI (`gui/main_window.py`)

Top-level tabs created in `MainWindow`:

- Recording
- Calibration
- Training
- Prediction
- Performance Review

The GUI orchestrates:

- Device connection/stream handling.
- Protocol-driven recording.
- Dataset creation + model training dialogs.
- Real-time prediction and optional TCP server.
- Game recording controls.
- Quattrocento import/training flows.

### Performance Review (`gui/widgets/performance_tab.py`)

The Performance Review tab supports comparison experiments with:

- Session-role assignment UI (train/val/test).
- Holdout and cross-validation oriented workflows.
- Deep model live training curves (MLP/CNN/AttentionNet/MSTNet).
- Aggregated metrics and confusion/report outputs.

## End-to-End Workflow

### 1) Record Sessions

1. Open GUI and select device type.
2. Configure subject/session/protocol settings.
3. Start recording and complete prompted protocol steps.
4. Save session artifacts under `data/sessions/<subject>/<session>/`.

### 2) Build Dataset

1. Select sessions (subject-level or explicit selection).
2. Set `window_size_ms` and `window_stride_ms`.
3. Generate dataset arrays and metadata in `data/datasets/<name>/`.

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
3. Stop recording to finalize CSV and session config output.

## Data Artifacts

### Sessions (`data/sessions/<subject>/<session>/`)

- `data.npy`
- `data.csv`
- `metadata.json`
- `gesture_set.json`

### Datasets (`data/datasets/<name>/`)

- `X.npy`
- `y.npy`
- `metadata.json`

### Models (`data/models/<name>/`)

- Serialized model files (`.pkl`, `.pt`, `.cbm`, depending on backend)
- `metadata.json`

### Game Recordings (`data/game_recordings/<subject>/`)

- CSV samples with prediction and game-state fields
- Session-level config/metadata companion files (when enabled by recorder)

## Scripting API

The package exports core symbols in `playagain_pipeline/__init__.py`, including:

- Gesture and protocol helpers.
- Recording session and data manager classes.
- Device manager abstractions.
- Model manager and feature extractor entry points.
- Calibration and configuration classes.

## Utility and Analysis Scripts

- `data/scripts/` contains conversion/import/plotting helpers for dataset work.
- `performance_assessment/` contains comparison tooling and notebook-based analysis.

## Extension Points

### Add a New Model

1. Implement a classifier class in `models/classifier.py` with the expected API.
2. Register it in `ModelManager.AVAILABLE_MODELS`.
3. Add corresponding `ModelConfig` parameters in `config/config.py`.
4. Expose/edit parameters in GUI dialogs when needed.

### Add a New Feature

1. Implement a `BaseFeatureExtractor` subclass.
2. Register via `@register_feature("feature_name")` in `models/feature_pipeline.py`.
3. Enable/configure through feature-selection tooling.

### Add a New Device Source

1. Extend `devices/emg_device.py` with a compatible device class.
2. Wire device selection/controls in `gui/main_window.py`.
3. Ensure emitted data shapes match downstream expectations.

## Notes

- Defaults and behavior evolve with code; treat this file as a maintained
  overview and confirm details against module docstrings when changing internals.
- If you are updating pipeline behavior, update this README and any related GUI
  help text in the same change.