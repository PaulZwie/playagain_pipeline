# PlayAgain Gesture Pipeline — Comprehensive Documentation

## Overview

The **PlayAgain Gesture Pipeline** (`playagain_pipeline`) is a complete Python application for EMG-based gesture recognition used in the PlayAgain rehabilitation game. It handles the entire workflow from recording raw EMG signals through a Muovi device, to training machine learning models, to real-time gesture prediction during gameplay.

The pipeline communicates with a Unity game (PlayAgain-Game2) via TCP, sending gesture predictions to control in-game actions and receiving ground truth data from the game for research recordings.

---

## Getting Started

### Requirements
- Python 3.10 or higher
- `pip` (or your preferred Python package manager)
- A virtual environment is recommended
- TCP port 5555 (default) open if using the prediction server

### Installation
```bash
cd /path/to/Master/Dataprocessing/playagain_pipeline
python -m venv venv                   # create virtual env (optional)
source venv/bin/activate              # macOS / Linux
pip install -r requirements.txt       # install dependencies
```

### Running
- **GUI mode:**
  ```bash
  python -m playagain_pipeline.run_gui
  ```

- **Headless prediction server:**
  ```bash
  python -m playagain_pipeline.prediction_server
  ```

- **Advanced dataset creation:**
  ```bash
  python -m playagain_pipeline.advanced_dataset_creation
  ```

### Development
See the sections below for module documentation and the overall architecture; code lives under the `playagain_pipeline` package and can be edited directly. Unit tests and notebooks are located alongside the code in this repository.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Project Structure](#project-structure)
3. [Core Concepts](#core-concepts)
4. [Module Documentation](#module-documentation)
   - [Entry Point (run_gui.py)](#entry-point)
   - [Configuration (config/)](#configuration)
   - [Core Data Types (core/)](#core-data-types)
   - [Devices (devices/)](#devices)
   - [Protocols (protocols/)](#protocols)
   - [Calibration (calibration/)](#calibration)
   - [Models & Feature Extraction (models/)](#models--feature-extraction)
   - [Prediction Server (prediction_server.py)](#prediction-server)
   - [Prediction Smoothing](#prediction-smoothing)
   - [Game Recorder (game_recorder.py)](#game-recorder)
   - [GUI (gui/)](#gui)
5. [Data Flow](#data-flow)
6. [How Training Works](#how-training-works)
7. [How Real-Time Prediction Works](#how-real-time-prediction-works)
8. [How Game Recording Works](#how-game-recording-works)
9. [File Formats](#file-formats)
10. [Extending the Pipeline](#extending-the-pipeline)

---

## Architecture Overview

```
┌──────────────┐      ┌──────────────────┐      ┌───────────────┐
│  Muovi EMG   │─────>│  Python Pipeline  │─────>│  Unity Game   │
│   Device     │ TCP  │   (this project)  │ TCP  │ (PlayAgain)   │
└──────────────┘      └──────────────────┘      └───────────────┘
                              │  ^                      │
                              │  │    ground truth       │
                              │  └──────────────────────┘
                              │
                    ┌─────────┴──────────┐
                    │                    │
              ┌─────┴─────┐      ┌──────┴──────┐
              │  Training │      │  Recording  │
              │  Pipeline │      │  (CSV/NPY)  │
              └───────────┘      └─────────────┘
```

**Key data flows:**
1. **Recording**: Muovi → Python → NPY/JSON files (training data)
2. **Prediction**: Muovi → Python (ML model) → Unity (TCP JSON)
3. **Game Recording**: Muovi + Model predictions + Unity ground truth → CSV

---

## Project Structure

```
playagain_pipeline/
├── __init__.py                  # Package root, exports all public symbols
├── run_gui.py                   # Entry point — launches the GUI
├── prediction_server.py         # TCP server for Unity + PredictionSmoother
├── game_recorder.py             # Records gameplay data to CSV
├── advanced_dataset_creation.py # Standalone script for advanced preprocessing
├── README.md                    # This file
│
├── config/
│   ├── __init__.py
│   └── config.py                # PipelineConfig, DeviceConfig, ModelConfig, etc.
│
├── core/
│   ├── __init__.py
│   ├── gesture.py               # Gesture, GestureSet, GestureCategory
│   ├── session.py               # RecordingSession, RecordingTrial, RecordingMetadata
│   └── data_manager.py          # DataManager — file I/O, dataset creation
│
├── devices/
│   ├── __init__.py
│   └── emg_device.py            # BaseEMGDevice, MuoviDevice, SyntheticEMGDevice, DeviceManager
│
├── protocols/
│   ├── __init__.py
│   └── protocol.py              # RecordingProtocol, ProtocolPhase, ProtocolStep, ProtocolConfig
│
├── calibration/
│   ├── __init__.py
│   └── calibrator.py            # AutoCalibrator, CalibrationProcessor, CalibrationResult
│
├── models/
│   ├── __init__.py
│   ├── classifier.py            # All ML models + ModelManager + EMGFeatureExtractor
│   └── feature_pipeline.py      # Modular feature extraction with registry pattern
│
├── gui/
│   ├── __init__.py
│   ├── main_window.py           # MainWindow — the full application UI
│   └── widgets/
│       ├── __init__.py
│       ├── emg_plot.py          # Real-time EMG visualization (Vispy)
│       ├── protocol_widget.py   # Recording protocol display + countdown
│       ├── calibration_dialog.py# Guided calibration dialog
│       ├── config_dialog.py     # Configuration dialog + bracelet visualization
│       ├── feature_selection.py # Feature selection dialog
│       └── training_dialog.py   # Advanced training dialog with live plots
│
├── utils/
│   └── __init__.py
│
└── data/                        # Runtime data directory
    ├── sessions/                # Recorded EMG sessions (NPY + metadata)
    │   └── <subject_id>/
    │       └── <session_id>/
    ├── datasets/                # ML-ready datasets (windowed features)
    ├── models/                  # Trained model files
    ├── calibrations/            # Calibration reference files
    └── game_recordings/         # Gameplay recording CSVs
        └── <subject_id>/
```

---

## Core Concepts

### Gestures

A **Gesture** is a hand/wrist movement that produces a characteristic EMG pattern. Each gesture has:
- A **name** (internal identifier, e.g., `"fist"`)
- A **display_name** (user-facing, e.g., `"Fist"`)
- A **label_id** (integer for ML, e.g., `1`)
- A **category** (enum: REST, FINGER, HAND, WRIST, GRIP, CUSTOM)

**GestureSet** is a collection of gestures with auto-assigned label IDs. The default set contains:

| Label ID | Name   | Description                        |
|----------|--------|------------------------------------|
| 0        | rest   | Relaxed hand (no activation)       |
| 1        | fist   | Closed fist (power grasp)          |
| 2        | pinch  | Thumb-index pinch grip             |
| 3        | tripod | Thumb-index-middle tripod grip     |

### Sessions

A **RecordingSession** captures a single recording run. It stores:
- Raw EMG data as numpy arrays (`data.npy`)
- Metadata (device info, timestamps, channel mapping) as `metadata.json`
- Trial annotations (which gesture was performed when) inside metadata
- The gesture set used (`gesture_set.json`)

Each session is saved to `data/sessions/<subject_id>/<session_id>/`.

### Datasets

A **Dataset** is an ML-ready transformation of one or more sessions. The `DataManager.create_dataset()` method:
1. Loads sessions and their trial annotations
2. Extracts fixed-size windows (e.g., 200ms at 2000Hz = 400 samples)
3. Slides the window with a configurable stride (e.g., 50ms)
4. Labels each window based on which trial it falls in
5. Optionally applies a preprocessing function (bandpass, normalization, etc.)

The result is `{X: np.ndarray, y: np.ndarray, metadata: dict}` saved as NPY files.

### Protocols

A **RecordingProtocol** defines the sequence of steps during a recording session:
- **PREPARATION** → **REST** → **CUE** → **HOLD** → **REST** → (repeat for each gesture × repetitions)

Timing is configurable (e.g., 5s hold, 3s rest). Gesture order can be randomized. During the HOLD phase, `is_recording` is True and the session records a trial for that gesture.

---

## Module Documentation

### Entry Point

**`run_gui.py`** (5 lines) — Simply imports and calls `main()` from `gui.main_window`. Run with:
```bash
python -m playagain_pipeline.run_gui
```

### Configuration

**`config/config.py`** — Hierarchical dataclass-based configuration system:

```python
PipelineConfig
├── DeviceConfig        # device_type, num_channels=32, sampling_rate=2000, ip, port
├── RecordingConfig     # window_size_ms=200, stride=50, protocol, gesture_set
├── CalibrationConfig   # enabled, num_gestures, duration, confidence_threshold
└── ModelConfig         # Per-model hyperparameters (SVM, RF, LDA, CatBoost, MLP, CNN, AttentionNet)
```

Supports JSON serialization: `config.save(path)` / `PipelineConfig.load(path)`.

The **`ModelConfig`** contains hyperparameters for all model types:
- **SVM**: kernel (rbf/linear/poly), C, gamma
- **Random Forest**: n_estimators, max_depth, min_samples_split
- **LDA**: solver (svd/lsqr/eigen), shrinkage
- **CatBoost**: iterations, learning_rate, depth, l2_leaf_reg
- **MLP**: hidden_layers list, epochs, batch_size, learning_rate, optimizer, early_stopping
- **CNN**: conv_filters, kernel_sizes, fc_layers, dropout, epochs
- **AttentionNet**: base_channels, branch_kernels, reduction_ratio (squeeze-excite attention)

### Core Data Types

**`core/gesture.py`** — Gesture definition and management:

```python
# Create the default 4-gesture set
gesture_set = create_default_gesture_set()  # rest, fist, pinch, tripod

# Or build custom
gs = GestureSet("my_gestures")
gs.add(Gesture(name="rest", category=GestureCategory.REST, label_id=0))
gs.add(Gesture(name="fist", category=GestureCategory.GRIP, label_id=1))
```

The module also provides `format_prompt()` and `format_pause_prompt()` functions for recording UI display, with `TASK_DURATION=5s` and `PAUSE_DURATION=8s`.

**`core/session.py`** — Recording session lifecycle:

```python
session = RecordingSession(
    session_id="2026-02-14_10:00:00_5rep",
    subject_id="VP_01",
    device_name="MUOVI",
    num_channels=32,
    sampling_rate=2000,
    gesture_set=gesture_set
)
session.start_recording()
session.start_trial("fist")      # Mark beginning of a trial
session.add_data(emg_chunk)      # Add data chunks as they arrive
session.end_trial()              # Mark end of trial
session.stop_recording()
session.save(Path("data/sessions/VP_01/session_1"))
```

Internally, `RecordingTrial` tracks `trial_id`, `gesture_name`, `gesture_label`, `start_sample`, `end_sample`, `start_time`, `end_time`, and `is_valid`. Trials can be marked invalid during recording (e.g., movement artifact detected).

**`core/data_manager.py`** — Central data I/O and dataset creation:

The `DataManager` manages the `data/` directory structure. Key methods:

- **`save_session(session)`** / **`load_session(subject_id, session_id)`** — Session persistence
- **`list_subjects()`** / **`list_sessions(subject_id)`** — Discovery
- **`create_dataset(name, sessions, window_size_ms, window_stride_ms, ...)`** — Converts raw sessions into ML-ready windowed arrays
- **`save_dataset(dataset)`** / **`load_dataset(name)`** — Dataset persistence
- **`get_train_test_split(dataset, test_size, stratify)`** — Sklearn wrapper

**Dataset creation** is the core preprocessing step. It:
1. Takes one or more `RecordingSession` objects
2. For each valid trial, extracts sliding windows of EMG data
3. Labels each window with the trial's gesture label
4. Returns `{X: shape(n_windows, window_samples, n_channels), y: shape(n_windows,), metadata: {...}}`

The `preprocessing_fn` parameter allows injecting custom filtering (e.g., bandpass) during windowing.

### Devices

**`devices/emg_device.py`** — EMG device abstraction layer:

**`BaseEMGDevice(QObject)`** is the abstract interface with Qt signals:
- `data_ready(np.ndarray)` — Emitted when new data chunk arrives (shape: samples x channels)
- `connected(bool)` — Connection state change
- `error(str)` — Error messages
- `ground_truth_changed(str)` — Ground truth label change (used in session replay mode)

**`MuoviDevice(BaseEMGDevice)`** — Hardware interface for the Muovi/Muovi Plus EMG bracelet:
- Uses the `device_interfaces` package for hardware communication
- Handles the Muovi's peculiar architecture: Unity/Python acts as **TCP server** on port 54321, and the Muovi bracelet **connects to it** as a client
- Connection flow: `connect_device()` starts TCP server → Muovi powers on and connects → `connected` signal fires → `start_streaming()` begins data flow
- Supports both Muovi (32 EMG + 6 aux channels) and Muovi Plus (64 EMG channels)

**`SyntheticEMGDevice(BaseEMGDevice)`** — Testing device with two modes:
1. **Procedural generation**: Creates EMG-like signals with configurable gesture patterns (different frequency/amplitude for fist, pinch, etc.) and realistic burst noise
2. **Session replay**: Loads a saved session (NPY file) and replays its data in a loop, emitting `ground_truth_changed` signals at trial boundaries — perfect for testing the prediction pipeline without hardware

**`DeviceManager`** — Factory that creates devices by `DeviceType` enum (SYNTHETIC, MUOVI, MUOVI_PLUS), manages data callbacks, and handles connection lifecycle.

### Protocols

**`protocols/protocol.py`** — Recording protocol definition:

```python
protocol_config = create_standard_protocol()  # 5 reps per gesture
protocol = RecordingProtocol(gesture_set, protocol_config)

# Generates a sequence of steps:
# [Preparation 3s] -> [Rest 3s] -> [Cue "Fist" 1s] -> [Hold "Fist" 5s] ->
# [Rest 3s] -> [Cue "Pinch" 1s] -> [Hold "Pinch" 5s] -> ... -> [Complete]
```

**`ProtocolPhase`** enum: PREPARATION, REST, CUE, HOLD, RELEASE, FEEDBACK, COMPLETE

**`ProtocolStep`** dataclass: Contains phase, gesture, duration, display message, trial/rep index, and `is_recording` flag (True only during HOLD).

The `ProtocolWidget` in the GUI drives through steps with a timer, displaying emoji-labeled gesture prompts with countdown.

### Calibration

**`calibration/calibrator.py`** — Electrode orientation detection:

**Problem**: The Muovi bracelet can be placed on the forearm at any rotational angle. The same gesture produces different channel patterns depending on electrode orientation relative to muscles.

**Solution**: The calibration system measures EMG patterns from standardized finger movements and uses **circular cross-correlation** to find the rotation offset between the current placement and a reference.

**`CalibrationProcessor`** algorithm:
1. `compute_rms_envelope()` — Sliding-window RMS via convolution
2. `compute_activation_pattern()` — Mean RMS per channel, normalized to [0, 1]
3. `find_rotation_offset()` — For each possible rotation (0 to N-1 channels), compute Pearson correlation of activation patterns across all common gestures, pick the rotation with highest average correlation
4. `create_channel_mapping()` — Apply modular rotation to channel indices

**`AutoCalibrator`** provides a high-level interface: save/load reference calibrations, auto-apply during prediction.

**`CalibrationDialog`** (GUI) — Guides the user through 8 individual finger gestures (rest, index/middle/ring/pinky/thumb flex, index extend, wrist flex) with 3s countdown + 3s recording each.

### Models & Feature Extraction

**`models/classifier.py`** (1681 lines) — The ML framework:

#### Feature Extraction

**`EMGFeatureExtractor`** computes 6 time-domain features per channel:

| Feature | Formula | Description |
|---------|---------|-------------|
| RMS | sqrt(mean(x^2)) | Root Mean Square — signal energy |
| MAV | mean(abs(x)) | Mean Absolute Value — signal amplitude |
| VAR | mean((x - mean(x))^2) | Variance — signal variability |
| WL | sum(abs(diff(x))) | Waveform Length — signal complexity |
| ZC | count of sign changes | Zero Crossings — frequency estimate |
| SSC | count of slope sign changes | Slope Sign Changes — frequency estimate |

For a 32-channel window, this produces 32 x 6 = 192 features. The extraction is fully vectorized with numpy for speed.

#### Classifier Implementations

All classifiers inherit from **`BaseClassifier(ABC)`** with interface:
- `extract_features(X)` — Raw EMG windows -> feature vectors
- `train(X, y, **kwargs)` — Train on feature vectors
- `predict(X)` / `predict_proba(X)` — Inference
- `save(path)` / `load(path)` — Persistence

**`SVMClassifier`**: StandardScaler -> sklearn SVC (RBF kernel by default). Robust baseline.

**`RandomForestClassifier`**: sklearn RandomForest with configurable n_estimators, max_depth. Returns feature importances for interpretability.

**`LDAClassifier`**: Linear Discriminant Analysis with optional shrinkage. Fast, works well when classes are linearly separable.

**`CatBoostClassifier`**: Gradient boosted trees. Handles categorical features natively, automatic GPU detection. Saves models in `.cbm` format.

**`MLPClassifier`**: PyTorch Sequential MLP with configurable hidden layers, dropout, and early stopping. Supports Adam/SGD optimizers. Training loop with validation split and epoch-level logging.

**`CNNClassifier`**: PyTorch 1D-CNN architecture:
```
Input(samples, channels)
-> Conv1d -> BatchNorm -> ReLU -> MaxPool
-> Conv1d -> BatchNorm -> ReLU -> MaxPool
-> AdaptiveAvgPool1d(1) -> Flatten
-> FC -> ReLU -> Dropout -> FC -> Softmax
```

**`AttentionNetClassifier`**: Inception-style architecture with channel attention (squeeze-and-excite). Uses multiple parallel 1D convolution branches with different kernel sizes to capture multi-scale temporal patterns, then applies channel attention to weight their importance. Extends `CNNClassifier`.

#### ModelManager

Factory + persistence layer. `AVAILABLE_MODELS` dict maps string keys to classes:
```python
{"svm": SVMClassifier, "random_forest": RandomForestClassifier,
 "lda": LDAClassifier, "catboost": CatBoostClassifier,
 "mlp": MLPClassifier, "cnn": CNNClassifier, "attention_net": AttentionNetClassifier}
```

Methods: `create_model(type, name)`, `train_model(model, dataset)`, `load_model(name)`, `list_models()`.

#### Feature Pipeline

**`models/feature_pipeline.py`** — Modular feature extraction with a decorator-based registry:

```python
@register_feature("rms")
def compute_rms(data, **kwargs):
    return np.sqrt(np.mean(data**2, axis=0))
```

8 registered features: RMS, MAV, Variance, Waveform Length, Zero Crossings, Slope Sign Changes, Integrated EMG, Simple Square Integral. Features can be enabled/disabled and reordered through the `FeaturePipeline` class.

### Prediction Server

**`prediction_server.py`** — TCP server for Unity game integration:

The `PredictionServer` bridges the Python ML pipeline and the Unity game:

1. **Starts a TCP server** on `127.0.0.1:5555`
2. **Accepts connections** from the Unity game (`PipelineGestureClient`)
3. **Sends handshake** with model info (name, class names, num_classes)
4. **For each EMG data chunk**: runs prediction -> applies smoothing -> broadcasts JSON to all clients
5. **Receives game state** from Unity (ground truth, requested gesture, camera blocking state)

**Protocol** (newline-delimited JSON):

```
Python -> Unity (predictions):
{"gesture":"fist","gesture_id":1,"confidence":0.95,"probabilities":{"rest":0.02,"fist":0.95,"pinch":0.02,"tripod":0.01},"timestamp":1739539200.123}

Unity -> Python (ground truth):
{"type":"game_state","gesture_requested":"fist","ground_truth":true,"camera_blocking":true,"timestamp":123.456}
```

**Bidirectional communication**: The TCP connection is full-duplex. Python reads from connected clients in separate reader threads (`_client_reader_loop`) while writing predictions from the main data flow. This allows Unity to send ground truth data back through the same connection.

**Callbacks**: External consumers (like `GameRecorder`) register via:
- `add_prediction_callback(fn)` — Called after each (smoothed) prediction
- `add_game_state_callback(fn)` — Called when Unity sends ground truth

### Prediction Smoothing

**`PredictionSmoother`** (in `prediction_server.py`) prevents rapid gesture switching caused by brief EMG signal similarities (e.g., tripod momentarily classified as fist).

**Algorithm:**
1. **Exponential Moving Average (EMA)** on the probability vector:
   `ema_t = alpha * p_t + (1 - alpha) * ema_(t-1)`
   where p_t is the raw probability vector and alpha controls smoothing (lower = smoother).

2. **Stability window**: The smoothed winner (argmax of EMA) must be consistent for at least `min_stable_ms` milliseconds before the output switches to it.

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 0.3 (30%) | EMA weight for new predictions. 0.1 = very smooth, 0.9 = almost raw |
| `min_stable_ms` | 150 ms | Minimum stability time before switching gesture |

**Example**: At 111 Hz prediction rate, a brief 2-frame misclassification (~18ms) would NOT cause a switch because it doesn't exceed the 150ms stability window. But a genuine 200ms gesture hold WOULD switch.

Both the PredictionServer (affecting Unity) and the GUI display use their own smoother instances with the same parameters, configurable from the GUI.

### Game Recorder

**`game_recorder.py`** — Records synchronized gameplay data to CSV:

The `GameRecorder` captures three data streams simultaneously during gameplay:

1. **Raw EMG data** — From the device, at the device's sample rate
2. **Model predictions** — The (smoothed) gesture predicted by the ML model
3. **Ground truth from Unity** — What gesture the game is requesting and when

**CSV Format:**
```csv
Timestamp,PredictedGesture,PredictedGestureId,Confidence,Prob_rest,Prob_fist,Prob_pinch,Prob_tripod,GroundTruthActive,RequestedGesture,CameraBlocking,EMG_Ch0,...,EMG_Ch31
0.000100,rest,0,0.9800,0.9800,0.0100,0.0050,0.0050,0,none,0,1.234e-04,...
12.500100,fist,1,0.9200,0.0300,0.9200,0.0300,0.0200,1,fist,1,-2.345e-04,...
```

**Ground truth timing**: The ground truth (`GroundTruthActive=1`) starts when the camera finishes its transition to the interaction/blocking position (not when the animal first appears). This matches the player's perception — the gesture is "requested" when the camera has settled and the instruction is visible. Ground truth ends when the feeding timer counts all the way in (gesture completed successfully).

**Data flow:**
```
Muovi Device --> _on_data_received() --> PredictionServer.on_emg_data()
                       |                         |
                       |                    +----+---- prediction callback --> GameRecorder.on_prediction()
                       |                    |
                       |               Unity sends game_state --> GameRecorder.on_game_state()
                       |
                       +--> GameRecorder.on_emg_data()  (records row with current prediction + ground truth)
```

The ordering in `_on_data_received()` is **critical**: `PredictionServer.on_emg_data()` is called BEFORE `GameRecorder.on_emg_data()`, ensuring the recorder has the latest prediction when it writes each sample.

### GUI

**`gui/main_window.py`** — The full application UI with 4 tabs:

#### Recording Tab
- **Session Settings**: Subject ID, notes
- **Device**: Device type selection (Synthetic/Muovi/Muovi Plus), channel count, sampling rate
- **Session Replay**: For synthetic device — select a previously recorded session to replay its data
- **Gestures**: Gesture set selection
- **Protocol**: Quick (3 reps) / Standard (5 reps) / Extended (10 reps)
- **Controls**: Start/Stop recording

#### Calibration Tab
- **Status Display**: Current calibration (rotation offset, confidence)
- **Actions**: Start calibration (launches dialog), save as reference, load from file

#### Training Tab
- **Dataset**: Create dataset from sessions (select by subject or specific sessions, configure window size and stride)
- **Model**: Select model type, train
- **Trained Models**: List of saved models

#### Prediction Tab
- **Model**: Load a trained model for prediction
- **Prediction Display**: Large text showing current gesture and confidence
- **Controls**: Start/Stop prediction
- **Prediction Smoothing**: Enable/disable, configure alpha and stability window
- **Unity TCP Server**: Host/port configuration, start/stop server
- **Game Recording**: Subject ID, session name, start/stop recording, live stats

#### Widgets

- **`EMGPlotWindow`**: Separate window with real-time EMG visualization using Vispy. Features circular buffer, 30 FPS update timer (decoupled from data rate), ground truth label display for session replay.
- **`ProtocolWidget`**: Displays the current recording protocol step with emoji gesture prompts, phase-colored labels, progress bar, and countdown timer.
- **`CalibrationDialog`**: Guided dialog for electrode calibration.
- **`ConfigurationDialog`**: Tabbed dialog for all pipeline configuration.
- **`BraceletVisualizationWidget`**: Custom QPainter widget showing electrode positions with muscle group colors and rotation indicator.
- **`TrainingProgressDialog`**: Advanced training with real-time loss/accuracy plots (pyqtgraph), hyperparameter editing, results table.
- **`FeatureSelectionDialog`**: Enable/disable/configure individual features.

---

## Data Flow

### Recording Flow

```
                     Protocol Widget drives timing
                            |
                            v
User sees gesture prompt --> Protocol step (HOLD phase, is_recording=True)
                            |
Muovi --> data_ready signal --> _on_data_received()
                                    |
                                    +--> EMGPlotWindow (visualization)
                                    +--> RecordingSession.add_data()
                                            |
                                            v
                                    start_trial() / end_trial()
                                            |
                                            v
                                    session.save() --> data.npy + metadata.json
```

### Prediction Flow

```
Muovi --> data_ready signal --> _on_data_received()
                                    |
                                    +--> PredictionWorker (background thread)
                                    |         |
                                    |         v
                                    |    model.predict(buffer) --> (pred, proba)
                                    |         |
                                    |         v
                                    |    GUI smoother --> display in UI
                                    |
                                    +--> PredictionServer.on_emg_data(data)
                                              |
                                              v
                                         model.predict(buffer) --> raw prediction
                                              |
                                              v
                                         PredictionSmoother.smooth() --> stable prediction
                                              |
                                              +--> prediction callbacks (GameRecorder)
                                              +--> TCP broadcast to Unity clients
```

---

## How Training Works

1. **Record sessions**: Use the Recording tab to record EMG data with the protocol. Each session captures gesture data with trial annotations.

2. **Create dataset**: In the Training tab, click "Create Dataset from Sessions":
   - Select subjects or specific sessions
   - Configure window size (default 200ms) and stride (default 50ms)
   - The DataManager extracts sliding windows from each trial and labels them

3. **Train model**: Select a model type and click "Train Model":
   - Features are extracted (RMS, MAV, VAR, WL, ZC, SSC) — 192-dimensional feature vector per window
   - Data is split 80/20 for train/validation (stratified)
   - Model is trained and evaluated
   - Results show training and validation accuracy

4. **Advanced Training**: For fine-tuning, use Tools -> Advanced Training:
   - Select dataset AND model type from dropdowns
   - Adjust hyperparameters per model type
   - Watch real-time loss/accuracy plots during training
   - Save the best model

---

## How Real-Time Prediction Works

1. **Load model**: In the Prediction tab, select and load a trained model
2. **Start prediction**: Click "Start Prediction" — creates a `PredictionWorker` thread
3. **Data arrives**: Each EMG chunk updates a rolling buffer (200ms window)
4. **Prediction**: The worker extracts features from the buffer and runs `model.predict()`
5. **Smoothing**: The `PredictionSmoother` applies EMA + stability filtering
6. **Display**: The smoothed gesture and confidence are shown in the GUI
7. **Unity**: If the TCP server is running, the smoothed prediction is broadcast as JSON

---

## How Game Recording Works

Game recording captures everything needed to evaluate gesture recognition performance during actual gameplay:

1. **Start the Unity TCP Server** in the Prediction tab
2. **Start the Unity game** — it connects to the Python server automatically
3. **Click "Start Game Recording"** — creates a `GameRecorder`
4. **During gameplay**:
   - EMG data flows through the prediction pipeline
   - Smoothed predictions are recorded via prediction callback
   - Unity sends `game_state` messages when ground truth changes
   - All data is written to CSV with synchronized timestamps
5. **Click "Stop Game Recording"** — flushes buffer and closes CSV file

**Ground truth timing** is carefully handled:
- Ground truth starts: When the camera finishes its swing to the interaction view (not when the animal first appears and the camera starts moving)
- Ground truth ends: When the feeding timer is fully counted in (gesture completed)
- The specific gesture requested (fist, pinch, tripod) is transmitted from Unity to Python

---

## File Formats

### Session Files (`data/sessions/<subject>/<session>/`)

| File | Format | Description |
|------|--------|-------------|
| `data.npy` | NumPy binary | Raw EMG data, shape (total_samples, channels) |
| `data.csv` | CSV | Same data in human-readable format |
| `metadata.json` | JSON | Session info, trial annotations, device config |
| `gesture_set.json` | JSON | Gesture definitions used in this session |

### Dataset Files (`data/datasets/<name>/`)

| File | Format | Description |
|------|--------|-------------|
| `X.npy` | NumPy binary | Feature matrix, shape (n_windows, window_samples, channels) |
| `y.npy` | NumPy binary | Labels, shape (n_windows,) |
| `metadata.json` | JSON | Dataset info (num_samples, num_classes, window params, class mapping) |

### Model Files (`data/models/<name>/`)

| File | Format | Description |
|------|--------|-------------|
| `model.*` | Various | Trained model (`.pkl` for sklearn, `.pt` for PyTorch, `.cbm` for CatBoost) |
| `metadata.json` | JSON | Model info (type, accuracies, class_names, hyperparameters, training_history) |

### Game Recording CSV (`data/game_recordings/<subject>/`)

| Column | Type | Description |
|--------|------|-------------|
| Timestamp | float | Seconds since recording start |
| PredictedGesture | string | Smoothed predicted gesture name |
| PredictedGestureId | int | Predicted gesture class ID |
| Confidence | float | Prediction confidence [0, 1] |
| Prob_<class> | float | Per-class probability (one column per class) |
| GroundTruthActive | int | 1 if the game is requesting a gesture, 0 otherwise |
| RequestedGesture | string | Which gesture the game wants ("fist", "pinch", etc., or "none") |
| CameraBlocking | int | 1 if camera is in interaction/blocking view |
| EMG_Ch<N> | float | Raw EMG channel value (scientific notation) |

---

## Extending the Pipeline

### Adding a New Model

1. Create a new class inheriting from `BaseClassifier` in `classifier.py`
2. Implement `extract_features()`, `train()`, `predict()`, `predict_proba()`, `save()`, `load()`
3. Register it in `ModelManager.AVAILABLE_MODELS`:
   ```python
   AVAILABLE_MODELS = {
       ...,
       "my_model": MyModelClassifier,
   }
   ```
4. Add hyperparameters to `ModelConfig` in `config.py`
5. Add UI for hyperparameters in `HyperparameterWidget` in `training_dialog.py`

### Adding a New Device

1. Create a new class inheriting from `BaseEMGDevice` in `emg_device.py`
2. Implement `connect_device()`, `disconnect()`, `start_streaming()`, `stop_streaming()`
3. Emit `data_ready` signal with shape (samples, channels)
4. Add to `DeviceType` enum and `DeviceManager` factory

### Adding a New Feature

Use the registry pattern in `feature_pipeline.py`:
```python
@register_feature("my_feature")
def compute_my_feature(data, **kwargs):
    """Compute my custom feature per channel."""
    return np.some_computation(data, axis=0)
```

---

## License

MIT License
