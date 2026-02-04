# EMG Gesture Recording Pipeline

A modular, extensible pipeline for recording EMG data, training gesture classification models, and performing real-time gesture prediction.

## Features

- **Flexible Gesture Definition**: Easily add, modify, or remove gestures for different experiments
- **Configurable Recording Protocols**: Define timing, repetitions, and randomization
- **Calibration System**: Detect electrode orientation and automatically correct channel mapping
- **Advanced Data Preprocessing**: Apply custom filtering, feature extraction, and normalization during dataset creation
- **Multiple ML Models**: Train SVM, Random Forest, or LDA classifiers
- **Real-time Prediction**: Stream EMG data and get live gesture predictions
- **Modular Design**: Easy to extend with new features, models, or devices

## Installation

```bash
cd playagain_pipeline
pip install -r requirements.txt
```

### Dependencies

- Python 3.9+
- NumPy
- SciPy
- PySide6
- pyqtgraph
- scikit-learn

## Quick Start

### 1. Launch the GUI

```bash
python -m playagain_pipeline.run_gui
# or
python playagain_pipeline/run_gui.py
```

### 2. Using the API Directly

```python
from playagain_pipeline.core import (
    create_default_gesture_set,
    RecordingSession,
    DataManager
)
from playagain_pipeline.protocols import RecordingProtocol, create_standard_protocol
from playagain_pipeline.devices import DeviceManager, DeviceType
from playagain_pipeline.models import ModelManager

# Create a gesture set
gesture_set = create_default_gesture_set()

# Setup device
device_manager = DeviceManager()
device = device_manager.create_device(
    DeviceType.SYNTHETIC,  # Use synthetic for testing
    num_channels=32,
    sampling_rate=2000
)

# Create recording session
session = RecordingSession(
    session_id="session_001",
    subject_id="subject_01",
    device_name="synthetic",
    num_channels=32,
    sampling_rate=2000,
    gesture_set=gesture_set
)

# Create protocol
protocol_config = create_standard_protocol()
protocol = RecordingProtocol(gesture_set, protocol_config)

# Start recording...
```

## Project Structure

```
gesture_pipeline/
├── __init__.py              # Package initialization
├── run_gui.py               # GUI entry point
├── advanced_dataset_creation.py  # Advanced dataset creation with preprocessing
├── requirements.txt         # Dependencies
│
├── core/                    # Core components
│   ├── gesture.py           # Gesture definitions
│   ├── session.py           # Recording session management
│   └── data_manager.py      # Data storage and datasets
│
├── protocols/               # Recording protocols
│   └── protocol.py          # Protocol definitions
│
├── calibration/             # Calibration system
│   └── calibrator.py        # Electrode orientation detection
│
├── devices/                 # Device interfaces
│   └── emg_device.py        # EMG device abstraction
│
├── models/                  # ML models
│   └── classifier.py        # Gesture classifiers
│
├── gui/                     # GUI components
│   ├── main_window.py       # Main application window
│   └── widgets/             # Reusable widgets
│       ├── emg_plot.py      # EMG visualization
│       └── protocol_widget.py # Protocol display
│
├── config/                  # Configuration
│   └── config.py            # Pipeline configuration
│
├── data/                    # Data storage
│   ├── calibrations/        # Calibration data
│   ├── datasets/            # Processed datasets
│   ├── models/              # Trained models
│   └── sessions/            # Recording sessions
│
├── device_interfaces/       # Device interface implementations
│   ├── devices/             # Device classes
│   ├── dicts/               # Device dictionaries
│   ├── enums/               # Device enumerations
│   ├── gui/                 # Device GUI components
│   └── tests/               # Device tests
│
├── gesture_pipeline_data/   # Runtime data directory
│   ├── calibrations/        # Calibration data
│   ├── datasets/            # Processed datasets
│   ├── models/              # Trained models
│   └── sessions/            # Recording sessions
│
└── utils/                   # Utilities
    └── __init__.py
```

## Gestures

### Default Gesture Set

- **Rest**: Relaxed hand position
- **Fist**: Closed fist
- **Index-Thumb Pinch**: Index finger to thumb
- **Three Fingers-Thumb**: Index, middle, ring to thumb

### Adding Custom Gestures

```python
from playagain_pipeline.core import Gesture, GestureSet, GestureCategory

# Create custom gesture set
gesture_set = GestureSet(name="my_gestures")

# Add gestures
gesture_set.add_gesture(Gesture(
    name="custom_gesture",
    display_name="My Custom Gesture",
    description="Description of how to perform it",
    category=GestureCategory.CUSTOM,
    duration_hint=3.0
))
```

## Protocols

### Pre-defined Protocols

- **Quick**: 3 repetitions, 2s holds (for testing)
- **Standard**: 5 repetitions, 3s holds (recommended)
- **Extended**: 10 repetitions, 4s holds (thorough)

### Custom Protocols

```python
from playagain_pipeline.protocols import ProtocolConfig, RecordingProtocol

config = ProtocolConfig(
    name="my_protocol",
    preparation_time=3.0,
    cue_time=1.0,
    hold_time=4.0,
    release_time=0.5,
    rest_time=2.0,
    repetitions_per_gesture=8,
    randomize_order=True
)

protocol = RecordingProtocol(gesture_set, config)
```

## Calibration

The calibration system detects electrode orientation by comparing activation patterns:

```python
from playagain_pipeline.calibration import AutoCalibrator

calibrator = AutoCalibrator(
    data_dir="calibration_data",
    num_channels=32,
    sampling_rate=2000
)

# Perform calibration with gesture data
calibration_data = {
    "fist": fist_emg_data,
    "extension": extension_emg_data,
    # ...
}

result = calibrator.calibrate(calibration_data, device_name="muovi")

# Apply to new data
corrected_data = calibrator.apply_calibration(raw_data)
```

## Training Models

```python
from playagain_pipeline.core import DataManager
from playagain_pipeline.models import ModelManager

# Create dataset from sessions
data_manager = DataManager("gesture_pipeline_data")
dataset = data_manager.create_dataset(
    name="my_dataset",
    window_size_ms=200,
    window_stride_ms=50
)

# Train model
model_manager = ModelManager("gesture_pipeline_data/models")
model = model_manager.create_model("svm")  # or "random_forest", "lda"
results = model_manager.train_model(model, dataset)

print(f"Training accuracy: {results['training_accuracy']:.2%}")
print(f"Validation accuracy: {results['validation_accuracy']:.2%}")
```

## Advanced Dataset Creation and Preprocessing

The pipeline supports advanced data preprocessing during the sessions-to-datasets transformation, allowing you to apply custom filtering, feature extraction, and normalization to EMG data before creating ML-ready datasets.

### Basic Dataset Creation

```python
from playagain_pipeline.core import DataManager

data_manager = DataManager("gesture_pipeline_data")

# Create basic dataset with custom parameters
dataset = data_manager.create_dataset(
    name="my_dataset",
    subject_ids=["VP_00", "VP_01"],  # Select specific subjects
    window_size_ms=200,  # 200ms windows
    window_stride_ms=50,  # 50ms overlap
    include_invalid=False  # Only use valid trials
)
```

### Custom Preprocessing Functions

Apply preprocessing to EMG data before windowing:

```python
import numpy as np
from scipy import signal
from sklearn.preprocessing import StandardScaler

def bandpass_filter(data, low_freq=20, high_freq=500, fs=2000):
    """Apply bandpass filter to EMG data."""
    nyquist = fs / 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    return signal.filtfilt(b, a, data, axis=0)

def rectify_and_smooth(data, window_size=50):
    """Rectify and apply moving average smoothing."""
    rectified = np.abs(data)
    kernel = np.ones(window_size) / window_size
    return np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), axis=0, arr=rectified)

def normalize_channels(data):
    """Z-score normalize each channel."""
    scaler = StandardScaler()
    original_shape = data.shape
    data_2d = data.reshape(-1, data.shape[-1])
    normalized = scaler.fit_transform(data_2d)
    return normalized.reshape(original_shape)

# Create dataset with preprocessing
def my_preprocessing(data):
    filtered = bandpass_filter(data)
    processed = rectify_and_smooth(filtered)
    return processed

dataset = data_manager.create_dataset(
    name="preprocessed_dataset",
    preprocessing_fn=my_preprocessing,
    window_size_ms=200,
    window_stride_ms=50
)
```

### Feature Extraction

Extract features like RMS during dataset creation:

```python
def extract_rms_features(data, window_size=50):
    """Extract RMS features from EMG data."""
    def rms_window(arr):
        return np.sqrt(np.mean(arr**2, axis=0))
    
    rms_features = []
    for i in range(0, len(data) - window_size + 1, window_size // 2):
        window = data[i:i + window_size]
        rms = rms_window(window)
        rms_features.append(rms)
    
    return np.array(rms_features)

def rms_preprocessing(data):
    filtered = bandpass_filter(data)
    rectified = np.abs(filtered)
    return extract_rms_features(rectified, window_size=40)

dataset = data_manager.create_dataset(
    name="rms_features_dataset",
    preprocessing_fn=rms_preprocessing,
    window_size_ms=100,
    window_stride_ms=50
)
```

### Advanced Preprocessing Pipeline

Create multi-stage preprocessing pipelines:

```python
def advanced_preprocessing(data):
    # Stage 1: Filtering
    filtered = bandpass_filter(data, low_freq=30, high_freq=400, fs=2000)
    
    # Stage 2: Rectification
    rectified = np.abs(filtered)
    
    # Stage 3: Normalization
    normalized = normalize_channels(rectified)
    
    # Stage 4: Smoothing
    smoothed = rectify_and_smooth(normalized, window_size=30)
    
    return smoothed

dataset = data_manager.create_dataset(
    name="advanced_dataset",
    preprocessing_fn=advanced_preprocessing,
    window_size_ms=250,
    window_stride_ms=100
)
```

### Using the Advanced Dataset Creation Script

Run the included script for examples of different preprocessing techniques:

```bash
cd playagain_pipeline
python advanced_dataset_creation.py
```

This script demonstrates:
- Basic filtering and rectification
- RMS feature extraction
- Multi-stage preprocessing pipelines
- Custom session selection

### GUI Dataset Creation

The GUI's "Create Dataset from Sessions..." button now opens a dialog allowing you to:
- Set custom window sizes and strides
- Include/exclude invalid trials
- Select specific subjects to include
- Name your datasets

## Real-time Prediction

```python
from playagain_pipeline.models import ModelManager

# Load trained model
model_manager = ModelManager("gesture_pipeline_data/models")
model = model_manager.load_model("my_trained_model")

# Make predictions
predictions = model.predict(emg_window)  # Shape: (1, samples, channels)
probabilities = model.predict_proba(emg_window)
```

## Device Support

### Synthetic Device (for testing)

```python
device = device_manager.create_device(
    DeviceType.SYNTHETIC,
    num_channels=32,
    sampling_rate=2000
)
```

### Real Devices

The pipeline is designed to integrate with the `device_interfaces` package from the PlayAgain project:

- Muovi / Muovi Plus
- Quattrocento
- Other OT Bioelettronica devices

## Extending the Pipeline

### Adding New ML Models

```python
from playagain_pipeline.models import BaseClassifier


class MyCustomClassifier(BaseClassifier):
    def extract_features(self, X):
        # Implement feature extraction
        pass

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        # Implement training
        pass

    def predict(self, X):
        # Implement prediction
        pass

    def predict_proba(self, X):
        # Implement probability prediction
        pass
```

### Adding New Devices

```python
from playagain_pipeline.devices import BaseEMGDevice


class MyDevice(BaseEMGDevice):
    def connect(self):
        # Implement connection
        pass

    def disconnect(self):
        # Implement disconnection
        pass

    def start_streaming(self):
        # Implement streaming start
        pass

    def stop_streaming(self):
        # Implement streaming stop
        pass
```

## Configuration

```python
from playagain_pipeline.config import PipelineConfig

# Load/save configuration
config = PipelineConfig()
config.device.num_channels = 64
config.recording.window_size_ms = 250
config.save("my_config.json")

# Load configuration
config = PipelineConfig.load("my_config.json")
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

