"""
Device interfaces for EMG data acquisition.

This module provides a unified interface for different EMG devices,
including real hardware and synthetic data generators.
"""

from abc import abstractmethod
from typing import Optional, Dict, Any, Callable
from enum import Enum, auto
from dataclasses import dataclass
import numpy as np
from PySide6.QtCore import QObject, Signal, QTimer, QMutex, QMutexLocker

# Detect the current OS and inject local package paths into sys.path so that
# device_interfaces can be imported without hard-coded absolute paths in
# requirements.txt (works on macOS, Windows, and Linux alike).
from playagain_pipeline.utils.platform_utils import inject_local_packages, get_os_name
_injected = inject_local_packages()

# Try to import device_interfaces for real device support
try:
    from device_interfaces.devices.muovi import Muovi
    from device_interfaces.enums.muovi import (
        MuoviWorkingMode, MuoviDetectionMode, MuoviStream,
        MuoviAvailableChannels, MuoviPlusAvailableChannels
    )
    from device_interfaces.enums.device import Device
    DEVICE_INTERFACES_AVAILABLE = True
except ImportError:
    DEVICE_INTERFACES_AVAILABLE = False
    # Silently handle - user will see error when trying to use real devices


class DeviceType(Enum):
    """Supported device types."""
    SYNTHETIC = auto()
    MUOVI = auto()
    MUOVI_PLUS = auto()
    QUATTROCENTO = auto()
    CUSTOM = auto()


@dataclass
class DeviceConfig:
    """Configuration for an EMG device."""
    device_type: DeviceType
    num_channels: int
    sampling_rate: int
    samples_per_frame: int = 100

    # Connection settings
    ip_address: str = "0.0.0.0"
    port: int = 54321

    # Additional settings
    extra_settings: Dict[str, Any] = None

    def __post_init__(self):
        if self.extra_settings is None:
            self.extra_settings = {}


class BaseEMGDevice(QObject):
    """
    Abstract base class for EMG devices.

    Provides a unified interface for different EMG hardware.
    """

    # Signals
    data_ready = Signal(np.ndarray)  # Emitted when new data is available
    connected = Signal(bool)          # Emitted when connection status changes
    error = Signal(str)               # Emitted on errors
    ground_truth_changed = Signal(str)  # Emitted when ground truth gesture changes (for session replay)

    def __init__(self, config: DeviceConfig, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.config = config
        self._is_connected = False
        self._is_streaming = False

        # Keep physical/raw channel count separate from processed output count.
        self.config.extra_settings.setdefault("physical_num_channels", config.num_channels)
        self.config.extra_settings.setdefault("bipolar_mode", False)
        self.config.extra_settings.setdefault("data_already_bipolar", False)

    def _is_bipolar_active(self) -> bool:
        """Return True only when bipolar transform should be applied to incoming data."""
        return (
            bool(self.config.extra_settings.get("bipolar_mode", False))
            and not bool(self.config.extra_settings.get("data_already_bipolar", False))
        )

    def _get_output_channels(self, input_channels: int) -> int:
        """Compute output channel count after optional bipolar transform."""
        if not self._is_bipolar_active() or input_channels < 2:
            return input_channels
        if input_channels % 32 == 0:
            return (input_channels // 32) * 16
        return input_channels // 2 if input_channels % 2 == 0 else input_channels

    def _normalize_samples_by_channels(self, data: np.ndarray) -> np.ndarray:
        """Normalize incoming data to shape (samples, channels)."""
        if data.ndim != 2:
            return data

        expected_ch = self.physical_num_channels
        if data.shape[1] == expected_ch:
            return data
        if data.shape[0] == expected_ch:
            return data.T

        # Fallback heuristic for unknown packets: channels are usually the smaller axis.
        if data.shape[0] < data.shape[1]:
            return data.T
        return data

    def _apply_bipolar(self, data: np.ndarray) -> np.ndarray:
        """Applies bipolar mode (top - bottom) if configured."""
        if not self._is_bipolar_active():
            return data

        n = data.shape[1]
        if n < 2:
            return data

        bipolar_data = []

        # Preferred layout for Muovi: groups of 32 channels with top(17-32)-bottom(1-16).
        if n % 32 == 0:
            for i in range(0, n, 32):
                top = data[:, i + 16:i + 32]
                bottom = data[:, i:i + 16]
                bipolar_data.append(top - bottom)

        # Generic fallback for even channel counts.
        elif n % 2 == 0:
            half = n // 2
            bipolar_data.append(data[:, half:] - data[:, :half])

        if bipolar_data:
            return np.hstack(bipolar_data).astype(data.dtype, copy=False)
        return data

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_streaming(self) -> bool:
        return self._is_streaming

    @property
    def num_channels(self) -> int:
        return self._get_output_channels(self.physical_num_channels)

    @property
    def physical_num_channels(self) -> int:
        return int(self.config.extra_settings.get("physical_num_channels", self.config.num_channels))

    @property
    def sampling_rate(self) -> int:
        return self.config.sampling_rate

    @abstractmethod
    def connect_device(self) -> bool:
        """Connect to the device. Returns True on success."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the device."""
        pass

    @abstractmethod
    def start_streaming(self) -> bool:
        """Start data streaming. Returns True on success."""
        pass

    @abstractmethod
    def stop_streaming(self) -> None:
        """Stop data streaming."""
        pass

    def get_device_info(self) -> Dict[str, Any]:
        """Get device information."""
        return {
            "device_type": self.config.device_type.name,
            "num_channels": self.num_channels,
            "physical_num_channels": self.physical_num_channels,
            "bipolar_mode": bool(self.config.extra_settings.get("bipolar_mode", False)),
            "sampling_rate": self.config.sampling_rate,
            "is_connected": self._is_connected,
            "is_streaming": self._is_streaming
        }


# Only define MuoviDevice if device_interfaces is available
if DEVICE_INTERFACES_AVAILABLE:
    class MuoviDevice(BaseEMGDevice):
        """
        Wrapper for the Muovi/Muovi Plus EMG bracelet.

        Uses the device_interfaces package for actual device communication.
        """

        def __init__(
            self,
            is_muovi_plus: bool = False,
            ip_address: str = "0.0.0.0",
            port: int = 54321,
            parent: Optional[QObject] = None
        ):
            # Determine channels based on device type
            num_channels = 64 if is_muovi_plus else 32
            sampling_rate = 2000  # EMG mode default

            config = DeviceConfig(
                device_type=DeviceType.MUOVI_PLUS if is_muovi_plus else DeviceType.MUOVI,
                num_channels=num_channels,
                sampling_rate=sampling_rate,
                ip_address=ip_address,
                port=port
            )
            super().__init__(config, parent)

            self._is_muovi_plus = is_muovi_plus
            self._muovi = Muovi(is_muovi_plus=is_muovi_plus, parent=parent)

            # Connect signals
            self._muovi.connected_signal.connect(self._on_connected)
            self._muovi.configured_signal.connect(self._on_configured)
            self._muovi.streaming_signal.connect(self._on_streaming)
            self._muovi.data_available_signal.connect(self._on_data_available)

        def _on_connected(self, connected: bool):
            """Handle physical connection status change."""
            self._is_connected = connected

            if connected:
                # Don't tell GUI we are ready yet. Configure first.
                self._configure_for_emg()
            else:
                # Disconnection is immediate
                self.connected.emit(False)

        def _on_configured(self, configured: bool):
            """Handle configuration completion."""
            if configured:
                # Update our config with actual device parameters
                info = self._muovi.get_device_information()
                if info:
                    self.config.sampling_rate = info.get("sampling_frequency", 2000)
                    physical_channels = info.get("biosignal_channels", self.config.num_channels)
                    self.config.num_channels = physical_channels
                    self.config.extra_settings["physical_num_channels"] = physical_channels

                # NOW we emit connected(True) because we are actually ready
                self.connected.emit(True)

        def _on_streaming(self, streaming: bool):
            self._is_streaming = streaming

        def _on_data_available(self, data: np.ndarray):
            """Handle incoming data."""
            # Extract EMG data (returns None if packet contains no EMG)
            emg_data = self._muovi.extract_emg_data(data, milli_volts=True)

            if emg_data is not None and emg_data.size > 0:
                emg_data = self._normalize_samples_by_channels(emg_data)

                bipolar_data = self._apply_bipolar(emg_data)
                self.data_ready.emit(bipolar_data)

        def _configure_for_emg(self):
            """Configure the Muovi for EMG recording."""
            params = {
                "working_mode": MuoviWorkingMode.EMG,
                "detection_mode": MuoviDetectionMode.MONOPOLAR_GAIN_8,
                "streaming_mode": MuoviStream.STOP,
            }
            self._muovi.configure_device(params)

        def connect_device(self) -> bool:
            """Connect to the Muovi device."""
            try:
                if not self._muovi.is_connected:
                    self._muovi.toggle_connection(
                        (self.config.ip_address, self.config.port)
                    )
                return True
            except Exception as e:
                self.error.emit(f"Connection error: {e}")
                return False

        def disconnect(self) -> None:
            """Disconnect from the Muovi device."""
            self.stop_streaming()
            if self._muovi.is_connected:
                self._muovi.toggle_connection()
            self._is_connected = False
            self.connected.emit(False)

        def start_streaming(self) -> bool:
            """Start EMG data streaming."""
            if not self._is_connected:
                self.error.emit("Device not connected")
                return False

            try:
                if not self._muovi.is_streaming:
                    self._muovi.toggle_streaming()
                return True
            except Exception as e:
                self.error.emit(f"Streaming error: {e}")
                return False

        def stop_streaming(self) -> None:
            """Stop EMG data streaming."""
            if self._is_streaming:
                if self._muovi.is_streaming:
                    self._muovi.toggle_streaming()
                self._is_streaming = False

        def get_device_info(self) -> Dict[str, Any]:
            """Get device information."""
            info = super().get_device_info()

            # Add Muovi-specific info
            muovi_info = self._muovi.get_device_information()
            if muovi_info:
                info.update({
                    "biosignal_channels": muovi_info.get("biosignal_channels"),
                    "aux_channels": muovi_info.get("aux_channels"),
                    "samples_per_frame": muovi_info.get("samples_per_frame"),
                })

            return info
else:
    # In playagain_pipeline/devices/emg_device.py

    class MuoviDevice(BaseEMGDevice):
        """
        Wrapper for the Muovi/Muovi Plus EMG bracelet.
        """

        def __init__(
                self,
                is_muovi_plus: bool = False,
                ip_address: str = "0.0.0.0",
                port: int = 54321,
                parent: Optional[QObject] = None
        ):
            # Determine channels based on device type
            num_channels = 64 if is_muovi_plus else 32
            sampling_rate = 2000  # EMG mode default

            config = DeviceConfig(
                device_type=DeviceType.MUOVI_PLUS if is_muovi_plus else DeviceType.MUOVI,
                num_channels=num_channels,
                sampling_rate=sampling_rate,
                ip_address=ip_address,
                port=port
            )
            super().__init__(config, parent)

            self._is_muovi_plus = is_muovi_plus
            self._muovi = Muovi(is_muovi_plus=is_muovi_plus, parent=parent)

            # Connect signals
            self._muovi.connected_signal.connect(self._on_connected)
            self._muovi.configured_signal.connect(self._on_configured)
            self._muovi.streaming_signal.connect(self._on_streaming)
            self._muovi.data_available_signal.connect(self._on_data_available)

        def _on_connected(self, connected: bool):
            """Handle physical connection status change."""
            self._is_connected = connected

            if connected:
                # FIX 1: Don't tell GUI we are ready yet. Configure first.
                print(f"Physical device connected. Configuring for EMG...")
                self._configure_for_emg()
            else:
                # Disconnection is immediate
                self.connected.emit(False)

        def _on_configured(self, configured: bool):
            """Handle configuration completion."""
            if configured:
                # Update our config with actual device parameters
                info = self._muovi.get_device_information()
                if info:
                    self.config.sampling_rate = info.get("sampling_frequency", 2000)
                    physical_channels = info.get("biosignal_channels", self.config.num_channels)
                    self.config.num_channels = physical_channels
                    self.config.extra_settings["physical_num_channels"] = physical_channels

                # FIX 2: NOW we emit connected(True) because we are actually ready
                print("Device configured successfully. Ready to stream.")
                self.connected.emit(True)

        def _on_streaming(self, streaming: bool):
            self._is_streaming = streaming

        def _on_data_available(self, data: np.ndarray):
            """Handle incoming data."""
            # Extract EMG data (returns None if packet contains no EMG)
            emg_data = self._muovi.extract_emg_data(data, milli_volts=True)

            if emg_data is not None and emg_data.size > 0:
                emg_data = self._normalize_samples_by_channels(emg_data)

                bipolar_data = self._apply_bipolar(emg_data)
                self.data_ready.emit(bipolar_data)

        def _configure_for_emg(self):
            params = {
                "working_mode": MuoviWorkingMode.EMG,
                "detection_mode": MuoviDetectionMode.MONOPOLAR_GAIN_8,
                "streaming_mode": MuoviStream.STOP,
            }
            self._muovi.configure_device(params)

        def connect_device(self) -> bool:
            try:
                if not self._muovi.is_connected:
                    self._muovi.toggle_connection(
                        (self.config.ip_address, self.config.port)
                    )
                return True
            except Exception as e:
                self.error.emit(f"Connection error: {e}")
                return False

        def disconnect(self) -> None:
            self.stop_streaming()
            self._muovi._disconnect_from_device()
            self._is_connected = False
            self.connected.emit(False)

        def start_streaming(self) -> bool:
            if not self._is_connected:
                self.error.emit("Device not connected")
                return False
            try:
                self._muovi._start_streaming()
                return True
            except Exception as e:
                self.error.emit(f"Streaming error: {e}")
                return False

        def stop_streaming(self) -> None:
            if self._is_streaming:
                try:
                    self._muovi._stop_streaming()
                except:
                    pass
            self._is_streaming = False


class SyntheticEMGDevice(BaseEMGDevice):
    """
    Synthetic EMG device for testing and development.

    Generates realistic synthetic EMG signals with configurable patterns,
    or replays existing session data in a loop.

    Thread-safe implementation with ground truth signaling for session replay.
    """

    def __init__(
        self,
        num_channels: int = 32,
        sampling_rate: int = 2000,
        samples_per_frame: int = 100,
        use_session_data: bool = False,
        session_subject_id: Optional[str] = None,
        session_id: Optional[str] = None,
        data_dir: Optional[str] = None,
        parent: Optional[QObject] = None
    ):
        config = DeviceConfig(
            device_type=DeviceType.SYNTHETIC,
            num_channels=num_channels,
            sampling_rate=sampling_rate,
            samples_per_frame=samples_per_frame
        )
        super().__init__(config, parent)

        # Thread safety
        self._data_mutex = QMutex()

        # Timer for data generation
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._generate_data)

        # Session replay mode
        self._use_session_data = use_session_data
        self._session_data: Optional[np.ndarray] = None
        self._session_index = 0
        self._session_subject_id = session_subject_id
        self._session_id = session_id
        self._data_dir = data_dir or "gesture_pipeline_data"

        # Session trial info for ground truth
        self._session_trials = []  # List of (start_sample, end_sample, label_name)
        self._current_ground_truth = "Unknown"

        if use_session_data:
            self._load_session_data()
        else:
            # Original synthetic generation parameters
            self._current_time = 0.0
            self._noise_level = 0.1
            self._signal_amplitude = 100.0

            # Channel-specific parameters for realistic EMG
            np.random.seed(42)  # For reproducibility
            self._channel_frequencies = [20 + i * 3 for i in range(num_channels)]
            self._channel_phases = np.random.uniform(0, 2*np.pi, num_channels)
            self._channel_amplitudes = 0.5 + np.random.uniform(-0.2, 0.2, num_channels)

            # Gesture simulation
            self._active_gesture: Optional[str] = None
            self._gesture_patterns: Dict[str, np.ndarray] = self._create_gesture_patterns()

    def _create_gesture_patterns(self) -> Dict[str, np.ndarray]:
        """Create activation patterns for different gestures."""
        patterns = {}
        n = self.config.num_channels

        # Rest - low uniform activity
        patterns["rest"] = np.ones(n) * 0.1

        # Fist - strong activation on flexors
        fist = np.zeros(n)
        fist[0:n//4] = 0.8  # Simulate flexor activation
        fist[n//4:n//2] = 0.5
        patterns["fist"] = fist + np.random.uniform(0, 0.1, n)

        # Index-thumb pinch
        pinch = np.zeros(n)
        pinch[0:n//8] = 0.6
        pinch[n//2:n//2 + n//8] = 0.4
        patterns["index_thumb"] = pinch + np.random.uniform(0, 0.1, n)

        # Three finger pinch
        three_finger = np.zeros(n)
        three_finger[0:n//4] = 0.7
        three_finger[n//4:n//2] = 0.5
        three_finger[n//2:3*n//4] = 0.3
        patterns["three_finger_thumb"] = three_finger + np.random.uniform(0, 0.1, n)

        # Open hand
        open_hand = np.zeros(n)
        open_hand[n//2:] = 0.6  # Extensors
        patterns["open_hand"] = open_hand + np.random.uniform(0, 0.1, n)

        return patterns

    def set_gesture(self, gesture_name: Optional[str]) -> None:
        """Set the current gesture for simulation."""
        self._active_gesture = gesture_name

    def set_noise_level(self, level: float) -> None:
        """Set noise level (0.0 to 1.0)."""
        self._noise_level = max(0.0, min(1.0, level))

    def set_amplitude(self, amplitude: float) -> None:
        """Set signal amplitude."""
        self._signal_amplitude = amplitude

    def connect_device(self) -> bool:
        """Simulate connection."""
        self._is_connected = True
        self.connected.emit(True)
        return True

    def disconnect(self) -> None:
        """Simulate disconnection."""
        self.stop_streaming()
        self._is_connected = False
        self.connected.emit(False)

    def start_streaming(self) -> bool:
        """Start generating synthetic data."""
        if not self._is_connected:
            self.error.emit("Device not connected")
            return False

        # Calculate timer interval
        frame_duration_ms = int(
            1000 * self.config.samples_per_frame / self.config.sampling_rate
        )
        frame_duration_ms = max(1, frame_duration_ms)

        self._timer.start(frame_duration_ms)
        self._is_streaming = True
        self._current_time = 0.0
        return True

    def stop_streaming(self) -> None:
        """Stop generating data."""
        self._timer.stop()
        self._is_streaming = False

    def _get_ground_truth_for_sample(self, sample_index: int) -> str:
        """Get the ground truth gesture label for a given sample index."""
        for start, end, label in self._session_trials:
            if start <= sample_index < end:
                return label
        return "Unknown"

    def _generate_data(self) -> None:
        """Generate and emit synthetic EMG data or replay session data."""
        samples = self.config.samples_per_frame

        with QMutexLocker(self._data_mutex):
            if self._use_session_data and self._session_data is not None:
                # Replay session data in a loop
                if self._session_index + samples > len(self._session_data):
                    # Loop back to beginning
                    self._session_index = 0

                data = self._session_data[self._session_index:self._session_index + samples]
                current_idx = self._session_index
                self._session_index += samples

                # If we don't have enough samples at the end, wrap around
                if data.shape[0] < samples:
                    remaining = samples - data.shape[0]
                    data = np.vstack([data, self._session_data[:remaining]])
                    self._session_index = remaining

                # Emit ground truth if it changed
                new_ground_truth = self._get_ground_truth_for_sample(current_idx)
                if new_ground_truth != self._current_ground_truth:
                    self._current_ground_truth = new_ground_truth
                    self.ground_truth_changed.emit(new_ground_truth)
            else:
                # Original synthetic generation
                channels = self.config.num_channels

                # Time vector for this frame
                dt = 1.0 / self.config.sampling_rate
                t = np.linspace(
                    self._current_time,
                    self._current_time + samples * dt,
                    samples,
                    endpoint=False
                )
                self._current_time += samples * dt

                # Generate base EMG signal
                data = np.zeros((samples, channels))

                for ch in range(channels):
                    # EMG-like signal (multiple frequency components)
                    freq = self._channel_frequencies[ch]
                    phase = self._channel_phases[ch]
                    amp = self._channel_amplitudes[ch]

                    # Carrier signal
                    carrier = amp * np.sin(2 * np.pi * freq * t + phase)

                    # Add harmonics
                    carrier += 0.3 * amp * np.sin(2 * np.pi * 2 * freq * t + phase)
                    carrier += 0.1 * amp * np.sin(2 * np.pi * 3 * freq * t + phase)

                    # Add noise
                    noise = np.random.randn(samples) * self._noise_level

                    data[:, ch] = carrier + noise

                # Apply gesture modulation
                if self._active_gesture and self._active_gesture in self._gesture_patterns:
                    pattern = self._gesture_patterns[self._active_gesture]
                    data = data * pattern * self._signal_amplitude
                else:
                    # Rest pattern
                    data = data * self._gesture_patterns["rest"] * self._signal_amplitude

                # Add some random burst noise (EMG artifact simulation)
                if np.random.random() < 0.01:  # 1% chance per frame
                    burst_ch = np.random.randint(0, channels)
                    burst_amp = np.random.uniform(2, 5)
                    data[:, burst_ch] *= burst_amp

        bipolar_data = self._apply_bipolar(data)
        self.data_ready.emit(bipolar_data)

    def _load_session_data(self) -> None:
        """Load session data for replay, including trial info for ground truth."""
        if not self._session_subject_id or not self._session_id:
            # Auto-select first available session if not specified
            from pathlib import Path
            data_dir = Path(self._data_dir)
            sessions_dir = data_dir / "sessions"

            if sessions_dir.exists():
                subjects = [d for d in sessions_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
                if subjects:
                    subject_dir = subjects[0]
                    sessions = [d for d in subject_dir.iterdir() if d.is_dir()]
                    if sessions:
                        self._session_subject_id = subject_dir.name
                        self._session_id = sessions[0].name

        if not self._session_subject_id or not self._session_id:
            raise ValueError("No session data found and no session specified")

        # Load session data
        from pathlib import Path
        import json
        session_path = Path(self._data_dir) / "sessions" / self._session_subject_id / self._session_id
        data_path = session_path / "data.npy"

        if not data_path.exists():
            raise FileNotFoundError(f"Session data not found: {data_path}")

        self._session_data = np.load(data_path)
        self._session_index = 0

        # Load trial info for ground truth labels
        self._session_trials = []

        # Load gesture set to map labels to names
        gesture_names = {}
        gesture_set_path = session_path / "gesture_set.json"
        if gesture_set_path.exists():
            try:
                with open(gesture_set_path, 'r') as f:
                    gesture_data = json.load(f)
                    for g in gesture_data.get("gestures", []):
                        gesture_names[g.get("label_id")] = g.get("display_name", g.get("name", "Unknown"))
            except Exception as e:
                print(f"Warning: Could not load gesture set: {e}")

        # Load trials from metadata.json (trials are embedded in metadata)
        metadata_path = session_path / "metadata.json"
        session_meta = {}
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                session_meta = metadata.get("metadata", {})
                custom_meta = session_meta.get("custom_metadata", {}) if isinstance(session_meta, dict) else {}
                if isinstance(custom_meta, dict):
                    session_signal_mode = custom_meta.get("signal_mode")
                    session_bipolar = bool(
                        custom_meta.get("bipolar_mode", session_signal_mode == "bipolar")
                    )
                    self.config.extra_settings["data_already_bipolar"] = session_bipolar
                    if isinstance(session_signal_mode, str):
                        self.config.extra_settings["session_signal_mode"] = session_signal_mode

                # Trials are stored directly in the metadata file
                trials = metadata.get("trials", [])
                for trial in trials:
                    start = trial.get("start_sample", 0)
                    end = trial.get("end_sample", 0)
                    # Try gesture_name first, then fall back to label lookup
                    label_name = trial.get("gesture_name")
                    if not label_name:
                        label_id = trial.get("gesture_label", 0)
                        label_name = gesture_names.get(label_id, f"Class {label_id}")
                    self._session_trials.append((start, end, label_name))

                print(f"Loaded {len(self._session_trials)} trials for ground truth")
            except Exception as e:
                print(f"Warning: Could not load trial info for ground truth: {e}")

        # Validate session data matches device configuration
        if self._session_data.shape[1] != self.physical_num_channels:
            # Update physical channel count to match replay data shape.
            print(
                f"Note: Adjusting channel count from {self.physical_num_channels} "
                f"to {self._session_data.shape[1]} to match session data"
            )
            self.config.num_channels = self._session_data.shape[1]
            self.config.extra_settings["physical_num_channels"] = self._session_data.shape[1]


class DeviceManager:
    """
    Manages EMG device connections and provides a unified interface.
    """

    def __init__(self):
        self._device: Optional[BaseEMGDevice] = None
        self._data_callback: Optional[Callable[[np.ndarray], None]] = None

    def create_device(
        self,
        device_type: DeviceType,
        num_channels: int = 32,
        sampling_rate: int = 2000,
        **kwargs
    ) -> BaseEMGDevice:
        """
        Create and configure an EMG device.

        Args:
            device_type: Type of device to create
            num_channels: Number of channels
            sampling_rate: Sampling rate in Hz
            **kwargs: Additional device-specific arguments
                - ip_address: IP address for network devices (default: "0.0.0.0")
                - port: Port for network devices (default: 54321)
                - samples_per_frame: For synthetic device

        Returns:
            Configured device instance
        """
        if device_type == DeviceType.SYNTHETIC:
            self._device = SyntheticEMGDevice(
                num_channels=num_channels,
                sampling_rate=sampling_rate,
                samples_per_frame=kwargs.get("samples_per_frame", 100),
                use_session_data=kwargs.get("use_session_data", False),
                session_subject_id=kwargs.get("session_subject_id"),
                session_id=kwargs.get("session_id"),
                data_dir=kwargs.get("data_dir")
            )
        elif device_type == DeviceType.MUOVI:
            if not DEVICE_INTERFACES_AVAILABLE:
                raise ImportError(
                    "device_interfaces package not available. "
                    "Install it from PlayAgain-Game1/device-interfaces-main or use SYNTHETIC device."
                )
            self._device = MuoviDevice(
                is_muovi_plus=False,
                ip_address=kwargs.get("ip_address", "0.0.0.0"),
                port=kwargs.get("port", 54321)
            )
        elif device_type == DeviceType.MUOVI_PLUS:
            if not DEVICE_INTERFACES_AVAILABLE:
                raise ImportError(
                    "device_interfaces package not available. "
                    "Install it from PlayAgain-Game1/device-interfaces-main or use SYNTHETIC device."
                )
            self._device = MuoviDevice(
                is_muovi_plus=True,
                ip_address=kwargs.get("ip_address", "0.0.0.0"),
                port=kwargs.get("port", 54321)
            )
        else:
            raise NotImplementedError(
                f"Device type {device_type.name} not yet implemented. "
                "Available: SYNTHETIC, MUOVI, MUOVI_PLUS"
            )

        if "bipolar_mode" in kwargs:
            self._device.config.extra_settings["bipolar_mode"] = kwargs["bipolar_mode"]
        self._device.config.extra_settings.setdefault("physical_num_channels", num_channels)

        return self._device

    @property
    def device(self) -> Optional[BaseEMGDevice]:
        """Get the current device."""
        return self._device

    def set_data_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """Set callback for incoming data."""
        self._data_callback = callback
        if self._device:
            self._device.data_ready.connect(callback)

    def connect_and_start(self) -> bool:
        """Connect to device and start streaming."""
        if self._device is None:
            return False

        if not self._device.connect_device():
            return False

        return self._device.start_streaming()

    def stop_and_disconnect(self) -> None:
        """Stop streaming and disconnect."""
        if self._device:
            self._device.stop_streaming()
            self._device.disconnect()

    @staticmethod
    def is_device_interfaces_available() -> bool:
        """Check if device_interfaces package is available for real devices."""
        return DEVICE_INTERFACES_AVAILABLE
