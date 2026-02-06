"""
EMG visualization widget for real-time signal display.

Uses VispyBiosignalPlot for efficient real-time plotting of multi-channel EMG data.
"""

import numpy as np

from typing import Optional
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox,
                                QCheckBox, QMainWindow, QScrollArea)
from PySide6.QtCore import Slot, Signal, QMutex
from gui_custom_elements.vispy.biosignal_plot import VispyBiosignalPlot


class EMGPlotWidget(QWidget):
    """
    Widget for displaying real-time EMG signals.

    Features:
    - Multi-channel display with configurable layout
    - Auto-scaling and manual range control
    - Channel selection and highlighting
    - Uses VispyBiosignalPlot for efficient visualization
    """

    def __init__(
        self,
        num_channels: int = 32,
        sampling_rate: int = 2000,
        display_seconds: float = 10.0,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)

        self.num_channels = num_channels
        self.sampling_rate = sampling_rate
        self.display_seconds = display_seconds
        self.display_samples = int(display_seconds * sampling_rate)

        # Data buffer - circular buffer
        self._data_buffer = np.zeros((self.display_samples, num_channels))
        self._buffer_index = 0
        self._buffer_full = False


        # Channel visibility
        self._channel_visible = [True] * num_channels

        # Frame counter for throttling updates
        self._frame_count = 0
        self._update_every_n_frames = 2  # Update plot every N data frames for performance

        # Setup UI
        self._setup_ui()

    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Control bar
        control_layout = QHBoxLayout()

        control_layout.addWidget(QLabel("Display (s):"))
        self.display_spin = QSpinBox()
        self.display_spin.setRange(1, 30)
        self.display_spin.setValue(int(self.display_seconds))
        self.display_spin.valueChanged.connect(self._on_display_time_changed)
        control_layout.addWidget(self.display_spin)

        self.autoscale_check = QCheckBox("Auto-scale")
        self.autoscale_check.setChecked(True)
        control_layout.addWidget(self.autoscale_check)

        control_layout.addStretch()
        layout.addLayout(control_layout)

        # Plot widget using VispyBiosignalPlot
        self.plot_widget = VispyBiosignalPlot()
        layout.addWidget(self.plot_widget)

    @Slot(int)
    def _on_display_time_changed(self, value: int):
        """Handle display time change."""
        self.display_seconds = value
        new_samples = int(value * self.sampling_rate)

        # Resize buffer
        if new_samples != self.display_samples:
            old_data = self._data_buffer
            self._data_buffer = np.zeros((new_samples, self.num_channels))

            # Copy existing data
            copy_samples = min(new_samples, len(old_data))
            self._data_buffer[-copy_samples:] = old_data[-copy_samples:]

            self.display_samples = new_samples

    def update_data(self, data: np.ndarray):
        """
        Update the plot with new data.

        Args:
            data: New EMG data (samples, channels)
        """
        if data is None or data.size == 0:
            return

        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        # Handle channel mismatch
        if data.shape[1] != self.num_channels:
            min_ch = min(data.shape[1], self.num_channels)
            if min_ch == 0:
                return
            temp_data = np.zeros((data.shape[0], self.num_channels))
            temp_data[:, :min_ch] = data[:, :min_ch]
            data = temp_data

        n_samples = data.shape[0]

        # Add new data to circular buffer
        if n_samples > self.display_samples:
            # If more data than buffer, take the latest
            self._data_buffer[:, :] = data[-self.display_samples:, :]
            self._buffer_index = 0
            self._buffer_full = True
        else:
            end_index = (self._buffer_index + n_samples) % self.display_samples
            if end_index < self._buffer_index:
                # Wrap around
                remaining = self.display_samples - self._buffer_index
                self._data_buffer[self._buffer_index:] = data[:remaining]
                self._data_buffer[:end_index] = data[remaining:]
                self._buffer_full = True
            else:
                self._data_buffer[self._buffer_index:end_index] = data
                if end_index == 0:
                    self._buffer_full = True
            self._buffer_index = end_index

        # Throttle plot updates for performance
        self._frame_count += 1
        if self._frame_count >= self._update_every_n_frames:
            self._frame_count = 0
            self._update_plots()

    def _update_plots(self):
        """Update the plot with current buffer data."""
        # Get ordered data
        ordered_data = self._get_ordered_data()

        # Update plot with the data
        if hasattr(self.plot_widget, 'update_data'):
            self.plot_widget.update_data(ordered_data)

    def set_channel_visible(self, channel: int, visible: bool):
        """Set visibility of a specific channel."""
        if 0 <= channel < self.num_channels:
            self._channel_visible[channel] = visible

    def set_num_channels(self, num_channels: int):
        """Update the number of channels."""
        if num_channels == self.num_channels:
            return

        self.num_channels = num_channels
        self._data_buffer = np.zeros((self.display_samples, num_channels))
        self._channel_visible = [True] * num_channels

        # Reconfigure plot widget
        self.plot_widget.configure(
            lines=self.num_channels,
            sampling_freuqency=self.sampling_rate,
            display_time=int(self.display_seconds),
        )

    def clear(self):
        """Clear all data."""
        self._data_buffer = np.zeros((self.display_samples, self.num_channels))
        self._buffer_index = 0
        self._buffer_full = False
        self._update_plots()

    def _get_ordered_data(self):
        """Get the data in chronological order (oldest to newest)."""
        if not self._buffer_full:
            return self._data_buffer[:self._buffer_index]
        else:
            return np.concatenate((self._data_buffer[self._buffer_index:], self._data_buffer[:self._buffer_index]), axis=0)



class EMGPlotWindow(QMainWindow):
    """
    Separate window for EMG plot with channel controls.

    Features channel checkboxes directly beside the plot for easy toggling.
    Uses VispyBiosignalPlot for efficient visualization.
    """

    closed = Signal()  # Emitted when window is closed

    def __init__(
        self,
        num_channels: int = 32,
        sampling_rate: int = 2000,
        display_seconds: float = 10.0,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)

        self.setWindowTitle("EMG Signal Visualization")
        self.setMinimumSize(1200, 800)

        self.num_channels = num_channels
        self.sampling_rate = sampling_rate
        self.display_seconds = display_seconds
        self.display_samples = int(display_seconds * sampling_rate)

        # Data buffer - circular buffer
        self._data_buffer = np.zeros((self.display_samples, num_channels))
        self._buffer_index = 0
        self._buffer_full = False

        # Channel visibility
        self._channel_visible = [True] * num_channels

        # Frame counter for throttling updates
        self._frame_count = 0
        self._update_every_n_frames = 3

        # Thread safety for data buffer
        self._data_mutex = QMutex()


        # Setup UI
        self._setup_ui()

    def _setup_ui(self):
        """Setup the user interface."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Left side - Plot
        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)

        # Control bar
        control_layout = QHBoxLayout()

        control_layout.addWidget(QLabel("Display (s):"))
        self.display_spin = QSpinBox()
        self.display_spin.setRange(1, 30)
        self.display_spin.setValue(int(self.display_seconds))
        self.display_spin.valueChanged.connect(self._on_display_time_changed)
        control_layout.addWidget(self.display_spin)

        control_layout.addStretch()
        plot_layout.addLayout(control_layout)

        # Plot widget using VispyBiosignalPlot
        self.plot_widget = VispyBiosignalPlot()
        self.plot_widget.configure(
            lines=self.num_channels,
            sampling_freuqency=self.sampling_rate,
            display_time=int(self.display_seconds),
        )
        plot_layout.addWidget(self.plot_widget)

        # Ground truth label display (for session replay)
        self._ground_truth_label = QLabel("Ground Truth: -")
        self._ground_truth_label.setStyleSheet(
            "font-size: 16px; font-weight: bold; padding: 5px; "
            "background-color: #e0e0e0; border-radius: 5px;"
        )
        plot_layout.addWidget(self._ground_truth_label)

        main_layout.addWidget(plot_container, stretch=3)

        # Right side - Channel controls
        channel_container = QWidget()
        channel_layout = QVBoxLayout(channel_container)
        channel_layout.setContentsMargins(5, 5, 5, 5)

        channel_label = QLabel("Channels")
        channel_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        channel_layout.addWidget(channel_label)


        # Scrollable area for channel checkboxes
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumWidth(150)

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(2)

        self.channel_checks = []
        for i in range(self.num_channels):
            check = QCheckBox(f"Ch {i}")
            check.setChecked(True)
            check.toggled.connect(lambda checked, ch=i: self._on_channel_toggled(ch, checked))
            scroll_layout.addWidget(check)
            self.channel_checks.append(check)

        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_widget)
        channel_layout.addWidget(scroll_area)

        main_layout.addWidget(channel_container, stretch=0)

    @Slot(str)
    def set_ground_truth(self, label: str):
        """Update the ground truth label display."""
        self._ground_truth_label.setText(f"Ground Truth: {label}")
        # Change background color based on gesture
        if label == "Unknown" or label == "-":
            self._ground_truth_label.setStyleSheet(
                "font-size: 16px; font-weight: bold; padding: 5px; "
                "background-color: #e0e0e0; border-radius: 5px;"
            )
        elif "rest" in label.lower():
            self._ground_truth_label.setStyleSheet(
                "font-size: 16px; font-weight: bold; padding: 5px; "
                "background-color: #a8e6cf; border-radius: 5px;"
            )
        else:
            self._ground_truth_label.setStyleSheet(
                "font-size: 16px; font-weight: bold; padding: 5px; "
                "background-color: #ffd3a5; border-radius: 5px;"
            )

    @Slot(int, bool)
    def _on_channel_toggled(self, channel: int, checked: bool):
        """Handle channel checkbox toggle."""
        self._channel_visible[channel] = checked
        self._update_plots()

    @Slot(int)
    def _on_display_time_changed(self, value: int):
        """Handle display time change."""
        self.display_seconds = value
        new_samples = int(value * self.sampling_rate)

        if new_samples != self.display_samples:
            old_data = self._data_buffer
            self._data_buffer = np.zeros((new_samples, self.num_channels))

            copy_samples = min(new_samples, len(old_data))
            self._data_buffer[-copy_samples:] = old_data[-copy_samples:]

            self.display_samples = new_samples

    def update_data(self, data: np.ndarray):
        """Update the plot with new data."""
        if data is None or data.size == 0:
            return

        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        # Handle channel mismatch
        if data.shape[1] != self.num_channels:
            min_ch = min(data.shape[1], self.num_channels)
            if min_ch == 0:
                return
            temp_data = np.zeros((data.shape[0], self.num_channels))
            temp_data[:, :min_ch] = data[:, :min_ch]
            data = temp_data

        n_samples = data.shape[0]

        # Add new data to circular buffer
        if n_samples > self.display_samples:
            # If more data than buffer, take the latest
            self._data_buffer[:, :] = data[-self.display_samples:, :]
            self._buffer_index = 0
            self._buffer_full = True
        else:
            end_index = (self._buffer_index + n_samples) % self.display_samples
            if end_index < self._buffer_index:
                # Wrap around
                remaining = self.display_samples - self._buffer_index
                self._data_buffer[self._buffer_index:] = data[:remaining]
                self._data_buffer[:end_index] = data[remaining:]
                self._buffer_full = True
            else:
                self._data_buffer[self._buffer_index:end_index] = data
                if end_index == 0:
                    self._buffer_full = True
            self._buffer_index = end_index

        # Throttle plot updates
        self._frame_count += 1
        if self._frame_count >= self._update_every_n_frames:
            self._frame_count = 0
            self._update_plots()

    def _update_plots(self):
        """Update all plot curves."""
        # Get ordered data
        ordered_data = self._get_ordered_data()

        # Filter visible channels if needed
        if not all(self._channel_visible):
            filtered_data = np.zeros_like(ordered_data)
            for i in range(self.num_channels):
                if self._channel_visible[i]:
                    filtered_data[:, i] = ordered_data[:, i]
            ordered_data = filtered_data

        # Update plot with the data
        if hasattr(self.plot_widget, 'update_data'):
            self.plot_widget.update_data(ordered_data)

    def set_num_channels(self, num_channels: int):
        """Update the number of channels."""
        if num_channels == self.num_channels:
            return

        self.num_channels = num_channels
        self._data_buffer = np.zeros((self.display_samples, num_channels))
        self._channel_visible = [True] * num_channels

        # Reconfigure plot widget
        self.plot_widget.configure(
            lines=self.num_channels,
            sampling_freuqency=self.sampling_rate,
            display_time=int(self.display_seconds),
        )

    def clear(self):
        """Clear all data."""
        self._data_buffer = np.zeros((self.display_samples, self.num_channels))
        self._buffer_index = 0
        self._buffer_full = False
        self._update_plots()

    def _get_ordered_data(self):
        """Get the data in chronological order (oldest to newest)."""
        if not self._buffer_full:
            return self._data_buffer[:self._buffer_index]
        else:
            return np.concatenate((self._data_buffer[self._buffer_index:], self._data_buffer[:self._buffer_index]), axis=0)

    def closeEvent(self, event):
        """Handle window close event."""
        self.closed.emit()
        event.accept()

