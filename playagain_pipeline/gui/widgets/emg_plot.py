"""
EMG visualization widget for real-time signal display.

Uses pyqtgraph for efficient real-time plotting of multi-channel EMG data.
"""

from typing import Optional, List
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QCheckBox, QMainWindow, QScrollArea
from PySide6.QtCore import Slot, Signal
import pyqtgraph as pg


class EMGPlotWidget(QWidget):
    """
    Widget for displaying real-time EMG signals.

    Features:
    - Multi-channel display with configurable layout
    - Auto-scaling and manual range control
    - Channel selection and highlighting
    - RMS envelope overlay option
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

        # Data statistics for auto-scaling
        self._running_std = 1.0

        # Channel visibility
        self._channel_visible = [True] * num_channels

        # Frame counter for throttling updates
        self._frame_count = 0
        self._update_every_n_frames = 3  # Update plot every N data frames for performance

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

        # Plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        self.plot_widget.setLabel('left', 'Channels')

        # Disable auto range on mouse/wheel - let checkbox control it
        self.plot_widget.setMouseEnabled(x=True, y=True)
        self.plot_widget.enableAutoRange(enable=False)

        # Set y-axis ticks for channels
        self._update_y_axis_ticks()

        layout.addWidget(self.plot_widget)

        # Create plot curves for each channel
        self._curves: List[pg.PlotDataItem] = []
        self._hlines: List[pg.InfiniteLine] = []

        # Color map for channels
        colors = self._generate_colors(self.num_channels)

        for i in range(self.num_channels):
            curve = self.plot_widget.plot(
                pen=pg.mkPen(color=colors[i], width=1),
                name=f'Ch {i}'
            )
            self._curves.append(curve)

        # Time axis
        self._time_axis = np.linspace(
            -self.display_seconds, 0, self.display_samples
        )

        # Force Y range now so ticks/hlines/labels map correctly
        self.plot_widget.setYRange(-0.5, self.num_channels - 0.5)
        # Update ticks and create guide lines and labels
        self._update_y_axis_ticks()
        self._create_hlines()
        self._clear_ylabels()
        self._create_ylabels()

    def _generate_colors(self, n: int) -> List[tuple]:
        """Generate n distinct colors."""
        colors = []
        for i in range(n):
            hue = int(360 * i / n)
            colors.append(pg.hsvColor(hue / 360.0, 0.8, 0.8))
        return colors

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
            self._time_axis = np.linspace(-self.display_seconds, 0, new_samples)

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

        # Update statistics for scaling (exponential moving average)
        recent_data = self._data_buffer[-min(self.display_samples, 4000):]
        if recent_data.size > 0 and np.any(recent_data != 0):
            new_std = np.std(recent_data)
            if new_std > 1e-10:
                # Exponential moving average for smooth scaling
                self._running_std = 0.9 * self._running_std + 0.1 * new_std

        # Throttle plot updates for performance
        self._frame_count += 1
        if self._frame_count >= self._update_every_n_frames:
            self._frame_count = 0
            self._update_plots()

    def _create_ylabels(self):
        """Create TextItem labels positioned next to each channel trace."""
        # Clear existing
        self._clear_ylabels()
        for i in range(self.num_channels):
            lbl = pg.TextItem(text=f'Ch {i}', color='k', anchor=(1, 0.5))
            # initial position at leftmost time and channel y
            lbl.setPos(self._time_axis[0] if hasattr(self, '_time_axis') else -self.display_seconds, i)
            lbl.setZValue(100)
            self.plot_widget.addItem(lbl)
            self._ylabels.append(lbl)

    def _clear_ylabels(self):
        """Remove existing labels from plot."""
        for lbl in getattr(self, '_ylabels', []):
            try:
                self.plot_widget.removeItem(lbl)
            except Exception:
                pass
        self._ylabels = []

    def _update_plots(self):
        """Update all plot curves."""
        # Get ordered data
        ordered_data = self._get_ordered_data()
        n_available = ordered_data.shape[0]

        # Adjust time axis if buffer not full
        if not self._buffer_full:
            time_axis = np.linspace(-self.display_seconds * (n_available / self.display_samples), 0, n_available)
        else:
            time_axis = self._time_axis

        # Update label positions to align with left edge of plotted data
        if getattr(self, '_ylabels', None):
            x_pos = time_axis[0] if len(time_axis) > 0 else -self.display_seconds
            for i, lbl in enumerate(self._ylabels):
                if i < len(self._ylabels):
                    # Use reversed offset to match data display order
                    offset = self.num_channels - 1 - i
                    lbl.setPos(x_pos, offset)
                    lbl.setVisible(self._channel_visible[i])

        # Calculate scaling factor based on data statistics
        scale = self._running_std if self._running_std > 1e-10 else 1.0

        for i in range(self.num_channels):
            if self._channel_visible[i] and i < len(self._curves):
                # Get channel data
                channel_data = ordered_data[:, i].copy()

                # Remove DC offset (center around mean)
                channel_data = channel_data - np.mean(channel_data)

                # Scale data to fit nicely (normalize by running std)
                if scale > 1e-10:
                    channel_data = channel_data / (scale * 6)  # 6 sigma range

                # Apply offset for channel separation - reverse order so Ch 0 is at bottom
                offset = self.num_channels - 1 - i
                display_data = channel_data + offset

                self._curves[i].setData(time_axis, display_data)
                self._curves[i].setVisible(True)
            elif i < len(self._curves):
                self._curves[i].setVisible(False)

        # Set axis ranges
        if self.autoscale_check.isChecked():
            y_min = -0.5
            y_max = self.num_channels - 0.5
            self.plot_widget.setYRange(y_min, y_max)
            self.plot_widget.setXRange(time_axis[0], time_axis[-1])

    def set_channel_visible(self, channel: int, visible: bool):
        """Set visibility of a specific channel."""
        if 0 <= channel < self.num_channels:
            self._channel_visible[channel] = visible

    def set_num_channels(self, num_channels: int):
        """Update the number of channels."""
        if num_channels == self.num_channels:
            return

        # Clear existing curves
        for curve in self._curves:
            self.plot_widget.removeItem(curve)

        self.num_channels = num_channels
        self._data_buffer = np.zeros((self.display_samples, num_channels))
        self._channel_visible = [True] * num_channels
        self._running_std = 1.0

        # Recreate curves
        self._curves = []
        colors = self._generate_colors(num_channels)

        for i in range(num_channels):
            curve = self.plot_widget.plot(
                pen=pg.mkPen(color=colors[i], width=1)
            )
            self._curves.append(curve)

        # Update y-axis ticks
        self._update_y_axis_ticks()
        # Recreate horizontal guide lines
        self._create_hlines()
        # Recreate per-channel text labels
        self._clear_ylabels()
        self._create_ylabels()

    def clear(self):
        """Clear all data."""
        self._data_buffer = np.zeros((self.display_samples, self.num_channels))
        self._running_std = 1.0
        self._update_plots()

    def _create_hlines(self):
        """Create horizontal guide lines at each integer channel offset."""
        # Clear existing
        self._clear_hlines()
        pen = pg.mkPen(color=(200, 200, 200), width=1, style=pg.QtCore.Qt.DotLine)
        for i in range(self.num_channels):
            # Use reversed offset to match data display order
            offset = self.num_channels - 1 - i
            line = pg.InfiniteLine(pos=offset, angle=0, pen=pen)
            line.setZValue(-100)  # behind data
            line.setVisible(self._channel_visible[i] if i < len(self._channel_visible) else True)
            self.plot_widget.addItem(line)
            self._hlines.append(line)
        # Ensure labels are recreated to match
        if hasattr(self, '_ylabels'):
            self._clear_ylabels()
            self._create_ylabels()

    def _clear_hlines(self):
        """Remove existing horizontal lines from plot."""
        for line in getattr(self, '_hlines', []):
            try:
                self.plot_widget.removeItem(line)
            except Exception:
                pass
        self._hlines = []

    def _get_ordered_data(self):
        """Get the data in chronological order (oldest to newest)."""
        if not self._buffer_full:
            return self._data_buffer[:self._buffer_index]
        else:
            return np.concatenate((self._data_buffer[self._buffer_index:], self._data_buffer[:self._buffer_index]), axis=0)

    def _update_y_axis_ticks(self):
        """Update y-axis ticks to show channel numbers."""
        # Reverse order so Ch 0 is at bottom, Ch N-1 at top
        ticks = [[(i, f'Ch {self.num_channels - 1 - i}') for i in range(self.num_channels)]]
        self.plot_widget.getAxis('left').setTicks(ticks)


class EMGPlotWindow(QMainWindow):
    """
    Separate window for EMG plot with channel controls.

    Features channel checkboxes directly beside the plot for easy toggling.
    """

    closed = Signal()  # Emitted when window is closed

    def __init__(
        self,
        num_channels: int = 32,
        sampling_rate: int = 2000,
        display_seconds: float = 5.0,
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

        # Data statistics for auto-scaling
        self._running_std = 1.0

        # Channel visibility
        self._channel_visible = [True] * num_channels

        # Frame counter for throttling updates
        self._frame_count = 0
        self._update_every_n_frames = 3

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

        self.autoscale_check = QCheckBox("Auto-scale")
        self.autoscale_check.setChecked(True)
        control_layout.addWidget(self.autoscale_check)

        control_layout.addStretch()
        plot_layout.addLayout(control_layout)

        # Plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        self.plot_widget.setLabel('left', 'Channels')
        self.plot_widget.setMouseEnabled(x=True, y=True)
        self.plot_widget.enableAutoRange(enable=False)

        # Set y-axis ticks for channels
        self._update_y_axis_ticks()

        plot_layout.addWidget(self.plot_widget)
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

        # Create plot curves
        self._curves: List[pg.PlotDataItem] = []
        self._hlines: List[pg.InfiniteLine] = []

        colors = self._generate_colors(self.num_channels)

        for i in range(self.num_channels):
            curve = self.plot_widget.plot(
                pen=pg.mkPen(color=colors[i], width=1),
                name=f'Ch {i}'
            )
            self._curves.append(curve)

        # Time axis
        self._time_axis = np.linspace(
            -self.display_seconds, 0, self.display_samples
        )

        # Force Y range and create ticks/guide lines/labels
        self.plot_widget.setYRange(-0.5, self.num_channels - 0.5)
        self._update_y_axis_ticks()
        self._create_hlines()
        self._clear_ylabels()
        self._create_ylabels()

    def _generate_colors(self, n: int) -> List[tuple]:
        """Generate n distinct colors."""
        colors = []
        for i in range(n):
            hue = int(360 * i / n)
            colors.append(pg.hsvColor(hue / 360.0, 0.8, 0.8))
        return colors

    @Slot(int, bool)
    def _on_channel_toggled(self, channel: int, checked: bool):
        """Handle channel checkbox toggle."""
        self._channel_visible[channel] = checked
        if channel < len(self._curves):
            self._curves[channel].setVisible(checked)
        # also toggle corresponding horizontal guide line if present
        if 0 <= channel < len(getattr(self, '_hlines', [])):
            self._hlines[channel].setVisible(checked)

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
            self._time_axis = np.linspace(-self.display_seconds, 0, new_samples)

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

        # Update statistics
        recent_data = self._data_buffer[-min(self.display_samples, 4000):]
        if recent_data.size > 0 and np.any(recent_data != 0):
            new_std = np.std(recent_data)
            if new_std > 1e-10:
                self._running_std = 0.9 * self._running_std + 0.1 * new_std

        # Throttle plot updates
        self._frame_count += 1
        if self._frame_count >= self._update_every_n_frames:
            self._frame_count = 0
            self._update_plots()

    def _create_ylabels(self):
        """Create TextItem labels positioned next to each channel trace (window)."""
        # Clear existing
        self._clear_ylabels()
        self._ylabels = []
        for i in range(self.num_channels):
            lbl = pg.TextItem(text=f'Ch {i}', color='k', anchor=(1, 0.5))
            # Use reversed offset to match data display order
            offset = self.num_channels - 1 - i
            lbl.setPos(self._time_axis[0] if hasattr(self, '_time_axis') else -self.display_seconds, offset)
            lbl.setZValue(100)
            self.plot_widget.addItem(lbl)
            self._ylabels.append(lbl)

    def _clear_ylabels(self):
        """Remove existing labels from plot (window)."""
        for lbl in getattr(self, '_ylabels', []):
            try:
                self.plot_widget.removeItem(lbl)
            except Exception:
                pass
        self._ylabels = []

    def _update_plots(self):
        """Update all plot curves."""
        # Get ordered data
        ordered_data = self._get_ordered_data()
        n_available = ordered_data.shape[0]

        # Adjust time axis if buffer not full
        if not self._buffer_full:
            time_axis = np.linspace(-self.display_seconds * (n_available / self.display_samples), 0, n_available)
        else:
            time_axis = self._time_axis

        # Update label positions to align with left edge of plotted data
        if getattr(self, '_ylabels', None):
            x_pos = time_axis[0] if len(time_axis) > 0 else -self.display_seconds
            for i, lbl in enumerate(self._ylabels):
                if i < len(self._ylabels):
                    # Use reversed offset to match data display order
                    offset = self.num_channels - 1 - i
                    lbl.setPos(x_pos, offset)
                    lbl.setVisible(self._channel_visible[i])

        # Calculate scaling factor based on data statistics
        scale = self._running_std if self._running_std > 1e-10 else 1.0

        for i in range(self.num_channels):
            if self._channel_visible[i] and i < len(self._curves):
                # Get channel data
                channel_data = ordered_data[:, i].copy()

                # Remove DC offset (center around mean)
                channel_data = channel_data - np.mean(channel_data)

                # Scale data to fit nicely (normalize by running std)
                if scale > 1e-10:
                    channel_data = channel_data / (scale * 6)  # 6 sigma range

                # Apply offset for channel separation - reverse order so Ch 0 is at bottom
                offset = self.num_channels - 1 - i
                display_data = channel_data + offset

                self._curves[i].setData(time_axis, display_data)
                self._curves[i].setVisible(True)
            elif i < len(self._curves):
                self._curves[i].setVisible(False)

        # Set axis ranges
        if self.autoscale_check.isChecked():
            y_min = -0.5
            y_max = self.num_channels - 0.5
            self.plot_widget.setYRange(y_min, y_max)
            self.plot_widget.setXRange(time_axis[0], time_axis[-1])

    def _update_rms_plots(self, ordered_data = None, time_axis = None):
        """Update RMS envelope plots."""
        pass

    def set_num_channels(self, num_channels: int):
        """Update the number of channels."""
        if num_channels == self.num_channels:
            return

        # This would require recreating the UI - for now, just update data buffer
        self.num_channels = num_channels
        self._data_buffer = np.zeros((self.display_samples, num_channels))
        self._channel_visible = [True] * num_channels
        self._running_std = 1.0

        # Update y-axis ticks
        self._update_y_axis_ticks()
        # Recreate horizontal guide lines
        self._create_hlines()
        # Recreate per-channel text labels
        self._clear_ylabels()
        self._create_ylabels()

    def clear(self):
        """Clear all data."""
        self._data_buffer = np.zeros((self.display_samples, self.num_channels))
        self._running_std = 1.0
        self._update_plots()

    def _update_y_axis_ticks(self):
        """Update y-axis ticks to show channel numbers (for window)."""
        # Reverse order so Ch 0 is at bottom, Ch N-1 at top
        ticks = [[(i, f'Ch {self.num_channels - 1 - i}') for i in range(self.num_channels)]]
        self.plot_widget.getAxis('left').setTicks(ticks)

    def _create_hlines(self):
        """Create horizontal guide lines at each integer channel offset (for window)."""
        self._clear_hlines()
        pen = pg.mkPen(color=(200, 200, 200), width=1, style=pg.QtCore.Qt.DotLine)
        self._hlines = []
        for i in range(self.num_channels):
            # Use reversed offset to match data display order
            offset = self.num_channels - 1 - i
            line = pg.InfiniteLine(pos=offset, angle=0, pen=pen)
            line.setZValue(-100)
            line.setVisible(self._channel_visible[i] if i < len(self._channel_visible) else True)
            self.plot_widget.addItem(line)
            self._hlines.append(line)

    def _clear_hlines(self):
        """Remove existing horizontal lines from plot (for window)."""
        for line in getattr(self, '_hlines', []):
            try:
                self.plot_widget.removeItem(line)
            except Exception:
                pass
        self._hlines = []

    def _get_ordered_data(self):
        """Get the data in chronological order (oldest to newest)."""
        if not self._buffer_full:
            return self._data_buffer[:self._buffer_index]
        else:
            return np.concatenate((self._data_buffer[self._buffer_index:], self._data_buffer[:self._buffer_index]), axis=0)
