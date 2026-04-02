"""
Configuration Dialog for device and pipeline settings.

Includes bracelet rotation visualization and electrode mapping display.
"""

import math
from typing import Optional, Dict, Any
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
    QPushButton, QTabWidget, QWidget, QDialogButtonBox,
    QLineEdit, QFormLayout, QSlider
)
from PySide6.QtCore import Qt, Signal, Slot, QRectF, QPointF
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QFont
from playagain_pipeline.gui.gui_style import apply_app_style


class BraceletVisualizationWidget(QWidget):
    """
    Widget that visualizes the electrode bracelet as a circular diagram.

    Shows electrode positions with correct layout:
    - Electrodes are in rows of 2 (inner and outer ring)
    - Counting goes down the left side first (0,1, 2,3, 4,5...)
    - Then down the right side (16,17, 18,19...)
    - This represents a band with 2 electrodes next to each other that wraps around the arm
    """

    def __init__(self, num_electrodes: int = 32, parent=None):
        super().__init__(parent)
        self.num_electrodes = num_electrodes
        self.rotation_offset = 0.0  # Degrees
        self.reference_rotation = 0.0  # Training reference rotation
        self.highlighted_channels = []

        # Muscle group colors (approximate mapping for forearm)
        self._update_muscle_groups()
        self.setMinimumSize(300, 300)

    def set_num_electrodes(self, num: int):
        """Set number of electrodes."""
        self.num_electrodes = num
        self._update_muscle_groups()
        self.update()

    def _update_muscle_groups(self):
        """Update muscle group mappings based on electrode count."""
        n = self.num_electrodes
        quarter = max(1, n // 4)
        self.muscle_groups = {
            "Flexors": (QColor(255, 100, 100, 180), list(range(0, quarter))),
            "Extensors": (QColor(100, 100, 255, 180), list(range(quarter, 2*quarter))),
            "Radial": (QColor(100, 255, 100, 180), list(range(2*quarter, 3*quarter))),
            "Ulnar": (QColor(255, 255, 100, 180), list(range(3*quarter, n))),
        }

    def set_rotation(self, degrees: float):
        """Set current bracelet rotation offset."""
        self.rotation_offset = degrees
        self.update()

    def set_reference_rotation(self, degrees: float):
        """Set training reference rotation."""
        self.reference_rotation = degrees
        self.update()

    def set_highlighted_channels(self, channels: list):
        """Set channels to highlight."""
        self.highlighted_channels = channels
        self.update()

    def _get_electrode_position(self, electrode_idx: int):
        """
        Get the angular position for an electrode based on the band layout.

        The bracelet has 2 rows (inner and outer), electrodes are numbered:
        - First half (0 to n/2-1): goes down the left side
        - Second half (n/2 to n-1): goes down the right side

        Pairs are: (0,1), (2,3), etc. where even=outer, odd=inner ring
        """
        n = self.num_electrodes
        half = n // 2
        pairs_per_side = half // 2  # How many electrode pairs per side

        # Determine which side and which pair
        if electrode_idx < half:
            # Left side (top to bottom)
            side = "left"
            pair_idx = electrode_idx // 2
            is_outer = (electrode_idx % 2) == 0  # Even = outer row
        else:
            # Right side (top to bottom)
            side = "right"
            pair_idx = (electrode_idx - half) // 2
            is_outer = (electrode_idx % 2) == 0  # Even = outer row

        # Calculate angle: left side is 90-270 degrees, right side is 270-450 (or -90 to 90)
        angle_range = 180.0  # Each side covers 180 degrees

        if side == "left":
            # Left side: from top (90°) to bottom (270°)
            base_angle = 90 + (pair_idx / max(1, pairs_per_side - 1)) * angle_range if pairs_per_side > 1 else 180
        else:
            # Right side: from top (-90° / 270°) to bottom (90°)
            base_angle = -90 + (pair_idx / max(1, pairs_per_side - 1)) * angle_range if pairs_per_side > 1 else 0

        return base_angle, is_outer

    def paintEvent(self, event):
        """Paint the bracelet visualization."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Calculate dimensions
        size = min(self.width(), self.height()) - 20
        center_x = self.width() / 2
        center_y = self.height() / 2
        outer_radius = size / 2 - 30  # Leave room for labels
        inner_radius = outer_radius * 0.7
        electrode_radius = min(14, size / self.num_electrodes * 1.5)

        # Radius for outer and inner electrode rows
        outer_electrode_radius = (outer_radius + inner_radius) / 2 + 10
        inner_electrode_radius = (outer_radius + inner_radius) / 2 - 10

        # Draw outer circle (bracelet ring)
        painter.setPen(QPen(QColor(80, 80, 80), 3))
        painter.setBrush(QBrush(QColor(220, 220, 220)))
        painter.drawEllipse(QRectF(
            center_x - outer_radius, center_y - outer_radius,
            outer_radius * 2, outer_radius * 2
        ))

        # Draw inner circle (arm)
        painter.setBrush(QBrush(QColor(250, 250, 250)))
        painter.drawEllipse(QRectF(
            center_x - inner_radius, center_y - inner_radius,
            inner_radius * 2, inner_radius * 2
        ))

        # Draw reference orientation marker (at top)
        ref_angle = math.radians(self.reference_rotation - 90)
        ref_x = center_x + (outer_radius + 20) * math.cos(ref_angle)
        ref_y = center_y + (outer_radius + 20) * math.sin(ref_angle)
        painter.setPen(QPen(QColor(0, 150, 0), 2))
        painter.setBrush(QBrush(QColor(0, 200, 0)))
        painter.drawEllipse(QPointF(ref_x, ref_y), 8, 8)

        # Draw "REF" label
        painter.setFont(QFont("Arial", 8))
        painter.setPen(QColor(0, 100, 0))
        painter.drawText(int(ref_x - 10), int(ref_y - 12), "REF")

        # Draw electrodes with correct band layout
        for i in range(self.num_electrodes):
            base_angle, is_outer = self._get_electrode_position(i)

            # Apply rotation offset
            angle = math.radians(base_angle + self.rotation_offset)

            # Choose radius based on inner/outer row
            radius = outer_electrode_radius if is_outer else inner_electrode_radius

            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)

            # Determine color based on muscle group
            color = QColor(150, 150, 150)
            for group_name, (group_color, channels) in self.muscle_groups.items():
                if i in channels:
                    color = group_color
                    break

            # Highlight selected channels
            if i in self.highlighted_channels:
                painter.setPen(QPen(QColor(255, 0, 0), 3))
            else:
                painter.setPen(QPen(QColor(50, 50, 50), 1))

            painter.setBrush(QBrush(color))
            painter.drawEllipse(QPointF(x, y), electrode_radius, electrode_radius)

            # Draw electrode number
            painter.setPen(QColor(0, 0, 0))
            font_size = max(7, int(electrode_radius * 0.7))
            painter.setFont(QFont("Arial", font_size))

            # Center the text in the electrode
            text = str(i)
            text_offset_x = -font_size * len(text) / 3
            text_offset_y = font_size / 3
            painter.drawText(int(x + text_offset_x), int(y + text_offset_y), text)

        # Draw current rotation indicator
        rot_angle = math.radians(self.rotation_offset - 90)
        arrow_len = inner_radius - 10
        arrow_x = center_x + arrow_len * math.cos(rot_angle)
        arrow_y = center_y + arrow_len * math.sin(rot_angle)

        painter.setPen(QPen(QColor(200, 0, 0), 3))
        painter.drawLine(QPointF(center_x, center_y), QPointF(arrow_x, arrow_y))

        # Draw legend
        legend_x = 10
        legend_y = self.height() - 80
        painter.setFont(QFont("Arial", 9))

        for i, (name, (color, _)) in enumerate(self.muscle_groups.items()):
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(QColor(50, 50, 50), 1))
            painter.drawRect(legend_x, legend_y + i * 18, 12, 12)
            painter.setPen(QColor(0, 0, 0))
            painter.drawText(legend_x + 18, legend_y + i * 18 + 10, name)


class ConfigurationDialog(QDialog):
    """
    Dialog for configuring device and pipeline settings.

    Features:
    - Device configuration (channels, sampling rate)
    - Recording settings
    - Calibration settings with bracelet visualization
    - Model configuration
    """

    config_changed = Signal(dict)

    def __init__(self, current_config: Optional[Dict[str, Any]] = None, parent=None):
        super().__init__(parent)
        self.current_config = current_config or {}

        self._setup_ui()
        apply_app_style(self, self.current_config.get("ui_theme", "bright"))
        self.setWindowTitle("Pipeline Configuration")
        self.setMinimumSize(700, 500)

        if current_config:
            self._load_config(current_config)

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Tabs for different configuration sections
        tabs = QTabWidget()

        # Device tab
        device_widget = QWidget()
        device_layout = QFormLayout(device_widget)

        self.device_type_combo = QComboBox()
        self.device_type_combo.addItems(["Synthetic", "Muovi", "Muovi Plus"])
        device_layout.addRow("Device Type:", self.device_type_combo)

        self.num_channels_spin = QSpinBox()
        self.num_channels_spin.setRange(1, 128)
        self.num_channels_spin.setValue(32)
        device_layout.addRow("Number of Channels:", self.num_channels_spin)

        self.sampling_rate_spin = QSpinBox()
        self.sampling_rate_spin.setRange(100, 10000)
        self.sampling_rate_spin.setValue(2000)
        self.sampling_rate_spin.setSuffix(" Hz")
        device_layout.addRow("Sampling Rate:", self.sampling_rate_spin)

        self.ip_edit = QLineEdit("0.0.0.0")
        device_layout.addRow("IP Address:", self.ip_edit)

        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(54321)
        device_layout.addRow("Port:", self.port_spin)

        tabs.addTab(device_widget, "Device")

        # Recording tab
        recording_widget = QWidget()
        recording_layout = QFormLayout(recording_widget)

        self.window_size_spin = QSpinBox()
        self.window_size_spin.setRange(50, 1000)
        self.window_size_spin.setValue(200)
        self.window_size_spin.setSuffix(" ms")
        recording_layout.addRow("Window Size:", self.window_size_spin)

        self.stride_spin = QSpinBox()
        self.stride_spin.setRange(10, 500)
        self.stride_spin.setValue(50)
        self.stride_spin.setSuffix(" ms")
        recording_layout.addRow("Window Stride:", self.stride_spin)

        self.gesture_duration_spin = QDoubleSpinBox()
        self.gesture_duration_spin.setRange(1.0, 30.0)
        self.gesture_duration_spin.setValue(5.0)
        self.gesture_duration_spin.setSuffix(" s")
        recording_layout.addRow("Gesture Duration:", self.gesture_duration_spin)

        self.rest_duration_spin = QDoubleSpinBox()
        self.rest_duration_spin.setRange(1.0, 30.0)
        self.rest_duration_spin.setValue(3.0)
        self.rest_duration_spin.setSuffix(" s")
        recording_layout.addRow("Rest Duration:", self.rest_duration_spin)

        tabs.addTab(recording_widget, "Recording")

        # Calibration tab with bracelet visualization
        calibration_widget = QWidget()
        calibration_layout = QHBoxLayout(calibration_widget)

        # Left side - controls
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)

        rotation_group = QGroupBox("Bracelet Rotation")
        rotation_layout = QVBoxLayout(rotation_group)

        self.rotation_slider = QSlider(Qt.Orientation.Horizontal)
        self.rotation_slider.setRange(0, 359)
        self.rotation_slider.setValue(0)
        self.rotation_slider.valueChanged.connect(self._on_rotation_changed)
        rotation_layout.addWidget(self.rotation_slider)

        self.rotation_label = QLabel("Current: 0°")
        rotation_layout.addWidget(self.rotation_label)

        self.ref_rotation_spin = QSpinBox()
        self.ref_rotation_spin.setRange(0, 359)
        self.ref_rotation_spin.setValue(0)
        self.ref_rotation_spin.setSuffix("°")
        self.ref_rotation_spin.valueChanged.connect(self._on_ref_rotation_changed)
        rotation_layout.addWidget(QLabel("Training Reference:"))
        rotation_layout.addWidget(self.ref_rotation_spin)

        controls_layout.addWidget(rotation_group)

        # Auto-calibrate button
        auto_calibrate_btn = QPushButton("Auto-Calibrate")
        auto_calibrate_btn.setToolTip("Automatically detect optimal rotation")
        controls_layout.addWidget(auto_calibrate_btn)

        controls_layout.addStretch()
        calibration_layout.addWidget(controls_widget)

        # Right side - visualization
        self.bracelet_widget = BraceletVisualizationWidget()
        self.bracelet_widget.set_num_electrodes(32)
        calibration_layout.addWidget(self.bracelet_widget, stretch=2)

        tabs.addTab(calibration_widget, "Calibration")

        # Model tab
        model_widget = QWidget()
        model_layout = QFormLayout(model_widget)

        self.default_model_combo = QComboBox()
        self.default_model_combo.addItems(["CatBoost", "SVM", "Random Forest", "LDA"])
        model_layout.addRow("Default Model Type:", self.default_model_combo)

        self.test_ratio_spin = QDoubleSpinBox()
        self.test_ratio_spin.setRange(0.1, 0.5)
        self.test_ratio_spin.setValue(0.2)
        self.test_ratio_spin.setSingleStep(0.05)
        model_layout.addRow("Test/Validation Ratio:", self.test_ratio_spin)

        self.auto_save_check = QCheckBox()
        self.auto_save_check.setChecked(True)
        model_layout.addRow("Auto-save trained models:", self.auto_save_check)

        tabs.addTab(model_widget, "Model")

        # Appearance tab
        appearance_widget = QWidget()
        appearance_layout = QFormLayout(appearance_widget)
        self.theme_combo = QComboBox()
        self.theme_combo.addItem("Bright", "bright")
        self.theme_combo.addItem("Dark", "dark")
        self.theme_combo.currentIndexChanged.connect(self._on_theme_changed)
        appearance_layout.addRow("Theme:", self.theme_combo)
        tabs.addTab(appearance_widget, "Appearance")

        layout.addWidget(tabs)

        # Dialog buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Apply
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self._on_apply)
        layout.addWidget(buttons)

        # Connect channel change to bracelet
        self.num_channels_spin.valueChanged.connect(self._on_channels_changed)

    @Slot(int)
    def _on_rotation_changed(self, value: int):
        """Handle rotation slider change."""
        self.rotation_label.setText(f"Current: {value}°")
        self.bracelet_widget.set_rotation(value)

    @Slot(int)
    def _on_ref_rotation_changed(self, value: int):
        """Handle reference rotation change."""
        self.bracelet_widget.set_reference_rotation(value)

    @Slot(int)
    def _on_channels_changed(self, value: int):
        """Handle channel count change."""
        self.bracelet_widget.set_num_electrodes(value)

    @Slot()
    def _on_apply(self):
        """Apply current configuration."""
        config = self.get_config()
        self.config_changed.emit(config)

    @Slot()
    def _on_theme_changed(self):
        apply_app_style(self, self.theme_combo.currentData())

    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration."""
        return {
            "device": {
                "type": self.device_type_combo.currentText(),
                "num_channels": self.num_channels_spin.value(),
                "sampling_rate": self.sampling_rate_spin.value(),
                "ip_address": self.ip_edit.text(),
                "port": self.port_spin.value()
            },
            "recording": {
                "window_size_ms": self.window_size_spin.value(),
                "stride_ms": self.stride_spin.value(),
                "gesture_duration_s": self.gesture_duration_spin.value(),
                "rest_duration_s": self.rest_duration_spin.value()
            },
            "calibration": {
                "rotation_offset": self.rotation_slider.value(),
                "reference_rotation": self.ref_rotation_spin.value()
            },
            "model": {
                "default_type": self.default_model_combo.currentText(),
                "test_ratio": self.test_ratio_spin.value(),
                "auto_save": self.auto_save_check.isChecked()
            },
            "ui_theme": self.theme_combo.currentData(),
        }

    def _load_config(self, config: Dict[str, Any]):
        """Load configuration into UI."""
        if "device" in config:
            dev = config["device"]
            if "type" in dev:
                idx = self.device_type_combo.findText(dev["type"])
                if idx >= 0:
                    self.device_type_combo.setCurrentIndex(idx)
            if "num_channels" in dev:
                self.num_channels_spin.setValue(dev["num_channels"])
            if "sampling_rate" in dev:
                self.sampling_rate_spin.setValue(dev["sampling_rate"])
            if "ip_address" in dev:
                self.ip_edit.setText(dev["ip_address"])
            if "port" in dev:
                self.port_spin.setValue(dev["port"])

        if "recording" in config:
            rec = config["recording"]
            if "window_size_ms" in rec:
                self.window_size_spin.setValue(rec["window_size_ms"])
            if "stride_ms" in rec:
                self.stride_spin.setValue(rec["stride_ms"])
            if "gesture_duration_s" in rec:
                self.gesture_duration_spin.setValue(rec["gesture_duration_s"])
            if "rest_duration_s" in rec:
                self.rest_duration_spin.setValue(rec["rest_duration_s"])

        if "calibration" in config:
            cal = config["calibration"]
            if "rotation_offset" in cal:
                self.rotation_slider.setValue(int(cal["rotation_offset"]))
            if "reference_rotation" in cal:
                self.ref_rotation_spin.setValue(int(cal["reference_rotation"]))

        if "model" in config:
            mod = config["model"]
            if "default_type" in mod:
                idx = self.default_model_combo.findText(mod["default_type"])
                if idx >= 0:
                    self.default_model_combo.setCurrentIndex(idx)
            if "test_ratio" in mod:
                self.test_ratio_spin.setValue(mod["test_ratio"])
            if "auto_save" in mod:
                self.auto_save_check.setChecked(mod["auto_save"])

        if "ui_theme" in config:
            idx = self.theme_combo.findData(config["ui_theme"])
            if idx >= 0:
                self.theme_combo.setCurrentIndex(idx)
