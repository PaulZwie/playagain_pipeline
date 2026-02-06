"""
Feature Selection Dialog.

Allows selecting and configuring features for the EMG pipeline.
"""

from typing import Dict, Any, List
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QLabel, QGroupBox, QPushButton, QDialogButtonBox, QWidget,
    QCheckBox, QSpinBox, QDoubleSpinBox, QFormLayout, QScrollArea
)
from PySide6.QtCore import Qt, Signal

from playagain_pipeline.models.feature_pipeline import (
    FeaturePipeline, get_registered_features, FeatureConfig
)


class FeatureSelectionDialog(QDialog):
    """
    Dialog for selecting which features to use in the pipeline.
    Simplified version with inline parameter editing.
    """

    pipeline_updated = Signal(FeaturePipeline)

    def __init__(self, current_pipeline: FeaturePipeline = None, parent=None):
        super().__init__(parent)
        self.pipeline = current_pipeline if current_pipeline else FeaturePipeline.create_default()
        self.registered_features = get_registered_features()

        self.setWindowTitle("Feature Selection")
        self.setMinimumSize(500, 400)

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Instructions
        info_label = QLabel(
            "Select features for the EMG classification pipeline.\n"
            "Check features to enable, adjust parameters as needed."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Scrollable feature list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        self.features_layout = QVBoxLayout(scroll_content)

        # Create a widget for each available feature
        self._feature_widgets = {}
        for name in self.registered_features:
            feature_widget = self._create_feature_widget(name)
            self.features_layout.addWidget(feature_widget)
            self._feature_widgets[name] = feature_widget

        self.features_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        # Quick selection buttons
        quick_layout = QHBoxLayout()

        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self._select_all)
        quick_layout.addWidget(select_all_btn)

        select_none_btn = QPushButton("Select None")
        select_none_btn.clicked.connect(self._select_none)
        quick_layout.addWidget(select_none_btn)

        select_default_btn = QPushButton("Reset to Default")
        select_default_btn.clicked.connect(self._reset_to_default)
        quick_layout.addWidget(select_default_btn)

        layout.addLayout(quick_layout)

        # Dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._apply_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Initialize checkboxes based on current pipeline
        self._update_from_pipeline()

    def _create_feature_widget(self, name: str) -> QGroupBox:
        """Create a widget for a single feature with inline parameters."""
        group = QGroupBox()
        layout = QHBoxLayout(group)
        layout.setContentsMargins(5, 5, 5, 5)

        # Checkbox for enable/disable
        checkbox = QCheckBox(name.upper())
        checkbox.setObjectName(f"check_{name}")
        checkbox.setMinimumWidth(80)
        layout.addWidget(checkbox)

        # Description
        descriptions = {
            "mav": "Mean Absolute Value",
            "rms": "Root Mean Square",
            "wl": "Waveform Length",
            "zc": "Zero Crossings",
            "ssc": "Slope Sign Changes",
            "var": "Variance",
            "iemg": "Integrated EMG",
            "ar": "Autoregressive Coefficients",
            "hjorth": "Hjorth Parameters",
            "freq_features": "Frequency Domain Features"
        }
        desc = descriptions.get(name, "")
        desc_label = QLabel(desc)
        desc_label.setStyleSheet("color: gray;")
        layout.addWidget(desc_label)

        layout.addStretch()

        # Inline parameters for features that need them
        if name in ("zc", "ssc"):
            threshold_label = QLabel("Threshold:")
            layout.addWidget(threshold_label)

            threshold_spin = QDoubleSpinBox()
            threshold_spin.setObjectName(f"threshold_{name}")
            threshold_spin.setRange(0.001, 1.0)
            threshold_spin.setSingleStep(0.001)
            threshold_spin.setDecimals(4)
            threshold_spin.setValue(0.01)
            threshold_spin.setMaximumWidth(80)
            layout.addWidget(threshold_spin)

        elif name == "ar":
            order_label = QLabel("Order:")
            layout.addWidget(order_label)

            order_spin = QSpinBox()
            order_spin.setObjectName(f"order_{name}")
            order_spin.setRange(1, 10)
            order_spin.setValue(4)
            order_spin.setMaximumWidth(60)
            layout.addWidget(order_spin)

        return group

    def _update_from_pipeline(self):
        """Update UI from current pipeline state."""
        # Get currently enabled features
        enabled_features = {f.name for f in self.pipeline.get_features() if f.enabled}

        for name, widget in self._feature_widgets.items():
            checkbox = widget.findChild(QCheckBox, f"check_{name}")
            if checkbox:
                checkbox.setChecked(name in enabled_features)

            # Update parameters if present
            for config in self.pipeline.get_features():
                if config.name == name:
                    threshold_spin = widget.findChild(QDoubleSpinBox, f"threshold_{name}")
                    if threshold_spin and "threshold" in config.params:
                        threshold_spin.setValue(config.params["threshold"])

                    order_spin = widget.findChild(QSpinBox, f"order_{name}")
                    if order_spin and "order" in config.params:
                        order_spin.setValue(config.params["order"])

    def _select_all(self):
        """Select all features."""
        for widget in self._feature_widgets.values():
            checkbox = widget.findChild(QCheckBox)
            if checkbox:
                checkbox.setChecked(True)

    def _select_none(self):
        """Deselect all features."""
        for widget in self._feature_widgets.values():
            checkbox = widget.findChild(QCheckBox)
            if checkbox:
                checkbox.setChecked(False)

    def _reset_to_default(self):
        """Reset to default feature selection."""
        self.pipeline = FeaturePipeline.create_default()
        self._update_from_pipeline()

    def _apply_and_accept(self):
        """Apply current selection to pipeline and close."""
        # Clear pipeline and rebuild from UI
        self.pipeline = FeaturePipeline()

        for name, widget in self._feature_widgets.items():
            checkbox = widget.findChild(QCheckBox, f"check_{name}")
            if checkbox and checkbox.isChecked():
                params = {}

                # Get threshold if present
                threshold_spin = widget.findChild(QDoubleSpinBox, f"threshold_{name}")
                if threshold_spin:
                    params["threshold"] = threshold_spin.value()

                # Get order if present
                order_spin = widget.findChild(QSpinBox, f"order_{name}")
                if order_spin:
                    params["order"] = order_spin.value()

                try:
                    self.pipeline.add_feature(name, **params)
                except Exception as e:
                    print(f"Error adding feature {name}: {e}")

        self.pipeline_updated.emit(self.pipeline)
        self.accept()


    def get_pipeline(self) -> FeaturePipeline:
        """Get the configured pipeline."""
        return self.pipeline
