from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Union
from PySide6.QtWidgets import QWidget, QGroupBox
from PySide6.QtCore import Signal
from device_interfaces.devices.mindrove import MindRoveBracelet
from device_interfaces.enums.device import LoggerLevel
from device_interfaces.gui.ui_compiled.mindrove_widget import Ui_MindRoveForm
import numpy as np

if TYPE_CHECKING:
    from enum import Enum


class MindRoveWidget(QWidget):
    ready_read_signal = Signal(np.ndarray)
    device_connected_signal = Signal(bool)
    device_configured_signal = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.ui = Ui_MindRoveForm()
        self.ui.setupUi(self)

        self.parent_object = parent

        # Device Setup
        self.device = MindRoveBracelet()
        self.device.data_available_signal.connect(self.update)
        self.device_params: dict = {}

        # Push Buttons
        self.connect_button = self.ui.commandConnectionPushButton
        self.connect_button.clicked.connect(self.toggle_connection)
        self.device.connected_signal.connect(self.toggle_connected)

        self.configure_button = self.ui.commandConfigurationPushButton
        self.configure_button.clicked.connect(self.configure_device)
        self.configure_button.setEnabled(False)
        self.device.configured_signal.connect(self.toggle_configured)

        self.stream_button = self.ui.commandStreamPushButton
        self.stream_button.clicked.connect(self.toggle_streaming)
        self.stream_button.setEnabled(False)

    def toggle_connection(self):
        if not self.device.is_connected:
            self.connect_button.setEnabled(False)

        self.device.toggle_connection()

    def toggle_connected(self, is_connected: bool) -> None:
        self.connect_button.setEnabled(True)
        if is_connected:
            self.connect_button.setText("Disconnect")
            self.connect_button.setChecked(True)
            self.configure_button.setEnabled(True)
            self.device.log_info("Connected")
        else:
            self.connect_button.setText("Connect")
            self.connect_button.setChecked(False)
            self.configure_button.setEnabled(False)
            self.stream_button.setEnabled(False)
            self.device.log_info("Disconnected")

        self.device_connected_signal.emit(is_connected)

    def configure_device(self) -> None:
        self.device.configure_device()

    def toggle_configured(self, is_configured: bool) -> None:
        if is_configured:
            self.stream_button.setEnabled(True)
            self.device.log_info("Configured")
        else:
            self.device.reset_configuration()

        self.device_configured_signal.emit(is_configured)

    def toggle_streaming(self) -> None:
        self.device.toggle_streaming()
        if self.device.is_streaming:
            self.stream_button.setText("Stop Streaming")
            self.stream_button.setChecked(True)
            self.configure_button.setEnabled(False)
            self.device.log_info("Streaming")
        else:
            self.stream_button.setText("Stream")
            self.stream_button.setChecked(False)
            self.configure_button.setEnabled(True)
            self.device.log_info("Stopped Streaming")

    def update(self, data: np.ndarray) -> None:
        self.ready_read_signal.emit(data)

    def extract_emg_data(self, data: np.ndarray, milli_volts) -> np.ndarray:
        """
        Extracts the EMG Signals from the transmitted data.

        Args:
            data (np.ndarray):
                Raw data that got transmitted.

            milli_volts (bool, optional):
                If True, the EMG data is converted to milli volts.
                Defaults to False.

        Returns:
            np.ndarray:
                Extracted EMG channels.
        """
        return self.device.extract_emg_data(data)

    def extract_aux_data(self, data: np.ndarray, index: int = 0) -> np.ndarray:
        """
        Extract a defined AUX channel from the transmitted data.

        Args:
            data (np.ndarray):
                Raw data that got transmitted.
            index (int, optional): Index of the AUX channel to be extracted.
                Defaults to 0.

        Returns:
            np.ndarray:
                Extracted AUX channel data.
        """
        return self.device.extract_aux_data(data, index)

    def get_device_information(self) -> Dict[str, Enum | int | float | str]:
        """
        Gets the current configuration of the device.

        Returns:
            Dict[str, Enum | int | float | str]:
                Dictionary that holds information about the
                current device configuration and status.
        """

        return self.device.get_device_information()

    def force_disconnect(self) -> None:
        self.device.force_disconnect()
