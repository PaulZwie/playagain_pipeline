from __future__ import annotations
from typing import TYPE_CHECKING, Union, Dict, Tuple
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Signal
from device_interfaces.devices.noxon import Noxon
from device_interfaces.enums.noxon import *
from device_interfaces.enums.device import LoggerLevel
from device_interfaces.gui.ui_compiled.noxon_widget import (
    Ui_NoxonForm,
)
import numpy as np

if TYPE_CHECKING:
    pass


class NoxonWidget(QWidget):
    ready_read_signal = Signal(np.ndarray)
    device_connected_signal = Signal(bool)
    device_configured_signal = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.ui = Ui_NoxonForm()
        self.ui.setupUi(self)

        self.parent_object = parent

        # Device Setup
        self.device = Noxon()
        self.device.data_available_signal.connect(self.update)
        self.device_params: dict = {}
        self._initialize_device_params()

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

        # Connection parameters
        self.connection_group_box = self.ui.connectionGroupBox
        self.connection_ip_line_edit = self.ui.connectionIPLineEdit
        self.default_connection_ip = self.ui.connectionIPLineEdit.text()
        self.connection_ip_line_edit.editingFinished.connect(self.device.check_valid_ip)
        self.connection_port_line_edit = self.ui.connectionPortLineEdit
        self.default_connection_port = int(self.ui.connectionPortLineEdit.text())
        self.connection_port_line_edit.editingFinished.connect(
            self.device.check_valid_port
        )

        # Acquisition parameters
        self.acquisition_group_box = self.ui.acquisitionGroupBox
        self.acquisition_sampling_frequency_combo_box = (
            self.ui.acquisitionSamplingFrequencyComboBox
        )
        self.acquisition_number_of_channels_combo_box = (
            self.ui.acquisitionNumberOfChannelsComboBox
        )

        # Input parameters
        self.input_group_box = self.ui.inputGroupBox
        self.input_low_resolution_radio_button = self.ui.inputLowResolutionRadioButton
        self.input_high_resolution_radio_button = self.ui.inputHighResolutionRadioButton
        self.input_detection_mode_combo_box = self.ui.inputDetectionModeComboBox
        self.input_gain_combo_box = self.ui.inputGainComboBox

        # Configuration parameters
        self.configuration_group_boxes = [
            self.acquisition_group_box,
            self.input_group_box,
        ]

    def toggle_connection(self):
        if not self.device.is_connected:
            self.connect_button.setEnabled(False)

        self.device.toggle_connection(
            (
                self.connection_ip_line_edit.text(),
                int(self.connection_port_line_edit.text()),
            )
        )

    def toggle_connected(self, is_connected: bool) -> None:
        self.connect_button.setEnabled(True)
        if is_connected:
            self.connect_button.setText("Disconnect")
            self.connect_button.setChecked(True)
            self.configure_button.setEnabled(True)
            self.device.log_info("Connected", "INFO")
        else:
            self.connect_button.setText("Connect")
            self.connect_button.setChecked(False)
            self.configure_button.setEnabled(False)
            self.stream_button.setEnabled(False)
            self.device.log_info("Disconnected", "INFO")

        self.connection_group_box.setEnabled(not self.connection_group_box.isEnabled())

    def configure_device(self) -> None:
        self.device_params["sampling_frequency_mode"] = NOXONSamplingFrequency(
            self.acquisition_sampling_frequency_combo_box.currentIndex()
        )
        self.device_params["number_of_channels_mode"] = NOXONChannels(
            self.acquisition_number_of_channels_combo_box.currentIndex()
        )
        self.device_params["acquisition_mode"] = NOXONMode(
            self.input_detection_mode_combo_box.currentIndex()
        )
        self.device_params["resolution_mode"] = NOXONResolution(
            int(self.input_high_resolution_radio_button.isChecked())
        )
        self.device_params["gain_mode"] = NOXONGain(
            self.input_gain_combo_box.currentIndex()
        )

        self.device.configure_device(self.device_params)

    def toggle_configured(self, is_configured: bool) -> None:
        if is_configured:
            self.stream_button.setEnabled(True)
            self.device.log_info("Configured")
        else:
            self.device.reset_configuration()

        self.device_configured_signal.emit(is_configured)

    def _toggle_configuration_group_boxes(self) -> None:
        for group_box in self.configuration_group_boxes:
            group_box.setEnabled(not group_box.isEnabled())

    def toggle_streaming(self) -> None:
        self.device.toggle_streaming()
        if self.device.is_streaming:
            self.stream_button.setText("Stop Streaming")
            self.stream_button.setChecked(True)
            self.configure_button.setEnabled(False)
            self.device.log_info("Streaming")
            self._toggle_configuration_group_boxes()
        else:
            self.stream_button.setText("Stream")
            self.stream_button.setChecked(False)
            self.configure_button.setEnabled(True)
            self._toggle_configuration_group_boxes()
            self.device.log_info("Stopped Streaming")

    def update(self, data: np.ndarray) -> None:
        self.ready_read_signal.emit(data)

    def extract_emg_data(
        self, data: np.ndarray, milli_volts: bool = False
    ) -> np.ndarray:
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
        return self.device.extract_emg_data(data, milli_volts)

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

    def _initialize_device_params(self) -> None:
        self.device_params = {
            "action_mode": NOXONAction.SET,
            "sampling_frequency_mode": NOXONSamplingFrequency.HIGH,
            "number_of_channels_mode": NOXONChannels.ULTRA,
            "acquisition_mode": NOXONMode.MONOPOLAR,
            "resolution_mode": NOXONResolution.LOW,
            "filtering_mode": NOXONFilter.ON,
            "gain_mode": NOXONGain.DEFAULT,
            "trigger_mode": NOXONTrigger.DEFAULT,
            "recording_mode": NOXONRecording.STOP,
            "transmission_mode": NOXONTransmission.STOP,
        }
