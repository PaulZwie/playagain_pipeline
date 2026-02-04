from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Union
from PySide6.QtWidgets import QWidget, QGroupBox
from PySide6.QtCore import Signal
from device_interfaces.devices.muovi import Muovi
from device_interfaces.dicts.muovi import *
from device_interfaces.enums.muovi import *
from device_interfaces.enums.device import LoggerLevel
from device_interfaces.gui.ui_compiled.muovi_widget import Ui_MuoviForm
import numpy as np

if TYPE_CHECKING:
    pass


class MuoviWidget(QWidget):
    ready_read_signal = Signal(np.ndarray)
    device_connected_signal = Signal(bool)
    device_configured_signal = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.ui = Ui_MuoviForm()
        self.ui.setupUi(self)

        self.parent_object = parent

        # Device Setup
        self.device = Muovi(is_muovi_plus=False)
        self.device.data_available_signal.connect(self.update)
        self.device_params: dict = {}
        self._initialize_device_params()

        # Push Buttons
        self.connect_push_button = self.ui.commandConnectionPushButton
        self.connect_push_button.clicked.connect(self.toggle_connection)
        self.device.connected_signal.connect(self.toggle_connected)

        self.configure_push_button = self.ui.commandConfigurationPushButton
        self.configure_push_button.clicked.connect(self.configure_device)
        self.configure_push_button.setEnabled(False)
        self.device.configured_signal.connect(self.toggle_configured)

        self.stream_push_button = self.ui.commandStreamPushButton
        self.stream_push_button.clicked.connect(self.toggle_streaming)
        self.stream_push_button.setEnabled(False)
        self.device.streaming_signal.connect(self.streaming_toggled)

        # Connection parameters
        self.connection_group_box = self.ui.connectionGroupBox
        self.connection_ip_combo_box = self.ui.connectionIPComboBox
        self.connection_port_label = self.ui.connectionPortLabel
        self.connection_update_push_button = self.ui.connectionUpdatePushButton
        self.connection_update_push_button.clicked.connect(
            lambda: (
                self.connection_ip_combo_box.clear(),
                self.connection_ip_combo_box.addItems(
                    self.device.get_server_wifi_ip_address()
                ),
            )
        )

        self.connection_ip_combo_box.clear()
        self.connection_ip_combo_box.addItems(self.device.get_server_wifi_ip_address())

        self.connection_port_label.setText(
            str(
                MUOVI_NETWORK_CHARACTERISTICS_DICT[
                    MuoviNetworkCharacteristics.EXTERNAL_NETWORK
                ]["port"]
            )
        )

        # Input parameters
        self.input_parameters_group_box = self.ui.inputGroupBox
        self.input_working_mode_combo_box = self.ui.inputWorkingModeComboBox
        self.input_detection_mode_combo_box = self.ui.inputDetectionModeComboBox

        # Configuration parameters
        self.configuration_group_boxes: list[QGroupBox] = [
            self.input_parameters_group_box,
        ]

    def toggle_connection(self):
        if not self.device.is_connected:
            self.connect_push_button.setEnabled(False)

        self.device.toggle_connection(
            (
                self.connection_ip_combo_box.currentText(),
                int(self.connection_port_label.text()),
            )
        )

    def toggle_connected(self, is_connected: bool) -> None:
        self.connect_push_button.setEnabled(True)
        if is_connected:
            self.connect_push_button.setText("Disconnect")
            self.connect_push_button.setChecked(True)
            self.configure_push_button.setEnabled(True)
            self.connection_group_box.setEnabled(False)
            self.device.log_info("Connected")
        else:
            self.connect_push_button.setText("Connect")
            self.connect_push_button.setChecked(False)
            self.configure_push_button.setEnabled(False)
            self.stream_push_button.setEnabled(False)
            self.connection_group_box.setEnabled(True)
            self.device.log_info("Disconnected")

        self.device_connected_signal.emit(is_connected)

    def configure_device(self) -> None:
        self.device_params["working_mode"] = MuoviWorkingMode(
            self.input_working_mode_combo_box.currentIndex()
        )
        self.device_params["detection_mode"] = MuoviDetectionMode(
            self.input_detection_mode_combo_box.currentIndex()
        )

        self.device.configure_device(self.device_params)

    def toggle_configured(self, is_configured: bool) -> None:
        if is_configured:
            self.stream_push_button.setEnabled(True)
            self.device.log_info("Configured")
        else:
            self.device.reset_configuration()

        self.device_configured_signal.emit(is_configured)

    def _toggle_configuration_group_boxes(self) -> None:
        for group_box in self.configuration_group_boxes:
            group_box.setEnabled(not group_box.isEnabled())

    def toggle_streaming(self) -> None:
        self.stream_push_button.setEnabled(False)
        self.device.toggle_streaming()

    def streaming_toggled(self, is_streaming: bool) -> None:
        self.stream_push_button.setEnabled(True)
        if is_streaming:
            self.stream_push_button.setText("Stop Streaming")
            self.stream_push_button.setChecked(True)
            self.configure_push_button.setEnabled(False)
            self.device.log_info("Streaming")
            self._toggle_configuration_group_boxes()
        else:
            self.stream_push_button.setText("Stream")
            self.stream_push_button.setChecked(False)
            self.configure_push_button.setEnabled(True)
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
            "working_mode": MuoviWorkingMode.EMG,
            "detection_mode": MuoviDetectionMode.MONOPOLAR_GAIN_8,
            "streaming_mode": MuoviStream.STOP,
        }
