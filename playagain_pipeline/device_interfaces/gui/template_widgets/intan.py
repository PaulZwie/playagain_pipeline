from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Union
from PySide6.QtWidgets import (
    QWidget,
    QGroupBox,
    QCheckBox,
    QPushButton,
    QLabel,
    QComboBox,
    QLineEdit,
)
from PySide6.QtCore import Signal
from device_interfaces.enums.device import LoggerLevel
from device_interfaces.enums.intan import IntanAvailablePorts
from device_interfaces.devices.intan import IntanRHDController
from device_interfaces.gui.ui_compiled.intan_widget import (
    Ui_IntanRHDControllerForm,
)
import numpy as np
from enum import Enum


if TYPE_CHECKING:
    pass


class IntanRHDControllerWidget(QWidget):
    ready_read_signal = Signal(np.ndarray)
    device_connected_signal = Signal(bool)
    device_configured_signal = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.ui = Ui_IntanRHDControllerForm()
        self.ui.setupUi(self)

        self.parent_object = parent

        # Device Setup
        self.device: IntanRHDController = IntanRHDController()
        self.device.data_available_signal.connect(self.update)
        self.device.configuration_available_signal.connect(
            self._update_device_configuration
        )
        self.device_params: dict = {}

        # Push Buttons
        self.connect_button: QPushButton = self.ui.commandConnectionPushButton
        self.connect_button.clicked.connect(self.toggle_connection)
        self.device.connected_signal.connect(self.toggle_connected)
        self.device.connected_signal.connect(self.device_connected_signal)

        self.configure_button: QPushButton = self.ui.commandConfigurationPushButton
        self.configure_button.clicked.connect(self.configure_device)
        self.configure_button.setEnabled(False)
        self.device.configured_signal.connect(self.toggle_configured)

        self.stream_button: QPushButton = self.ui.commandStreamPushButton
        self.stream_button.clicked.connect(self.toggle_streaming)
        self.stream_button.setEnabled(False)

        # Acquisition Parameters
        self.acquisition_group_box: QGroupBox = self.ui.acquisitionParametersGroupBox
        self.sampling_frequency_label: QLabel = (
            self.ui.acquisitionSamplingFrequencyLabel
        )

        # Input Parameters
        self.input_group_box: QGroupBox = self.ui.inputParametersGroupBox
        self.input_group_box.setEnabled(False)

        # Grid Selection
        self.grid_selection_group_box: QGroupBox = self.ui.gridSelectionGroupBox
        self.grid_selection_check_box_list: list[QCheckBox] = [
            self.ui.gridACheckBox,
            self.ui.gridBCheckBox,
            self.ui.gridCCheckBox,
            self.ui.gridDCheckBox,
            self.ui.gridECheckBox,
            self.ui.gridFCheckBox,
            self.ui.gridGCheckBox,
            self.ui.gridHCheckBox,
        ]
        [
            check_box.setChecked(False)
            for check_box in self.grid_selection_check_box_list
        ]
        self.device_update_push_button: QPushButton = self.ui.deviceUpdatePushButton
        self.device_update_push_button.clicked.connect(self.get_configuration)
        self.device_update_push_button.setEnabled(False)

        # Connection parameters
        self.connection_group_box: QGroupBox = self.ui.connectionGroupBox
        self.connection_ip_line_edit: QLineEdit = self.ui.connectionIPLineEdit
        self.default_connection_ip: str = self.connection_ip_line_edit.text()
        self.connection_ip_line_edit.textChanged.connect(self._check_ip_input)
        self.connection_port_line_edit: QLineEdit = self.ui.connectionPortLineEdit
        self.default_connection_port: int = self.connection_port_line_edit.text()
        self.connection_port_line_edit.textChanged.connect(self._check_port_input)
        self.connection_use_waveform_check_box: QCheckBox = (
            self.ui.connectionUseWaveformCheckBox
        )
        self.connection_use_waveform_check_box.toggled.connect(
            self._toggle_use_waveform
        )
        self.connection_use_waveform_check_box.setChecked(True)
        self.connection_use_spike_check_box: QCheckBox = (
            self.ui.connectionUseSpikeCheckBox
        )
        self.connection_use_spike_check_box.setChecked(False)
        self.connection_use_spike_check_box.setEnabled(False)
        self.connection_use_spike_check_box.toggled.connect(
            self._toggle_use_spike_detection
        )

        # Configuration GroupBoxes
        self.configuration_group_boxes: list[QGroupBox] = [
            self.acquisition_group_box,
            self.input_group_box,
            self.grid_selection_group_box,
        ]

    def get_configuration(self) -> Dict[str, Union[Enum, int, float, str]]:
        """
        Gets the current configuration of the device.

        Returns:
            Dict[str, Union[Enum, int, float, str]]:
                Dictionary that holds information about the
                current device configuration and status.
        """
        return self.device.get_configuration()

    def _update_device_configuration(
        self, configuration: Dict[str, Union[Enum, int, float, str]]
    ) -> None:
        self.sampling_frequency_label.setText(str(configuration["sampling_frequency"]))
        for index, check_box in enumerate(self.grid_selection_check_box_list):
            check_box.setEnabled(
                configuration["port_information"][IntanAvailablePorts(index)]["enabled"]
            )
            check_box.setChecked(
                configuration["port_information"][IntanAvailablePorts(index)]["enabled"]
            )

    def toggle_connection(self):
        self.device.toggle_connection(
            (
                self.connection_ip_line_edit.text(),
                int(self.connection_port_line_edit.text()),
            )
        )

    def toggle_connected(self, is_connected: bool) -> None:
        if is_connected:
            self.connect_button.setText("Disconnect")
            self.connect_button.setChecked(True)
            self.configure_button.setEnabled(True)
            self.device.log_info("Connected", "INFO")
            self.device_update_push_button.setEnabled(True)
        else:
            self.connect_button.setText("Connect")
            self.connect_button.setChecked(False)
            self.configure_button.setEnabled(False)
            self.device.log_info("Disconnected", "INFO")
            self.device_update_push_button.setEnabled(False)

        self.connection_group_box.setEnabled(not self.connection_group_box.isEnabled())

    def configure_device(self) -> None:
        self.device_params = {
            "enabled_ports": [
                IntanAvailablePorts(index)
                for index, check_box in enumerate(self.grid_selection_check_box_list)
                if check_box.isChecked()
            ]
        }
        self.device.configure_device(self.device_params)

    def toggle_configured(self, is_configured: bool) -> None:
        if is_configured:
            self.stream_button.setEnabled(True)
            self.device.log_info("Configured")
        else:
            self.device.reset_configuration()

        self._toggle_configuration_group_boxes()
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
        else:
            self.stream_button.setText("Stream")
            self.stream_button.setChecked(False)
            self.stream_button.setEnabled(False)
            self.configure_button.setEnabled(True)
            self.device.log_info("Stopped Streaming")

        self._toggle_configuration_group_boxes()

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

    def _check_ip_input(self) -> None:
        if not self.device.check_valid_ip(self.connection_ip_line_edit.text()):
            self.connection_ip_line_edit.setText(str(self.default_connection_ip))
            self.device.log_info("Invalid IP", LoggerLevel.WARNING)

    def _check_port_input(self) -> None:
        if not self.device.check_valid_port(self.connection_port_line_edit.text()):
            self.connection_port_line_edit.setText(str(self.default_connection_port))
            self.device.log_info("Invalid port", LoggerLevel.WARNING)

    def _toggle_use_spike_detection(self, checked: bool) -> None:
        self.device.use_spike_detection = checked

    def _toggle_use_waveform(self, checked: bool) -> None:
        self.device.use_waveform = checked

    def closeEvent(self, event) -> None:
        self.force_disconnect()
