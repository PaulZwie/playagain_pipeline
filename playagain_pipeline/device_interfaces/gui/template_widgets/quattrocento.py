"""
Template widget for the Quattrocento device.
Developer: Dominik I. Braun
Contact: dome.braun@fau.de
Last Update: 2023-12-21
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Union, Dict, Tuple
from device_interfaces.devices.quattrocento import Quattrocento
from device_interfaces.dicts.quattrocento import *
from device_interfaces.enums.device import LoggerLevel
from device_interfaces.enums.quattrocento import *
from PySide6.QtCore import Signal
from device_interfaces.gui.ui_compiled.quattrocento_widget import Ui_QuattrocentoForm
from PySide6.QtWidgets import QWidget
import numpy as np

if TYPE_CHECKING:
    pass


class QuattrocentoWidget(QWidget):
    ready_read_signal = Signal(np.ndarray)
    device_connected_signal = Signal(bool)
    device_configured_signal = Signal(bool)
    device_streaming_signal = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.ui = Ui_QuattrocentoForm()
        self.ui.setupUi(self)

        self.parent_object = parent

        # Device Setup
        self.device = Quattrocento()
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
        self.connection_ip_line_edit = self.ui.connectionIPLineEdit
        self.default_connection_ip = self.ui.connectionIPLineEdit.text()
        self.connection_ip_line_edit.editingFinished.connect(self._check_ip_input)
        self.connection_port_line_edit = self.ui.connectionPortLineEdit
        self.default_connection_port = int(self.ui.connectionPortLineEdit.text())
        self.connection_port_line_edit.editingFinished.connect(self._check_port_input)

        # Acquisition parameters
        self.acquisition_group_box = self.ui.acquisitionGroupBox
        self.acquisition_sampling_frequency_combo_box = (
            self.ui.acquisitionSamplingFrequencyComboBox
        )
        self.acquisition_number_of_channels_combo_box = (
            self.ui.acquisitionNumberOfChannelsComboBox
        )
        self.acquisition_decimator_check_box = self.ui.acquisitionDecimatorCheckBox
        self.acquisition_recording_check_box = self.ui.acquisitionRecordingCheckBox

        # Grid Selection
        self.grid_selection_group_box = self.ui.gridSelectionGroupBox
        self.grid_selection_check_box_list = [
            self.ui.gridOneCheckBox,
            self.ui.gridTwoCheckBox,
            self.ui.gridThreeCheckBox,
            self.ui.gridFourCheckBox,
            self.ui.gridFiveCheckBox,
            self.ui.gridSixCheckBox,
        ]

        # Input parameters
        self.input_group_box = self.ui.inputGroupBox
        self.input_channel_combo_box = self.ui.inputChannelComboBox
        self.input_high_pass_combo_box = self.ui.inputHighPassComboBox
        self.input_high_pass_default = self.ui.inputHighPassComboBox.currentIndex()
        self.input_low_pass_combo_box = self.ui.inputLowPassComboBox
        self.input_low_pass_default = self.ui.inputLowPassComboBox.currentIndex()
        self.input_detection_mode_combo_box = self.ui.inputDetectionModeComboBox
        self.input_detection_mode_default = (
            self.ui.inputDetectionModeComboBox.currentIndex()
        )
        self.input_configuration_push_button = self.ui.inputConfigurationPushButton
        self.input_configuration_push_button.clicked.connect(
            self._configure_selected_input
        )

        # Configuration parameters
        self.configuration_group_boxes = [
            self.acquisition_group_box,
            self.grid_selection_group_box,
            self.input_group_box,
        ]

    def toggle_connection(self):
        self.connect_push_button.setEnabled(False)
        self.device.toggle_connection(
            (
                self.connection_ip_line_edit.text(),
                int(self.connection_port_line_edit.text()),
            )
        )

    def toggle_connected(self, is_connected: bool) -> None:
        self.connect_push_button.setEnabled(True)
        if is_connected:
            self.connect_push_button.setText("Disconnect")
            self.connect_push_button.setChecked(True)
            self.configure_push_button.setEnabled(True)
            self.device.log_info("Connected", "INFO")
        else:
            self.connect_push_button.setText("Connect")
            self.connect_push_button.setChecked(False)
            self.configure_push_button.setEnabled(False)
            self.stream_push_button.setEnabled(False)
            self.device.log_info("Disconnected", "INFO")

        self.connection_group_box.setEnabled(not self.connection_group_box.isEnabled())

    def configure_device(self) -> None:
        self.device_params["acquisiton_configuration"] = {
            "decim_mode": QuattrocentoDecim(
                int(self.acquisition_decimator_check_box.isChecked())
            ),
            "recording_mode": QuattrocentoRecording(
                int(self.acquisition_recording_check_box.isChecked())
            ),
            "sampling_frequency_mode": QuattrocentoSamplingFrequency(
                self.acquisition_sampling_frequency_combo_box.currentIndex()
            ),
            "number_of_channels_mode": QuattrocentoNumberOfChannels(
                self.acquisition_number_of_channels_combo_box.currentIndex()
            ),
            "acquisition_mode": QuattrocentoAcquisition.INACTIVE,
        }

        self.device_params["grids"] = [
            i
            for i, check_box in enumerate(self.grid_selection_check_box_list)
            if check_box.isChecked()
        ]

        self.device.configure_device(self.device_params)

    def toggle_configured(self, is_configured: bool) -> None:
        if is_configured:
            self.stream_push_button.setEnabled(True)
            self.device.log_info("Configured")
        else:
            self.device.reset_configuration()

        self._toggle_configuration_group_boxes()
        self.device_configured_signal.emit(is_configured)

    def _configure_selected_input(self) -> None:
        current_input_channel = self.input_channel_combo_box.currentText()
        self.device_params[INPUT_CHANNEL_CONFIGURATION_DICT[current_input_channel]] = {
            "high_pass_filter": QuattrocentoHighPassFilter(
                self.input_high_pass_combo_box.currentIndex()
            ),
            "low_pass_filter": QuattrocentoLowPassFilter(
                self.input_low_pass_combo_box.currentIndex()
            ),
            "mode": QuattrocentoDetectionMode(
                self.input_detection_mode_combo_box.currentIndex()
            ),
        }

        self.device.log_info(
            f"Input {current_input_channel} successfully configured.", LoggerLevel.INFO
        )

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
            self._toggle_configuration_group_boxes(False)
        else:
            self.stream_push_button.setText("Stream")
            self.stream_push_button.setChecked(False)
            self.configure_push_button.setEnabled(True)
            self._toggle_configuration_group_boxes(True)
            self.device.log_info("Stopped Streaming")

        self._toggle_configuration_group_boxes()

    def update(self, data: np.ndarray) -> None:
        self.ready_read_signal.emit(data)

    def extract_emg_data(
        self, data: np.ndarray, milli_volts: bool = True
    ) -> np.ndarray:
        """
        Extracts the EMG Signals from the transmitted data.

        Args:
            data (np.ndarray):
                Raw data that got transmitted.

            milli_volts (bool, optional):
                If True, the EMG data is converted to milli volts.
                Defaults to True.

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

    def _initialize_device_params(self) -> None:
        self.device_params = {
            "acquisiton_configuration": {
                "decim_mode": QuattrocentoDecim.ACTIVE,
                "recording_mode": QuattrocentoRecording.STOP,
                "sampling_frequency_mode": QuattrocentoSamplingFrequency.MEDIUM,
                "number_of_channels_mode": QuattrocentoNumberOfChannels.ULTRA,
                "acquisition_mode": QuattrocentoAcquisition.INACTIVE,
            },
            "analog_output_input_selection_configuration": {
                "analog_output_gain": QuattrocentoAnalogOutputGain.LOW,
                "input_selection": QuattrocentoSourceInput.IN_I,
            },
            "analog_output_channel_selection_configuration": {
                "channel_selection": 0,
            },
            "in_top_left_configuration": {
                "muscle": QuattrocentoMuscle.NOT_DEFINED,
                "sensor": QuattrocentoSensor.SIXTYFOUR_ELECTRODE_GRID_10MM,
                "adaptor": QuattrocentoAdaptor.SIXTYFOUR_AD1x64,
                "side": QuattrocentoSide.NOT_DEFINED,
                "high_pass_filter": QuattrocentoHighPassFilter.MEDIUM,
                "low_pass_filter": QuattrocentoLowPassFilter.MEDIUM,
                "mode": QuattrocentoDetectionMode.MONOPOLAR,
            },
            "in_top_right_configuration": {
                "muscle": QuattrocentoMuscle.NOT_DEFINED,
                "sensor": QuattrocentoSensor.SIXTYFOUR_ELECTRODE_GRID_10MM,
                "adaptor": QuattrocentoAdaptor.SIXTYFOUR_AD1x64,
                "side": QuattrocentoSide.NOT_DEFINED,
                "high_pass_filter": QuattrocentoHighPassFilter.MEDIUM,
                "low_pass_filter": QuattrocentoLowPassFilter.MEDIUM,
                "mode": QuattrocentoDetectionMode.MONOPOLAR,
            },
            "multiple_in1_configuration": {
                "muscle": QuattrocentoMuscle.NOT_DEFINED,
                "sensor": QuattrocentoSensor.SIXTYFOUR_ELECTRODE_GRID_10MM,
                "adaptor": QuattrocentoAdaptor.SIXTYFOUR_AD1x64,
                "side": QuattrocentoSide.NOT_DEFINED,
                "high_pass_filter": QuattrocentoHighPassFilter.MEDIUM,
                "low_pass_filter": QuattrocentoLowPassFilter.MEDIUM,
                "mode": QuattrocentoDetectionMode.MONOPOLAR,
            },
            "multiple_in2_configuration": {
                "muscle": QuattrocentoMuscle.NOT_DEFINED,
                "sensor": QuattrocentoSensor.SIXTYFOUR_ELECTRODE_GRID_10MM,
                "adaptor": QuattrocentoAdaptor.SIXTYFOUR_AD1x64,
                "side": QuattrocentoSide.NOT_DEFINED,
                "high_pass_filter": QuattrocentoHighPassFilter.MEDIUM,
                "low_pass_filter": QuattrocentoLowPassFilter.MEDIUM,
                "mode": QuattrocentoDetectionMode.MONOPOLAR,
            },
            "multiple_in3_configuration": {
                "muscle": QuattrocentoMuscle.NOT_DEFINED,
                "sensor": QuattrocentoSensor.SIXTYFOUR_ELECTRODE_GRID_10MM,
                "adaptor": QuattrocentoAdaptor.SIXTYFOUR_AD1x64,
                "side": QuattrocentoSide.NOT_DEFINED,
                "high_pass_filter": QuattrocentoHighPassFilter.MEDIUM,
                "low_pass_filter": QuattrocentoLowPassFilter.MEDIUM,
                "mode": QuattrocentoDetectionMode.MONOPOLAR,
            },
            "multiple_in4_configuration": {
                "muscle": QuattrocentoMuscle.NOT_DEFINED,
                "sensor": QuattrocentoSensor.SIXTYFOUR_ELECTRODE_GRID_10MM,
                "adaptor": QuattrocentoAdaptor.SIXTYFOUR_AD1x64,
                "side": QuattrocentoSide.NOT_DEFINED,
                "high_pass_filter": QuattrocentoHighPassFilter.MEDIUM,
                "low_pass_filter": QuattrocentoLowPassFilter.MEDIUM,
                "mode": QuattrocentoDetectionMode.MONOPOLAR,
            },
            "grids": [i for i in range(6)],
        }


from device_interfaces.devices.quattrocento import QuattrocentoLight
from device_interfaces.gui.ui_compiled.quattrocento_light_widget import (
    Ui_QuattrocentoLightForm,
)


class QuattrocentoLightWidget(QWidget):
    ready_read_signal = Signal(np.ndarray)
    device_connected_signal = Signal(bool)
    device_configured_signal = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.ui = Ui_QuattrocentoLightForm()
        self.ui.setupUi(self)

        self.parent_object = parent

        # Device setup
        self.device = QuattrocentoLight()
        self.device.data_available_signal.connect(self.update)
        self.device_params: dict = {}

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
        self.connection_ip_label = self.ui.connectionIPLabel
        self.connection_port_label = self.ui.connectionPortLabel

        # Acquisition parameters
        self.acquisition_group_box = self.ui.acquisitionGroupBox
        self.acquisition_sampling_frequency_combo_box = (
            self.ui.acquisitionSamplingFrequencyComboBox
        )
        self.acquisition_streaming_frequency_combo_box = (
            self.ui.acquisitionStreamingFrequencyComboBox
        )

        # Grid parameters
        self.grid_selection_group_box = self.ui.gridSelectionGroupBox
        self.grid_selection_check_box_list = [
            self.ui.gridOneCheckBox,
            self.ui.gridTwoCheckBox,
            self.ui.gridThreeCheckBox,
            self.ui.gridFourCheckBox,
            self.ui.gridFiveCheckBox,
            self.ui.gridSixCheckBox,
        ]

        [
            check_box.setChecked(False)
            for check_box in self.grid_selection_check_box_list
        ]
        self.grid_selection_check_box_list[2].setChecked(True)
        self.grid_selection_check_box_list[3].setChecked(True)

        # Configuration parameters
        self.configuration_group_boxes = [
            self.acquisition_group_box,
            self.grid_selection_group_box,
        ]

    def toggle_connection(self):
        if not self.device.is_connected:
            self.connect_push_button.setEnabled(False)

        self.device.toggle_connection(
            (self.connection_ip_label.text(), int(self.connection_port_label.text())),
        )

    def toggle_connected(self, is_connected: bool) -> None:
        self.connect_push_button.setEnabled(True)
        if is_connected:
            self.connect_push_button.setText("Disconnect")
            self.connect_push_button.setChecked(True)
            self.configure_push_button.setEnabled(True)
            self.device.log_info("Connected")
        else:
            self.connect_push_button.setText("Connect")
            self.connect_push_button.setChecked(False)
            self.configure_push_button.setEnabled(False)
            self.stream_push_button.setEnabled(False)
            self.device.log_info("Disconnected")

        self.device_connected_signal.emit(is_connected)

    def configure_device(self) -> None:
        self.device_params["grids"] = [
            i
            for i, check_box in enumerate(self.grid_selection_check_box_list)
            if check_box.isChecked()
        ]
        self.device_params["streaming_frequency"] = (
            QUATTROCENTO_LIGHT_STREAMING_FREQUENCY_DICT[
                QuattrocentoLightStreamingFrequency(
                    self.acquisition_streaming_frequency_combo_box.currentIndex()
                )
            ]
        )
        self.device_params["sampling_frequency"] = (
            QUATTROCENTO_LIGHT_SAMPLING_FREQUENCY_DICT[
                QuattrocentoLightSamplingFrequency(
                    self.acquisition_sampling_frequency_combo_box.currentIndex()
                )
            ]
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
        else:
            self.stream_push_button.setText("Stream")
            self.stream_push_button.setChecked(False)
            self.configure_push_button.setEnabled(True)
            self.device.log_info("Stopped Streaming")

        self._toggle_configuration_group_boxes()

    def update(self, data: Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]) -> None:
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
