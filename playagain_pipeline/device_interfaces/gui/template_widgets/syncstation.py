from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Union
from PySide6.QtWidgets import QWidget, QGroupBox, QComboBox, QCheckBox
from PySide6.QtCore import Signal
from device_interfaces.devices.syncstation import SyncStation
from device_interfaces.dicts.syncstation import *
from device_interfaces.enums.syncstation import *
from device_interfaces.enums.device import LoggerLevel
from device_interfaces.gui.ui_compiled.syncstation_widget import (
    Ui_SyncStationForm,
)
import numpy as np
from functools import partial

if TYPE_CHECKING:
    pass


class SyncStationWidget(QWidget):
    ready_read_signal = Signal(np.ndarray)
    device_connected_signal = Signal(bool)
    device_configured_signal = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.ui = Ui_SyncStationForm()
        self.ui.setupUi(self)

        self.parent_object = parent

        # Device Setup
        self.device = SyncStation()
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
        self.connection_ip_address_label = self.ui.connectionIPAddressLabel
        self.connection_port_label = self.ui.connectionPortLabel

        # Input parameters
        self.input_parameters_group_box = self.ui.inputGroupBox
        self.input_working_mode_combo_box = self.ui.inputWorkingModeComboBox
        self.input_working_mode_combo_box.setCurrentIndex(1)

        self.probes_tab_widget = self.ui.probesTabWidget
        self.probes_tab_widget.setCurrentIndex(0)
        self.probes_dict: Dict[
            SyncStationProbeConfigMode, dict[str, Union[QComboBox, QCheckBox]]
        ] = self._configure_probes_dict()

        for key, value in self.probes_dict.items():
            value["enable_probe"].stateChanged.connect(
                partial(self._update_probe_params, key)
            )

        self._set_default_probe_params()

        # Configuration parameters
        self.configuration_group_boxes: list[QGroupBox] = [
            self.input_parameters_group_box,
        ]

    def toggle_connection(self):
        if not self.device.is_connected:
            self.connect_button.setEnabled(False)

        self.device.toggle_connection(
            (
                self.connection_ip_address_label.text(),
                int(self.connection_port_label.text()),
            )
        )

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

        self.connection_group_box.setEnabled(not self.connection_group_box.isEnabled())
        self.device_connected_signal.emit(is_connected)

    def configure_device(self) -> None:
        self.device_params["working_mode"] = SyncStationWorkingMode(
            self.input_working_mode_combo_box.currentIndex()
        )

        count_enabled = 0
        for key, value in self.probes_dict.items():
            is_enabled = value["enable_probe"].isChecked()
            self.device_params["optional_bytes_configuration_A"][key][
                "enable_probe"
            ] = SyncStationEnableProbeMode(int(is_enabled))
            self.device_params["optional_bytes_configuration_A"][key][
                "detection_mode"
            ] = SyncStationDetectionMode(value["detection_mode"].currentIndex())

            if is_enabled:
                count_enabled += 1

        self.device_params["optional_bytes_a_mode"] = SyncStationOptionalBytesAMode(
            count_enabled
        )

        self.device_params["optional_bytes_configuration_B"] = {
            "latency": 200,
        }

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
            "rec_on_mode": SyncStationRecOnMode.OFF,
            "acqusition_mode": SyncStationAcquisitionMode.STOP,
            "working_mode": SyncStationWorkingMode.EMG,
            "optional_bytes_a_mode": SyncStationOptionalBytesAMode.FOUR,
            "optional_bytes_configuration_A": {
                SyncStationProbeConfigMode.MUOVI_PROBE_ONE: {
                    "detection_mode": SyncStationDetectionMode.MONOPOLAR_GAIN_LOW,
                    "enable_probe": SyncStationEnableProbeMode.ENABLE,
                },
                SyncStationProbeConfigMode.MUOVI_PROBE_TWO: {
                    "detection_mode": SyncStationDetectionMode.MONOPOLAR_GAIN_LOW,
                    "enable_probe": SyncStationEnableProbeMode.ENABLE,
                },
                SyncStationProbeConfigMode.MUOVI_PROBE_THREE: {
                    "detection_mode": SyncStationDetectionMode.MONOPOLAR_GAIN_LOW,
                    "enable_probe": SyncStationEnableProbeMode.DISABLE,
                },
                SyncStationProbeConfigMode.MUOVI_PROBE_FOUR: {
                    "detection_mode": SyncStationDetectionMode.MONOPOLAR_GAIN_LOW,
                    "enable_probe": SyncStationEnableProbeMode.DISABLE,
                },
                SyncStationProbeConfigMode.MUOVI_PLUS_PROBE_ONE: {
                    "detection_mode": SyncStationDetectionMode.MONOPOLAR_GAIN_LOW,
                    "enable_probe": SyncStationEnableProbeMode.DISABLE,
                },
                SyncStationProbeConfigMode.MUOVI_PLUS_PROBE_TWO: {
                    "detection_mode": SyncStationDetectionMode.MONOPOLAR_GAIN_LOW,
                    "enable_probe": SyncStationEnableProbeMode.DISABLE,
                },
                SyncStationProbeConfigMode.DUE_PLUS_PROBE_ONE: {
                    "detection_mode": SyncStationDetectionMode.MONOPOLAR_GAIN_LOW,
                    "enable_probe": SyncStationEnableProbeMode.DISABLE,
                },
                SyncStationProbeConfigMode.DUE_PLUS_PROBE_TWO: {
                    "detection_mode": SyncStationDetectionMode.MONOPOLAR_GAIN_LOW,
                    "enable_probe": SyncStationEnableProbeMode.DISABLE,
                },
                SyncStationProbeConfigMode.DUE_PLUS_PROBE_THREE: {
                    "detection_mode": SyncStationDetectionMode.MONOPOLAR_GAIN_LOW,
                    "enable_probe": SyncStationEnableProbeMode.DISABLE,
                },
                SyncStationProbeConfigMode.DUE_PLUS_PROBE_FOUR: {
                    "detection_mode": SyncStationDetectionMode.MONOPOLAR_GAIN_LOW,
                    "enable_probe": SyncStationEnableProbeMode.DISABLE,
                },
                SyncStationProbeConfigMode.DUE_PLUS_PROBE_FIVE: {
                    "detection_mode": SyncStationDetectionMode.MONOPOLAR_GAIN_LOW,
                    "enable_probe": SyncStationEnableProbeMode.DISABLE,
                },
                SyncStationProbeConfigMode.DUE_PLUS_PROBE_SIX: {
                    "detection_mode": SyncStationDetectionMode.MONOPOLAR_GAIN_LOW,
                    "enable_probe": SyncStationEnableProbeMode.DISABLE,
                },
                SyncStationProbeConfigMode.DUE_PLUS_PROBE_SEVEN: {
                    "detection_mode": SyncStationDetectionMode.MONOPOLAR_GAIN_LOW,
                    "enable_probe": SyncStationEnableProbeMode.DISABLE,
                },
                SyncStationProbeConfigMode.DUE_PLUS_PROBE_EIGHT: {
                    "detection_mode": SyncStationDetectionMode.MONOPOLAR_GAIN_LOW,
                    "enable_probe": SyncStationEnableProbeMode.DISABLE,
                },
                SyncStationProbeConfigMode.DUE_PLUS_PROBE_NINE: {
                    "detection_mode": SyncStationDetectionMode.MONOPOLAR_GAIN_LOW,
                    "enable_probe": SyncStationEnableProbeMode.DISABLE,
                },
                SyncStationProbeConfigMode.DUE_PLUS_PROBE_TEN: {
                    "detection_mode": SyncStationDetectionMode.MONOPOLAR_GAIN_LOW,
                    "enable_probe": SyncStationEnableProbeMode.DISABLE,
                },
            },
            "optional_bytes_configuration_B": {"latency": 200},
        }

    def _configure_probes_dict(self):
        return {
            SyncStationProbeConfigMode.MUOVI_PROBE_ONE: {
                "enable_probe": self.ui.muoviProbeOneEnableCheckBox,
                "detection_mode": self.ui.muoviProbeOneDetectionModeComboBox,
            },
            SyncStationProbeConfigMode.MUOVI_PROBE_TWO: {
                "enable_probe": self.ui.muoviProbeTwoEnableCheckBox,
                "detection_mode": self.ui.muoviProbeTwoDetectionModeComboBox,
            },
            SyncStationProbeConfigMode.MUOVI_PROBE_THREE: {
                "enable_probe": self.ui.muoviProbeThreeEnableCheckBox,
                "detection_mode": self.ui.muoviProbeThreeDetectionModeComboBox,
            },
            SyncStationProbeConfigMode.MUOVI_PROBE_FOUR: {
                "enable_probe": self.ui.muoviProbeFourEnableCheckBox,
                "detection_mode": self.ui.muoviProbeFourDetectionModeComboBox,
            },
            SyncStationProbeConfigMode.MUOVI_PLUS_PROBE_ONE: {
                "enable_probe": self.ui.muoviPlusProbeOneEnableCheckBox,
                "detection_mode": self.ui.muoviPlusProbeOneDetectionModeComboBox,
            },
            SyncStationProbeConfigMode.MUOVI_PLUS_PROBE_TWO: {
                "enable_probe": self.ui.muoviPlusProbeTwoEnableCheckBox,
                "detection_mode": self.ui.muoviPlusProbeTwoDetectionModeComboBox,
            },
            SyncStationProbeConfigMode.DUE_PLUS_PROBE_ONE: {
                "enable_probe": self.ui.duePlusProbeOneEnableCheckBox,
                "detection_mode": self.ui.duePlusProbeOneDetectionModeComboBox,
            },
            SyncStationProbeConfigMode.DUE_PLUS_PROBE_TWO: {
                "enable_probe": self.ui.duePlusProbeTwoEnableCheckBox,
                "detection_mode": self.ui.duePlusProbeTwoDetectionModeComboBox,
            },
            SyncStationProbeConfigMode.DUE_PLUS_PROBE_THREE: {
                "enable_probe": self.ui.duePlusProbeThreeEnableCheckBox,
                "detection_mode": self.ui.duePlusProbeThreeDetectionModeComboBox,
            },
            SyncStationProbeConfigMode.DUE_PLUS_PROBE_FOUR: {
                "enable_probe": self.ui.duePlusProbeFourEnableCheckBox,
                "detection_mode": self.ui.duePlusProbeFourDetectionModeComboBox,
            },
            SyncStationProbeConfigMode.DUE_PLUS_PROBE_FIVE: {
                "enable_probe": self.ui.duePlusProbeFiveEnableCheckBox,
                "detection_mode": self.ui.duePlusProbeFiveDetectionModeComboBox,
            },
            SyncStationProbeConfigMode.DUE_PLUS_PROBE_SIX: {
                "enable_probe": self.ui.duePlusProbeSixEnableCheckBox,
                "detection_mode": self.ui.duePlusProbeSixDetectionModeComboBox,
            },
            SyncStationProbeConfigMode.DUE_PLUS_PROBE_SEVEN: {
                "enable_probe": self.ui.duePlusProbeSevenEnableCheckBox,
                "detection_mode": self.ui.duePlusProbeSevenDetectionModeComboBox,
            },
            SyncStationProbeConfigMode.DUE_PLUS_PROBE_EIGHT: {
                "enable_probe": self.ui.duePlusProbeEightEnableCheckBox,
                "detection_mode": self.ui.duePlusProbeEightDetectionModeComboBox,
            },
            SyncStationProbeConfigMode.DUE_PLUS_PROBE_NINE: {
                "enable_probe": self.ui.duePlusProbeNineEnableCheckBox,
                "detection_mode": self.ui.duePlusProbeNineDetectionModeComboBox,
            },
            SyncStationProbeConfigMode.DUE_PLUS_PROBE_TEN: {
                "enable_probe": self.ui.duePlusProbeTenEnableCheckBox,
                "detection_mode": self.ui.duePlusProbeTenDetectionModeComboBox,
            },
        }

    def _update_probe_params(self, probe: SyncStationProbeConfigMode, state) -> None:
        self.probes_dict[probe]["detection_mode"].setEnabled(state == 2)

    def _set_default_probe_params(self) -> None:
        for values in self.probes_dict.values():
            values["detection_mode"].setCurrentIndex(1)
            values["enable_probe"].setChecked(True)
            values["enable_probe"].setChecked(False)

        self.probes_dict[SyncStationProbeConfigMode.MUOVI_PROBE_ONE][
            "enable_probe"
        ].setChecked(True)

        self.probes_dict[SyncStationProbeConfigMode.MUOVI_PROBE_TWO][
            "enable_probe"
        ].setChecked(True)
