"""
Class for testing the real-time interface of the sessantaquattro.
Developer: Dominik I. Braun
Contact: dome.braun@fau.de
Last Update: 2023-10-20
"""

from PySide6.QtWidgets import QMainWindow, QApplication
from PySide6.QtGui import QCloseEvent
import sys
from device_interfaces.devices.quattrocento import Quattrocento
from device_interfaces.gui.ui_compiled.quattrocento_test import Ui_MainWindow
from typing import TYPE_CHECKING, Union, Dict, Tuple
from device_interfaces.enums.quattrocento import *
import numpy as np
import time


class SessantaquattroInterface(QMainWindow):
    def __init__(self):
        super(SessantaquattroInterface, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.device = Quattrocento()

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

        self.plot = self.ui.vispyPlotWidget
        self.plot.measure_fps()
        self.device.data_available_signal.connect(self.plot_data)

    def toggle_connection(self):
        ip = "169.254.1.10"
        port = 23456

        self.device.toggle_connection((ip, port))

    def toggle_connected(self, is_connected: bool) -> None:
        if is_connected:
            self.connect_button.setText("Disconnect")
            self.connect_button.setChecked(True)
            self.configure_button.setEnabled(True)
            self.device.log_info("Connected", "INFO")
        else:
            self.connect_button.setText("Connect")
            self.connect_button.setChecked(False)
            self.configure_button.setEnabled(False)

    def configure_device(self) -> None:
        params = {
            "acquisiton_configuration": {
                "decim_mode": QuattrocentoDecim.ACTIVE,
                "recording_mode": QuattrocentoRecording.STOP,
                "sampling_frequency_mode": QuattrocentoSamplingFrequency.MEDIUM,
                "number_of_channels_mode": QuattrocentoNumberOfChannels.LOW,
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
        }

        self.device.configure_device(params)

    def toggle_configured(self, is_configured: bool) -> None:
        if is_configured:
            self.stream_button.setEnabled(True)
            self.device.log_info("Configured")
        else:
            self.device.reset_configuration()

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

    def request_configuration(self) -> None:
        self.device.get_configuration()

    def read_configuration(
        self,
        configuration: Dict[str, int],
    ) -> None:
        self.device.log_info(f"Configuration received: {configuration}", "INFO")

    def plot_data(self, data: np.ndarray) -> None:
        print(data.shape)
        # emg_data = self.device.extract_emg_data(data) / 10
        # self.plot.set_plot_data(emg_data)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.device:
            if self.device.is_streaming:
                self.toggle_streaming()
            if self.device.is_connected:
                self.toggle_connection()

        event.accept()


def main():
    appQt = QApplication(sys.argv)

    win = SessantaquattroInterface()
    win.show()
    appQt.exec()


if __name__ == "__main__":
    main()
