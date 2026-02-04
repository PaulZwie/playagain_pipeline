"""
Class for testing the real-time interface of the sessantaquattro.
Developer: Dominik I. Braun
Contact: dome.braun@fau.de
Last Update: 2023-10-20
"""

from PySide6.QtWidgets import QMainWindow, QApplication
import sys
from device_interfaces.devices.sessantaquattro import Sessantaquattro
from device_interfaces.gui.ui_compiled.sessantaquattro_test import Ui_MainWindow
from typing import TYPE_CHECKING, Union, Dict, Tuple
from device_interfaces.enums.sessantaquattro import *
import numpy as np
import time


class SessantaquattroInterface(QMainWindow):
    def __init__(self):
        super(SessantaquattroInterface, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.device = Sessantaquattro()

        # Push Buttons
        self.connect_button = self.ui.connectDevicePushButton
        self.connect_button.clicked.connect(self.toggle_connection)
        self.device.connected_signal.connect(self.toggle_connected)

        self.configure_button = self.ui.configurePushButton
        self.configure_button.clicked.connect(self.configure_device)
        self.configure_button.setEnabled(False)
        self.device.configured_signal.connect(self.toggle_configured)

        self.request_configuration_button = self.ui.requestConfigurationPushButton
        self.request_configuration_button.clicked.connect(self.request_configuration)
        self.request_configuration_button.setEnabled(False)
        self.device.configuration_available_signal.connect(self.read_configuration)

        self.reset_configuration_button = self.ui.resetConfigurationPushButton
        self.reset_configuration_button.clicked.connect(self.device.reset_configuration)

        self.stream_button = self.ui.streamPushButton
        self.stream_button.clicked.connect(self.toggle_streaming)
        self.stream_button.setEnabled(False)

        self.plot = self.ui.vispyPlotWidget
        sampling_frequency = 500
        self.plot.configure_lines_plot(
            frame_len=48, display_time=1, fs=sampling_frequency, lines=8
        )
        self.plot.measure_fps()
        self.device.data_available_signal.connect(self.plot_data)

    def toggle_connection(self):
        ip = "0.0.0.0"  # "192.168.246.227"
        port = 45454
        self.device.toggle_connection((ip, port))

    def toggle_connected(self, is_connected: bool) -> None:
        if is_connected:
            self.connect_button.setText("Disconnect")
            self.connect_button.setChecked(True)
            self.configure_button.setEnabled(True)
            self.request_configuration_button.setEnabled(True)
            self.device.log_info("Connected", "INFO")
        else:
            self.connect_button.setText("Connect")
            self.connect_button.setChecked(False)
            self.configure_button.setEnabled(False)
            self.request_configuration_button.setEnabled(False)

    def configure_device(self) -> None:
        params = {
            "action_mode": SSQTAction.SET,
            "sampling_frequency_mode": SSQTSamplingFrequency.LOW,
            "number_of_channels_mode": SSQTChannels.LOW,
            "acquisition_mode": SSQTMode.TEST,
            "resolution_mode": SSQTResolution.LOW,
            "filtering_mode": SSQTFilter.ON,
            "gain_mode": SSQTGain.DEFAULT,
            "trigger_mode": SSQTTrigger.DEFAULT,
            "recording_mode": SSQTRecording.STOP,
            "transmission_mode": SSQTTransmission.STOP,
        }

        self.device.configure_device(params)

    def toggle_configured(self, is_configured: bool) -> None:
        if is_configured:
            self.stream_button.setEnabled(True)
            self.device.log_info("Configured", "INFO")
        else:
            self.device.reset_configuration()

    def toggle_streaming(self) -> None:
        self.device.toggle_streaming()
        if self.device.is_streaming:
            self.stream_button.setText("Stop Streaming")
            self.stream_button.setChecked(True)
            self.configure_button.setEnabled(False)
            self.request_configuration_button.setEnabled(False)

            self.device.log_info("Streaming", "INFO")
        else:
            self.stream_button.setText("Stream")
            self.stream_button.setChecked(False)
            self.configure_button.setEnabled(True)
            self.request_configuration_button.setEnabled(True)

            self.device.log_info("Stopped Streaming", "INFO")

    def request_configuration(self) -> None:
        self.device.get_configuration()

    def read_configuration(
        self,
        configuration: Dict[str, int],
    ) -> None:
        self.device.log_info(f"Configuration received: {configuration}", "INFO")

    def plot_data(self, data: np.ndarray) -> None:
        emg_data = self.device.extract_emg_data(data, milli_volts=True) / 10
        self.plot.set_plot_data(emg_data)


def main():
    appQt = QApplication(sys.argv)

    win = SessantaquattroInterface()
    win.show()
    appQt.exec()


if __name__ == "__main__":
    main()
