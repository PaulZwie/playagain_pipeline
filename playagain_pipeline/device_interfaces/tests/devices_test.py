from __future__ import annotations
from typing import TYPE_CHECKING
from device_interfaces.enums.device import Device
from device_interfaces.gui.ui_compiled.devices_test import Ui_DevicesTest
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QMainWindow, QApplication
import sys
import time
import numpy as np
import os

if TYPE_CHECKING:
    from device_interfaces.gui.template_widgets.devices import DeviceWidget
    from gui_custom_elements.vispy_plot_widget import VispyPlotWidget


class DevicesTest(QMainWindow):
    def __init__(self):
        super().__init__()

        self.ui = Ui_DevicesTest()
        self.ui.setupUi(self)

        # Plot Setup
        self.plot: VispyPlotWidget = self.ui.vispyPlotWidget
        self.plot.refresh_plot()
        self.plot_toggle_check_box = self.ui.vispyPlotToggleCheckBox
        self.plot_toggle_check_box.styleSheet = (
            "QCheckBox{background-color: rgba(255, 255, 255, 0.2); color: black;}"
        )

        self.display_time = 5
        self.sampling_frequency: int = None
        self.lines: int = None

        # Device Setup
        self.device_widget: DeviceWidget = self.ui.deviceWidget
        self.device_widget.ready_read_signal.connect(self.update)
        self.device_widget.device_connected_signal.connect(self.toggle_device_connected)
        self.device_widget.device_configured_signal.connect(self._prepare_plot)
        self.device_widget.device_selection_combo_box.setCurrentIndex(
            Device.MUOVI.value
        )

        # Timing
        self.buffer = []
        self.start_time = time.time()

    def toggle_device_connected(self, connected: bool):
        print("Device connected:", connected)
        if connected:
            self.buffer = []
        else:
            if len(self.buffer) > 0:
                emg = np.hstack([x[1] for x in self.buffer])
                timings = np.array([x[0] for x in self.buffer])

                if not os.path.exists("data"):
                    os.makedirs("data")
                np.save("data/device_test_emg.npy", emg)  # Save data
                np.save("data/device_test_timings.npy", timings)  # Save timings

                print("Data saved")

    def update(self, data: np.ndarray):
        # EMG Data
        # print(f"FPS: {1 / (time.time() - self.start_time + 1e-10)}")
        self.start_time = time.time()
        # print("Data shape:", data.shape)
        emg_data = self.ui.deviceWidget.extract_emg_data(data)
        # print("EMG shape:", emg_data.shape)
        # print(emg_data)

        if self.plot_toggle_check_box.isChecked():
            self.plot.set_plot_data(emg_data[: self.lines])

        self.buffer.append((time.time(), emg_data))
        print(f"Buffer length: {len(self.buffer)}")

    def _prepare_plot(self, configured: bool):
        device_information = self.device_widget.get_device_information()
        self.device_widget.get_current_widget().device.log_info(device_information)
        self.sampling_frequency = device_information["sampling_frequency"]
        self.lines = device_information["biosignal_channels"]

        self.buffer = []

        if self.lines > 32:
            self.lines = 32

        self.plot.refresh_plot()
        self.plot.configure_lines_plot(
            self.display_time,
            fs=self.sampling_frequency,
            lines=self.lines,
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        self.device_widget.closeEvent(event)
        return super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    win = DevicesTest()
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
