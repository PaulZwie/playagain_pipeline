from __future__ import annotations
from typing import Dict, Union, TYPE_CHECKING
from device_interfaces.gui.ui_compiled.muovi_test import Ui_MuoviTest
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QMainWindow, QApplication
import sys
import time
import numpy as np

if TYPE_CHECKING:
    from device_interfaces.gui.template_widgets.devices import DeviceWidget
    from gui_custom_elements.vispy_plot_widget import VispyPlotWidget


class MuoviTest(QMainWindow):
    def __init__(self):
        super().__init__()

        self.ui = Ui_MuoviTest()
        self.ui.setupUi(self)

        # Plot Setup
        self.plot: VispyPlotWidget = self.ui.vispyPlotWidget
        # self.plot.measure_fps()
        self.display_time = 2

        # Device Setup
        self.deviceWidget: DeviceWidget = self.ui.deviceInterfaceWidget
        self.deviceWidget.ready_read_signal.connect(self.update)
        self.deviceWidget.device_changed_signal.connect(self._prepare_plot)
        self.deviceWidget.device_selection_combo_box.setCurrentIndex(5)
        self.deviceWidget.device_connected_signal.connect(self.toggle_device_connected)

        # Initialize
        self._prepare_plot()

        # Timing
        self.buffer = []

    def toggle_device_connected(self, connected: bool):
        if connected:
            self.buffer = []
        else:
            emg = np.hstack([x[1] for x in self.buffer])
            timings = np.array([x[0] for x in self.buffer])
            np.save("data/muovi_test_emg.npy", emg)  # Save data
            np.save("data/muovi_test_timings.npy", timings)  # Save timings

    def update(self, data: np.ndarray):
        # EMG Data
        emg_data = self.ui.deviceInterfaceWidget.extract_emg_data(data)
        self.plot.set_plot_data(emg_data / 1000)

        self.buffer.append((time.time(), emg_data))

    def _prepare_plot(self):
        sampling_frequency = 2000
        lines = 32
        if sampling_frequency and lines:
            self.plot.refresh_plot()
            self.plot.configure_lines_plot(
                self.display_time,
                fs=sampling_frequency,
                lines=lines,
            )

    def closeEvent(self, event: QCloseEvent) -> None:
        self.deviceWidget.closeEvent(event)
        return super().closeEvent(event)


def main():
    appQt = QApplication(sys.argv)

    win = MuoviTest()
    win.show()
    appQt.exec()


if __name__ == "__main__":
    main()
