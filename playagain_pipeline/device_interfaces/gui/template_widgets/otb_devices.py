from __future__ import annotations
from typing import TYPE_CHECKING, Union, Dict
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QWidget
from device_interfaces.enums.device import OTBDevice
from device_interfaces.gui.ui_compiled.otb_devices_widget import Ui_OTBDeviceWidgetForm
from PySide6.QtCore import Signal
import numpy as np
from enum import Enum

if TYPE_CHECKING:
    from device_interfaces.gui.template_widgets.quattrocento import (
        QuattrocentoLightWidget,
    )
    from device_interfaces.gui.template_widgets.muovi import MuoviWidget
    from device_interfaces.gui.template_widgets.muovi_plus import MuoviPlusWidget


class OTBDeviceWidget(QWidget):
    ready_read_signal = Signal(np.ndarray)
    device_connected_signal = Signal(bool)
    device_configured_signal = Signal(bool)
    device_changed_signal = Signal(None)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.ui = Ui_OTBDeviceWidgetForm()
        self.ui.setupUi(self)

        self.parent_object = parent

        self.device_stacked_widget = self.ui.deviceStackedWidget

        self.device_selection_combo_box = self.ui.deviceSelectionComboBox
        self.device_selection_combo_box.currentIndexChanged.connect(
            self.update_stacked_widget
        )

        # self.device_connected_signal.connect(self.test)

        self.device_selection_combo_box.setCurrentIndex(
            OTBDevice.QUATTROCENTO_LIGHT.value
        )

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
        return self.get_current_widget().extract_emg_data(data, milli_volts)

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
        return self.get_current_widget().extract_aux_data(data, index)

    def get_device_information(self) -> Dict[str, Enum | int | float | str]:
        """
        Gets the current configuration of the device.

        Returns:
            Dict[str, Enum | int | float | str]:
                Dictionary that holds information about the
                current device configuration and status.
        """

        return self.get_current_widget().get_device_information()

    def update_stacked_widget(self, index: int):
        current_widget = self.get_current_widget()
        current_widget.force_disconnect()
        try:
            current_widget.ready_read_signal.disconnect(self.update)
            current_widget.device_connected_signal.disconnect(
                self.device_connected_signal
            )
            current_widget.device_configured_signal.disconnect(
                self.device_configured_signal
            )

        except Exception:
            pass

        self.device_stacked_widget.setCurrentIndex(index)
        current_widget = self.get_current_widget()
        current_widget.ready_read_signal.connect(self.update)
        current_widget.device_connected_signal.connect(self.device_connected_signal)
        current_widget.device_configured_signal.connect(self.device_configured_signal)

        self.device_changed_signal.emit()

    def get_current_widget(self) -> Union[
        QuattrocentoLightWidget,
        MuoviWidget,
        MuoviPlusWidget,
    ]:
        return self.device_stacked_widget.currentWidget()

    def closeEvent(self, event: QCloseEvent) -> None:
        self.get_current_widget().closeEvent(event)
