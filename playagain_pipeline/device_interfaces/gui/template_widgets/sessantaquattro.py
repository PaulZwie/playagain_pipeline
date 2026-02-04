from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Union
from PySide6.QtWidgets import QWidget, QGroupBox
from PySide6.QtCore import Signal
from device_interfaces.devices.sessantaquattro import Sessantaquattro
from device_interfaces.enums.device import LoggerLevel
from device_interfaces.gui.ui_compiled.sessantaquattro_widget import (
    Ui_SessantaquattroForm,
)
from enum import Enum
import numpy as np

if TYPE_CHECKING:
    pass


class SessantaquattroWidget(QWidget):
    ready_read_signal = Signal(np.ndarray)
    device_connected_signal = Signal(bool)
    device_configured_signal = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.ui = Ui_SessantaquattroForm()
        self.ui.setupUi(self)

        self.device = Sessantaquattro()

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
