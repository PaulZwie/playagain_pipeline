"""
Device class for real-time interfacing the MindRove device.
Developer: Dominik I. Braun
Contact: dome.braun@fau.de
Last Update: 2024-04-02
"""

# Python Libraries
from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING, Union, Dict, Tuple
import numpy as np
from PySide6.QtCore import QTimer

# MindRove libraries
from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds
from mindrove.data_filter import DataFilter, FilterTypes, DetrendOperations

# Local Libraries
from device_interfaces.devices.device import BaseDevice
from device_interfaces.dicts.intan import *
from device_interfaces.enums.intan import *
from device_interfaces.enums.device import Device, CommunicationProtocol

if TYPE_CHECKING:
    from PySide6.QtWidgets import QMainWindow, QWidget


class MindRoveBracelet(BaseDevice):
    """
    MindRove Bracelet class derived from BaseDevice class.

    The MindRove Bracelet is a WiFi (TCP/IP) device for recording and stimulating.
    """

    def __init__(
        self,
        parent: Union[QMainWindow, QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self.name: Device = Device.MINDROVE
        self.communication_protocol: CommunicationProtocol = CommunicationProtocol.TCPIP

        BoardShim.enable_dev_board_logger()
        self.board_parameters = MindRoveInputParams()
        self.board_id = BoardIds.MINDROVE_WIFI_BOARD
        self.board_shim = BoardShim(self.board_id, self.board_parameters)

        self.biosignal_channels = None
        self.aux_channels = None
        self.biosignal_channel_indices: list[int] = None
        self.aux_channel_indices: list[int] = None
        self._conversion_factor = 30000  # Arbitrary!!!

        self.sampling_frequency = None

        self.buffer_size = None

        # Check for new data thread
        self.check_for_new_data_timer = QTimer()
        self.check_for_new_data_timer.timeout.connect(self._check_for_new_data)
        self.time_out = 50  # ms

    def _connect_to_device(self, settings: Tuple[str, int] = None) -> None:
        """
        Open the device connection.
        """
        self.board_shim.prepare_session()
        self.is_connected = True
        self.connected_signal.emit(True)

    def _disconnect_from_device(self) -> None:
        """
        Close the device connection.
        """
        self.board_shim.release_session()
        self.is_connected = False
        self.connected_signal.emit(False)

    def configure_device(
        self, params: Dict[str, Enum | Dict[str, Enum]] = None
    ) -> None:
        """
        Configure the device with the given parameters.
        """

        self.biosignal_channel_indices = BoardShim.get_emg_channels(self.board_id)
        self.aux_channel_indices = BoardShim.get_accel_channels(self.board_id)
        self.biosignal_channels = len(self.biosignal_channel_indices)
        self.aux_channels = len(self.aux_channel_indices)
        self.sampling_frequency = BoardShim.get_sampling_rate(self.board_id)

        self.buffer_size = int(self.sampling_frequency * self.time_out / 1000)

        self.is_configured = True
        self.configured_signal.emit(True)

    def _start_streaming(self) -> None:
        """
        Start the data streaming.
        """
        self.board_shim.start_stream()
        self.check_for_new_data_timer.start(self.time_out)
        self.is_streaming = True
        self.streaming_signal.emit(True)

    def _stop_streaming(self) -> None:
        """
        Stop the data streaming.
        """
        self.board_shim.stop_stream()
        self.check_for_new_data_timer.stop()
        self.is_streaming = False
        self.streaming_signal.emit(False)

    def _check_for_new_data(self) -> None:
        """
        Check for new data from the device.
        """
        if self.board_shim.get_board_data_count() < self.buffer_size:
            return

        self._read_data()

    def _read_data(self) -> None:
        """
        Read the data from the device.
        """
        data = self.board_shim.get_current_board_data(self.buffer_size)
        self._process_data(data)

    def _process_data(self, input: np.ndarray) -> None:
        """
        Process the data from the device.
        """
        for count, channel in enumerate(self.biosignal_channel_indices):
            DataFilter.detrend(input[channel], DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(
                input[channel],
                self.sampling_frequency,
                center_freq=255,
                band_width=490,
                order=2,
                filter_type=FilterTypes.BUTTERWORTH.value,
                ripple=0,
            )
            DataFilter.perform_bandstop(
                input[channel],
                self.sampling_frequency,
                center_freq=50,
                band_width=4,
                order=2,
                filter_type=FilterTypes.BUTTERWORTH.value,
                ripple=0,
            )

        self.data_available_signal.emit(input.astype(np.float32))

    def extract_emg_data(self, data: np.ndarray) -> np.ndarray:
        """
        Extract the EMG data from the input data.

        Args:
            data (np.ndarray): The input data.

        Returns:
            np.ndarray: The extracted EMG data.
        """
        return (
            data[self.biosignal_channel_indices] / self._conversion_factor
            if self.biosignal_channel_indices
            else None
        )

    def extract_aux_data(self, data: np.ndarray, index: int = 0) -> np.ndarray:
        """
        Extract the AUX data from the input data.

        Args:
            data (np.ndarray): The input data.
            index (int): The index of the AUX channel.

        Returns:
            np.ndarray: The extracted AUX data.
        """
        return (
            data[self.aux_channel_indices[index]] if self.aux_channel_indices else None
        )

    def get_device_information(self) -> Dict[str, Enum | int | float | str]:
        """
        Gets the current configuration of the device.

        Returns:
            Dict[str, Enum | int | float | str]:
                Dictionary that holds information about the
                current device configuration and status.
        """

        return {
            "name": self.name,
            "communication_protocol": self.communication_protocol,
            "sampling_frequency": self.sampling_frequency,
            "biosignal_channels": self.biosignal_channels,
            "aux_channels": self.aux_channels,
        }
