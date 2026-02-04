"""
Device class for real-time interfacing the Muovi device.
Developer: Dominik I. Braun
Contact: dome.braun@fau.de
Last Update: 2024-01-09
"""

# Python Libraries
from __future__ import annotations
from typing import TYPE_CHECKING, Union, Dict, Tuple
from PySide6.QtNetwork import QTcpSocket, QTcpServer, QHostAddress
from PySide6.QtCore import QByteArray, QTimer
import numpy as np

# Local Libraries
from device_interfaces.devices.device import BaseDevice
from device_interfaces.dicts.muovi import *
from device_interfaces.enums.muovi import *
from device_interfaces.enums.device import Device, CommunicationProtocol, LoggerLevel

if TYPE_CHECKING:
    # Python Libraries
    from PySide6.QtWidgets import QMainWindow, QWidget
    from PySide6.QtNetwork import QAbstractSocket


class Muovi(BaseDevice):
    def __init__(
        self,
        is_muovi_plus: bool,
        parent: Union[QMainWindow, QWidget] = None,
    ) -> None:
        """
        Muovi device class derived from BaseDevice class.

        Args:
            is_muovi_plus (bool):
                True if the device is a Muovi Plus, False if not.

            parent (Union[QMainWindow, QWidget], optional):
                Parent widget to which the device is assigned to.
                Defaults to None.

        The Muovi class is using a TCP/IP protocol to communicate with the device.

        Communication Protocol:
        1 control byte sequences are sent to the device to start the data transfer.
        Control Byte 1:
        [<0> <0> <0> <0> <WORKING_MODE> <DETECTION_MODE1> <DETECTION_MODE0> <STREAM>]

        Descriptions can be found in device_interfaces/enums/Muovi.py
        """
        super().__init__(parent)

        # Device Parameters
        self.name: Device = Device.MUOVI if not is_muovi_plus else Device.MUOVI_PLUS
        self.communication_protocol: CommunicationProtocol = CommunicationProtocol.TCPIP

        # Configuration modes
        self.working_mode: MuoviWorkingMode = None
        self.detection_mode: MuoviDetectionMode = None
        self.streaming_mode: MuoviStream = None

        # Configuration parameters
        self.sampling_frequency: int = None
        self.bytes_in_sample: int = None
        self.number_of_channels: int = None
        self.frame_len: int = None
        self.buffer_size: int = None
        self.command: int = None
        self._conversion_factor: float = 0.000286

        # Define socket
        self.interface: QTcpServer = None
        self.socket: QTcpSocket = None
        self.received_bytes: QByteArray = bytearray()

        # Timer for connection timeout
        self.timeout_timer: QTimer = QTimer()
        self.timeout_timer.setSingleShot(True)
        self.timeout_timer.timeout.connect(self._server_connection_timeout)
        self.timeout_timer.setInterval(1000)

    def _reset_device_parameters(self) -> Dict[str, Union[Enum, Dict[str, Enum]]]:
        """
        Resets the device parameters to Default values.

        Returns:
            Dict[str, Union[Enum, Dict[str, Enum]]]:
                Default values of the device attributes.

                The first one are the attributes (configuration mode) name,
                and the second its respective value.
        """

        params = {
            "working_mode": MuoviWorkingMode.EMG,
            "detection_mode": MuoviDetectionMode.MONOPOLAR_GAIN_8,
            "streaming_mode": MuoviStream.STOP,
        }

        self._update_configuration_parameters(params)

        return params

    def _connect_to_device(self, settings: Tuple[str, int] = None) -> None:
        """
        Function to attempt a connection to the devices

        Args:
            settings (Tuple[str, int], optional):
                If CommunicationProtocol.TCPIP:
                Tuple[0] = IP -> string
                Tuple[1] = Port -> int

                If CommunicationProtocol.SERIAL pr CommunicationProtocol.USB:
                Tuple[0] = COM Port -> string
                Tuple[1] = Baudrate -> int

                Defaults to None.
        """
        self.interface = QTcpServer(self)
        if not self.interface.listen(QHostAddress(settings[0]), settings[1]):
            self.log_info("Could not connect to server address", LoggerLevel.ERROR)
        else:
            self.log_info(
                f"Server is listening to {settings[0]}:{settings[1]}", LoggerLevel.INFO
            )

        self.interface.newConnection.connect(self._make_request)
        self.interface.acceptError.connect(self._server_error)

        self.timeout_timer.start()

    def _server_error(self, error: QAbstractSocket.SocketError) -> None:
        self.log_info(error, LoggerLevel.ERROR)

    def _server_connection_timeout(self) -> None:
        self.log_info("Connection to server timed out.", LoggerLevel.ERROR)
        self._disconnect_from_device()

    def _make_request(self, settings: Tuple[str, int] = None) -> bool:
        """
        Requests a connection or checks if someone connected to the server.
        After connection is successful, the Signal connected_signal emits True
        and sets the current state is_connected to True.

        In the case of Muovi:
        The Muovi resets itself after receiving a new configuration.
        Therefore, after reconnection and if is_connection is true, the state
        is_configured is set to true. Furthermore, the Signal configured_signal
        emits True.

        Args:
            settings (Tuple[str, int], optional):
                If CommunicationProtocol.TCPIP:
                Tuple[0] = IP -> string
                Tuple[1] = Port -> int

                If CommunicationProtocol.SERIAL pr CommunicationProtocol.USB:
                Tuple[0] = COM Port -> string
                Tuple[1] = Baudrate -> int

                Defaults to None.

        Returns:
            bool:
                Returns True if request was successfully. False if not.
        """
        self.socket = self.interface.nextPendingConnection()
        if self.socket:
            self.socket.readyRead.connect(self._read_data)
            if not self.is_connected:
                self.is_connected = True
                self.connected_signal.emit(True)
                self.timeout_timer.stop()
                self.log_info(f"{self.name.name} has successfully connected!")
            elif not self.is_configured:
                self.is_configured = True
                self.configured_signal.emit(True)

    def _disconnect_from_device(self) -> None:
        """
        Closes the connection to the device.

        self.interface closes and is set to None.
        Device state is_connected is set to False.
        Signal connected_signal emits False.
        """

        if self.socket:
            self.socket.close()
            self.socket = None
        if self.interface:
            self.interface.close()
            self.interface = None
        self.is_connected = False
        self.is_configured = False
        self.connected_signal.emit(False)

    def configure_device(self, params: Dict[str, Union[Enum, Dict[str, Enum]]]) -> None:
        """
        Sends a configuration byte sequence based on selected params to the device.
        An overview of possible configurations can be seen in enums/{device}.

        E.g., enums/sessantaquattro.py


        Args:
            params (Dict[str, Union[Enum, Dict[str, Enum]]]):
                Dictionary that holds the configuration settings
                to which the device should be configured to.

                The first one should be the attributes (configuration mode) name,
                and the second its respective value. Orient yourself on the
                enums of the device to choose the correct configuration settings.
        """
        self._update_configuration_parameters(params)

        # Check if detection mode is valid for working mode (Case EEG -> MONOPOLAR_GAIN_4 => MONOPOLAR_GAIN_8)
        if self.working_mode == MuoviWorkingMode.EEG:
            self.detection_mode = (
                MuoviDetectionMode.MONOPOLAR_GAIN_8
                if self.detection_mode == MuoviDetectionMode.MONOPOLAR_GAIN_4
                else self.detection_mode
            )

        # Set configuration parameters for data transfer
        self.sampling_frequency = MUOVI_WORKING_MODE_CHARACTERISTICS_DICT[
            self.working_mode
        ]["sampling_frequency"]
        self.bytes_in_sample = MUOVI_WORKING_MODE_CHARACTERISTICS_DICT[
            self.working_mode
        ]["bytes_per_sample"]
        self.frame_len = MUOVI_FRAME_LEN_DICT[self.name][self.working_mode]
        if self.name == Device.MUOVI:
            self.number_of_channels = MuoviAvailableChannels.ALL.value
            self.biosignal_channels = MuoviAvailableChannels.BIOSIGNALS.value
            self.aux_channels = MuoviAvailableChannels.AUXILIARY.value
        elif self.name == Device.MUOVI_PLUS:
            self.number_of_channels = MuoviPlusAvailableChannels.ALL.value
            self.biosignal_channels = MuoviPlusAvailableChannels.BIOSIGNALS.value
            self.aux_channels = MuoviPlusAvailableChannels.AUXILIARY.value

        self.buffer_size = (
            self.number_of_channels * self.frame_len * self.bytes_in_sample
        )

        self._configure_command()

        success = self._send_configuration_to_device()

    def _send_configuration_to_device(self) -> bool:
        command_bytes = self._integer_to_bytes(self.command)
        success = self.socket.write(command_bytes)
        self.socket.waitForBytesWritten(1000)

        if success:
            self.log_info(
                f"{self.name.name} configured with: {command_bytes} - {self.command}"
            )

            return True

        return False

    def _configure_command(self) -> None:
        self.command = self.working_mode.value << 3
        self.command += self.detection_mode.value << 1
        self.command += self.streaming_mode.value

        # self.log_info(f"Command: {bin(self.command)}")

    def _update_configuration_parameters(
        self, params: Dict[str, Union[Enum, Dict[str, Enum]]]
    ) -> None:
        """
        Updates the device attributes with the new configuration parameters.

        Args:
            params (Dict[str, Union[Enum, Dict[str, Enum]]]):
                Dictionary that holds the configuration settings
                to which the device should be configured to.

                The first one should be the attributes (configuration mode) name,
                and the second its respective value. Orient yourself on the
                enums of the device to choose the correct configuration settings.
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.log_info(
                    f"Attribute '{key}' not found in the class of {self.name.name}",
                    LoggerLevel.ERROR,
                )

    def get_configuration(self) -> None:
        """
        Sends the command to get the current configuration of the device.
        """
        self.log_info(
            f"{self.name.name} does not have a command to get the configuration.",
            LoggerLevel.WARNING,
        )

    def _start_streaming(self) -> None:
        """
        Sends the command to start the streaming to the device.

        if successful:
            Device state is_streaming is set to True.
            Signal streaming_signal emits True.
        """
        if self.command:
            self.command += 1
            self._send_configuration_to_device()
            self.is_streaming = True
            self.streaming_signal.emit(True)

    def _stop_streaming(self) -> None:
        """
        Sends the command to stop the streaing to the device

        if successful:
            Device state is_streaming is set to False.
            Signal streaming_signal emits False.
        """
        if self.command:
            self.command -= 1
            self._send_configuration_to_device()
            self.is_streaming = False
            self.streaming_signal.emit(False)

    def clear_socket(self) -> None:
        """Reads all the bytes from the buffer."""
        self.socket.readAll()

    def _read_data(self) -> None:
        """
        This function is called when bytes are ready to be read in the buffer.
        After reading the bytes from the buffer, _process_data is called to
        decode and process the raw data.
        """
        if not self.is_streaming:
            packet = self.socket.readAll()

        else:
            while self.socket.bytesAvailable() > self.buffer_size:
                packet = self.socket.read(self.buffer_size)
                if not packet:
                    self.received_bytes = bytearray()
                    continue

                self.received_bytes.extend(packet)

                if len(self.received_bytes) % self.buffer_size == 0:
                    self._process_data(self.received_bytes)
                    self.received_bytes = bytearray()

                else:
                    self.received_bytes = bytearray()

    def _process_data(self, input: bytearray, configuration: bool = False) -> None:
        """
        Decodes the transmitted bytes and convert them to respective
        output format (e.g., mV).

        Emits the processed data through the Signal data_available_signal
        which can be connected to a function using:
        {self.device}.data_available_signal.connect(your_custom_function).

        This works perfectly fine outside of this class.

        Your custom function your_custom_function needs to have a parameter
        "data" which is of type np.ndarray.

        Args:
            input (QByteArray):
                Bytearray of the transmitted raw data.
        """
        if configuration:
            # TODO How are they encoded and what do the bytes represent? -> Check with OT Bioelettronica
            pass  # Is not possible with Muovi

        else:
            data = self._bytes_to_integers(input)

            data = data.reshape(self.number_of_channels, -1, order="F").astype(
                np.double
            )

            self.data_available_signal.emit(data)

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
        if self.number_of_channels is None or self.number_of_channels == 0:
            return None

        match self.name:
            case Device.MUOVI:
                return (
                    data[
                        : self.number_of_channels
                        - MuoviAvailableChannels.AUXILIARY.value
                    ]
                    * self._conversion_factor
                    if milli_volts
                    else data[
                        : self.number_of_channels
                        - MuoviAvailableChannels.AUXILIARY.value
                    ]
                )
            case Device.MUOVI_PLUS:
                return (
                    data[
                        : self.number_of_channels
                        - MuoviPlusAvailableChannels.AUXILIARY.value
                    ]
                    * self._conversion_factor
                    if milli_volts
                    else data[
                        : self.number_of_channels
                        - MuoviPlusAvailableChannels.AUXILIARY.value
                    ]
                )

    def extract_aux_data(self, data: np.ndarray, index: int = 0) -> np.ndarray:
        match self.name:
            case Device.MUOVI:
                return data[MuoviAvailableChannels.BIOSIGNALS.value + index]
            case Device.MUOVI_PLUS:
                return data[MuoviPlusAvailableChannels.BIOSIGNALS.value + index]

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
            "samples_per_frame": self.frame_len,
            "biosignal_channels": self.biosignal_channels,
            "aux_channels": self.aux_channels,
        }

    # Convert integer to bytes
    def _integer_to_bytes(self, command: int) -> bytes:
        return int(command).to_bytes(1, byteorder="big")

    # Convert channels from bytes to integers
    def _bytes_to_integers(
        self,
        data: QByteArray,
    ) -> np.ndarray:
        channel_values = []
        # Separate channels from byte-string. One channel has
        # "bytes_in_sample" many bytes in it.
        for channel_index in range(len(data) // 2):
            channel_start = channel_index * self.bytes_in_sample
            channel_end = (channel_index + 1) * self.bytes_in_sample
            channel = data[channel_start:channel_end]

            # Convert channel's byte value to integer
            if self.working_mode == MuoviWorkingMode.EMG:
                value = self._decode_int16(channel)
            elif self.working_mode == MuoviWorkingMode.EEG:
                value = self._decode_int24(channel)
            channel_values.append(value)

        return np.array(channel_values)

    def _decode_int16(self, bytes_value: QByteArray) -> int:
        value = None
        # Combine 2 bytes to a 16 bit integer value
        value = bytes_value[0] * 256 + bytes_value[1]
        # See if the value is negative and make the two's complement
        if value >= 32768:
            value -= 65536
        return value

    # Convert byte-array value to an integer value and apply two's complement
    def _decode_int24(self, bytes_value: QByteArray) -> int:
        value = None
        # Combine 3 bytes to a 24 bit integer value
        value = bytes_value[0] * 65536 + bytes_value[1] * 256 + bytes_value[2]
        # See if the value is negative and make the two's complement
        if value >= 8388608:
            value -= 16777216
        return value
