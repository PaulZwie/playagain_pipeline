"""
Device class for real-time interfacing the Noxon device.
Developer: Dominik I. Braun
Contact: dome.braun@fau.de
Last Update: 2024-01-07
"""

# Python Libraries
from __future__ import annotations
from typing import TYPE_CHECKING, Union, Dict, Tuple
from PySide6.QtNetwork import QTcpSocket, QTcpServer, QHostAddress
from PySide6.QtCore import QByteArray, QTimer
import numpy as np

# Local Libraries
from device_interfaces.devices.device import BaseDevice
from device_interfaces.dicts.nswitch import GET_GAIN_DICT
from device_interfaces.enums.device import Device, CommunicationProtocol, LoggerLevel
from device_interfaces.enums.nswitch import *

if TYPE_CHECKING:
    # Python Libraries
    from PySide6.QtWidgets import QMainWindow, QWidget
    from PySide6.QtNetwork import QAbstractSocket


class NSwitch(BaseDevice):
    """
    NSwitch device class derived from BaseDevice class.

    The NSwitch class is using a TCP/IP protocol to communicate with the device.

    Communication Protocol:
    2 control byte sequences are sent to the device to start the data transfer.
    Control Byte 1:
    [<GETSET> <FSAMP1> <FSAMP0> <NCH1> <NCH0> <MODE2> <MODE1> <Mode0>]

    Control Byte 2:
    [<HRES> <HPF> <GAIN1> <GAIN0> <TRIG1> <TRIG0> <REC> <GO/STOP>]

    Together:
    [<GETSET> <FSAMP1> <FSAMP0> <NCH1> <NCH0> <MODE2> <MODE1> <Mode0> <HRES> <HPF> <GAIN1> <GAIN0> <TRIG1> <TRIG0> <REC> <GO/STOP>]

    Descriptions can be found in device_interfaces/enums/nswitch.py
    """

    def __init__(
        self,
        parent: Union[QMainWindow, QWidget] = None,
    ) -> None:
        super().__init__(parent)

        # Device Parameters
        self.name: Device = Device.NSWITCH
        self.communication_protocol: CommunicationProtocol = CommunicationProtocol.TCPIP

        self._conversion_factor = None
        self.magic_number = 0x620F

        # Configuration modes
        self.action_mode: NSWITCHAction = None
        self.sampling_frequency_mode: NSWITCHSamplingFrequency = None
        self.number_of_channels_mode: NSWITCHChannels = None
        self.acquisition_mode: NSWITCHMode = None
        self.resolution_mode: NSWITCHResolution = None
        self.filtering_mode: NSWITCHFilter = None
        self.gain_mode: NSWITCHGain = None
        self.transmission_mode: NSWITCHTransmission = None
        self.configuration_mode: NSWITCHConfigurationMode = None
        self.reference_voltage_mode: NSWITCHReferenceVoltageMode = None

        # Configuration parameters
        self.sampling_frequency: int = None
        self.bytes_per_sample: int = None
        self.samples_per_frame: int = 10  # 10 samples per block
        self.number_of_channels: int = None
        self.buffer_size: int = None
        self.gain: int = None
        self.reference_voltage = 3.3  # Vref of the ADC
        self.command: int = None

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
            "action_mode": NSWITCHAction.SET,
            "sampling_frequency_mode": NSWITCHSamplingFrequency.HIGH,
            "number_of_channels_mode": NSWITCHChannels.ULTRA,
            "acquisition_mode": NSWITCHMode.MONOPOLAR,
            "resolution_mode": NSWITCHResolution.LOW,
            "filtering_mode": NSWITCHFilter.ON,
            "gain_mode": NSWITCHGain.DEFAULT,
            "transmission_mode": NSWITCHTransmission.STOP,
            "configuration_mode": NSWITCHConfigurationMode.IGNORE,
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

        In the case of Sessantaquattro:
        The sessantaquattro resets itself after receiving a new configuration.
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
                self.timeout_timer.stop()
                self.connected_signal.emit(True)

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

        self.configuration_mode = NSWITCHConfigurationMode.SET_CONFIGURATION

        # Set configuration parameters for data transfer
        success = self._set_sampling_frequency_parameter()
        if not success:
            return

        success = self._set_number_of_channels()
        if not success:
            return

        success = self._set_resolution()
        if not success:
            return

        self.buffer_size = (
            self.number_of_channels * self.bytes_per_sample * self.samples_per_frame
            + 2 * self.bytes_per_sample  # Magic Number and Timestamp
        )

        self.biosignal_channels = self.number_of_channels
        self.aux_channels = 0
        if self.gain_mode == (
            NSWITCHGain.GAIN_4 or NSWITCHGain.GAIN_8 or NSWITCHGain.GAIN_12
        ):
            raise NotImplementedError(
                f"Gain mode {self.gain_mode} not implemented yet!"
            )
        self.gain = GET_GAIN_DICT[self.gain_mode]
        self.resolution = 2 ** (8 * self.bytes_per_sample)  # 2**16 or 2**24
        self.reference_voltage = (
            2.4
            if self.reference_voltage_mode == NSWITCHReferenceVoltageMode.VREF_2_4
            else 4
        )
        self._conversion_factor = (
            (2 * self.reference_voltage) / self.gain / (self.resolution - 1)
        )

        self._configure_command()
        success = self._send_configuration_to_device()

    def _send_configuration_to_device(self) -> bool:
        command_bytes = self._integer_to_bytes(self.command)

        success = self.socket.write(command_bytes)
        self.socket.waitForBytesWritten(1000)

        if success:
            self.log_info(f"{self.name} configured with: {command_bytes}")

            return True

        return False

    def _configure_command(self) -> None:
        self.command = self.action_mode.value << 15
        self.command += self.sampling_frequency_mode.value << 13
        self.command += self.number_of_channels_mode.value << 11
        self.command += self.acquisition_mode.value << 8
        self.command += self.resolution_mode.value << 7
        self.command += self.filtering_mode.value << 6
        self.command += self.gain_mode.value << 4
        self.command += self.reference_voltage_mode.value << 2
        self.command += self.configuration_mode.value << 1
        self.command += self.transmission_mode.value

        self.log_info(f"Command: {bin(self.command)}")

    def _set_number_of_channels(self) -> bool:
        mode = self.number_of_channels_mode
        match mode:
            case NSWITCHChannels.LOW:
                self.number_of_channels = 2

            case NSWITCHChannels.MEDIUM:
                self.number_of_channels = 4

            case NSWITCHChannels.HIGH:
                self.number_of_channels = 8

            case NSWITCHChannels.ULTRA:
                raise NotImplementedError(
                    f"{mode} mode for number of channels not implemented yet!"
                )
                self.number_of_channels = 16

            case _:
                self.log_info(
                    f"Number of Channels Mode: {mode} not defined", LoggerLevel.ERROR
                )
                return False

        self.biosignal_channels = self.number_of_channels
        self.aux_channels = 0

        return True

    def _set_sampling_frequency_parameter(self) -> bool:
        mode = self.sampling_frequency_mode
        match mode:
            case NSWITCHSamplingFrequency.LOW:
                self.sampling_frequency = 500

            case NSWITCHSamplingFrequency.MEDIUM:
                self.sampling_frequency = 1000

            case NSWITCHSamplingFrequency.HIGH:
                self.sampling_frequency = 2000

            case NSWITCHSamplingFrequency.ULTRA:
                self.sampling_frequency = 4000

            case _:
                self.log_info(
                    f"Sampling Frequency Mode: {mode} not defined", LoggerLevel.ERROR
                )
                return False

        return True

    def _set_resolution(self) -> bool:
        mode = self.resolution_mode
        match mode:
            case NSWITCHResolution.LOW:
                self.bytes_per_sample = 2
            case NSWITCHResolution.HIGH:
                self.bytes_per_sample = 3
            case _:
                self.log_info(f"Resolution Mode: {mode} not defined", LoggerLevel.ERROR)
                return False
        return True

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
                    f"Attribute '{key}' not found in the class", LoggerLevel.ERROR
                )

    def get_configuration(self) -> None:
        """
        Sends the command to get the current configuration of the device.
        """
        set_action = int(0b1 << 15)
        in_bytes = self._integer_to_bytes(set_action)
        transmitted = self.socket.write(in_bytes)
        self.socket.waitForBytesWritten()

        self.log_info(f"Transmitted bytes: {transmitted} - {in_bytes}")

    def _stop_streaming(self) -> None:
        """
        Sends the command to stop the streaing to the device

        if successful:
            Device state is_streaming is set to False.
            Signal streaming_signal emits False.
        """
        if self.is_configured:
            self.transmission_mode = NSWITCHTransmission.STOP
            self._configure_command()
            self._send_configuration_to_device()
            self.is_streaming = False
            self.streaming_signal.emit(False)
        if self.is_connected:
            self.clear_socket()

    def _start_streaming(self) -> None:
        """
        Sends the command to start the streaming to the device.

        if successful:
            Device state is_streaming is set to True.
            Signal streaming_signal emits True.
        """
        if self.command:
            self.transmission_mode = NSWITCHTransmission.START
            self._configure_command()
            self._send_configuration_to_device()

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
            try:
                msg = packet.data().decode("utf-8")
            except UnicodeDecodeError:
                return
            if not msg:
                return
            if msg == "NSWITCH":
                self.is_configured = True
                self.configured_signal.emit(True)
                self.configuration_mode = NSWITCHConfigurationMode.IGNORE
            if msg == "RUNNING":
                self.is_streaming = True
                self.streaming_signal.emit(True)
        else:
            while self.socket.bytesAvailable() > self.buffer_size:
                packet = self.socket.read(self.buffer_size)
                if not packet:
                    continue
                self.received_bytes.extend(packet)

                if len(self.received_bytes) % self.buffer_size == 0:
                    self._process_data(self.received_bytes)
                    self.received_bytes = bytearray()

    def _process_data(self, input: QByteArray, configuration: bool = False) -> None:
        """
        Decodes the transmitted bytes and convert them to respective
        output format (e.g., mV).

        Emits the processed data through the Signal data_available_signal
        which can be connected to a function using:
        {self.device}.data_available_signal.connect(your_custom_function).

        This works perfectly fine outside of this class.

        Your custom function your_custom_function needs to have a parameter
        "data" which is of type np.ndarray.


        In case that the current configuration of the device was requested,
        the configuration is provided through the Signal
        configuration_available_signal that emits the current parameters
        in a dictionary.

        Args:
            input (QByteArray):
                Bytearray of the transmitted raw data.
        """

        data = self._bytes_to_integers(input)

        data = data.reshape(self.number_of_channels, -1, order="F").astype(np.double)

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
        return (
            data[: self.biosignal_channels] * self._conversion_factor
            if milli_volts
            else data[: self.biosignal_channels]
        )

    def extract_aux_data(self, data: np.ndarray, index: int = 0) -> np.ndarray:
        raise NotImplementedError(
            f"{self.name} does not have any AUX data implemented!"
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
            "samples_per_frame": self.samples_per_frame,
            "biosignal_channels": self.biosignal_channels,
            "aux_channels": self.aux_channels,
        }

    # Convert integer to bytes
    def _integer_to_bytes(self, command: int) -> bytes:
        return int(command).to_bytes(2, byteorder="big")

    # Convert channels from bytes to integers
    def _bytes_to_integers(
        self,
        data: QByteArray,
    ) -> np.ndarray:
        # Read magic number and timestamp
        magic_number_bytes = data[: self.bytes_per_sample]
        magic_number = (
            self._decode_int16(magic_number_bytes)
            if self.resolution_mode == NSWITCHResolution.LOW
            else self._decode_int24(magic_number_bytes)
        )

        if magic_number != self.magic_number:
            self.log_info(
                f"Magic number not found: {hex(magic_number)}", LoggerLevel.ERROR
            )
            return np.array([])

        # Read timestamp
        timestamp_byte = data[self.bytes_per_sample : 2 * self.bytes_per_sample]
        timestamp = (
            self._decode_int16(timestamp_byte)
            if self.resolution_mode == NSWITCHResolution.LOW
            else self._decode_int24(timestamp_byte)
        )

        data = data[2 * self.bytes_per_sample :]
        number_of_bytes = len(data)
        channel_values = []

        # Convert channel's byte value to integer
        match self.resolution_mode:
            case NSWITCHResolution.LOW:
                for channel_index in range(number_of_bytes // 2):
                    channel_start = channel_index * self.bytes_per_sample
                    channel_end = (channel_index + 1) * self.bytes_per_sample
                    channel = data[channel_start:channel_end]

                    value = self._decode_int16(channel)
                    channel_values.append(value)
            case NSWITCHResolution.HIGH:
                for channel_index in range(number_of_bytes // 3):
                    channel_start = channel_index * self.bytes_per_sample
                    channel_end = (channel_index + 2) * self.bytes_per_sample
                    channel = data[channel_start:channel_end]

                    value = self._decode_int24(channel)
                    channel_values.append(value)

        return np.array(channel_values)

    def _decode_int16(self, bytes_value: QByteArray) -> int:
        value = None
        # Combine 2 bytes to a 16 bit integer value
        value = bytes_value[0] * 2**8 + bytes_value[1]
        # See if the value is negative and make the two's complement
        if value >= 2**15:
            value -= 2**16
        return value

    # Convert byte-array value to an integer value and apply two's complement
    def _decode_int24(self, bytes_value: QByteArray) -> int:
        value = None
        # Combine 3 bytes to a 24 bit integer value
        value = bytes_value[0] * 2**16 + bytes_value[1] * 2**8 + bytes_value[2]
        # See if the value is negative and make the two's complement
        if value >= 2**23:
            value -= 2**24
        return value
