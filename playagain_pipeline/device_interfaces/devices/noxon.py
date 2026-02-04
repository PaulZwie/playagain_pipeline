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
from PySide6.QtCore import QByteArray
import numpy as np

# Local Libraries
from device_interfaces.devices.device import BaseDevice
from device_interfaces.dicts.noxon import FRAME_LEN_NO_ACC_DICT
from device_interfaces.enums.device import Device, CommunicationProtocol, LoggerLevel
from device_interfaces.enums.noxon import *

if TYPE_CHECKING:
    # Python Libraries
    from PySide6.QtWidgets import QMainWindow, QWidget
    from PySide6.QtNetwork import QAbstractSocket


class Noxon(BaseDevice):
    """
    Noxon device class derived from BaseDevice class.

    The Noxon class is using a TCP/IP protocol to communicate with the device.

    Communication Protocol:
    2 control byte sequences are sent to the device to start the data transfer.
    Control Byte 1:
    [<GETSET> <FSAMP1> <FSAMP0> <NCH1> <NCH0> <MODE2> <MODE1> <Mode0>]

    Control Byte 2:
    [<HRES> <HPF> <GAIN1> <GAIN0> <TRIG1> <TRIG0> <REC> <GO/STOP>]

    Together:
    [<GETSET> <FSAMP1> <FSAMP0> <NCH1> <NCH0> <MODE2> <MODE1> <Mode0> <HRES> <HPF> <GAIN1> <GAIN0> <TRIG1> <TRIG0> <REC> <GO/STOP>]

    Descriptions can be found in device_interfaces/enums/noxon.py
    """

    def __init__(
        self,
        parent: Union[QMainWindow, QWidget] = None,
    ) -> None:
        super().__init__(parent)

        # Device Parameters
        self.name: Device = Device.NOXON
        self.communication_protocol: CommunicationProtocol = CommunicationProtocol.TCPIP

        self._conversion_factor = 1

        # Configuration modes
        self.action_mode: NOXONAction = None
        self.sampling_frequency_mode: NOXONSamplingFrequency = None
        self.number_of_channels_mode: NOXONChannels = None
        self.acquisition_mode: NOXONMode = None
        self.resolution_mode: NOXONResolution = None
        self.filtering_mode: NOXONFilter = None
        self.gain_mode: NOXONGain = None
        self.trigger_mode: NOXONTrigger = None
        self.recording_mode: NOXONRecording = None
        self.transmission_mode: NOXONTransmission = None

        # Configuration parameters
        self.sampling_frequency: int = None
        self.bytes_in_sample: int = None
        self.number_of_channels: int = None
        self.buffer_size: int = None
        self.command: int = None

        # Define socket
        self.interface: QTcpServer = None
        self.socket: QTcpSocket = None
        self.received_bytes: QByteArray = bytearray()

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
            "action_mode": NOXONAction.SET,
            "sampling_frequency_mode": NOXONSamplingFrequency.HIGH,
            "number_of_channels_mode": NOXONChannels.ULTRA,
            "acquisition_mode": NOXONMode.MONOPOLAR,
            "resolution_mode": NOXONResolution.LOW,
            "filtering_mode": NOXONFilter.ON,
            "gain_mode": NOXONGain.DEFAULT,
            "trigger_mode": NOXONTrigger.DEFAULT,
            "recording_mode": NOXONRecording.STOP,
            "transmission_mode": NOXONTransmission.STOP,
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

    def _server_error(self, error: QAbstractSocket.SocketError) -> None:
        self.log_info(error, LoggerLevel.ERROR)

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
                self.connected_signal.emit(True)

    def _disconnect_from_device(self) -> None:
        """
        Closes the connection to the device.

        self.interface closes and is set to None.
        Device state is_connected is set to False.
        Signal connected_signal emits False.
        """

        self.socket.close()
        self.socket = None
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
            self.number_of_channels
            * self.bytes_in_sample
            * FRAME_LEN_NO_ACC_DICT[self.number_of_channels_mode]
        )

        self._configure_command()

        success = self._send_configuration_to_device()

    def _send_configuration_to_device(self) -> bool:
        command_bytes = self._integer_to_bytes(self.command)

        success = self.socket.write(command_bytes)
        self.socket.waitForBytesWritten(1000)

        if success:
            self.log_info(f"Noxon configured with: {command_bytes}")

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
        self.command += self.trigger_mode.value << 2
        self.command += self.recording_mode.value << 1
        self.command += self.transmission_mode.value

        self.log_info(f"Command: {bin(self.command)}")

    def _set_number_of_channels(self) -> bool:
        mode = self.number_of_channels_mode
        match mode:
            case NOXONChannels.LOW:
                if self.acquisition_mode == NOXONMode.BIPOLAR:
                    self.number_of_channels = 8
                else:
                    self.number_of_channels = 12

            case NOXONChannels.MEDIUM:
                if self.acquisition_mode == NOXONMode.BIPOLAR:
                    self.number_of_channels = 12
                else:
                    self.number_of_channels = 20

            case NOXONChannels.HIGH:
                if self.acquisition_mode == NOXONMode.BIPOLAR:
                    self.number_of_channels = 20
                else:
                    self.number_of_channels = 36

            case NOXONChannels.ULTRA:
                if self.acquisition_mode == NOXONMode.BIPOLAR:
                    self.number_of_channels = 36
                else:
                    self.number_of_channels = 68

            case _:
                self.log_info(
                    f"Number of Channels Mode: {mode} not defined", LoggerLevel.ERROR
                )
                return False

        return True

    def _set_sampling_frequency_parameter(self) -> bool:
        mode = self.sampling_frequency_mode
        match mode:
            case NOXONSamplingFrequency.LOW:
                if self.acquisition_mode == NOXONMode.ACCELEROMETER:
                    self.sampling_frequency = 2000
                else:
                    self.sampling_frequency = 500

            case NOXONSamplingFrequency.MEDIUM:
                if self.acquisition_mode == NOXONMode.ACCELEROMETER:
                    self.sampling_frequency = 4000
                else:
                    self.sampling_frequency = 1000

            case NOXONSamplingFrequency.HIGH:
                if self.acquisition_mode == NOXONMode.ACCELEROMETER:
                    self.sampling_frequency = 8000
                else:
                    self.sampling_frequency = 2000

            case NOXONSamplingFrequency.ULTRA:
                if self.acquisition_mode == NOXONMode.ACCELEROMETER:
                    self.sampling_frequency = 16000
                else:
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
            case NOXONResolution.LOW:
                self.bytes_in_sample = 2
            case NOXONResolution.HIGH:
                self.bytes_in_sample = 3
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
        if self.command:
            self.command -= 1
            self._send_configuration_to_device()
            self.is_streaming = False
            self.streaming_signal.emit(False)

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

    def clear_socket(self) -> None:
        """Reads all the bytes from the buffer."""
        self.socket.readAll()

    def _read_data(self) -> None:
        """
        This function is called when bytes are ready to be read in the buffer.
        After reading the bytes from the buffer, _process_data is called to
        decode and process the raw data.
        """
        # TODO Catch configured settings from device
        if not self.is_streaming:
            packet = self.socket.readAll()
            msg = packet.data().decode("utf-8")
            if not msg:
                return
            if msg == "NOXON":
                self.log_info("Device is configured", LoggerLevel.INFO)
                self.is_configured = True
                self.configured_signal.emit(True)
        else:
            while self.socket.bytesAvailable() > self.buffer_size:
                packet = self.socket.read(self.buffer_size)
                if not packet:
                    self.received_bytes = bytearray()
                self.received_bytes.extend(packet)

                if len(self.received_bytes) % self.buffer_size == 0:
                    self._process_data(self.received_bytes)
                    self.received_bytes = bytearray()

                else:
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
        if configuration:
            # TODO How are they encoded and what do the bytes represent? -> Check with OT Bioelettronica
            config = np.frombuffer(input, dtype=np.int16)

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
        return (
            data[: self.number_of_channels - 4] * self._conversion_factor
            if milli_volts
            else data[: self.number_of_channels - 4]
        )

    def extract_aux_data(self, data: np.ndarray, index: int = 0) -> np.ndarray:
        return data[-4 + index]

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

    # Convert integer to bytes
    def _integer_to_bytes(self, command: int) -> bytes:
        return int(command).to_bytes(2, byteorder="big")

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
            if self.resolution_mode == NOXONResolution.LOW:
                value = self._decode_int16(channel)
            else:
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
