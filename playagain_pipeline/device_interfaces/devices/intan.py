"""
Device class for real-time interfacing the Intan RHD Controller device.
Developer: Dominik I. Braun
Contact: dome.braun@fau.de
Last Update: 2024-04-04
"""

# Python Libraries
from __future__ import annotations
from typing import TYPE_CHECKING, Union, Dict, Tuple
from PySide6.QtCore import QByteArray, QIODevice
from PySide6.QtNetwork import QTcpSocket, QHostAddress
import numpy as np
from enum import Enum

# Local Libraries
from device_interfaces.devices.device import BaseDevice
from device_interfaces.dicts.intan import *
from device_interfaces.enums.intan import *
from device_interfaces.enums.device import Device, CommunicationProtocol, LoggerLevel

if TYPE_CHECKING:
    from PySide6.QtWidgets import QMainWindow, QWidget


class IntanRHDController(BaseDevice):
    """
    Intan RHD Controller class derived from BaseDevice class.

    The Intan RHD Controller is a TCP/IP device for recording and stimulating.
    """

    def __init__(
        self,
        parent: Union[QMainWindow, QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self.name: Device = Device.INTAN_RHD_CONTROLLER
        self.communication_protocol: CommunicationProtocol = CommunicationProtocol.TCPIP

        # Fixed device parameters
        self.samples_per_frame: int = 128  # Fixed 128

        # Configuration modes

        # Configuration parameters
        self.sampling_frequency: int = None
        self.number_of_channels: int = None
        self.biosignal_channels: int = None
        self.aux_channels: int = (
            None  # Default value (2 An. In, 2 An. Out, 2 Dig. In 2 Dig. Out)
        )
        self.biosignal_channel_indices: list[int] = None
        self.aux_channel_indices: list[int] = None
        self._conversion_factor = 0.195 * 10 ** (-3)  # Conversion factor to mV
        self.enabled_ports: list[IntanAvailablePorts] = None
        self.port_configurations_dict: Dict[IntanAvailablePorts, bool | int] = None
        self.requested_configuration_received: list[IntanAvailablePorts] = None
        self.request_configuration_order: list[IntanAvailablePorts] = None
        self.configuration_command: str = None

        # Command
        self.command_interface: QTcpSocket = None
        self.command_buffer_size: int = 1024

        # Waveform
        self.use_waveform: bool = False
        self.waveform_interface: QTcpSocket = None
        self.waveform_ip_address: QHostAddress = QHostAddress.LocalHost
        self.waveform_port: int = 5001
        self.waveform_bytes_per_frame: int = None
        self.waveform_bytes_per_frame: int = None
        self.waveform_buffer_size: int = None
        self.waveform_received_bytes: bytearray = None
        self._waveform_magic_number: int = 0x2EF07A08

        # Spike
        self.use_spike_detection: bool = False
        self.spike_interface: QTcpSocket = None
        self.spike_ip_address: QHostAddress = QHostAddress.LocalHost
        self.spike_port: int = 5002
        self.spike_bytes_per_frame: int = None
        self.spike_bytes_per_frame: int = None
        self.spike_buffer_size: int = None
        self.spike_received_bytes: bytearray = None

    def _reset_device_parameters(self) -> Dict[str, Enum | Dict[str, Enum]]:
        return super()._reset_device_parameters()

    def _connect_to_device(self, settings: Tuple[str | int] = None) -> None:
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

                Defaults to (LocalHost="127.0.0.1", 5000) for
                Intan RHD Recording Controller.

                self.is_connected is set to True after sampling_frequency was
                obtained successfully (in self._read_command).
                self.connected_signal emits True after connection was established.
        """
        self.command_interface = QTcpSocket(self)
        success = self._make_request(settings)

        if not success:
            return

        self.get_configuration()

    def _make_request(
        self, settings: Tuple[str, int] = (QHostAddress.LocalHost, 5000)
    ) -> bool:
        """
        Requests a connection or checks if someone connected to the server.
        After connection is successful, the Signal connected_signal emits True
        and sets the current state is_connected to True.

        Args:
            settings (Tuple[str, int], optional):
                If CommunicationProtocol.TCPIP:
                Tuple[0] = IP -> string
                Tuple[1] = Port -> int

                If CommunicationProtocol.SERIAL pr CommunicationProtocol.USB:
                Tuple[0] = COM Port -> string
                Tuple[1] = Baudrate -> int

                Defaults to (LocalHost="127.0.0.1", 5000) for Intan RHD Recording Controller.

        Returns:
            bool:
                Returns True if request was successfully. False if not.
        """
        self.command_interface.connectToHost(
            settings[0], settings[1], QIODevice.ReadWrite
        )

        if not self.command_interface.waitForConnected(1000):
            self.log_info("Connection to device failed.", LoggerLevel.ERROR)
            return False

        self.command_interface.readyRead.connect(self._read_command_data)

        # Spike
        if self.use_spike_detection:
            self.spike_interface = QTcpSocket(self)
            self.spike_interface.connectToHost(
                self.spike_ip_address, self.spike_port, QIODevice.ReadWrite
            )

            if not self.spike_interface.waitForConnected(1000):
                self.log_info("Connection to Spike Port failed.", LoggerLevel.ERROR)
                return False

            self.spike_interface.readyRead.connect(self._read_spike_data)

        # Waveform
        if self.use_waveform:
            self.waveform_interface = QTcpSocket(self)
            self.waveform_interface.connectToHost(
                self.waveform_ip_address, self.waveform_port, QIODevice.ReadWrite
            )

            if not self.waveform_interface.waitForConnected(1000):
                self.log_info("Connection to Waveform Port failed.", LoggerLevel.ERROR)
                return False

            self.waveform_interface.readyRead.connect(self._read_data)

        self.log_info("Connection to device established.", LoggerLevel.INFO)

        self.clear_socket()
        return True

    def _disconnect_from_device(self) -> None:
        if self.is_streaming:
            self._stop_streaming()

        self.command_interface.disconnectFromHost()

        if self.use_spike_detection:
            self.spike_interface.disconnectFromHost()

        if self.use_waveform:
            self.waveform_interface.disconnectFromHost()

        self.is_connected = False
        self.connected_signal.emit(False)

    def configure_device(self, params: Dict[str, Enum | Dict[str, Enum]]) -> None:
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
        self._configure_command()

        if self.use_waveform:
            bytes_per_sample = 4 + self.number_of_channels * 2
            self.waveform_buffer_size = bytes_per_sample * self.samples_per_frame + 4

        if self.use_spike_detection:
            bytes_per_sample = 4 + self.number_of_channels * 2
            self.spike_buffer_size = bytes_per_sample * self.samples_per_frame + 4

        self._send_configuration_to_device()

        self.is_configured = True
        self.configured_signal.emit(True)

    def _send_configuration_to_device(
        self,
    ) -> None:
        self.command_interface.write(self.configuration_command.encode("utf-8"))

    def _configure_command(self) -> None:
        self.configuration_command = ""

        # Biosignal channels TODO: Add indices
        self.biosignal_channels = 0
        for port in self.enabled_ports:
            channels = self.port_configurations_dict[port]["channels"]
            self.biosignal_channels += channels
            for i in range(channels):
                self.configuration_command += f"set {port.name.split('_')[-1].lower()}-{i:03}.tcpdataoutputenabled true;"

        # Analog Channels
        self.aux_channels = 0

        # Analog In
        number_of_analog_in_channels = len(IntanAnalogInChannels)
        self.aux_channels += number_of_analog_in_channels
        for analog_in in range(number_of_analog_in_channels):
            self.configuration_command += (
                f"set analog-in-{analog_in + 1}.tcpdataoutputenabled true;"
            )

        # Analog Out
        # number_of_analog_out_channels = len(IntanAnalogOutChannels)
        # self.aux_channels += number_of_analog_out_channels
        # for analog_out in range(number_of_analog_out_channels):
        #     self.configuration_command += (
        #         f"set analog-out-{analog_out + 1:02}.tcpdataoutputenabled true;"
        #     )

        # Digital In
        # number_of_digital_in_channels = len(IntanDigitalInChannels)
        # self.aux_channels += number_of_digital_in_channels
        # for digital_in in range(number_of_digital_in_channels):
        #     self.configuration_command += (
        #         f"set digital-in-{digital_in + 1:02}.tcpdataoutputenabled true;"
        #     )

        # # Digital Out
        # number_of_digital_out_channels = len(IntanDigitalOutputChannels)
        # self.aux_channels += number_of_digital_out_channels
        # for digital_out in range(number_of_digital_out_channels):
        #     self.configuration_command += (
        #         f"set digital-out-{digital_out + 1:02}.tcpdataoutputenabled true;"
        #     )

        # Remove last semi-colon
        # self.configuration_command = self.configuration_command[:-1]
        self.number_of_channels = self.biosignal_channels + self.aux_channels
        self.biosignal_channel_indices = list(range(self.biosignal_channels))
        self.aux_channel_indices = list(
            range(self.biosignal_channels, self.number_of_channels)
        )

    def _update_configuration_parameters(
        self, params: Dict[str, Enum | Dict[str, Enum]]
    ) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.log_info(
                    f"Attribute '{key}' not found in the class of {self.name.name}",
                    LoggerLevel.ERROR,
                )

    def _query_sampling_frequency(self) -> None:
        self.command_interface.write(b"get sampleratehertz;")
        self.command_interface.waitForBytesWritten(1000)

    def get_configuration(self) -> None:
        """
        Sends the command to get the current configuration of the device.
        """
        self.port_configurations_dict = {}
        # Set TCP numberDataBlockPerWrite
        self.command_interface.write(b"set tcpnumberdatablocksperwrite 1;")

        success = self.command_interface.waitForBytesWritten(1000)
        if not success:
            self.log_info(
                "Failed to set the tcpnumberdatablocksperwrite.", LoggerLevel.ERROR
            )

        # Rescan new ports
        self.command_interface.write(b"execute RescanPorts;")
        success = self.command_interface.waitForBytesWritten(1000)
        if not success:
            self.log_info("Failed to rescan ports.", LoggerLevel.ERROR)

        # Get connected headstages
        self.requested_configuration_received = []
        self.request_configuration_order = []
        self.sampling_frequency = None
        for port in IntanAvailablePorts:
            self.command_interface.write(
                f"get {port.name.split('_')[-1].lower()}.numberamplifierchannels;".encode(
                    "utf-8"
                )
            )
            self.request_configuration_order.append(port)

        success = self.command_interface.waitForBytesWritten(1000)
        if not success:
            self.log_info(
                "Failed to get the current port configuration.", LoggerLevel.ERROR
            )

        self._query_sampling_frequency()

    def _start_streaming(self) -> None:
        """
        Sends the command to start the streaming to the device.

        if successful:
            Device state is_streaming is set to True.
            Signal streaming_signal emits True.
        """
        self.command_interface.write(INTAN_RUN_MODES[IntanRunMode.RUN])
        success = self.command_interface.waitForBytesWritten(1000)

        if not success:
            return

        if self.use_waveform:
            self.waveform_received_bytes = bytearray()

        if self.use_spike_detection:
            self.spike_received_bytes = bytearray()

        self.is_streaming = True
        self.streaming_signal.emit(True)

    def _stop_streaming(self) -> None:
        """
        Sends the command to stop the streaing to the device

        if successful:
            Device state is_streaming is set to False.
            Signal streaming_signal emits False.
        """
        self.command_interface.write(INTAN_RUN_MODES[IntanRunMode.STOP])
        success = self.command_interface.waitForBytesWritten(1000)

        if not success:
            return

        self.is_streaming = False
        self.streaming_signal.emit(False)

    def clear_socket(self) -> None:
        """Reads all the bytes from the buffer."""
        self.command_interface.write(b"execute clearalldataoutputs;")
        success = self.command_interface.waitForBytesWritten(1000)
        if not success:
            self.log_info("Failed to clear the socket.", LoggerLevel.ERROR)

        if self.command_interface.bytesAvailable():
            self.command_interface.readAll()

        if self.use_spike_detection and self.spike_interface.bytesAvailable():
            self.spike_interface.readAll()

        if self.use_waveform and self.waveform_interface.bytesAvailable():
            self.waveform_interface.readAll()

    def _read_command_data(self) -> None:
        msg = self.command_interface.readAll().data()
        command_return = str(msg, "utf-8")

        command_prefixes = ["Return:", "Error:", "Unrecognized"]
        for keyword in command_prefixes:
            command_return = command_return.replace(keyword, " " + keyword)

        splits = command_return.split()
        for i, split in enumerate(splits):
            if not split in command_prefixes:
                continue
            match split:
                case "Return:":
                    match splits[i + 1]:
                        case "runmode":
                            if splits[2] == "Run":
                                self._stop_streaming()

                        case "SampleRateHertz":
                            self.sampling_frequency = int(splits[i + 2])

                        case "NumberAmplifierChannels":
                            port = self.request_configuration_order.pop(0)
                            self.port_configurations_dict[port] = {
                                "enabled": True if splits[i + 2] > "0" else False,
                                "channels": int(splits[i + 2]),
                            }
                            self.requested_configuration_received.append(port)

                    if (
                        len(self.requested_configuration_received)
                        == len(IntanAvailablePorts)
                        and self.sampling_frequency is not None
                    ):
                        configurations_dict = {
                            "sampling_frequency": self.sampling_frequency,
                            "port_information": self.port_configurations_dict,
                        }
                        self.configuration_available_signal.emit(configurations_dict)
                        self.requested_configuration_received = None

                        self.is_connected = True
                        self.connected_signal.emit(True)

                case "Error:":
                    self.log_info(f"Error: {command_return}", LoggerLevel.ERROR)

                case "Unrecognized":
                    self.log_info(f"Unrecognized parameter.", LoggerLevel.ERROR)

    def _read_spike_data(self) -> None:
        return

    def _read_data(self) -> None:
        """
        This function is called when bytes are ready to be read in the buffer.
        After reading the bytes from the buffer, _process_data is called to
        decode and process the raw data.
        """

        if not self.is_streaming:
            self.waveform_interface.readAll()
            return

        while self.waveform_interface.bytesAvailable() > self.waveform_buffer_size:
            packet = self.waveform_interface.read(self.waveform_buffer_size)

            if not packet:
                continue

            self.waveform_received_bytes.extend(packet)

            if len(self.waveform_received_bytes) % self.waveform_buffer_size == 0:
                self._process_data(self.waveform_received_bytes)
                self.waveform_received_bytes = bytearray()

    def _process_data(self, input: bytearray) -> None:
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

        magic_number = int.from_bytes(input[:4], "little", signed=False)
        if magic_number != self._waveform_magic_number:
            self.log_info("Invalid magic number.", LoggerLevel.ERROR)
            return

        data = input[4:]

        output = []

        for frame in range(self.samples_per_frame):
            # TODO: Check for sampling frequency
            if (
                self.sampling_frequency == 4000 and frame % 2 == 1
            ):  # Skip every second frame block at 4k Hz to get 2k Hz
                continue

            time_stamp = int.from_bytes(data[:4], "little", signed=False)

            data = data[4:]

            # Read the samples per channel from the buffer
            samples = (
                np.frombuffer(data[: self.number_of_channels * 2], dtype=np.uint16)
                .reshape(self.number_of_channels, 1)
                .astype(np.double)
                - 32768
            )

            output.append(samples)

            data = data[self.number_of_channels * 2 :]

        output = np.hstack(output)

        self.data_available_signal.emit(output)

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

        if len(self.biosignal_channel_indices) > 0:
            return (
                data[self.biosignal_channel_indices] * self._conversion_factor
                if milli_volts
                else data[self.biosignal_channel_indices]
            )

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
        if self.aux_channel_indices:
            return data[self.aux_channel_indices[index]] * self._conversion_factor

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


class IntanRHDControllerFiles(BaseDevice):
    """
    Intan RHD Controller class derived from BaseDevice class.

    The Intan RHD Controller is a TCP/IP device for recording and stimulating.
    """

    def __init__(
        self,
        parent: Union[QMainWindow, QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self.name: Device = Device.INTAN_RHD_CONTROLLER_FILES
        self.communication_protocol: CommunicationProtocol = CommunicationProtocol.TCPIP

    def _reset_device_parameters(self) -> Dict[str, Enum | Dict[str, Enum]]:
        return super()._reset_device_parameters()

    def _connect_to_device(self, settings: Tuple[str | int] = None) -> None:
        return super()._connect_to_device(settings)

    def _disconnect_from_device(self) -> None:
        return super()._disconnect_from_device()

    def configure_device(self, params: Dict[str, Enum | Dict[str, Enum]]) -> None:
        return super().configure_device(params)

    def _send_configuration_to_device(
        self, params: Dict[str, Enum | Dict[str, Enum]]
    ) -> None:
        return super()._send_configuration_to_device(params)

    def _configure_command(self) -> None:
        return

    def _update_configuration_parameters(
        self, params: Dict[str, Enum | Dict[str, Enum]]
    ) -> None:
        return super()._update_configuration_parameters(params)

    def _start_streaming(self) -> None:
        return super()._start_streaming()

    def _stop_streaming(self) -> None:
        return super()._stop_streaming()

    def clear_socket(self) -> None:
        return super().clear_socket()

    def _read_data(self) -> None:
        return super()._read_data()

    def _process_data(self, input: QByteArray) -> None:
        return super()._process_data(input)

    def extract_emg_data(
        self, data: np.ndarray, milli_volts: TYPE_CHECKING = False
    ) -> np.ndarray:
        return super().extract_emg_data(data, milli_volts)

    def extract_aux_data(self, data: np.ndarray, index: int = 0) -> np.ndarray:
        return super().extract_aux_data(data, index)

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
