"""
SyncStation class for real-time interface to the syncstation device from OT Bioelettronica.

Developer: Dominik I. Braun
Contact: dome.braun@fau.de
Last Update: 2024-02-15
"""

# Python Libraries
from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING, Union, Dict, Tuple
from PySide6.QtNetwork import QTcpSocket, QHostAddress
import numpy as np

# Local Libraries
from device_interfaces.devices.device import BaseDevice
from device_interfaces.dicts.syncstation import *
from device_interfaces.enums.syncstation import *
from device_interfaces.enums.device import LoggerLevel, CommunicationProtocol, Device

if TYPE_CHECKING:
    from PySide6.QtWidgets import QMainWindow, QWidget


class SyncStation(BaseDevice):
    def __init__(self, parent: Union[QMainWindow, QWidget] = None) -> None:
        """
        SyncStation class derived from BaseDevice class.

        Args:
            parent (Union[QMainWindow, QWidget], optional):
            Parent widget to which the device is assigned to.
            Defaults to None.


        The SyncStation class is using a TCP/IP protocol to communicate with the device.

        The commands to start data transfer and configure the wireless devices is a message string
        including the number of bytes in the message and a CRC8. The message sting can be of two types: StartStop
        Command, OptSettings Command

        The StartStop Command is structured as:
        START BYTE A (with bit 7 = 0)
        CONTROL BYTE1: config muovi probe n.1 (opt)
        CONTROL BYTE2: config muovi probe n.2 (opt)
        CONTROL BYTE3: config muovi probe n.3 (opt)
        CONTROL BYTE4: config muovi probe n.4 (opt)
        CONTROL BYTE5: config muovi+/sessantaquattro/sessantaquattro+ n.1 (opt)
        CONTROL BYTE6: config muovi+/sessantaquattro/sessantaquattro+ n.2 (opt)
        CONTROL BYTE7: config due+ probe n.1 (opt)
        CONTROL BYTE8: config due+ probe n.2 (opt)
        CONTROL BYTE9: config due+ probe n.3 (opt)
        CONTROL BYTE10: config due+ probe n.4 (opt)
        CONTROL BYTE11: config due+ probe n.5 (opt)
        CONTROL BYTE12: config due+ probe n.6 (opt)
        CONTROL BYTE13: config due+ probe n.7 (opt)
        CONTROL BYTE14: config due+ probe n.8 (opt)
        CONTROL BYTE15: config due+ probe n.9 (opt)
        CONTROL BYTE16: config due+ probe n.10 (opt)
        CRC8

        The OptSettings Command is structured as:
        START BYTE B (with bit 7 = 1)

        OPTION BYTE1: set the waiting time (Latency) for the SyncStation (opt)
        The value can range from 1 to 200. It set the latency in terms of accumulate blocks into the internal buffer
        of the SyncStation waiting for the corresponding sample from one or more wireless devices. Refer to the
        next pages for additional details

        OPTION BYTE2: to be defined (opt)
        OPTION BYTE3: to be defined (opt)
        OPTION BYTE4: to be defined (opt)
            CRC8

        Accessory channel 1 (ch. 37 for Muovi, ch. 69 for muovi+/sessantaquattro) - 2/3 bytes
        In case of 3 bytes -> first byte holds only 0/ not present when resolution is 16 bit
        [(ZEROS7:0>)<TRIG><TR_CODE7:0><0><BUF6:0>]
        Accessory channel 2 is a sample counter

        Accessory channel 1 (SyncStation) - 2 bytes
        [<TRIG><TR_CODE7:0><BUF7:0>]
        Accessory channel 2 (SyncStation) is a sample counter
        """
        super().__init__(parent)

        # Device Parameters
        self.name: Device = Device.SYNCSTATION
        self.protocol: CommunicationProtocol = CommunicationProtocol.TCPIP

        # Communication parameters
        self.interface: QTcpSocket = None
        self.frame_size: int = None
        self.buffer_size: int = None
        self.sampling_frequency: int = None
        self.bytes_in_sample: int = None
        self.number_of_channels: int = None
        self.biosignal_channel_indices: list[int] = None
        self.aux_channel_indices: int = None
        self.number_of_bytes: int = None

        # Configuration commands
        self.command_bytearray_A: bytearray = None
        self.command_bytearray_B: bytearray = None

        # START BYTE A
        self.rec_on_mode: SyncStationRecOnMode = None
        self.working_mode: SyncStationWorkingMode = None
        self.optional_bytes_a_mode: SyncStationOptionalBytesAMode = None
        self.acqusition_mode: SyncStationAcquisitionMode = None

        # OPTIONAL BYTES A
        self.optional_bytes_configuration_A: Dict[
            SyncStationProbeConfigMode,
            dict[
                str,
                Union[
                    SyncStationDetectionMode,
                    SyncStationEnableProbeMode,
                ],
            ],
        ] = None

        # START BYTE B
        self.optional_bytes_b_mode: SyncStationOptionalBytesBMode = None

        # OPTIONAL BYTES B
        self.optional_bytes_configuration_B: Dict[str, int] = None

        # Simulate data
        self.read_bytes_counter = 0
        self.read_bytes_cycles = 10
        # self.read_bytes = bytearray(np.load("data\syncstation_read_bytes_example.npy"))

    def _reset_device_parameters(self) -> Dict[str, Enum | Dict[str, Enum]]:
        """
        Resets the device parameters to Default values.

        Returns:
            Dict[str, Union[Enum, Dict[str, Enum]]]:
                Default values of the device attributes.

                The first one are the attributes (configuration mode) name,
                and the second its respective value.
        """

        params = {
            "rec_on_mode": SyncStationRecOnMode.OFF,
            "acqusition_mode": SyncStationAcquisitionMode.STOP,
            "working_mode": SyncStationWorkingMode.EMG,
            "optional_bytes_a_mode": SyncStationOptionalBytesAMode.FOUR,
            "optional_bytes_configuration_A": {
                SyncStationProbeConfigMode.MUOVI_PROBE_ONE: {
                    "detection_mode": SyncStationDetectionMode.MONOPOLAR_GAIN_LOW,
                    "enable_probe": SyncStationEnableProbeMode.ENABLE,
                },
                SyncStationProbeConfigMode.MUOVI_PROBE_TWO: {
                    "detection_mode": SyncStationDetectionMode.MONOPOLAR_GAIN_LOW,
                    "enable_probe": SyncStationEnableProbeMode.ENABLE,
                },
                SyncStationProbeConfigMode.MUOVI_PROBE_THREE: {
                    "detection_mode": SyncStationDetectionMode.MONOPOLAR_GAIN_LOW,
                    "enable_probe": SyncStationEnableProbeMode.DISABLE,
                },
                SyncStationProbeConfigMode.MUOVI_PROBE_FOUR: {
                    "detection_mode": SyncStationDetectionMode.MONOPOLAR_GAIN_LOW,
                    "enable_probe": SyncStationEnableProbeMode.DISABLE,
                },
                SyncStationProbeConfigMode.MUOVI_PLUS_PROBE_ONE: {
                    "detection_mode": SyncStationDetectionMode.MONOPOLAR_GAIN_LOW,
                    "enable_probe": SyncStationEnableProbeMode.DISABLE,
                },
                SyncStationProbeConfigMode.MUOVI_PLUS_PROBE_TWO: {
                    "detection_mode": SyncStationDetectionMode.MONOPOLAR_GAIN_LOW,
                    "enable_probe": SyncStationEnableProbeMode.DISABLE,
                },
                SyncStationProbeConfigMode.DUE_PLUS_PROBE_ONE: {
                    "detection_mode": SyncStationDetectionMode.MONOPOLAR_GAIN_LOW,
                    "enable_probe": SyncStationEnableProbeMode.DISABLE,
                },
                SyncStationProbeConfigMode.DUE_PLUS_PROBE_TWO: {
                    "detection_mode": SyncStationDetectionMode.MONOPOLAR_GAIN_LOW,
                    "enable_probe": SyncStationEnableProbeMode.DISABLE,
                },
                SyncStationProbeConfigMode.DUE_PLUS_PROBE_THREE: {
                    "detection_mode": SyncStationDetectionMode.MONOPOLAR_GAIN_LOW,
                    "enable_probe": SyncStationEnableProbeMode.DISABLE,
                },
                SyncStationProbeConfigMode.DUE_PLUS_PROBE_FOUR: {
                    "detection_mode": SyncStationDetectionMode.MONOPOLAR_GAIN_LOW,
                    "enable_probe": SyncStationEnableProbeMode.DISABLE,
                },
                SyncStationProbeConfigMode.DUE_PLUS_PROBE_FIVE: {
                    "detection_mode": SyncStationDetectionMode.MONOPOLAR_GAIN_LOW,
                    "enable_probe": SyncStationEnableProbeMode.DISABLE,
                },
                SyncStationProbeConfigMode.DUE_PLUS_PROBE_SIX: {
                    "detection_mode": SyncStationDetectionMode.MONOPOLAR_GAIN_LOW,
                    "enable_probe": SyncStationEnableProbeMode.DISABLE,
                },
                SyncStationProbeConfigMode.DUE_PLUS_PROBE_SEVEN: {
                    "detection_mode": SyncStationDetectionMode.MONOPOLAR_GAIN_LOW,
                    "enable_probe": SyncStationEnableProbeMode.DISABLE,
                },
                SyncStationProbeConfigMode.DUE_PLUS_PROBE_EIGHT: {
                    "detection_mode": SyncStationDetectionMode.MONOPOLAR_GAIN_LOW,
                    "enable_probe": SyncStationEnableProbeMode.DISABLE,
                },
                SyncStationProbeConfigMode.DUE_PLUS_PROBE_NINE: {
                    "detection_mode": SyncStationDetectionMode.MONOPOLAR_GAIN_LOW,
                    "enable_probe": SyncStationEnableProbeMode.DISABLE,
                },
                SyncStationProbeConfigMode.DUE_PLUS_PROBE_TEN: {
                    "detection_mode": SyncStationDetectionMode.MONOPOLAR_GAIN_LOW,
                    "enable_probe": SyncStationEnableProbeMode.DISABLE,
                },
            },
            "optional_bytes_configuration_B": {"latency": 200},
        }

        self._update_configuration_parameters(params=params)

    def _connect_to_device(
        self,
        settings: Tuple[QHostAddress | int] = (QHostAddress("192.168.76.1"), 54320),
    ) -> None:
        """
        Function to attempt a connection to the devices

        Args:
            settings (Tuple[str, int], optional):
                If CommunicationProtocol.TCPIP:
                Tuple[0] = IP -> string
                Tuple[1] = Port -> int

                If CommunicationProtocol.SERIAL or CommunicationProtocol.USB:
                Tuple[0] = COM Port -> string
                Tuple[1] = Baudrate -> int

                Defaults to ("192.168.76.1", 54320) for the SyncStation.
        """

        self.interface = QTcpSocket(self)
        self.received_bytes: bytearray = bytearray()
        self._make_request(settings=settings)

    def _make_request(
        self,
        settings: Tuple[QHostAddress | int] = (QHostAddress("192.168.76.1"), 54320),
    ) -> None:
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

                Defaults to ("192.168.76.1", 54320) for the SyncStation.

        Returns:
            bool:
                Returns True if request was successfully. False if not.
        """

        self.interface.connectToHost(settings[0], settings[1])

        if not self.interface.waitForConnected(1000):
            self.log_info("Connection to device failed.", LoggerLevel.ERROR)
            self.is_connected = False
            self.connected_signal.emit(False)
            return False

        self.interface.readyRead.connect(self._read_data)

        self.is_connected = True
        self.connected_signal.emit(True)

        return True

    def _disconnect_from_device(self) -> None:
        """
        Closes the connection to the device.

        self.interface closes and is set to None.
        Device state is_connected is set to False.
        Signal connected_signal emits False.
        """

        self.interface.disconnectFromHost()
        self.interface.readyRead.disconnect(self._read_data)
        self.interface.close()
        self.interface.deleteLater()
        self.interface = None
        self.is_connected = False
        self.connected_signal.emit(False)

    def configure_device(self, params: Dict[str, Union[Enum, Dict[str, Enum]]]) -> None:
        """
        Sends a configuration byte sequence based on selected params to the device.
        An overview of possible configurations can be seen in enums/{device}.

        E.g., enums/sessantaquattro.py

        If SyncStation:
        Emits True through the Signal configured_signal
        and sets the current state is_configured to True if
        successfully configured.

        Args:
            params (Dict[str, Union[Enum, Dict[str, Enum]]]):
                Dictionary that holds the configuration settings
                to which the device should be configured to.

                The first one should be the attributes (configuration mode) name,
                and the second its respective value. Orient yourself on the
                enums of the device to choose the correct configuration settings.
        """

        self._update_configuration_parameters(params=params)

        success = self._configure_start_byte_sequence_A()

        if not success:
            self.log_info(
                "Configuration of start byte sequence A failed.", LoggerLevel.ERROR
            )
            return

        success = self._configure_start_byte_sequence_B()
        if not success:
            self.log_info(
                "Configuration of start byte sequence B failed.", LoggerLevel.ERROR
            )
            return

        success = self._send_configuration_to_device()

        if not success:
            self.log_info(
                f"Configuration: {self.command_bytearray_A} failed.", LoggerLevel.ERROR
            )
            self.configured_signal(False)
            self.is_configured = False
            return

        self.is_configured = True
        self.configured_signal.emit(True)

    def _send_configuration_to_device(self) -> bool:
        self.log_info(
            f"Device configuration sent: {[int.from_bytes(self.command_bytearray_A[i : i + 1], 'big') for i in range(len(self.command_bytearray_A))]}"
        )
        self.interface.write(self.command_bytearray_A)
        # self.interface.write(self.command_bytearray_B) # TODO: Not really necessary at the moment
        return self.interface.waitForBytesWritten(1000)

    def _configure_start_byte_sequence_B(self) -> bool:
        start_byte_b = 1 << 7

        self.command_bytearray_B = bytearray()
        optional_bytes_b_mode = (
            SyncStationOptionalBytesBMode.ONE
            if self.optional_bytes_configuration_B["latency"]
            else SyncStationOptionalBytesBMode.GET_FIRMWARE
        )
        start_byte_b += optional_bytes_b_mode.value << 1
        self.command_bytearray_B.append(start_byte_b)  #

        if optional_bytes_b_mode == SyncStationOptionalBytesBMode.ONE:
            self.command_bytearray_B.append(
                self.optional_bytes_configuration_B["latency"]
            )

        start_byte_b_ckc8 = self._crc_check(
            self.command_bytearray_B, len(self.command_bytearray_B)
        )
        self.command_bytearray_B.append(start_byte_b_ckc8)

        return True

    def _configure_start_byte_sequence_A(self) -> bool:
        start_byte_a = 0
        start_byte_a += self.rec_on_mode.value << 6

        self.sampling_frequency = SYNCSTATION_CHARACTERISTICS_DICT[
            "channel_information"
        ][self.working_mode]["sampling_frequency"]
        self.bytes_in_sample = SYNCSTATION_CHARACTERISTICS_DICT["channel_information"][
            self.working_mode
        ]["bytes_per_sample"]

        self.command_bytearray_A = bytearray()
        self.number_of_channels = 0
        self.number_of_bytes = 0

        self.biosignal_channels = 0
        self.aux_channels = 0
        self.biosignal_channel_indices = []
        self.aux_channel_indices = []

        for key, val in self.optional_bytes_configuration_A.items():
            probe_command = 0
            probe_command += key.value << 4
            probe_command += self.working_mode.value << 3
            probe_command += val["detection_mode"].value << 1
            probe_command += val["enable_probe"].value
            if val["enable_probe"] == SyncStationEnableProbeMode.ENABLE:
                self.command_bytearray_A.append(probe_command)
                channels = PROBE_CHARACTERISTICS_DICT[key]["channels"]
                biosignal_channels = PROBE_CHARACTERISTICS_DICT[key]["biosignal"]
                aux_channels = PROBE_CHARACTERISTICS_DICT[key]["aux"]

                self.biosignal_channel_indices.append(
                    np.arange(
                        self.number_of_channels,
                        self.number_of_channels + biosignal_channels,
                    )
                )
                self.aux_channel_indices.append(
                    np.arange(
                        self.number_of_channels + biosignal_channels,
                        self.number_of_channels + channels,
                    )
                )
                self.number_of_channels += channels
                self.biosignal_channels += biosignal_channels
                self.aux_channels += aux_channels

        self.biosignal_channel_indices = np.hstack(self.biosignal_channel_indices)
        self.aux_channel_indices = np.hstack(self.aux_channel_indices)

        self.number_of_bytes = self.number_of_channels * self.bytes_in_sample

        # Add SyncStation channels
        self.number_of_channels += SYNCSTATION_CHARACTERISTICS_DICT["channels"]
        self.aux_channels += SYNCSTATION_CHARACTERISTICS_DICT["channels"]

        self.number_of_bytes += (
            SYNCSTATION_CHARACTERISTICS_DICT["channels"]
            * SYNCSTATION_CHARACTERISTICS_DICT["bytes_per_sample"]
        )

        self.frame_size = int(
            self.number_of_bytes
            * self.sampling_frequency
            * (1 / SYNCSTATION_CHARACTERISTICS_DICT["FPS"])
        )
        self.buffer_size = SYNCSTATION_CHARACTERISTICS_DICT["package_size"]

        # self.interface.setReadBufferSize(self.buffer_size * 100)

        num_probes = len(self.command_bytearray_A)
        start_byte_a += num_probes << 1
        self.command_bytearray_A.insert(0, start_byte_a)
        start_byte_a_ckc8 = self._crc_check(
            self.command_bytearray_A, len(self.command_bytearray_A)
        )
        self.command_bytearray_A.append(start_byte_a_ckc8)

        return True

    # Function to calculate CRC8
    def _crc_check(self, command_bytes: bytearray, command_length: int) -> bytes:
        """
        Performs the Cyclic Redundancy Check (CRC) of the transmitted bytes.

        Translated function from example code provided by OT Bioelettronica.

        Args:
            command_bytes (bytearray):
                Bytearray of the transmitted bytes.

            command_length (int):
                Length of the transmitted bytes.

        Returns:
            bytes:
                CRC of the transmitted bytes.
        """

        crc = 0
        j = 0

        while command_length > 0:
            extracted_byte = command_bytes[j]
            for i in range(8, 0, -1):
                sum = crc % 2 ^ extracted_byte % 2
                crc = crc // 2

                if sum > 0:
                    crc_bin = format(crc, "08b")
                    a_bin = format(140, "08b")

                    str_list = []

                    for k in range(8):
                        str_list.append("0" if crc_bin[k] == a_bin[k] else "1")

                    crc = int("".join(str_list), 2)

                extracted_byte = extracted_byte // 2

            command_length -= 1
            j += 1

        return crc

    def _update_configuration_parameters(
        self, params: Dict[str, Enum | Dict[str, Enum]]
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
        pass

    def _stop_streaming(self) -> None:
        """
        Sends the command to stop the streaing to the device

        if successful:
            Device state is_streaming is set to False.
            Signal streaming_signal emits False.
        """

        self.command_bytearray_A[0] -= 1
        self.command_bytearray_A[-1] = self._crc_check(
            self.command_bytearray_A, len(self.command_bytearray_A) - 1
        )

        success = self._send_configuration_to_device()
        if not success:
            self.log_info("Streaming stop failed.", LoggerLevel.ERROR)
            return

        self.is_streaming = False
        self.streaming_signal.emit(False)

    def _start_streaming(self) -> None:
        """
        Sends the command to start the streaming to the device.

        if successful:
            Device state is_streaming is set to True.
            Signal streaming_signal emits True.
        """

        self.command_bytearray_A[0] += 1
        self.command_bytearray_A[-1] = self._crc_check(
            self.command_bytearray_A, len(self.command_bytearray_A) - 1
        )

        success = self._send_configuration_to_device()
        if not success:
            self.log_info("Streaming start failed.", LoggerLevel.ERROR)
            return

        self.is_streaming = True
        self.streaming_signal.emit(True)

    def clear_socket(self) -> None:
        """Reads all the bytes from the buffer."""

        self.interface.readAll()

    def _read_data(self) -> None:
        """
        This function is called when bytes are ready to be read in the buffer.
        After reading the bytes from the buffer, _process_data is called to
        decode and process the raw data.
        """
        if not self.is_streaming:
            packet = self.interface.readAll()

        else:
            if self.interface.bytesAvailable() > 0:

                packet = self.interface.readAll()
                packet_bytearray = bytearray(packet.data())

                if not packet_bytearray:
                    self.log_info("No packet received.")
                    return

                self.received_bytes.extend(packet_bytearray)

                while len(self.received_bytes) >= self.frame_size:
                    self._process_data(
                        bytearray(self.received_bytes)[: self.frame_size]
                    )
                    self.received_bytes = bytearray(self.received_bytes)[
                        self.frame_size :
                    ]

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


        In case that the current configuration of the device was requested,
        the configuration is provided through the Signal
        configuration_available_signal that emits the current parameters
        in a dictionary.

        Args:
            input (QByteArray):
                Bytearray of the transmitted raw data.
        """
        data: np.ndarray = np.frombuffer(input, dtype=np.uint8).astype(np.float32)

        samples = self.frame_size // self.number_of_bytes
        data = np.reshape(data, (samples, self.number_of_bytes)).T
        data = self._bytes_to_integers(data)

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
                Defaults to False.

        Returns:
            np.ndarray:
                Extracted EMG channels.
        """
        # Old code -- Possible to remove???
        # channels_to_read = 0
        # emg: list[np.ndarray] = []
        # for device in SyncStationProbeConfigMode:
        #     if (
        #         self.optional_bytes_configuration_A[device]["enable_probe"]
        #         == SyncStationEnableProbeMode.ENABLE
        #     ):
        #         channel_number = PROBE_CHARACTERISTICS_DICT[device]["channels"]
        #         biosignal_channels = PROBE_CHARACTERISTICS_DICT[device]["biosignal"]
        #         emg.append(
        #             data[channels_to_read : channels_to_read + biosignal_channels]
        #         )
        #         channels_to_read += channel_number
        # output = np.vstack(emg)
        # return output
        return (
            data[self.biosignal_channel_indices]
            if len(self.biosignal_channel_indices) > 0
            else None
        )  # * self._conversion_factor if milli_volts else emg

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
        # TODO: implement this function - or check for performance
        return data[self.aux_channel_indices[index]]  # * self._conversion_factor_aux

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
            "samples_per_frame": self.frame_size,
            "biosignal_channels": self.biosignal_channels,
            "aux_channels": self.aux_channels,
        }

    def _integer_to_bytes(self, command: int) -> bytes:
        return int(command).to_bytes(1, byteorder="big")

    # Convert channels from bytes to integers
    def _bytes_to_integers(
        self,
        data: np.ndarray,
    ) -> np.ndarray:
        samples = self.frame_size // self.number_of_bytes
        frame_data = np.zeros((self.number_of_channels, samples), dtype=np.float32)
        channels_to_read = 0
        for device in SyncStationProbeConfigMode:
            if (
                self.optional_bytes_configuration_A[device]["enable_probe"]
                == SyncStationEnableProbeMode.ENABLE
            ):
                channel_number = PROBE_CHARACTERISTICS_DICT[device]["channels"]
                # Convert channel's byte value to integer
                if self.working_mode == SyncStationWorkingMode.EMG:
                    channel_indices = (
                        np.arange(0, channel_number * 2, 2) + channels_to_read * 2
                    )
                    data_sub_matrix = self._decode_int16(data, channel_indices)
                    frame_data[
                        channels_to_read : channels_to_read + channel_number, :
                    ] = data_sub_matrix

                elif self.working_mode == SyncStationWorkingMode.EEG:
                    channel_indices = (
                        np.arange(0, channel_number * 3, 3) + channels_to_read * 2
                    )
                    data_sub_matrix = self._decode_int24(data, channel_indices)
                    frame_data[
                        channels_to_read : channels_to_read + channel_number, :
                    ] = data_sub_matrix

                channels_to_read += channel_number
                del data_sub_matrix
                del channel_indices

        syncstation_aux_bytes_number = (
            SYNCSTATION_CHARACTERISTICS_DICT["channels"]
            * SYNCSTATION_CHARACTERISTICS_DICT["bytes_per_sample"]
        )
        syncstation_aux_starting_byte = (
            self.number_of_bytes - syncstation_aux_bytes_number
        )
        channel_indices = np.arange(
            syncstation_aux_starting_byte,
            syncstation_aux_starting_byte + syncstation_aux_bytes_number,
            2,
        )
        data_sub_matrix = self._decode_int16(data, channel_indices)
        frame_data[channels_to_read : channels_to_read + 6, :] = data_sub_matrix
        return np.array(frame_data)

    def _decode_int24(
        self, data: np.ndarray, channel_indices: np.ndarray
    ) -> np.ndarray:
        data_sub_matrix = (
            data[channel_indices, :] * 2**16
            + data[channel_indices + 1, :] * 2**8
            + data[channel_indices + 2, :]
        )
        negative_indices = np.where(data_sub_matrix >= 2**23)
        data_sub_matrix[negative_indices] -= 2**24

        return data_sub_matrix

    def _decode_int16(
        self, data: np.ndarray, channel_indices: np.ndarray
    ) -> np.ndarray:
        data_sub_matrix = data[channel_indices, :] * 2**8 + data[channel_indices + 1, :]
        negative_indices = np.where(data_sub_matrix >= 2**15)
        data_sub_matrix[negative_indices] -= 2**16
        return data_sub_matrix

    def closeEvent(self, event):
        self._disconnect_from_device()
        event.accept()
