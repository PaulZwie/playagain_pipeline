"""
1)  Quattrocento Light class for real-time interface to 
    Quattrocento using OT Biolab Light.

2)  Quattrocento class for direct real-time interface to 
    Quattrocento without using OT Biolab Light.

Developer: Dominik I. Braun
Contact: dome.braun@fau.de
Last Update: 2023-10-23
"""

# Python Libraries
from __future__ import annotations
from typing import TYPE_CHECKING, Union, Dict, Tuple
from PySide6.QtNetwork import QTcpSocket, QHostAddress
from PySide6.QtCore import QIODevice, QByteArray
import numpy as np

# Local Libraries
from device_interfaces.devices.device import BaseDevice
from device_interfaces.enums.device import Device, CommunicationProtocol, LoggerLevel
from device_interfaces.enums.quattrocento import *

if TYPE_CHECKING:
    # Python Libraries
    from PySide6.QtWidgets import QMainWindow, QWidget
    from enum import Enum


class QuattrocentoLight(BaseDevice):
    def __init__(
        self,
        parent: Union[QMainWindow, QWidget] = None,
    ) -> None:
        """
        QuattrocentoLight device class derived from BaseDevice class.
        The QuattrocentoLight is using a TCP/IP protocol to communicate with the device.

        This class directly interfaces with the OT Biolab Light software from
        OT Bioelettronica. The configured settings of the device have to
        match the settings from the OT Biolab Light software!

        This class also works with the EMG simulator from the N-squared Lab.
        The repository can be found here:
        https://gitos.rrze.fau.de/n-squared-lab/software/emg-interface/emg_simulator/basic_emg_simulator
        """
        super().__init__(parent)

        # Device Parameters
        self.name: Device = Device.QUATTROCENTO_LIGHT
        self.communication_protocol: CommunicationProtocol = CommunicationProtocol.TCPIP

        # Fix parameters
        self.number_of_channels: int = 408  # 384 biosignals + 16 AUX + 1 Ramp + 7 None
        self.biosignal_channels: int = None
        self.aux_channels: int = 16  # 16 AUX channels

        self.grid_size: int = 64  # Electrode Grid size

        self._conversion_factor: float = 5 / (2**16) / 150 * 1000
        self._conversion_factor_aux: float = 5 / (2**16) / 0.5

        # Customizable parameters
        self.sampling_frequency: int = None
        self.streaming_frequency: int = None
        self.grids: list[int] = None

        # Calculated parameters
        # Buffer Size: 2 (int16 => 2x Bytes) * Frame Length
        # * Channels Available = Amount of Bytes per Frame
        self.buffer_size: int = None
        self.frame_len: int = None

        # Define socket
        self.interface: QTcpSocket = None

    def _connect_to_device(
        self, settings: Tuple[str, int] = (QHostAddress("127.0.0.1"), 31000)
    ) -> None:
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

                Defaults to (LocalHost="127.0.0.1", 31000) for Quattrocento Light.
        """
        self.interface = QTcpSocket(self)
        self.received_bytes: QByteArray = bytearray()

        self._make_request(settings=settings)

    def _make_request(
        self, settings: Tuple[str, int] = (QHostAddress.LocalHost, 31000)
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

                Defaults to (LocalHost="127.0.0.1", 31000) for Quattrocento Light.

        Returns:
            bool:
                Returns True if request was successfully. False if not.
        """
        self.interface.connectToHost(settings[0], settings[1], QIODevice.ReadWrite)

        if not self.interface.waitForConnected(1000):
            self.log_info("Connection to device failed.", LoggerLevel.ERROR)
            self.connected_signal.emit(False)
            self.is_connected = False
            return False

        self.interface.readyRead.connect(self._read_data)

        return True

    def _disconnect_from_device(self) -> None:
        """
        Closes the connection to the device.

        self.interface closes and is set to None.
        Device state is_connected is set to False.
        Signal connected_signal emits False.
        """
        self.interface.disconnectFromHost()
        self.interface.close()
        self.interface = None
        self.is_connected = False
        self.connected_signal.emit(False)

    def configure_device(self, params: Dict[str, Union[Enum, Dict[str, Enum]]]) -> None:
        """
        Sends a configuration byte sequence based on selected params to the device.
        An overview of possible configurations can be seen in enums/{device}.

        E.g., enums/sessantaquattro.py

        If Quattrocento or QuattrocentoLight:
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

        self.biosignal_channels = len(self.grids) * self.grid_size

        self.frame_len = self.sampling_frequency // self.streaming_frequency

        self.buffer_size = 2 * self.frame_len * self.number_of_channels

        self.is_configured = True
        self.configured_signal.emit(True)

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
        pass

    def _start_streaming(self) -> None:
        """
        Sends the command to start the streaming to the device.

        if successful: -> Is done in _read_data for Quattrocento
            Device state is_streaming is set to True.
            Signal streaming_signal emits True.
        """
        self.clear_socket()
        self.interface.write(b"startTX")
        self.is_streaming = True
        self.streaming_signal.emit(True)

    def _stop_streaming(self) -> None:
        """
        Sends the command to stop the streaing to the device

        if successful:
            Device state is_streaming is set to False.
            Signal streaming_signal emits False.
        """

        self.interface.write(b"stopTX")
        self.interface.waitForBytesWritten(1000)
        self.clear_socket()
        self.is_streaming = False
        self.streaming_signal.emit(False)

    def clear_socket(self) -> None:
        """Reads all the bytes from the buffer."""

        self.interface.readAll()

    def _read_data(self) -> None:
        """
        This function is called when bytes are ready to be read in the buffer.
        After reading the bytes from the buffer, _process_data is called to
        decode and process the raw data.
        """
        if not self.is_connected:
            if self.interface.bytesAvailable() == 8:
                packet = self.interface.read(8)
                if not packet:
                    return
                try:
                    decoded_msg = packet.data().decode("utf-8")
                    # print(decoded_msg, type(decoded_msg))
                    if decoded_msg == "OTBioLab":
                        self.is_connected = True
                        self.connected_signal.emit(True)
                        return
                except UnicodeDecodeError:
                    self.interface.readAll()
                    return
        if not self.is_streaming:
            self.clear_socket()
            return

        while self.interface.bytesAvailable() > self.buffer_size:
            packet = self.interface.read(self.buffer_size)
            if not packet:
                self.received_bytes = bytearray()
            self.received_bytes.extend(packet)

            if len(self.received_bytes) == self.buffer_size:
                self._process_data(self.received_bytes)
                self.received_bytes = bytearray()
            else:
                self.received_bytes = bytearray()

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

        # Decode the data
        data: np.ndarray = np.frombuffer(input, dtype=np.int16)

        # Rehsape it to the correct format
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

                No effect for QuattrocentoLight.

        Returns:
            np.ndarray:
                Extracted EMG channels.
        """
        return (
            (
                np.concatenate([data[ch * 64 : ch * 64 + 64] for ch in self.grids])
                * self._conversion_factor
            )
            if len(self.grids) > 0
            else None
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
        return data[index + 384] * self._conversion_factor_aux

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


class Quattrocento(BaseDevice):
    """
    Quattrocento device class derived from BaseDevice class.

    The Quattrocento class is using a TCP/IP protocol to communicate with the device.

    Communication Protocol:
    Command string composed by 40 bytes has to be send to the device.
    A detailed manual for the communication protocol can be found here:
    https://otbioelettronica.it/en/download/#55-148-wpfd-quattrocento

    Additional descriptions can be found in device_interfaces/enums/quattrocento.py

    ACQ_SETT:
    [<1> <DECIM> <REC_ON> <FSAMP1> <FSAMP0> <NCH1> <NCH0> <ACQ_ON>]

    AN_OUT_IN_SEL BYTE:
    [<0> <0> <ANOUT_GAIN1> <ANOUT_GAIN0> <INSEL3> <INSEL2> <INSEL1> <INSEL0>]

    AN_OUT_CH_SEL:
    [<0> <0> CHSEL5> <CHSEL4> <CHSEL3> <CHSEL2> <CHSEL1> <CHSEL0>]

    INX_CONF0:
    [<0> <MUS6> <MUS5> <MUS4> <MUS3> <MUS2> <MUS1> <MUS0>]

    INX_CONF1:
    [<SENS4> <SENS3> <SENS2> <SENS1> <SENS0> <ADAPT2> <ADAPT1> <ADAPT0>]

    INX_CONF2:
    [<SIDE1> <SIDE0> <HPF1> <HPF0> <LPF1> <LPF0> <MODE1> <MODE0>]

    """

    def __init__(
        self,
        parent: Union[QMainWindow, QWidget] = None,
    ) -> None:
        super().__init__(parent)

        # Device Parameters
        self.name: Device = Device.QUATTROCENTO
        self.communication_protocol: CommunicationProtocol = CommunicationProtocol.TCPIP

        # Fix parameters
        self._conversion_factor: float = 5 / (2**16) / 150 * 1000
        self._conversion_factor_aux: float = 5 / (2**16) / 0.5

        # Customizable parameters
        self.sampling_frequency: int = None
        self.frame_len: int = None
        self.number_of_channels: int = None
        self.biosignal_channels: int = 16
        self.aux_channels: int = None

        self.buffer_size: int = None
        self.grids: list[int] = None
        self.grid_size = (
            64  # TODO: This is more flexible if other configurations are allowed
        )

        # Configuration modes
        self.command_bytearray: bytearray = bytearray(40)

        # ------- ACQ_SETT ----------
        self.acquisiton_configuration: Dict[str, Enum] = {
            "decim_mode": None,
            "recording_mode": None,
            "sampling_frequency_mode": None,
            "number_of_channels_mode": None,
            "acquisition_mode": None,
        }

        # ---------- AN_OUT_IN_SEL ----------
        self.analog_output_input_selection_configuration: Dict[str, Enum] = {
            "analog_output_gain": None,
            "input_selection": None,
        }

        # ---------- AN_OUT_CH_SEL ----------
        self.analog_output_channel_selection_configuration: Dict[str, Enum] = {
            "channel_selection": None,
        }

        # Should be extended to IN1, IN2, ..., IN8
        # ---------- IN1-4 ----------
        self.in_top_left_configuration: Dict[str, Enum] = {
            "muscle": None,
            "sensor": None,
            "adaptor": None,
            "side": None,
            "high_pass_filter": None,
            "low_pass_filter": None,
            "mode": None,
        }

        # ---------- IN5-8 ----------
        self.in_top_right_configuration: Dict[str, Enum] = {
            "muscle": None,
            "sensor": None,
            "adaptor": None,
            "side": None,
            "high_pass_filter": None,
            "low_pass_filter": None,
            "mode": None,
        }

        # ------ MULTIPLE IN 1 ------
        self.multiple_in1_configuration: Dict[str, Enum] = {
            "muscle": None,
            "sensor": None,
            "adaptor": None,
            "side": None,
            "high_pass_filter": None,
            "low_pass_filter": None,
            "mode": None,
        }

        # ------ MULTIPLE IN 2 ------
        self.multiple_in2_configuration: Dict[str, Enum] = {
            "muscle": None,
            "sensor": None,
            "adaptor": None,
            "side": None,
            "high_pass_filter": None,
            "low_pass_filter": None,
            "mode": None,
        }

        # ------ MULTIPLE IN 3 ------
        self.multiple_in3_configuration: Dict[str, Enum] = {
            "muscle": None,
            "sensor": None,
            "adaptor": None,
            "side": None,
            "high_pass_filter": None,
            "low_pass_filter": None,
            "mode": None,
        }

        # ------ MULTIPLE IN 4 ------
        self.multiple_in4_configuration: Dict[str, Enum] = {
            "muscle": None,
            "sensor": None,
            "adaptor": None,
            "side": None,
            "high_pass_filter": None,
            "low_pass_filter": None,
            "mode": None,
        }

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
            "acquisiton_configuration": {
                "decim_mode": QuattrocentoDecim.ACTIVE,
                "recording_mode": QuattrocentoRecording.STOP,
                "sampling_frequency_mode": QuattrocentoSamplingFrequency.MEDIUM,
                "number_of_channels_mode": QuattrocentoNumberOfChannels.ULTRA,
                "acquisition_mode": QuattrocentoAcquisition.INACTIVE,
            },
            "analog_output_input_selection_configuration": {
                "analog_output_gain": QuattrocentoAnalogOutputGain.LOW,
                "input_selection": QuattrocentoSourceInput.IN_I,
            },
            "analog_output_channel_selection_configuration": {
                "channel_selection": 0,
            },
            "in_top_left_configuration": {
                "muscle": QuattrocentoMuscle.NOT_DEFINED,
                "sensor": QuattrocentoSensor.SIXTYFOUR_ELECTRODE_GRID_10MM,
                "adaptor": QuattrocentoAdaptor.SIXTYFOUR_AD1x64,
                "side": QuattrocentoSide.NOT_DEFINED,
                "high_pass_filter": QuattrocentoHighPassFilter.MEDIUM,
                "low_pass_filter": QuattrocentoLowPassFilter.MEDIUM,
                "mode": QuattrocentoDetectionMode.MONOPOLAR,
            },
            "in_top_right_configuration": {
                "muscle": QuattrocentoMuscle.NOT_DEFINED,
                "sensor": QuattrocentoSensor.SIXTYFOUR_ELECTRODE_GRID_10MM,
                "adaptor": QuattrocentoAdaptor.SIXTYFOUR_AD1x64,
                "side": QuattrocentoSide.NOT_DEFINED,
                "high_pass_filter": QuattrocentoHighPassFilter.MEDIUM,
                "low_pass_filter": QuattrocentoLowPassFilter.MEDIUM,
                "mode": QuattrocentoDetectionMode.MONOPOLAR,
            },
            "multiple_in1_configuration": {
                "muscle": QuattrocentoMuscle.NOT_DEFINED,
                "sensor": QuattrocentoSensor.SIXTYFOUR_ELECTRODE_GRID_10MM,
                "adaptor": QuattrocentoAdaptor.SIXTYFOUR_AD1x64,
                "side": QuattrocentoSide.NOT_DEFINED,
                "high_pass_filter": QuattrocentoHighPassFilter.MEDIUM,
                "low_pass_filter": QuattrocentoLowPassFilter.MEDIUM,
                "mode": QuattrocentoDetectionMode.MONOPOLAR,
            },
            "multiple_in2_configuration": {
                "muscle": QuattrocentoMuscle.NOT_DEFINED,
                "sensor": QuattrocentoSensor.SIXTYFOUR_ELECTRODE_GRID_10MM,
                "adaptor": QuattrocentoAdaptor.SIXTYFOUR_AD1x64,
                "side": QuattrocentoSide.NOT_DEFINED,
                "high_pass_filter": QuattrocentoHighPassFilter.MEDIUM,
                "low_pass_filter": QuattrocentoLowPassFilter.MEDIUM,
                "mode": QuattrocentoDetectionMode.MONOPOLAR,
            },
            "multiple_in3_configuration": {
                "muscle": QuattrocentoMuscle.NOT_DEFINED,
                "sensor": QuattrocentoSensor.SIXTYFOUR_ELECTRODE_GRID_10MM,
                "adaptor": QuattrocentoAdaptor.SIXTYFOUR_AD1x64,
                "side": QuattrocentoSide.NOT_DEFINED,
                "high_pass_filter": QuattrocentoHighPassFilter.MEDIUM,
                "low_pass_filter": QuattrocentoLowPassFilter.MEDIUM,
                "mode": QuattrocentoDetectionMode.MONOPOLAR,
            },
            "multiple_in4_configuration": {
                "muscle": QuattrocentoMuscle.NOT_DEFINED,
                "sensor": QuattrocentoSensor.SIXTYFOUR_ELECTRODE_GRID_10MM,
                "adaptor": QuattrocentoAdaptor.SIXTYFOUR_AD1x64,
                "side": QuattrocentoSide.NOT_DEFINED,
                "high_pass_filter": QuattrocentoHighPassFilter.MEDIUM,
                "low_pass_filter": QuattrocentoLowPassFilter.MEDIUM,
                "mode": QuattrocentoDetectionMode.MONOPOLAR,
            },
            "grids": [i for i in range(6)],
        }

        self._update_configuration_parameters(params=params)

        return params

    def _connect_to_device(
        self, settings: Tuple[QHostAddress, int] = (QHostAddress("169.254.1.10"), 23456)
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

                Defaults to ("169.254.1.10", 23456) for Quattrocento.
                Can be changed on the Webpage of the device.
        """

        self.interface = QTcpSocket(self)
        self.received_bytes: QByteArray = bytearray()
        self._make_request(settings=settings)

    def _make_request(
        self, settings: Tuple[str, int] = (QHostAddress("169.254.1.10"), 23456)
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

                Defaults to ("169.254.1.10", 23456) for Quattrocento.
                Can be changed on the Webpage of the device.

        Returns:
            bool:
                Returns True if request was successfully. False if not.
        """

        self.interface.connectToHost(settings[0], settings[1], QIODevice.ReadWrite)

        if not self.interface.waitForConnected(1000):
            self.log_info("Connection to device failed.", LoggerLevel.ERROR)
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

        If Quattrocento or QuattrocentoLight:
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

        success = self._set_sampling_frequency_parameter()
        if not success:
            return

        success = self._set_number_of_channels_parameter()
        if not success:
            return

        # 2 (int16 => 2x Bytes) * Number Of Channels = Amount of Bytes per Frame
        self.buffer_size = 2 * self.number_of_channels
        self.biosignal_channels = len(self.grids) * self.grid_size

        self._configure_command_bytearray()

        success = self._send_configuration_to_device()
        if not success:
            self.log_info(
                f"Configuration: {self.command_bytearray} failed.", LoggerLevel.ERROR
            )
            return False

        self.log_info(f"Configuration: {self.command_bytearray} successful.")

        self.is_configured = True
        self.configured_signal.emit(True)

    def _send_configuration_to_device(self) -> bool:
        self.interface.write(self.command_bytearray)
        return self.interface.waitForBytesWritten(1000)

    def _configure_command_bytearray(self) -> None:
        # Byte 1: Configure ACQ_SETT
        self.command_bytearray[0] = self._configure_acq_sett_byte()

        # Byte 2: Configure AN_OUT_IN_SEL
        self.command_bytearray[1] = self._configure_an_out_in_sel_byte()

        # Byte 3: Configure AN_OUT_CH_SEL
        self.command_bytearray[2] = self._configure_an_out_ch_sel_byte()

        # Byte 4-15: Configure IN1-4 -> TODO: change that to individual configuration
        muscle_byte = self._configure_muscle_byte(self.in_top_left_configuration)
        sensor_adaper_byte = self._configure_sensor_adaptor_byte(
            self.in_top_left_configuration
        )
        input_characteristics_byte = self._configure_input_characteristics_byte(
            self.in_top_left_configuration
        )
        for i in range(4):
            self.command_bytearray[3 + i * 3] = muscle_byte
            self.command_bytearray[3 + i * 3 + 1] = sensor_adaper_byte
            self.command_bytearray[3 + i * 3 + 2] = input_characteristics_byte

        # Byte 16-27: Configure IN5-8 -> TODO: change that to individual configuration
        muscle_byte = self._configure_muscle_byte(self.in_top_right_configuration)
        sensor_adaper_byte = self._configure_sensor_adaptor_byte(
            self.in_top_right_configuration
        )
        input_characteristics_byte = self._configure_input_characteristics_byte(
            self.in_top_right_configuration
        )
        for i in range(4):
            self.command_bytearray[15 + i * 3] = muscle_byte
            self.command_bytearray[15 + i * 3 + 1] = sensor_adaper_byte
            self.command_bytearray[15 + i * 3 + 2] = input_characteristics_byte

        # Byte 28-30: Configure MULTIPLE IN 1
        self.command_bytearray[27] = self._configure_muscle_byte(
            self.multiple_in1_configuration
        )
        self.command_bytearray[28] = self._configure_sensor_adaptor_byte(
            self.multiple_in1_configuration
        )
        self.command_bytearray[29] = self._configure_input_characteristics_byte(
            self.multiple_in1_configuration
        )

        # Byte 31-33: Configure MULTIPLE IN 2
        self.command_bytearray[30] = self._configure_muscle_byte(
            self.multiple_in2_configuration
        )
        self.command_bytearray[31] = self._configure_sensor_adaptor_byte(
            self.multiple_in2_configuration
        )
        self.command_bytearray[32] = self._configure_input_characteristics_byte(
            self.multiple_in2_configuration
        )

        # Byte 34-36: Configure MULTIPLE IN 3
        self.command_bytearray[33] = self._configure_muscle_byte(
            self.multiple_in3_configuration
        )
        self.command_bytearray[34] = self._configure_sensor_adaptor_byte(
            self.multiple_in3_configuration
        )
        self.command_bytearray[35] = self._configure_input_characteristics_byte(
            self.multiple_in3_configuration
        )

        # Byte 37-39: Configure MULTIPLE IN 4
        self.command_bytearray[36] = self._configure_muscle_byte(
            self.multiple_in4_configuration
        )
        self.command_bytearray[37] = self._configure_sensor_adaptor_byte(
            self.multiple_in4_configuration
        )
        self.command_bytearray[38] = self._configure_input_characteristics_byte(
            self.multiple_in4_configuration
        )

        # CRC Byte
        self.command_bytearray[39] = self._crc_check(self.command_bytearray, 39)

    def _configure_acq_sett_byte(self) -> int:
        acq_sett_byte = 1 << 7
        acq_sett_byte += self.acquisiton_configuration["decim_mode"].value << 6
        acq_sett_byte += self.acquisiton_configuration["recording_mode"].value << 5
        acq_sett_byte += (
            self.acquisiton_configuration["sampling_frequency_mode"].value << 3
        )
        acq_sett_byte += (
            self.acquisiton_configuration["number_of_channels_mode"].value << 1
        )
        acq_sett_byte += self.acquisiton_configuration["acquisition_mode"].value

        return acq_sett_byte

    def _configure_an_out_in_sel_byte(self) -> int:
        an_out_in_sel_byte = (
            self.analog_output_input_selection_configuration["analog_output_gain"].value
            << 4
        )
        an_out_in_sel_byte += self.analog_output_input_selection_configuration[
            "input_selection"
        ].value

        return an_out_in_sel_byte

    def _configure_an_out_ch_sel_byte(self) -> int:
        return self.analog_output_channel_selection_configuration["channel_selection"]

    def _configure_muscle_byte(self, input_configuration: dict[str, Enum]) -> int:
        return input_configuration["muscle"].value

    def _configure_sensor_adaptor_byte(
        self, input_configuration: dict[str, Enum]
    ) -> int:
        sensor_adaptor_byte = input_configuration["sensor"].value << 3
        sensor_adaptor_byte += input_configuration["adaptor"].value

        return sensor_adaptor_byte

    def _configure_input_characteristics_byte(
        self, input_configuration: dict[str, Enum]
    ) -> int:
        input_characteristics_byte = input_configuration["side"].value << 6
        input_characteristics_byte += input_configuration["high_pass_filter"].value << 4
        input_characteristics_byte += input_configuration["low_pass_filter"].value << 2
        input_characteristics_byte += input_configuration["mode"].value

        return input_characteristics_byte

    def _set_number_of_channels_parameter(self) -> bool:
        mode = self.acquisiton_configuration["number_of_channels_mode"]
        match mode:
            case QuattrocentoNumberOfChannels.LOW:
                self.number_of_channels = 120
            case QuattrocentoNumberOfChannels.MEDIUM:
                self.number_of_channels = 216
            case QuattrocentoNumberOfChannels.HIGH:
                self.number_of_channels = 312
            case QuattrocentoNumberOfChannels.ULTRA:
                self.number_of_channels = 408
            case _:
                self.log_info(
                    f"Number of Channels Mode: {mode} not defined",
                    LoggerLevel.ERROR,
                )
                return False

        return True

    def _set_sampling_frequency_parameter(self) -> bool:
        mode = self.acquisiton_configuration["sampling_frequency_mode"]
        match mode:
            case QuattrocentoSamplingFrequency.LOW:
                self.sampling_frequency = 512
            case QuattrocentoSamplingFrequency.MEDIUM:
                self.sampling_frequency = 2048
            case QuattrocentoSamplingFrequency.HIGH:
                self.sampling_frequency = 5120
            case QuattrocentoSamplingFrequency.ULTRA:
                self.sampling_frequency = 10240
            case _:
                self.log_info(
                    f"Sampling Frequency Mode: {mode} not defined", LoggerLevel.ERROR
                )
                return False

        return True

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
        pass

    def _stop_streaming(self) -> None:
        """
        Sends the command to stop the streaing to the device

        if successful:
            Device state is_streaming is set to False.
            Signal streaming_signal emits False.
        """

        # self.command_bytearray[0] -= 1 << 5
        self.command_bytearray[0] -= 1
        self.command_bytearray[39] = self._crc_check(self.command_bytearray, 39)

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

        # If this doesnt work, try to set the recording bit instead of acquisition bit
        # self.command_bytearray[0] += 1 << 5
        self.command_bytearray[0] += 1
        self.command_bytearray[39] = self._crc_check(self.command_bytearray, 39)

        self._send_configuration_to_device()

        self.is_streaming = True
        self.streaming_signal.emit(True)

        self.log_info("Streaming started.")

    def clear_socket(self) -> None:
        """Reads all the bytes from the buffer."""

        self.interface.readAll()

    def _read_data(self) -> None:
        """
        This function is called when bytes are ready to be read in the buffer.
        After reading the bytes from the buffer, _process_data is called to
        decode and process the raw data.
        """
        while self.interface.bytesAvailable() > self.buffer_size:
            packet = self.interface.readAll()
            if not packet:
                self.received_bytes = bytearray()
            self.received_bytes.extend(packet)

            if len(self.received_bytes) % self.buffer_size == 0:
                self._process_data(self.received_bytes)
                self.received_bytes = bytearray()

            else:
                self.received_bytes = bytearray()

    def _process_data(self, input: QByteArray) -> None:
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
        data: np.ndarray = np.frombuffer(input, dtype="<i2")
        data = data.reshape((self.number_of_channels, -1), order="F").astype(np.double)

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
        if not self.grids:
            return None

        return (
            np.concatenate([data[ch * 64 : ch * 64 + 64] for ch in self.grids])
            * self._conversion_factor
            if milli_volts
            else np.concatenate([data[ch * 64 : ch * 64 + 64] for ch in self.grids])
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
        return data[384 + index] * self._conversion_factor_aux

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
