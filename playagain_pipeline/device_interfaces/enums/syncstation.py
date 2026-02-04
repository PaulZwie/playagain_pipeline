"""
Enum classes for the SyncStation device.

Developer: Dominik I. Braun
Contact: dome.braun@fau.de
Last Update: 2024-02-15
"""

from enum import Enum


class SyncStationStartByteMode(Enum):
    """
    Enum class for the start byte mode of the SyncStation device.

    0 -> A:  Start Byte A
    1 -> B:  Start Byte B
    """

    A = 0
    B = 1


# Start Byte A
class SyncStationRecOnMode(Enum):
    """
    Enum class for the recording mode of the SyncStation device.

    0 -> OFF:  the PC is not recording the received signals from
                SyncStation. If the Timestapms.log is closed if it
                was previously opened
    1 -> ON:    the PC is recording the signals received by the SyncStation.
                When triggered, this bit reset the internal timer for the
                ramp counter sent on the Accessory Ch2 and start the log of
                the timestamps on the internal timestamps.log file.
    """

    OFF = 0
    ON = 1


class SyncStationOptionalBytesAMode(Enum):
    """
    Enum class for the control byte mode of the SyncStation device.
    Defines the size of the command string. The value set with these
    5 bits indicates how many CONTROL BYTES follows. The value can
    range from 1 to 16, it doesn`t include the CRC8 byte terminating
    the command string who must always be in the configuration string.

    1 -> ONE:  1 control byte follows
    2 -> TWO:  2 control bytes follow
    3 -> THREE:  3 control bytes follow
    4 -> FOUR:  4 control bytes follow
    5 -> FIVE:  5 control bytes follow
    6 -> SIX:  6 control bytes follow
    7 -> SEVEN:  7 control bytes follow
    8 -> EIGHT:  8 control bytes follow
    9 -> NINE:  9 control bytes follow
    10 -> TEN: 10 control bytes follow
    11 -> ELEVEN: 11 control bytes follow
    12 -> TWELVE: 12 control bytes follow
    13 -> THIRTEEN: 13 control bytes follow
    14 -> FOURTEEN: 14 control bytes follow
    15 -> FIFTEEN: 15 control bytes follow
    16 -> SIXTEEN: 16 control bytes follow
    """

    ERROR = 0
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    ELEVEN = 11
    TWELVE = 12
    THIRTEEN = 13
    FOURTEEN = 14
    FIFTEEN = 15
    SIXTEEN = 16


class SyncStationAcquisitionMode(Enum):
    """
    Enum class for the acquisition mode of the SyncStation device.

    0 -> STOP: stop data transfer on the TCP socket
    1 -> START: start data transfer of Auxiliary, Accessory and
                signals channels from muovi probes with EN
                bit = 1
    """

    STOP = 0
    START = 1


# START BYTE B
class SyncStationOptionalBytesBMode(Enum):
    """
    Enum class for the control byte mode of the SyncStation device.
    Defines the size of the command string. The value set with these
    5 bits indicates how many CONTROL BYTES follows. The value can
    range from 1 to 16, it doesn’t include the CRC8 byte terminating
    the command string who must always be in the configuration string.

    0 -> GET_FIRMWARE:  Get the firmware version of the SyncStation as
                        plain text
    1 -> TWO:  2 control bytes follow
    2 -> THREE:  3 control bytes follow
    3 -> FOUR:  4 control bytes follow
    4 -> FIVE:  5 control bytes follow
    """

    GET_FIRMWARE = 0
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4


# CONTROL BYTEX
class SyncStationProbeConfigMode(Enum):
    """
    Enum class for the probe configuration mode of the SyncStation device.

    0 -> MUOVI_PROBE_ONE:  Command sets the configuration of muovi probe 1
    1 -> MUOVI_PROBE_TWO:  Command sets the configuration of muovi probe 2
    2 -> MUOVI_PROBE_THREE:  Command sets the configuration of muovi probe 3
    3 -> MUOVI_PROBE_FOUR:  Command sets the configuration of muovi probe 4
    4 -> MUOVI_PLUS_PROBE_ONE:  Command sets the configuration of
                                muovi+/sessantaquattro(+) probe 1
    5 -> MUOVI_PLUS_PROBE_TWO:  Command sets the configuration of
                                muovi+/sessantaquattro(+) probe 2
    6 -> DUE_PLUS_PROBE_ONE:  Command sets the configuration of
                                due+/sessantotto(+) probe 1
    7 -> DUE_PLUS_PROBE_TWO:  Command sets the configuration of
                                due+/sessantotto(+) probe 2
    8 -> DUE_PLUS_PROBE_THREE:  Command sets the configuration of
                                due+/sessantotto(+) probe 3
    9 -> DUE_PLUS_PROBE_FOUR:  Command sets the configuration of
                                due+/sessantotto(+) probe 4
    10 -> DUE_PLUS_PROBE_FIVE:  Command sets the configuration of
                                due+/sessantotto(+) probe 5
    11 -> DUE_PLUS_PROBE_SIX:  Command sets the configuration of
                                due+/sessantotto(+) probe 6
    12 -> DUE_PLUS_PROBE_SEVEN:  Command sets the configuration of
                                due+/sessantotto(+) probe 7
    13 -> DUE_PLUS_PROBE_EIGHT:  Command sets the configuration of
                                due+/sessantotto(+) probe 8
    14 -> DUE_PLUS_PROBE_NINE:  Command sets the configuration of
                                due+/sessantotto(+) probe 9
    15 -> DUE_PLUS_PROBE_TEN:  Command sets the configuration of
                                due+/sessantotto(+) probe 10

    """

    MUOVI_PROBE_ONE = 0
    MUOVI_PROBE_TWO = 1
    MUOVI_PROBE_THREE = 2
    MUOVI_PROBE_FOUR = 3
    MUOVI_PLUS_PROBE_ONE = 4
    MUOVI_PLUS_PROBE_TWO = 5
    DUE_PLUS_PROBE_ONE = 6
    DUE_PLUS_PROBE_TWO = 7
    DUE_PLUS_PROBE_THREE = 8
    DUE_PLUS_PROBE_FOUR = 9
    DUE_PLUS_PROBE_FIVE = 10
    DUE_PLUS_PROBE_SIX = 11
    DUE_PLUS_PROBE_SEVEN = 12
    DUE_PLUS_PROBE_EIGHT = 13
    DUE_PLUS_PROBE_NINE = 14
    DUE_PLUS_PROBE_TEN = 15


class SyncStationWorkingMode(Enum):
    """
    Enum class for the sampling frequency mode of the SyncStation device.

    0 -> EEG:  EEG Mode Fsamp 500 Hz, DC coupled, 24 bit resolution
    1 -> EMG:  EMG Mode Fsamp 2000 Hz, high pass filter at 10 Hz*, 16 bit resolution
    """

    EEG = 0
    EMG = 1


class SyncStationDetectionMode(Enum):
    """
    Enum class for the detection mode of the SyncStation device.

    0 -> MONOPOLAR_GAIN_HIGH:   Monopolar mode with preamp gain 8. 32 monopolar
                                bioelectrical signals + 6 accessory signals.
                                Resolution is 286.1 nV and range +/-9.375 mV(2).
    1 -> MONOPOLAR_GAIN_LOW:    This option only affects EMG mode and firmware
                                version 3.2.0 or higher. If EEG is set, or previous
                                version of firmware is used, this mode is the same
                                as 00. Monopolar mode with preamp gain is 4.
                                32 monopolar bioelectrical signals + 6 accessory
                                signals.
                                Resolution is 572.2 nV and range +/-18.75 mV(2)
    2 -> IMPEDANCE_CHECK:   Impedance check on all monopolar bioelectrical signals +
                            IMU/AUX/accessory.
    3 -> TEST: Sends ramps on all bioelectrical + IMU/AUX/accessory channels

    (2) Preamp gain of 4 has a double input range and a slightly larger noise w.r.t. 
    preamp gain of 8. It can be used when DC component of EMG signals is higher and 
    generates saturation before the high pass filter resulting in flat signals. 
    The input range before the high pass filter is +/-600mV when the preamp is set 
    to 4 and +/-300mV when the preamp is set to 8.
    """

    MONOPOLAR_GAIN_HIGH = 0
    MONOPOLAR_GAIN_LOW = 1
    IMPEDANCE_CHECK = 2
    TEST = 3


class SyncStationEnableProbeMode(Enum):
    """
    Enum class for the mode to enable data transfer of corresponding probe
    to the SyncStation device.

    0 -> DISABLE:  Disable data transfer to the station
    1 -> ENABLE:  Enable data transfer to the station
    """

    DISABLE = 0
    ENABLE = 1
