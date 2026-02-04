"""
Enum classes for the Sessantaquattro device.

Developer: Dominik I. Braun
Contact: dome.braun@fau.de
Last Update: 2024-02-15
"""

from enum import Enum


class MuoviWorkingMode(Enum):
    """
    Enum class for the working mode of the Muovi device.

    0 -> EEG Mode: FSAMP 500 Hz, DC coupled, 24 bit resolution
    1 -> EMG Mode: FSAMP 2000 Hz, DC coupled, 16 bit resolution, High pass filtered at 10 Hz*1

    *1
    High pass filter implemented by firmware subtracting the exponential moving average, obtained by:
    Average_ChX[t] = (1-alpha) Average_ChX[t-1] + alpha ChX[t]
    Where alpha is equal to 1/25 for MODE = 0, 1 or 2. It is equal to 1/2 in case of Impedance check.
    For the standard modes, this result in a high pass filter with a cut-off frequency of 10.5 Hz, when sampling
    the signals at 2000 Hz. More in general the cut-off frequency is Fsamp/190

    """

    EEG = 0
    EMG = 1


class MuoviDetectionMode(Enum):
    """
    Enum class for the detection mode of the Muovi device.

    0 -> MONOPOLAR_GAIN_8:  Monopolar Mode.preamp gain 8. 32 monopolar bioelectrical signals + 6 accessory signals. Resolution is 286.1 nV and range +/-9.375 mV. **
    1 -> MONOPOLAR_GAIN_4:  Monopolar Mode (Only EMG -> EEG=>Mode 0). preamp gain 4. 32 bioelectrical signals + 6 accessory signals. Resolution is 572.2nV and range +/-18.75 mV. **
    2 -> IMPEDANCE_CHECK:   Impedance Check on all 32 + 6 channels.
    3 -> TEST:              Ramps on all 32 + 6 channels.

    **
    Preamp gain of 4 has a double input range and a slightly larger noise w.r.t. preamp gain of 8. It can be
    used when DC component of EMG signals is higher and generates saturation before the high pass filter
    resulting in flat signals. The input range before the high pass filter is +/-600mV when the preamp is set to
    4 and +/-300mV when the preamp is set to 8.
    """

    MONOPOLAR_GAIN_8 = 0
    MONOPOLAR_GAIN_4 = 1
    IMPEDANCE_CHECK = 2
    TEST = 3


class MuoviStream(Enum):
    """
    Enum class for the streaming mode of the Muovi device.

    0 -> STOP:  Stop streaming.
    1 -> GO:    Start streaming.
    """

    STOP = 0
    GO = 1


class MuoviAuxiliaryChannels(Enum):
    """
    Enum class for the auxiliary channels of the Muovi device.

    0 -> IMU_W:  Sampling frequency of the auxiliary channels.
    1 -> IMU_X:  Sampling frequency of the auxiliary channels.
    2 -> IMU_Y:  Sampling frequency of the auxiliary channels.
    3 -> IMU_Z:  Sampling frequency of the auxiliary channels.
    4 -> BUFFER: Buffer usage.
    5 -> SAMPLE: Sample counter.
    """

    IMU_W = 0
    IMU_X = 1
    IMU_Y = 2
    IMU_Z = 3
    BUFFER = 4
    SAMPLE = 5


class MuoviAvailableChannels(Enum):
    """
    Enum class for the available channels of the Muovi device.

    6 -> AUXILIARY: 6 auxiliary channels.
    32 -> BIOSIGNALS: 32 biosignals channels.
    """

    AUXILIARY = 6
    BIOSIGNALS = 32
    ALL = 38


class MuoviAvailableChannels(Enum):
    """
    Enum class for the available channels of the Muovi device.

    6 -> AUXILIARY: 6 auxiliary channels.
    32 -> BIOSIGNALS: 32 biosignals channels.
    """

    AUXILIARY = 6
    BIOSIGNALS = 32
    ALL = 38


class MuoviPlusAvailableChannels(Enum):
    """
    Enum class for the available channels of the Muovi device.

    6 -> AUXILIARY: 6 auxiliary channels.
    32 -> BIOSIGNALS: 32 biosignals channels.
    """

    AUXILIARY = 6
    BIOSIGNALS = 64
    ALL = 70


class MuoviNetworkCharacteristics(Enum):
    """
    Enum class for the network characteristics of the Muovi device.

    0 -> EXTERNAL_NETWORK:  The Muovi is connected to an external network.
    1 -> ACCESS_POINT:  The Muovi is hosting a network as an access point.
    """

    EXTERNAL_NETWORK = 0
    ACCESS_POINT = 1
