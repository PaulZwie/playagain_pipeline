"""
Enum classes for the Sessantaquattro device.

Developer: Dominik I. Braun
Contact: dome.braun@fau.de
Last Update: 2024-01-07
"""

from enum import Enum


class NSWITCHAction(Enum):
    """
    Enum class for the type of action to perform on the Noxon device.

    0 -> SET:   Set the parameters of the device. All the other bits and bytes
                are used to set new values to the Noxon settings.
    1 -> GET:   Get the parameters of the device.
                bit 14-3: not used
                bit 2-0:
                000 ->  reply with a sequence of 13 bytes indicating the actual settings.
                001 ->  reply with 2 bytes indicating the first and second digit of the
                        firmware version.
                010 ->  reply with a byte indicating the battery level as a percentage.
                011 - 111 -> to be defined.
    """

    SET = 0
    GET = 1


class NSWITCHSamplingFrequency(Enum):
    """
    Enum class for the sampling frequency modes of the Noxon device.

    0 -> LOW = 500 Hz
    0 -> MEDIUM = 1000 Hz
    0 -> HIGH = 2000 Hz
    0 -> ULTRA = 4000 Hz

    """

    LOW = 0
    MEDIUM = 1
    HIGH = 2
    ULTRA = 3


class NSWITCHChannels(Enum):
    """
    Enum class for the number of transferred channels of the Noxon device
    (refer to the user manual for additional details).

    0 -> LOW:       2
    1 -> MEDIUM:    4
    2 -> HIGH:      8
    3 -> ULTRA:     16
    """

    LOW = 0
    MEDIUM = 1
    HIGH = 2
    ULTRA = 3


class NSWITCHMode(Enum):
    """
    Enum class for the working modes of the Noxon device.

    0 -> MONOPOLAR:     All the channels are acquired with respect to the reference.
    1 -> BIPOLAR:       Using AD8x1SE adapter (differences between CH3-1 Ch4-2 CH7-5 ...)
    2 -> IMPEDANCE:     Impedance measurement mode.
    3 -> TEST:          Test mode.
    """

    MONOPOLAR = 0
    BIPOLAR = 1
    IMPEDANCE = 2
    TEST = 3


class NSWITCHResolution(Enum):
    """
    Enum class for the resolution of the Noxon device.

    0 -> LOW:   16 bits
    1 -> HIGH:  24 bits
    """

    LOW = 0
    HIGH = 1


class NSWITCHFilter(Enum):
    """
    Enum class for the high-pass filter of the Noxon device.

    0 -> OFF:   DC-Signals (to be used with the high resolution).
    1 -> ON:    High pass filter implemented by the microcontroller subtracting the exponential
                moving average, obtained by:
                $Average_ChX[t] = (1-\alpha) Average_ChX[t-1] + \alpha ChX[t]$,
                where \alpha is equal to \frac{1}{2}^5 for MODE = 0, 1 or 2. It is equal to \frac{1}{2} for MODE = 5 or 6.

                For the standard modes, this reuslt in a high pass filter with a cut-off frequency of 10.5 Hz,
                when sampling the signals at 2000 Hz. More in general the cut-off frequency is FSAMP/190.
    """

    OFF = 0
    ON = 1


class NSWITCHGain(Enum):
    """
    Enum class for the preamp gain of the Noxon device.
    There are differences depending on the resolution:

    0 -> DEFAULT = 6
    1 -> GAIN_1 = 1
    2 -> GAIN_2 = 2
    3 -> GAIN_3 = 3
    4 -> GAIN_4 = 4
    5 -> GAIN_8 = 8
    6 -> GAIN_12 = 12
    """

    DEFAULT = 0
    GAIN_1 = 1
    GAIN_2 = 2
    GAIN_3 = 3
    GAIN_4 = 4
    GAIN_8 = 5
    GAIN_12 = 6


class NSWITCHTrigger(Enum):
    """
    Enum class for the trigger mode of the Noxon device.
    Event trigger to start data transfer or acquisition on SD card.

    0 -> DEFAULT: The data transfer is controlled from GO/STOP bit, REC has no effect.
    1 -> INTERNAL: The data transfer ist triggered by the internal signal (phototransistor)
    2 -> EXTERNAL: The data transfer is triggered by the external signal (from the adapter)
    3 -> SDCARD: SD card acquisition starts/stops with the hardware button or with the REC bit.

    """

    DEFAULT = 0
    INTERNAL = 1
    EXTERNAL = 2
    SDCARD = 3


class NSWITCHRecording(Enum):
    """
    Enum class for the recording status of the Noxon device.
    Starts/Stops the acquisition on the MicroSD

    0 -> STOP:   Stop the recording. Works only if TRIG = 3 (SDCARD).
    1 -> START:   Start the recording. Works only if TRIG = 3 (SDCARD).
    """

    STOP = 0
    START = 1


class NSWITCHTransmission(Enum):
    """
    Enum class for the data transfer status of the Noxon device.
    Starts/Stops the data transfer on the TCP socket.

    0 -> STOP: Stop and close the socket.
    1 -> START: Start the data transfer.
    """

    STOP = 0
    START = 1


class NSWITCHConfigurationMode(Enum):
    """
    Enum class for the configuration of the NSWITCH device.

    0 -> IGNORE
    1 -> SET_CONFIGURATION
    """

    IGNORE = 0
    SET_CONFIGURATION = 1


class NSWITCHReferenceVoltageMode(Enum):
    """_summary_

    Args:
        Enum (_type_): _description_
    """

    VREF_2_4 = 0
    VREF_4 = 1
