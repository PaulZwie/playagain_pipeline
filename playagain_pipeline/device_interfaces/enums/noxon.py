"""
Enum classes for the Sessantaquattro device.

Developer: Dominik I. Braun
Contact: dome.braun@fau.de
Last Update: 2024-01-07
"""

from enum import Enum


class NOXONAction(Enum):
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


class NOXONSamplingFrequency(Enum):
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


class NOXONChannels(Enum):
    """
    Enum class for the number of transferred channels of the Noxon device
    (refer to the user manual for additional details).

    0 -> LOW:       4
    1 -> MEDIUM:    8
    2 -> HIGH:      16
    3 -> ULTRA:     32
    """

    LOW = 0
    MEDIUM = 1
    HIGH = 2
    ULTRA = 3


class NOXONMode(Enum):
    """
    Enum class for the working modes of the Noxon device.

    0 -> MONOPOLAR:     All the channels are acquired with respect to the reference.
    1 -> BIPOLAR:       Using AD8x1SE adapter (differences between CH3-1 Ch4-2 CH7-5 ...)
    2 -> DIFFERENTIAL:  All the channels are the differences between consecutive inputs
                        over groups of 32 channels. Channels 32 and 64 are monopolar.
                        NOTE: the configuration 64 CH, 16 bits resolution, HPF on and 2000 Hz will retrieve
                        all monopolar signals even if set on Differential.
    3 -> Accelerometer: Only 8 channels (plus 2 AUX and 2 accessory) are acquired and transferred
                        (even if NCH has a different value) with increased sampling frequency.
                        FSAMP in this mode has the values 2000, 4000, 8000, 16000 Hz.
    4 -> UNDEFINED:     Undefined mode.
    5 -> IMPEDANCE_ADV: Impedance check (advanced).
    6 -> IMPEDANCE:     Impedance check.
    7 -> TEST:          Test mode. Sends ramps on all channels
    """

    MONOPOLAR = 0
    BIPOLAR = 1
    DIFFERENTIAL = 2
    ACCELEROMETER = 3
    UNDEFINED = 4
    IMPEDANCE_ADV = 5
    IMPEDANCE = 6
    TEST = 7


class NOXONResolution(Enum):
    """
    Enum class for the resolution of the Noxon device.

    0 -> LOW:   16 bits
    1 -> HIGH:  24 bits
    """

    LOW = 0
    HIGH = 1


class NOXONFilter(Enum):
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


class NOXONGain(Enum):
    """
    Enum class for the preamp gain of the Noxon device.
    There are differences depending on the resolution:

    0 -> DEFAULT:
        If HRES = 1, Gain 2 and the resolution is 286.1 nV and range +/- 1200mV.
        If HRES = 0, Gain 8 and the resolution is 286.1 nV and range +/- 9.375 mV.

    1 -> LOW: Gain 4:
        If HRES=1 the resolution is 143 nV and range +/- 600mV.
        If HRES=0 the resolution is 572.2 nV and range +/-18.75 mV.

    2 -> MEDIUM: Gain = 6:
        If HRES=1 the resolution is 95.4 nV and range +/-400mV.
        If HRES=0 the resolution is 381.5 nV and range +/-12.5 mV.

    3 -> HIGH: Gain = 8:
        If HRES = 1: the resolution is 71.5 nV and range +/- 300mV.
        If HRES = 0: the resolution is 286.1 nV and range +/-9.375 mV.
    """

    DEFAULT = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3


class NOXONTrigger(Enum):
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


class NOXONRecording(Enum):
    """
    Enum class for the recording status of the Noxon device.
    Starts/Stops the acquisition on the MicroSD

    0 -> STOP:   Stop the recording. Works only if TRIG = 3 (SDCARD).
    1 -> START:   Start the recording. Works only if TRIG = 3 (SDCARD).
    """

    STOP = 0
    START = 1


class NOXONTransmission(Enum):
    """
    Enum class for the data transfer status of the Noxon device.
    Starts/Stops the data transfer on the TCP socket.

    0 -> STOP: Stop and close the socket.
    1 -> START: Start the data transfer.
    """

    STOP = 0
    START = 1
