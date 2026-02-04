from enum import Enum


class IntanSamplingFrequencyMode(Enum):
    """
    Enum class for the sampling frequency mode of the Intan device.

    0 -> LOW: Low sampling frequency mode.
    1 -> HIGH: High sampling frequency mode.
    """

    SAMPLING_1000 = 0
    SAMPLING_1250 = 1
    SAMPLING_1500 = 2
    SAMPLING_2000 = 3
    SAMPLING_2500 = 4
    SAMPLING_3000 = 5
    SAMPLING_3333 = 6
    SAMPLING_4000 = 7
    SAMPLING_5000 = 8
    SAMPLING_6250 = 9
    SAMPLING_8000 = 10
    SAMPLING_10000 = 11
    SAMPLING_12500 = 12
    SAMPLING_15000 = 13
    SAMPLING_20000 = 14
    SAMPLING_25000 = 15
    SAMPLING_30000 = 16


class IntanRunMode(Enum):
    """
    Enum class for the run mode of the Intan device.
    """

    RUN = 0
    STOP = 1


class IntanAvailablePorts(Enum):
    """
    Enum class for the available ports of the Intan device.
    """

    PORT_A = 0
    PORT_B = 1
    PORT_C = 2
    PORT_D = 3
    PORT_E = 4
    PORT_F = 5
    PORT_G = 6
    PORT_H = 7


class IntanAnalogInChannels(Enum):
    """
    Enum class for the available analog input channels of the Intan device.
    """

    ANALOG_IN_1 = 0
    ANALOG_IN_2 = 1
    # ANALOG_IN_3 = 2
    # ANALOG_IN_4 = 3
    # ANALOG_IN_5 = 4
    # ANALOG_IN_6 = 5
    # ANALOG_IN_7 = 6
    # ANALOG_IN_8 = 7


class IntanAnalogOutChannels(Enum):
    """
    Enum class for the available analog output channels of the Intan device.
    """

    ANALOG_OUT_1 = 0
    ANALOG_OUT_2 = 1
    ANALOG_OUT_3 = 2
    ANALOG_OUT_4 = 3
    ANALOG_OUT_5 = 4
    ANALOG_OUT_6 = 5
    ANALOG_OUT_7 = 6
    ANALOG_OUT_8 = 7


class IntanDigitalInChannels(Enum):
    """
    Enum class for the available digital input channels of the Intan device.
    """

    DIGITAL_IN_1 = 0
    DIGITAL_IN_2 = 1
    DIGITAL_IN_3 = 2
    DIGITAL_IN_4 = 3
    DIGITAL_IN_5 = 4
    DIGITAL_IN_6 = 5
    DIGITAL_IN_7 = 6
    DIGITAL_IN_8 = 7
    DIGITAL_IN_9 = 8
    DIGITAL_IN_10 = 9
    DIGITAL_IN_11 = 10
    DIGITAL_IN_12 = 11
    DIGITAL_IN_13 = 12
    DIGITAL_IN_14 = 13
    DIGITAL_IN_15 = 14
    DIGITAL_IN_16 = 15


class IntanDigitalOutputChannels(Enum):
    """
    Enum class for the available digital output channels of the Intan device.
    """

    DIGITAL_OUT_1 = 0
    DIGITAL_OUT_2 = 1
    DIGITAL_OUT_3 = 2
    DIGITAL_OUT_4 = 3
    DIGITAL_OUT_5 = 4
    DIGITAL_OUT_6 = 5
    DIGITAL_OUT_7 = 6
    DIGITAL_OUT_8 = 7
    DIGITAL_OUT_9 = 8
    DIGITAL_OUT_10 = 9
    DIGITAL_OUT_11 = 10
    DIGITAL_OUT_12 = 11
    DIGITAL_OUT_13 = 12
    DIGITAL_OUT_14 = 13
    DIGITAL_OUT_15 = 14
    DIGITAL_OUT_16 = 15
