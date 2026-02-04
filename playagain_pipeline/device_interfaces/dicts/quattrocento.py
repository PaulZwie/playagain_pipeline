"""
Dict classes for the Quattrocento device.

Developer: Dominik I. Braun
Contact: dome.braun@fau.de
Last Update: 2023-12-21
"""

from device_interfaces.enums.quattrocento import *

INPUT_CHANNEL_CONFIGURATION_DICT: dict[str, int] = {
    "IN1-4": "in_top_left_configuration",
    "IN5-8": "in_top_right_configuration",
    "MULTIPLE_IN_1": "multiple_in1_configuration",
    "MULTIPLE_IN_2": "multiple_in2_configuration",
    "MULTIPLE_IN_3": "multiple_in3_configuration",
    "MULTIPLE_IN_4": "multiple_in4_configuration",
}
"""
Dictionary to get sampling frequency for each mode. No accelerometer.
"""


QUATTROCENTO_LIGHT_SAMPLING_FREQUENCY_DICT: dict[
    QuattrocentoLightSamplingFrequency, int
] = {
    QuattrocentoLightSamplingFrequency.LOW: 512,
    QuattrocentoLightSamplingFrequency.MEDIUM: 2048,
    QuattrocentoLightSamplingFrequency.HIGH: 5120,
    QuattrocentoLightSamplingFrequency.ULTRA: 10240,
}
"""
Dictionary to get sampling frequency for each mode.
"""

QUATTROCENTO_LIGHT_STREAMING_FREQUENCY_DICT: dict[
    QuattrocentoLightStreamingFrequency, int
] = {
    QuattrocentoLightStreamingFrequency.ONE: 1,
    QuattrocentoLightStreamingFrequency.TWO: 2,
    QuattrocentoLightStreamingFrequency.FOUR: 4,
    QuattrocentoLightStreamingFrequency.EIGHT: 8,
    QuattrocentoLightStreamingFrequency.SIXTEEN: 16,
    QuattrocentoLightStreamingFrequency.THIRTYTWO: 32,
}
