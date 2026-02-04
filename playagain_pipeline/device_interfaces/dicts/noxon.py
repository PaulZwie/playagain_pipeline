"""
Dict classes for the Noxon device.

Developer: Dominik I. Braun
Contact: dome.braun@fau.de
Last Update: 2024-01-07
"""
from device_interfaces.enums.noxon import *

SAMPLING_FREQUENCY_NO_ACC_DICT: dict[NOXONSamplingFrequency, int] = {
    NOXONSamplingFrequency.LOW: 500,
    NOXONSamplingFrequency.MEDIUM: 1000,
    NOXONSamplingFrequency.HIGH: 2000,
    NOXONSamplingFrequency.ULTRA: 4000,
}
"""
Dictionary to get sampling frequency for each mode.
"""


CHANNEL_DICT: dict[NOXONChannels, int] = {
    NOXONChannels.LOW: 12,
    NOXONChannels.MEDIUM: 20,
    NOXONChannels.HIGH: 36,
    NOXONChannels.ULTRA: 68,
}
"""
Dictionary to get channel number for each channel mode.
"""

FRAME_LEN_NO_ACC_DICT: dict[NOXONChannels, int] = {
    NOXONChannels.LOW: 48,
    NOXONChannels.MEDIUM: 28,
    NOXONChannels.HIGH: 16,
    NOXONChannels.ULTRA: 8,
}
"""
Dictionary to get frame length for each channel mode.
"""
