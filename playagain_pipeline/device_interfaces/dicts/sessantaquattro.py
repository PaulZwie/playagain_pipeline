"""
Dict classes for the Sessantaquattro device.

Developer: Dominik I. Braun
Contact: dome.braun@fau.de
Last Update: 2023-12-07
"""
from device_interfaces.enums.sessantaquattro import *

SAMPLING_FREQUENCY_NO_ACC_DICT: dict[SSQTSamplingFrequency, int] = {
    SSQTSamplingFrequency.LOW: 500,
    SSQTSamplingFrequency.MEDIUM: 1000,
    SSQTSamplingFrequency.HIGH: 2000,
    SSQTSamplingFrequency.ULTRA: 4000,
}
"""
Dictionary to get sampling frequency for each mode. No accelerometer.
"""

SAMPLING_FREQUENCY_WITH_ACC_DICT: dict[SSQTSamplingFrequency, int] = {
    SSQTSamplingFrequency.LOW: 2000,
    SSQTSamplingFrequency.MEDIUM: 4000,
    SSQTSamplingFrequency.HIGH: 8000,
    SSQTSamplingFrequency.ULTRA: 16000,
}
"""
Dictionary to get sampling frequency for each mode. With accelerometer.
"""

CHANNEL_DICT: dict[SSQTChannels, int] = {
    SSQTChannels.LOW: 12,
    SSQTChannels.MEDIUM: 20,
    SSQTChannels.HIGH: 36,
    SSQTChannels.ULTRA: 68,
}
"""
Dictionary to get channel number for each channel mode.
"""

FRAME_LEN_NO_ACC_DICT: dict[SSQTChannels, int] = {
    SSQTChannels.LOW: 48,
    SSQTChannels.MEDIUM: 28,
    SSQTChannels.HIGH: 16,
    SSQTChannels.ULTRA: 8,
}
"""
Dictionary to get frame length for each channel mode.
"""
