"""
Dict classes for the Noxon device.

Developer: Dominik I. Braun
Contact: dome.braun@fau.de
Last Update: 2024-01-07
"""

from device_interfaces.enums.nswitch import *

SAMPLING_FREQUENCY_NO_ACC_DICT: dict[NSWITCHSamplingFrequency, int] = {
    NSWITCHSamplingFrequency.LOW: 500,
    NSWITCHSamplingFrequency.MEDIUM: 1000,
    NSWITCHSamplingFrequency.HIGH: 2000,
    NSWITCHSamplingFrequency.ULTRA: 4000,
}
"""
Dictionary to get sampling frequency for each mode.
"""


CHANNEL_DICT: dict[NSWITCHChannels, int] = {
    NSWITCHChannels.LOW: 2,
    NSWITCHChannels.MEDIUM: 4,
    NSWITCHChannels.HIGH: 8,
    NSWITCHChannels.ULTRA: 16,
}
"""
Dictionary to get channel number for each channel mode.
"""

FRAME_LEN_NO_ACC_DICT: dict[NSWITCHChannels, int] = {
    NSWITCHChannels.LOW: 48,
    NSWITCHChannels.MEDIUM: 28,
    NSWITCHChannels.HIGH: 16,
    NSWITCHChannels.ULTRA: 8,
}
"""
Dictionary to get frame length for each channel mode.
"""

GET_GAIN_DICT: dict[NSWITCHGain, int] = {
    NSWITCHGain.DEFAULT: 6,
    NSWITCHGain.GAIN_1: 1,
    NSWITCHGain.GAIN_2: 2,
    NSWITCHGain.GAIN_3: 3,
    NSWITCHGain.GAIN_4: 4,
    NSWITCHGain.GAIN_8: 8,
    NSWITCHGain.GAIN_12: 12,
}
"""
Dictionary to get gain for each gain mode.
"""
