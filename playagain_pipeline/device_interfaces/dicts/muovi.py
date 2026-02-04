"""
Dict classes for the Muovi device.

Developer: Dominik I. Braun
Contact: dome.braun@fau.de
Last Update: 2024-01-07
"""

from device_interfaces.enums.device import Device
from device_interfaces.enums.muovi import *

MUOVI_WORKING_MODE_CHARACTERISTICS_DICT: dict[MuoviWorkingMode, dict[str, int]] = {
    MuoviWorkingMode.EEG: {
        "sampling_frequency": 500,
        "bytes_per_sample": 3,
    },
    MuoviWorkingMode.EMG: {
        "sampling_frequency": 2000,
        "bytes_per_sample": 2,
    },
}
"""
Dictionary to get characteristics of the Muovi working mode.
"""

MUOVI_NETWORK_CHARACTERISTICS_DICT: dict[
    MuoviNetworkCharacteristics, dict[str, int]
] = {
    MuoviNetworkCharacteristics.EXTERNAL_NETWORK: {
        "port": 54321,
        "ip_address": None,
    },
    MuoviNetworkCharacteristics.ACCESS_POINT: {
        "port": 54321,
        "ip_address": "0.0.0.0",
    },
}
"""
Dictionary to get characteristics of the Muovi network mode.
"""

MUOVI_FRAME_LEN_DICT: dict[Device, dict[MuoviWorkingMode, int]] = {
    Device.MUOVI: {
        MuoviWorkingMode.EEG: 12,
        MuoviWorkingMode.EMG: 18,
    },
    Device.MUOVI_PLUS: {
        MuoviWorkingMode.EEG: 6,
        MuoviWorkingMode.EMG: 10,
    },
}
"""
Dictionary to get the frame length of the Muovi.
"""
