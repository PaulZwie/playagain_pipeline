from device_interfaces.enums.syncstation import *
from typing import Union, Dict

PROBE_CHARACTERISTICS_DICT: Dict[
    SyncStationProbeConfigMode, Dict[str, Union[str, int]]
] = {
    SyncStationProbeConfigMode.MUOVI_PROBE_ONE: {
        "name": "Muovi Probe 1",
        "channels": 38,
        "biosignal": 32,
        "aux": 6,
    },
    SyncStationProbeConfigMode.MUOVI_PROBE_TWO: {
        "name": "Muovi Probe 2",
        "channels": 38,
        "biosignal": 32,
        "aux": 6,
    },
    SyncStationProbeConfigMode.MUOVI_PROBE_THREE: {
        "name": "Muovi Probe 3",
        "channels": 38,
        "biosignal": 32,
        "aux": 6,
    },
    SyncStationProbeConfigMode.MUOVI_PROBE_FOUR: {
        "name": "Muovi Probe 4",
        "channels": 38,
        "biosignal": 32,
        "aux": 6,
    },
    SyncStationProbeConfigMode.MUOVI_PLUS_PROBE_ONE: {
        "name": "Muovi+ Probe 1",
        "channels": 70,
        "biosignal": 64,
        "aux": 6,
    },
    SyncStationProbeConfigMode.MUOVI_PLUS_PROBE_TWO: {
        "name": "Muovi+ Probe 2",
        "channels": 70,
        "biosignal": 64,
        "aux": 6,
    },
    SyncStationProbeConfigMode.DUE_PLUS_PROBE_ONE: {
        "name": "Due+ Probe 1",
        "channels": 8,
        "biosignal": 2,
        "aux": 6,
    },
    SyncStationProbeConfigMode.DUE_PLUS_PROBE_TWO: {
        "name": "Due+ Probe 2",
        "channels": 8,
        "biosignal": 2,
        "aux": 6,
    },
    SyncStationProbeConfigMode.DUE_PLUS_PROBE_THREE: {
        "name": "Due+ Probe 3",
        "channels": 8,
        "biosignal": 2,
        "aux": 6,
    },
    SyncStationProbeConfigMode.DUE_PLUS_PROBE_FOUR: {
        "name": "Due+ Probe 4",
        "channels": 8,
        "biosignal": 2,
        "aux": 6,
    },
    SyncStationProbeConfigMode.DUE_PLUS_PROBE_FIVE: {
        "name": "Due+ Probe 5",
        "channels": 8,
        "biosignal": 2,
        "aux": 6,
    },
    SyncStationProbeConfigMode.DUE_PLUS_PROBE_SIX: {
        "name": "Due+ Probe 6",
        "channels": 8,
        "biosignal": 2,
        "aux": 6,
    },
    SyncStationProbeConfigMode.DUE_PLUS_PROBE_SEVEN: {
        "name": "Due+ Probe 7",
        "channels": 8,
        "biosignal": 2,
        "aux": 6,
    },
    SyncStationProbeConfigMode.DUE_PLUS_PROBE_EIGHT: {
        "name": "Due+ Probe 8",
        "channels": 8,
        "biosignal": 2,
        "aux": 6,
    },
    SyncStationProbeConfigMode.DUE_PLUS_PROBE_NINE: {
        "name": "Due+ Probe 9",
        "channels": 8,
        "biosignal": 2,
        "aux": 6,
    },
    SyncStationProbeConfigMode.DUE_PLUS_PROBE_TEN: {
        "name": "Due+ Probe 10",
        "channels": 8,
        "biosignal": 2,
        "aux": 6,
    },
}

SYNCSTATION_CHARACTERISTICS_DICT: Dict[str, int] = {
    "channels": 6,
    "bytes_per_sample": 2,
    "channel_information": {
        SyncStationWorkingMode.EEG: {
            "sampling_frequency": 500,
            "bytes_per_sample": 3,
            "frame_size": 5,
        },
        SyncStationWorkingMode.EMG: {
            "sampling_frequency": 2000,
            "bytes_per_sample": 2,
            "frame_size": 10,
        },
    },
    "number_of_packages": 32,
    "package_size": 1460,
    "FPS": 50,
}
