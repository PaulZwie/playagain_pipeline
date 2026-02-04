"""Device interfaces for EMG acquisition."""

from playagain_pipeline.devices.emg_device import (
    DeviceType,
    DeviceConfig,
    BaseEMGDevice,
    SyntheticEMGDevice,
    MuoviDevice,
    DeviceManager,
    DEVICE_INTERFACES_AVAILABLE
)

__all__ = [
    "DeviceType",
    "DeviceConfig",
    "BaseEMGDevice",
    "SyntheticEMGDevice",
    "MuoviDevice",
    "DeviceManager",
    "DEVICE_INTERFACES_AVAILABLE"
]
