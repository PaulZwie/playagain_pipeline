from device_interfaces.enums.intan import *

INTAN_SAMPLING_FREQUENCIES: dict[IntanSamplingFrequencyMode, int] = {
    IntanSamplingFrequencyMode.SAMPLING_1000: 1000,
    IntanSamplingFrequencyMode.SAMPLING_1250: 1250,
    IntanSamplingFrequencyMode.SAMPLING_1500: 1500,
    IntanSamplingFrequencyMode.SAMPLING_2000: 2000,
    IntanSamplingFrequencyMode.SAMPLING_2500: 2500,
    IntanSamplingFrequencyMode.SAMPLING_3000: 3000,
    IntanSamplingFrequencyMode.SAMPLING_3333: 3333,
    IntanSamplingFrequencyMode.SAMPLING_4000: 4000,
    IntanSamplingFrequencyMode.SAMPLING_5000: 5000,
    IntanSamplingFrequencyMode.SAMPLING_6250: 6250,
    IntanSamplingFrequencyMode.SAMPLING_8000: 8000,
    IntanSamplingFrequencyMode.SAMPLING_10000: 10000,
    IntanSamplingFrequencyMode.SAMPLING_12500: 12500,
    IntanSamplingFrequencyMode.SAMPLING_15000: 15000,
    IntanSamplingFrequencyMode.SAMPLING_20000: 20000,
    IntanSamplingFrequencyMode.SAMPLING_25000: 25000,
    IntanSamplingFrequencyMode.SAMPLING_30000: 30000,
}

INTAN_RUN_MODES: dict[IntanRunMode, bytes] = {
    IntanRunMode.RUN: b"set runmode run;",
    IntanRunMode.STOP: b"set runmode stop;",
}
