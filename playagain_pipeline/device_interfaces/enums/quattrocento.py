"""
Enum classes for the Quattrocento device.

Developer: Dominik I. Braun
Contact: dome.braun@fau.de
Last Update: 2023-10-23
"""

"""
Quattrocento configuration command sequence.

Byte 1: ACQ_SETT
    - Sampling frequency
    - number of channels
    - start/stop acquisition
    - start/stop recording

Byte 2: AN_OUT_IN_SEL
    - Select the input source and gain for the analog output

Byte 3: AN_OUT_CH_SEL
    - Select the channel for the analog output source

Byte 4-6: IN1_CONF0/1/2
    - Configuration for the first IN input
        + high pass filter
        + low pass filter
        + detection mode
        + muscle (TBI)
        + side (TBI)
        + sensor
        + adapter

Byte 7-9: IN2_CONF0/1/2
    - Configuration for the second IN input
        + high pass filter
        + low pass filter
        + detection mode
        + muscle (TBI)
        + side (TBI)
        + sensor
        + adapter

Byte 10-12: IN3_CONF0/1/2
    - Configuration for the third IN input
        + high pass filter
        + low pass filter
        + detection mode
        + muscle (TBI)
        + side (TBI)
        + sensor
        + adapter

Byte 13-15: IN4_CONF0/1/2
    - Configuration for the fourth IN input
        + high pass filter
        + low pass filter
        + detection mode
        + muscle (TBI)
        + side (TBI)
        + sensor
        + adapter

Byte 16-18: IN5_CONF0/1/2
    - Configuration for the fifth IN input
        + high pass filter
        + low pass filter
        + detection mode
        + muscle (TBI)
        + side (TBI)
        + sensor
        + adapter

Byte 19-21: IN6_CONF0/1/2
    - Configuration for the sixth IN input
        + high pass filter
        + low pass filter
        + detection mode
        + muscle (TBI)
        + side (TBI)
        + sensor
        + adapter

Byte 22-24: IN7_CONF0/1/2
    - Configuration for the seventh IN input
        + high pass filter
        + low pass filter
        + detection mode
        + muscle (TBI)
        + side (TBI)
        + sensor
        + adapter

Byte 25-27: IN8_CONF0/1/2
    - Configuration for the eigth IN input
        + high pass filter
        + low pass filter
        + detection mode
        + muscle (TBI)
        + side (TBI)
        + sensor
        + adapter

Byte 28-30: MULTIPLE_IN1_CONF0/1/2
    - Configuration for the first MULTIPLE IN input
        + high pass filter
        + low pass filter
        + detection mode
        + muscle (TBI)
        + side (TBI)
        + sensor
        + adapter

Byte 31-33: MULTIPLE_IN2_CONF0/1/2
    - Configuration for the second MULTIPLE IN input
        + high pass filter
        + low pass filter
        + detection mode
        + muscle (TBI)
        + side (TBI)
        + sensor
        + adapter

Byte 34-36: MULTIPLE_IN3_CONF0/1/2
    - Configuration for the third MULTIPLE IN input
        + high pass filter
        + low pass filter
        + detection mode
        + muscle (TBI)
        + side (TBI)
        + sensor
        + adapter

Byte 37-39: MULTIPLE_IN4_CONF0/1/2
    - Configuration for the fourth MULTIPLE IN input
        + high pass filter
        + low pass filter
        + detection mode
        + muscle (TBI)
        + side (TBI)
        + sensor
        + adapter

Byte 40: CRC
    - Eigth bits CRC
"""

from enum import Enum


# ------- Command Byte Sequences --------
class QuattrocentoCommandSequence(Enum):
    """
    Enum class for the different kind of command sequences.

    0 -> ACQ_SETT:          General settings regarding the acquisition of
                            data.
    1 -> AN_OUT_IN_SEL:     Select the input source and gain for the analog
                            output.
    2 -> AN_OUT_CH_SEL:     Select the channel for the analog output source.
    3 -> IN_CONF:           Configuration for the eight IN inputs or
                            configuration for the four MULTIPLE IN inputs.
    4 -> CRC: Eight bits CRC.
    """

    ACQ_SETT = 0
    AN_OUT_IN_SEL = 1
    AN_OUT_CH_SEL = 2
    IN_CONF = 3
    CRC = 4


# ------------ ACQ_SETT BYTE ------------
# Bit 7 is fixed to 1


class QuattrocentoDecim(Enum):
    """
    Enum class for the decimation bit of the Quattrocento device.

    0 -> INACTIVE:  No decimation.
                    The required sampling frequency is obtained by sampling
                    the signals directly at the desired sampling frequency
    1 -> ACTIVE:    Decimation active.
                    The required sampling frequency is obtained by
                    sampling all the signals at 10240 Hz and then sending one
                    sample out of 2, 5 or 20, to obtain the desired number of
                    sample per second.
    """

    INACTIVE = 0
    ACTIVE = 1


class QuattrocentoRecording(Enum):
    """
    Enum class for the recording bit.

    If the Trigger OUT has to be used to synchronize the acquisition with
    other instruments, the recording has to be started when the trigger
    channel has a transition. In other words it is the quattrocento that
    generate a signal indicating to the computer when the data has to be
    recorded.

    0 -> STOP:  The user wants to stop an acquisition associated with the
                Trigger OUT.
    1 -> START: The user wants to start an acquisition associated with the
                Trigger OUT.
    """

    STOP = 0
    START = 1


class QuattrocentoSamplingFrequency(Enum):
    """
    Enum class for the sampling frequency of the Quattrocento device (2 bits).

    0 -> LOW: 512 Hz
    1 -> MEDIUM: 2048 Hz
    2 -> HIGH: 5120 Hz
    3 -> ULTRA: 10240 Hz
    """

    LOW = 0
    MEDIUM = 1
    HIGH = 2
    ULTRA = 3


class QuattrocentoNumberOfChannels(Enum):
    """
    Enum class for the number of channels of the Quattrocento device (2 bits).

    0 -> LOW: IN1, IN2 and MULTIPLE_IN1 are active.
    1 -> MEDIUM: IN1-IN4, MULTIPLE_IN1, MULTIPLE_IN2 are active.
    2 -> HIGH: IN1-IN6, MULTIPLE_IN1-MULTIPLE_IN2,
    3 -> ULTRA: IN1-IN8, MULTIPLE_IN1-MULTIPLE_IN4 are active.
    """

    LOW = 0
    MEDIUM = 1
    HIGH = 2
    ULTRA = 3


class QuattrocentoAcquisition(Enum):
    """
    Enum class for the acquisition bit.

    0 -> INACTIVE: Data sampling and transfer is not active.
    1 -> ACTIVE: Data sampling and transfer is active.
    """

    INACTIVE = 0
    ACTIVE = 1


# ---------- AN_OUT_IN_SEL BYTE ----------
# Bit 7 and bit 6 are fixed to 0.


class QuattrocentoAnalogOutputGain(Enum):
    """
    Enum class of the analog output gain of the Quattrocento device (2 bits).

    0 -> LOW: Gain = 1
    1 -> MEDIUM: Gain = 2
    2 -> HIGH: Gain = 4
    3 -> ULTRA: Gain = 16
    """

    LOW = 0
    MEDIUM = 1
    HIGH = 2
    ULTRA = 3


class QuattrocentoSourceInput(Enum):
    """
    Enum class for the source input of the analog output (4 bits).

    0000 -> IN_I
    0001 -> IN_II
    0010 -> IN_III
    0011 -> IN_IV
    0100 -> IN_V
    0101 -> IN_VI
    0110 -> IN_VII
    0111 -> IN_VIII
    1000 -> MULTIPLE_IN_I
    1001 -> MULTIPLE_IN_II
    1010 -> MULTIPLE_IN_III
    1011 -> MULTIPLE_IN_IV
    1100 -> AUX_IN
    """

    IN_I = 0
    IN_II = 1
    IN_III = 2
    IN_IV = 3
    IN_V = 4
    IN_VI = 5
    IN_VII = 6
    IN_VIII = 7
    MULTIPLE_IN_I = 8
    MULTIPLE_IN_II = 9
    MULTIPLE_IN_III = 10
    MULTIPLE_IN_IV = 11
    AUX_IN = 12


# ---------- AN_OUT_CH_SEL BYTE ----------
# Bit 7 and bit 6 are fixed to 0.


class QuattrocentoSourceChannel(Enum):
    """
    Enum class for the source channel for the analog output (6 bits).

    Considering the input selected by the AN_OUT_IN_SEL byte, this number
    indicates which channels of that input have to be provided at the ANALOG
    OUT BNC on the rear panel.

    NOTE: 0 indicates the first channel, 1 the second channel etc.
    """


# -- INX_CONF0 and MULTIPLE_INX_CONF0 BYTE --
# Bit 7 is fixed to 0.


class QuattrocentoMuscle(Enum):
    """
    Enum class for the muscle bit of the Quattrocento device (7 bits).

    0 -> NOT_DEFINED
    1 -> TEMPORALIS_ANTERIOR
    2 -> SUPERFICIAL_MASSETER
    3 -> SPLENIUS_CAPITIS
    4 -> UPPER_TRAPEZIUS
    5 -> MIDDLE_TRAPEZIUS
    6 -> LOWER_TRAPEZIUS
    7 -> RHOMBOIDEUS_MAJOR
    8 -> RHOMBOIDEUS_MINOR
    9 -> ANTERIOR_DELTOID
    10 -> POSTERIOR_DELTOID
    11 -> LATERAL_DELTOID
    12 -> INFRASPINATUS
    13 -> TERES_MAJOR
    14 -> ERECTOR_SPINAE
    15 -> LATISSIMUS_DORSI
    16 -> BIC_BR_LONG_HEAD
    17 -> BIC_BR_SHORT_HEAD
    18 -> TRIC_BR_LAT_HEAD
    19 -> TRIC_BR_MED_HEAD
    20 -> PRONATOR_TERES
    21 -> FLEX_CARPI_RADIALIS
    22 -> FLEX_CARPI_ULNARIS
    23 -> PALMARIS_LONGUS
    24 -> EXT_CARPI_RADIALIS
    25 -> EXT_CARPI_ULNARIS
    26 -> EXT_DIG_COMMUNIS
    27 -> BRACHIORADIALIS
    28 -> ABD_POLLICIS_BREV
    29 -> ABD_POLLICIS_LONG
    30 -> OPPONENS_POLLICIS
    31 -> ADDUCTOR_POLLICIS
    32 -> FLEX_POLLICIS_BREV
    33 -> ABD_DIGITI_MINIMI
    34 -> FLEX_DIGITI_MINIMI
    35 -> OPP_DIGITI_MINIMI
    36 -> DORSAL_INTEROSSEI
    37 -> PALMAR_INTEROSSEI
    38 -> LUMBRICAL
    39 -> RECTUS_ABDOMINIS
    40 -> EXT_ABDOM_OBLIQ
    41 -> SERRATUS_ANTERIOR
    42 -> PECTORALIS_MAJOR
    43 -> STERNOC_STER_HEAD
    44 -> STERNOC_CLAV_HEAD
    45 -> ANTERIOR_SCALENUS
    46 -> TENSOR_FASCIA_LATAE
    47 -> GASTROCN_LAT
    48 -> GASTROCN_MED
    49 -> BICEPS_FEMORIS
    50 -> SOLEUS
    51 -> SEMITENDINOSUS
    52 -> GLUTEUS_MAXIMUS
    53 -> GLUTEUS_MEDIUS
    54 -> VASTUS_LATERALIS
    55 -> VASTUS_MEDIALIS
    56 -> RECTUS_FEMORIS
    57 -> TIBIALIS_ANTERIOR
    58 -> PERONEUS_LONGUS
    59 -> SEMIMEMBRANOSUS
    60 -> GRACILIS
    61 -> EXT_ANAL_SPINCTER
    62 -> PUBORECTALIS
    63 -> URETHRAL_SPINCTER
    64 -> NOT_A_MUSCLE
    """

    NOT_DEFINED = 0
    TEMPORALIS_ANTERIOR = 1
    SUPERFICIAL_MASSETER = 2
    SPLENIUS_CAPITIS = 3
    UPPER_TRAPEZIUS = 4
    MIDDLE_TRAPEZIUS = 5
    LOWER_TRAPEZIUS = 6
    RHOMBOIDEUS_MAJOR = 7
    RHOMBOIDEUS_MINOR = 8
    ANTERIOR_DELTOID = 9
    POSTERIOR_DELTOID = 10
    LATERAL_DELTOID = 11
    INFRASPINATUS = 12
    TERES_MAJOR = 13
    ERECTOR_SPINAE = 14
    LATISSIMUS_DORSI = 15
    BIC_BR_LONG_HEAD = 16
    BIC_BR_SHORT_HEAD = 17
    TRIC_BR_LAT_HEAD = 18
    TRIC_BR_MED_HEAD = 19
    PRONATOR_TERES = 20
    FLEX_CARPI_RADIALIS = 21
    FLEX_CARPI_ULNARIS = 22
    PALMARIS_LONGUS = 23
    EXT_CARPI_RADIALIS = 24
    EXT_CARPI_ULNARIS = 25
    EXT_DIG_COMMUNIS = 26
    BRACHIORADIALIS = 27
    ABD_POLLICIS_BREV = 28
    ABD_POLLICIS_LONG = 29
    OPPONENS_POLLICIS = 30
    ADDUCTOR_POLLICIS = 31
    FLEX_POLLICIS_BREV = 32
    ABD_DIGITI_MINIMI = 33
    FLEX_DIGITI_MINIMI = 34
    OPP_DIGITI_MINIMI = 35
    DORSAL_INTEROSSEI = 36
    PALMAR_INTEROSSEI = 37
    LUMBRICAL = 38
    RECTUS_ABDOMINIS = 39
    EXT_ABDOM_OBLIQ = 40
    SERRATUS_ANTERIOR = 41
    PECTORALIS_MAJOR = 42
    STERNOC_STER_HEAD = 43
    STERNOC_CLAV_HEAD = 44
    ANTERIOR_SCALENUS = 45
    TENSOR_FASCIA_LATAE = 46
    GASTROCN_LAT = 47
    GASTROCN_MED = 48
    BICEPS_FEMORIS = 49
    SOLEUS = 50
    SEMITENDINOSUS = 51
    GLUTEUS_MAXIMUS = 52
    GLUTEUS_MEDIUS = 53
    VASTUS_LATERALIS = 54
    VASTUS_MEDIALIS = 55
    RECTUS_FEMORIS = 56
    TIBIALIS_ANTERIOR = 57
    PERONEUS_LONGUS = 58
    SEMIMEMBRANOSUS = 59
    GRACILIS = 60
    EXT_ANAL_SPINCTER = 61
    PUBORECTALIS = 62
    URETHRAL_SPINCTER = 63
    NOT_A_MUSCLE = 64


# ---------- INX_CONF1 BYTE ----------


class QuattrocentoSensor(Enum):
    """
    Enum class for the sensors for INPUT INX or MULTIPLE INX (5 bits).

    0 -> NOT_DEFINED
    1 -> SIXTEENMONOPOLAR_EEG
    2 -> MONOPOLAR_INTRAMUSCULAR_ELECTRODE
    3 -> BIPOLAR_ELECTRODE_CODE
    4 -> EIGHT_ACCELEROMETER
    5 -> BIPOLAR_ELECTRODE_DE1
    6 -> BIPOLAR_ELECTRODE_CDE
    7 -> BIPOLAR_ELECTRODE_OTHER
    8 -> FOUR_ELECTRODE_ARRAY_10MM
    9 -> EIGHT_ELECTRODE_ARRAY_5MM
    10 -> EIGTH_ELECTRODE_ARRAY_10MM
    11 -> SIXTYFOUR_ELECTRODE_GRID_2_54MM
    12 -> SIXTYFOUR_ELECTRODE_GRID_8MM
    13 -> SIXTYFOUR_ELECTRODE_GRID_10MM
    14 -> SIXTYFOUR_ELECTRODE_GRID_12_5MM
    15 -> SIXTEEN_ELECTRODE_ARRAY_2_5MM
    16 -> SIXTEEN_ELECTRODE_ARRAY_5MM
    17 -> SIXTEEN_ELECTRODE_ARRAY_10MM
    18 -> SIXTEEN_ELECTRODE_ARRAY_10MM_2
    19 -> SIXTEEN_ELECTRODE_RECTAL_PROBE
    20 -> FORTYEIGHT_ELECTRODE_RECTAL_PROBE
    21 -> TWELVE_ELECTRODE_ARMBAND
    22 -> SIXTEEN_ELECTRODE_ARMBAND
    23 -> OTHER_SENSOR
    """

    NOT_DEFINED = 0
    SIXTEENMONOPOLAR_EEG = 1
    MONOPOLAR_INTRAMUSCULAR_ELECTRODE = 2
    BIPOLAR_ELECTRODE_CODE = 3
    EIGHT_ACCELEROMETER = 4
    BIPOLAR_ELECTRODE_DE1 = 5
    BIPOLAR_ELECTRODE_CDE = 6
    BIPOLAR_ELECTRODE_OTHER = 7
    FOUR_ELECTRODE_ARRAY_10MM = 8
    EIGHT_ELECTRODE_ARRAY_5MM = 9
    EIGTH_ELECTRODE_ARRAY_10MM = 10
    SIXTYFOUR_ELECTRODE_GRID_2_54MM = 11
    SIXTYFOUR_ELECTRODE_GRID_8MM = 12
    SIXTYFOUR_ELECTRODE_GRID_10MM = 13
    SIXTYFOUR_ELECTRODE_GRID_12_5MM = 14
    SIXTEEN_ELECTRODE_ARRAY_2_5MM = 15
    SIXTEEN_ELECTRODE_ARRAY_5MM = 16
    SIXTEEN_ELECTRODE_ARRAY_10MM = 17
    SIXTEEN_ELECTRODE_ARRAY_10MM_2 = 18
    SIXTEEN_ELECTRODE_RECTAL_PROBE = 19
    FORTYEIGHT_ELECTRODE_RECTAL_PROBE = 20
    TWELVE_ELECTRODE_ARMBAND = 21
    SIXTEEN_ELECTRODE_ARMBAND = 22
    OTHER_SENSOR = 23


class QuattrocentoAdaptor(Enum):
    """
    Enum class for the adaptors for INPUT INX or MULTIPLE INX (3 bits).

    0 -> NOT_DEFINED
    1 -> SIXTEEN_AD1x16
    2 -> EIGTH_AD2x8
    3 -> FOUR_AD4x4
    4 -> SIXTYFOUR_AD1x64
    5 -> SIXTEEN_AD_8x2
    6 -> OTHER_ADAPTOR
    """

    NOT_DEFINED = 0
    SIXTEEN_AD1x16 = 1
    EIGTH_AD2x8 = 2
    FOUR_AD4x4 = 3
    SIXTYFOUR_AD1x64 = 4
    SIXTEEN_AD_8x2 = 5
    OTHER_ADAPTOR = 6


# ---------- INX_CONF2 BYTE ----------


class QuattrocentoSide(Enum):
    """
    Enum class for the side index for INPUT INX or MULTIPLE INX (2 bits).

    0 -> NOT_DEFINED
    1 -> LEFT
    2 -> RIGHT
    3 -> NONE
    """

    NOT_DEFINED = 0
    LEFT = 1
    RIGHT = 2
    NONE = 3


class QuattrocentoHighPassFilter(Enum):
    """
    Enum class for the high-pass filter of INPUT INX or MULTIPLE INX (2 bits).

    0 -> LOW: 0.7 Hz
    1 -> MEDIUM: 10 Hz
    2 -> HIGH: 100 Hz
    3 -> ULTRA: 200 Hz
    """

    LOW = 0
    MEDIUM = 1
    HIGH = 2
    ULTRA = 3


class QuattrocentoLowPassFilter(Enum):
    """
    Enum class for the low-pass filter of INPUT INX or MULTIPLE INX (2 bits).

    0 -> LOW: 130 Hz
    1 -> MEDIUM: 500 Hz
    2 -> HIGH: 900 Hz
    3 -> ULTRA: 4400 Hz
    """

    LOW = 0
    MEDIUM = 1
    HIGH = 2
    ULTRA = 3


class QuattrocentoDetectionMode(Enum):
    """
    Enum class for the detection mode for INPUT INX or MULTIPLE INX (2 bits).

    0 -> MONOPOLAR
    1 -> DIFFERENTIAL
    2 -> BIPOLAR
    """

    MONOPOLAR = 0
    DIFFERENTIAL = 1
    BIPOLAR = 2


class QuattrocentoLightSamplingFrequency(Enum):
    """
    Enum class for the sampling frequencies of the Quattrocento Light device.

    0 -> LOW: 512 Hz
    1 -> MEDIUM: 2048 Hz
    2 -> HIGH: 5120 Hz
    3 -> ULTRA: 10240 Hz
    """

    LOW = 0
    MEDIUM = 1
    HIGH = 2
    ULTRA = 3


class QuattrocentoLightStreamingFrequency(Enum):
    """
    Enum class for the streaming frequencies of the Quattrocento Light device

    0 -> ONE: 1 Hz
    1 -> TWO: 2 Hz
    2 -> FOUR: 4 Hz
    3 -> EIGHT: 8 Hz
    4 -> SIXTEEN: 16 Hz
    5 -> THIRTYTWO: 32 Hz
    """

    ONE = 0
    TWO = 1
    FOUR = 2
    EIGHT = 3
    SIXTEEN = 4
    THIRTYTWO = 5
