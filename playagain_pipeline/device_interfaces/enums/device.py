from enum import Enum


class CommunicationProtocol(Enum):
    TCPIP = 0
    UDP = 1
    SERIAL = 2
    Bluetooth = 3
    I2C = 4
    SPI = 5
    DEFAULT = 6


class Device(Enum):
    QUATTROCENTO = 0
    QUATTROCENTO_LIGHT = 1
    SESSANTAQUATTRO = 2
    INTAN_RHD_CONTROLLER = 3
    NOXON = 4
    MUOVI = 5
    MUOVI_PLUS = 6
    SYNCSTATION = 7
    MINDROVE = 8
    NSWITCH = 9
    INTAN_RHD_CONTROLLER_FILES = 10
    DEFAULT = 11


class OTBDevice(Enum):
    QUATTROCENTO_LIGHT = 0
    MUOVI = 1
    MUOVI_PLUS = 2


class LoggerLevel(Enum):
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4
    NOTSET = 5
    DEFAULT = 6
