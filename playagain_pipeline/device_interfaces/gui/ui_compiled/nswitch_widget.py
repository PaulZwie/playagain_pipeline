# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'nswitch_widget.ui'
##
## Created by: Qt User Interface Compiler version 6.6.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QComboBox, QGridLayout, QGroupBox,
    QLabel, QPushButton, QRadioButton, QSizePolicy,
    QSpacerItem, QWidget)

class Ui_NSwitchForm(object):
    def setupUi(self, NSwitchForm):
        if not NSwitchForm.objectName():
            NSwitchForm.setObjectName(u"NSwitchForm")
        NSwitchForm.resize(400, 540)
        self.gridLayout_2 = QGridLayout(NSwitchForm)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.acquisitionGroupBox = QGroupBox(NSwitchForm)
        self.acquisitionGroupBox.setObjectName(u"acquisitionGroupBox")
        self.gridLayout = QGridLayout(self.acquisitionGroupBox)
        self.gridLayout.setObjectName(u"gridLayout")
        self.label_2 = QLabel(self.acquisitionGroupBox)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)

        self.acquisitionSamplingFrequencyComboBox = QComboBox(self.acquisitionGroupBox)
        self.acquisitionSamplingFrequencyComboBox.addItem("")
        self.acquisitionSamplingFrequencyComboBox.addItem("")
        self.acquisitionSamplingFrequencyComboBox.addItem("")
        self.acquisitionSamplingFrequencyComboBox.addItem("")
        self.acquisitionSamplingFrequencyComboBox.setObjectName(u"acquisitionSamplingFrequencyComboBox")

        self.gridLayout.addWidget(self.acquisitionSamplingFrequencyComboBox, 0, 1, 1, 1)

        self.acquisitionNumberOfChannelsComboBox = QComboBox(self.acquisitionGroupBox)
        self.acquisitionNumberOfChannelsComboBox.addItem("")
        self.acquisitionNumberOfChannelsComboBox.addItem("")
        self.acquisitionNumberOfChannelsComboBox.addItem("")
        self.acquisitionNumberOfChannelsComboBox.addItem("")
        self.acquisitionNumberOfChannelsComboBox.setObjectName(u"acquisitionNumberOfChannelsComboBox")

        self.gridLayout.addWidget(self.acquisitionNumberOfChannelsComboBox, 1, 1, 1, 1)

        self.label = QLabel(self.acquisitionGroupBox)
        self.label.setObjectName(u"label")

        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)


        self.gridLayout_2.addWidget(self.acquisitionGroupBox, 1, 0, 1, 1)

        self.inputGroupBox = QGroupBox(NSwitchForm)
        self.inputGroupBox.setObjectName(u"inputGroupBox")
        self.gridLayout_4 = QGridLayout(self.inputGroupBox)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.voltageReferenceGroupBox = QGroupBox(self.inputGroupBox)
        self.voltageReferenceGroupBox.setObjectName(u"voltageReferenceGroupBox")
        self.gridLayout_6 = QGridLayout(self.voltageReferenceGroupBox)
        self.gridLayout_6.setObjectName(u"gridLayout_6")
        self.voltageReferenceLowRadioButton = QRadioButton(self.voltageReferenceGroupBox)
        self.voltageReferenceLowRadioButton.setObjectName(u"voltageReferenceLowRadioButton")
        self.voltageReferenceLowRadioButton.setChecked(True)

        self.gridLayout_6.addWidget(self.voltageReferenceLowRadioButton, 0, 0, 1, 1)

        self.voltageReferenceHighRadioButton = QRadioButton(self.voltageReferenceGroupBox)
        self.voltageReferenceHighRadioButton.setObjectName(u"voltageReferenceHighRadioButton")

        self.gridLayout_6.addWidget(self.voltageReferenceHighRadioButton, 0, 1, 1, 1)


        self.gridLayout_4.addWidget(self.voltageReferenceGroupBox, 3, 0, 1, 2)

        self.groupBox_2 = QGroupBox(self.inputGroupBox)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.gridLayout_5 = QGridLayout(self.groupBox_2)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.inputLowResolutionRadioButton = QRadioButton(self.groupBox_2)
        self.inputLowResolutionRadioButton.setObjectName(u"inputLowResolutionRadioButton")
        self.inputLowResolutionRadioButton.setChecked(False)

        self.gridLayout_5.addWidget(self.inputLowResolutionRadioButton, 0, 0, 1, 1)

        self.inputHighResolutionRadioButton = QRadioButton(self.groupBox_2)
        self.inputHighResolutionRadioButton.setObjectName(u"inputHighResolutionRadioButton")
        self.inputHighResolutionRadioButton.setChecked(True)

        self.gridLayout_5.addWidget(self.inputHighResolutionRadioButton, 0, 1, 1, 1)


        self.gridLayout_4.addWidget(self.groupBox_2, 0, 0, 1, 2)

        self.inputDetectionModeComboBox = QComboBox(self.inputGroupBox)
        self.inputDetectionModeComboBox.addItem("")
        self.inputDetectionModeComboBox.addItem("")
        self.inputDetectionModeComboBox.addItem("")
        self.inputDetectionModeComboBox.addItem("")
        self.inputDetectionModeComboBox.setObjectName(u"inputDetectionModeComboBox")

        self.gridLayout_4.addWidget(self.inputDetectionModeComboBox, 1, 1, 1, 1)

        self.label_5 = QLabel(self.inputGroupBox)
        self.label_5.setObjectName(u"label_5")

        self.gridLayout_4.addWidget(self.label_5, 2, 0, 1, 1)

        self.label_10 = QLabel(self.inputGroupBox)
        self.label_10.setObjectName(u"label_10")

        self.gridLayout_4.addWidget(self.label_10, 1, 0, 1, 1)

        self.inputGainComboBox = QComboBox(self.inputGroupBox)
        self.inputGainComboBox.addItem("")
        self.inputGainComboBox.addItem("")
        self.inputGainComboBox.addItem("")
        self.inputGainComboBox.addItem("")
        self.inputGainComboBox.addItem("")
        self.inputGainComboBox.addItem("")
        self.inputGainComboBox.addItem("")
        self.inputGainComboBox.setObjectName(u"inputGainComboBox")

        self.gridLayout_4.addWidget(self.inputGainComboBox, 2, 1, 1, 1)


        self.gridLayout_2.addWidget(self.inputGroupBox, 2, 0, 1, 1)

        self.connectionGroupBox = QGroupBox(NSwitchForm)
        self.connectionGroupBox.setObjectName(u"connectionGroupBox")
        self.gridLayout_7 = QGridLayout(self.connectionGroupBox)
        self.gridLayout_7.setObjectName(u"gridLayout_7")
        self.connectionPortLabel = QLabel(self.connectionGroupBox)
        self.connectionPortLabel.setObjectName(u"connectionPortLabel")

        self.gridLayout_7.addWidget(self.connectionPortLabel, 1, 1, 1, 1)

        self.label_6 = QLabel(self.connectionGroupBox)
        self.label_6.setObjectName(u"label_6")

        self.gridLayout_7.addWidget(self.label_6, 0, 0, 1, 1)

        self.label_7 = QLabel(self.connectionGroupBox)
        self.label_7.setObjectName(u"label_7")

        self.gridLayout_7.addWidget(self.label_7, 1, 0, 1, 1)

        self.connectionUpdatePushButton = QPushButton(self.connectionGroupBox)
        self.connectionUpdatePushButton.setObjectName(u"connectionUpdatePushButton")

        self.gridLayout_7.addWidget(self.connectionUpdatePushButton, 0, 2, 1, 1)

        self.connectionIPComboBox = QComboBox(self.connectionGroupBox)
        self.connectionIPComboBox.setObjectName(u"connectionIPComboBox")

        self.gridLayout_7.addWidget(self.connectionIPComboBox, 0, 1, 1, 1)


        self.gridLayout_2.addWidget(self.connectionGroupBox, 0, 0, 1, 1)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_2.addItem(self.verticalSpacer, 4, 0, 1, 1)

        self.commandsGroupBox = QGroupBox(NSwitchForm)
        self.commandsGroupBox.setObjectName(u"commandsGroupBox")
        self.gridLayout_3 = QGridLayout(self.commandsGroupBox)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.commandConnectionPushButton = QPushButton(self.commandsGroupBox)
        self.commandConnectionPushButton.setObjectName(u"commandConnectionPushButton")

        self.gridLayout_3.addWidget(self.commandConnectionPushButton, 0, 0, 1, 1)

        self.commandConfigurationPushButton = QPushButton(self.commandsGroupBox)
        self.commandConfigurationPushButton.setObjectName(u"commandConfigurationPushButton")

        self.gridLayout_3.addWidget(self.commandConfigurationPushButton, 1, 0, 1, 1)

        self.commandStreamPushButton = QPushButton(self.commandsGroupBox)
        self.commandStreamPushButton.setObjectName(u"commandStreamPushButton")

        self.gridLayout_3.addWidget(self.commandStreamPushButton, 2, 0, 1, 1)


        self.gridLayout_2.addWidget(self.commandsGroupBox, 3, 0, 1, 1)


        self.retranslateUi(NSwitchForm)

        self.acquisitionSamplingFrequencyComboBox.setCurrentIndex(0)
        self.acquisitionNumberOfChannelsComboBox.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(NSwitchForm)
    # setupUi

    def retranslateUi(self, NSwitchForm):
        NSwitchForm.setWindowTitle(QCoreApplication.translate("NSwitchForm", u"NSwitchForm", None))
        self.acquisitionGroupBox.setTitle(QCoreApplication.translate("NSwitchForm", u"Acquisiton Parameters", None))
        self.label_2.setText(QCoreApplication.translate("NSwitchForm", u"Number of Channels", None))
        self.acquisitionSamplingFrequencyComboBox.setItemText(0, QCoreApplication.translate("NSwitchForm", u"500", None))
        self.acquisitionSamplingFrequencyComboBox.setItemText(1, QCoreApplication.translate("NSwitchForm", u"1000", None))
        self.acquisitionSamplingFrequencyComboBox.setItemText(2, QCoreApplication.translate("NSwitchForm", u"2000", None))
        self.acquisitionSamplingFrequencyComboBox.setItemText(3, QCoreApplication.translate("NSwitchForm", u"4000", None))

        self.acquisitionNumberOfChannelsComboBox.setItemText(0, QCoreApplication.translate("NSwitchForm", u"2", None))
        self.acquisitionNumberOfChannelsComboBox.setItemText(1, QCoreApplication.translate("NSwitchForm", u"4", None))
        self.acquisitionNumberOfChannelsComboBox.setItemText(2, QCoreApplication.translate("NSwitchForm", u"8", None))
        self.acquisitionNumberOfChannelsComboBox.setItemText(3, QCoreApplication.translate("NSwitchForm", u"16", None))

        self.label.setText(QCoreApplication.translate("NSwitchForm", u"Sampling Frequency", None))
        self.inputGroupBox.setTitle(QCoreApplication.translate("NSwitchForm", u"Input Parameters", None))
        self.voltageReferenceGroupBox.setTitle(QCoreApplication.translate("NSwitchForm", u"Voltage reference VREF", None))
        self.voltageReferenceLowRadioButton.setText(QCoreApplication.translate("NSwitchForm", u"2.4V", None))
        self.voltageReferenceHighRadioButton.setText(QCoreApplication.translate("NSwitchForm", u"4V", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("NSwitchForm", u"Resolution", None))
        self.inputLowResolutionRadioButton.setText(QCoreApplication.translate("NSwitchForm", u"16 Bit", None))
        self.inputHighResolutionRadioButton.setText(QCoreApplication.translate("NSwitchForm", u"24 Bit", None))
        self.inputDetectionModeComboBox.setItemText(0, QCoreApplication.translate("NSwitchForm", u"MONOPOLAR", None))
        self.inputDetectionModeComboBox.setItemText(1, QCoreApplication.translate("NSwitchForm", u"BIPOLAR", None))
        self.inputDetectionModeComboBox.setItemText(2, QCoreApplication.translate("NSwitchForm", u"IMPEDANCE", None))
        self.inputDetectionModeComboBox.setItemText(3, QCoreApplication.translate("NSwitchForm", u"TEST", None))

        self.label_5.setText(QCoreApplication.translate("NSwitchForm", u"Gain", None))
        self.label_10.setText(QCoreApplication.translate("NSwitchForm", u"Mode", None))
        self.inputGainComboBox.setItemText(0, QCoreApplication.translate("NSwitchForm", u"6 (Default)", None))
        self.inputGainComboBox.setItemText(1, QCoreApplication.translate("NSwitchForm", u"1", None))
        self.inputGainComboBox.setItemText(2, QCoreApplication.translate("NSwitchForm", u"2", None))
        self.inputGainComboBox.setItemText(3, QCoreApplication.translate("NSwitchForm", u"3", None))
        self.inputGainComboBox.setItemText(4, QCoreApplication.translate("NSwitchForm", u"4", None))
        self.inputGainComboBox.setItemText(5, QCoreApplication.translate("NSwitchForm", u"8", None))
        self.inputGainComboBox.setItemText(6, QCoreApplication.translate("NSwitchForm", u"12", None))

        self.connectionGroupBox.setTitle(QCoreApplication.translate("NSwitchForm", u"Connection parameters", None))
        self.connectionPortLabel.setText(QCoreApplication.translate("NSwitchForm", u"1234", None))
        self.label_6.setText(QCoreApplication.translate("NSwitchForm", u"IP", None))
        self.label_7.setText(QCoreApplication.translate("NSwitchForm", u"Port", None))
        self.connectionUpdatePushButton.setText(QCoreApplication.translate("NSwitchForm", u"Update", None))
        self.commandsGroupBox.setTitle(QCoreApplication.translate("NSwitchForm", u"Commands", None))
        self.commandConnectionPushButton.setText(QCoreApplication.translate("NSwitchForm", u"Connect", None))
        self.commandConfigurationPushButton.setText(QCoreApplication.translate("NSwitchForm", u"Configure", None))
        self.commandStreamPushButton.setText(QCoreApplication.translate("NSwitchForm", u"Stream", None))
    # retranslateUi

