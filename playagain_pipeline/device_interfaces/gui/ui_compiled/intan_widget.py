# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'intan_widget.ui'
##
## Created by: Qt User Interface Compiler version 6.7.0
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
from PySide6.QtWidgets import (QApplication, QCheckBox, QGridLayout, QGroupBox,
    QLabel, QLineEdit, QPushButton, QSizePolicy,
    QSpacerItem, QWidget)

class Ui_IntanRHDControllerForm(object):
    def setupUi(self, IntanRHDControllerForm):
        if not IntanRHDControllerForm.objectName():
            IntanRHDControllerForm.setObjectName(u"IntanRHDControllerForm")
        IntanRHDControllerForm.resize(400, 524)
        self.gridLayout = QGridLayout(IntanRHDControllerForm)
        self.gridLayout.setObjectName(u"gridLayout")
        self.commandsGroupBox = QGroupBox(IntanRHDControllerForm)
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


        self.gridLayout.addWidget(self.commandsGroupBox, 6, 0, 1, 2)

        self.inputParametersGroupBox = QGroupBox(IntanRHDControllerForm)
        self.inputParametersGroupBox.setObjectName(u"inputParametersGroupBox")
        self.gridLayout_4 = QGridLayout(self.inputParametersGroupBox)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.label_6 = QLabel(self.inputParametersGroupBox)
        self.label_6.setObjectName(u"label_6")

        self.gridLayout_4.addWidget(self.label_6, 0, 0, 1, 1)


        self.gridLayout.addWidget(self.inputParametersGroupBox, 3, 0, 1, 2)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.gridLayout.addItem(self.verticalSpacer, 7, 0, 1, 1)

        self.acquisitionParametersGroupBox = QGroupBox(IntanRHDControllerForm)
        self.acquisitionParametersGroupBox.setObjectName(u"acquisitionParametersGroupBox")
        self.gridLayout_2 = QGridLayout(self.acquisitionParametersGroupBox)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.label = QLabel(self.acquisitionParametersGroupBox)
        self.label.setObjectName(u"label")

        self.gridLayout_2.addWidget(self.label, 1, 0, 1, 1)

        self.acquisitionSamplingFrequencyLabel = QLabel(self.acquisitionParametersGroupBox)
        self.acquisitionSamplingFrequencyLabel.setObjectName(u"acquisitionSamplingFrequencyLabel")

        self.gridLayout_2.addWidget(self.acquisitionSamplingFrequencyLabel, 1, 1, 1, 1)


        self.gridLayout.addWidget(self.acquisitionParametersGroupBox, 2, 0, 1, 2)

        self.gridSelectionGroupBox = QGroupBox(IntanRHDControllerForm)
        self.gridSelectionGroupBox.setObjectName(u"gridSelectionGroupBox")
        self.gridLayout_5 = QGridLayout(self.gridSelectionGroupBox)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.gridDCheckBox = QCheckBox(self.gridSelectionGroupBox)
        self.gridDCheckBox.setObjectName(u"gridDCheckBox")
        self.gridDCheckBox.setEnabled(False)

        self.gridLayout_5.addWidget(self.gridDCheckBox, 1, 3, 1, 1)

        self.gridHCheckBox = QCheckBox(self.gridSelectionGroupBox)
        self.gridHCheckBox.setObjectName(u"gridHCheckBox")
        self.gridHCheckBox.setEnabled(False)

        self.gridLayout_5.addWidget(self.gridHCheckBox, 2, 3, 1, 1)

        self.gridGCheckBox = QCheckBox(self.gridSelectionGroupBox)
        self.gridGCheckBox.setObjectName(u"gridGCheckBox")
        self.gridGCheckBox.setEnabled(False)

        self.gridLayout_5.addWidget(self.gridGCheckBox, 2, 2, 1, 1)

        self.gridECheckBox = QCheckBox(self.gridSelectionGroupBox)
        self.gridECheckBox.setObjectName(u"gridECheckBox")
        self.gridECheckBox.setEnabled(False)

        self.gridLayout_5.addWidget(self.gridECheckBox, 2, 0, 1, 1)

        self.gridFCheckBox = QCheckBox(self.gridSelectionGroupBox)
        self.gridFCheckBox.setObjectName(u"gridFCheckBox")
        self.gridFCheckBox.setEnabled(False)

        self.gridLayout_5.addWidget(self.gridFCheckBox, 2, 1, 1, 1)

        self.gridACheckBox = QCheckBox(self.gridSelectionGroupBox)
        self.gridACheckBox.setObjectName(u"gridACheckBox")
        self.gridACheckBox.setEnabled(False)

        self.gridLayout_5.addWidget(self.gridACheckBox, 1, 0, 1, 1)

        self.gridBCheckBox = QCheckBox(self.gridSelectionGroupBox)
        self.gridBCheckBox.setObjectName(u"gridBCheckBox")
        self.gridBCheckBox.setEnabled(False)

        self.gridLayout_5.addWidget(self.gridBCheckBox, 1, 1, 1, 1)

        self.gridCCheckBox = QCheckBox(self.gridSelectionGroupBox)
        self.gridCCheckBox.setObjectName(u"gridCCheckBox")
        self.gridCCheckBox.setEnabled(False)

        self.gridLayout_5.addWidget(self.gridCCheckBox, 1, 2, 1, 1)

        self.deviceUpdatePushButton = QPushButton(self.gridSelectionGroupBox)
        self.deviceUpdatePushButton.setObjectName(u"deviceUpdatePushButton")

        self.gridLayout_5.addWidget(self.deviceUpdatePushButton, 0, 0, 1, 1)


        self.gridLayout.addWidget(self.gridSelectionGroupBox, 4, 0, 1, 2)

        self.connectionGroupBox = QGroupBox(IntanRHDControllerForm)
        self.connectionGroupBox.setObjectName(u"connectionGroupBox")
        self.gridLayout_7 = QGridLayout(self.connectionGroupBox)
        self.gridLayout_7.setObjectName(u"gridLayout_7")
        self.label_5 = QLabel(self.connectionGroupBox)
        self.label_5.setObjectName(u"label_5")

        self.gridLayout_7.addWidget(self.label_5, 1, 0, 1, 1)

        self.connectionIPLineEdit = QLineEdit(self.connectionGroupBox)
        self.connectionIPLineEdit.setObjectName(u"connectionIPLineEdit")

        self.gridLayout_7.addWidget(self.connectionIPLineEdit, 0, 1, 1, 1)

        self.label_4 = QLabel(self.connectionGroupBox)
        self.label_4.setObjectName(u"label_4")

        self.gridLayout_7.addWidget(self.label_4, 0, 0, 1, 1)

        self.connectionPortLineEdit = QLineEdit(self.connectionGroupBox)
        self.connectionPortLineEdit.setObjectName(u"connectionPortLineEdit")

        self.gridLayout_7.addWidget(self.connectionPortLineEdit, 1, 1, 1, 1)

        self.connectionUseWaveformCheckBox = QCheckBox(self.connectionGroupBox)
        self.connectionUseWaveformCheckBox.setObjectName(u"connectionUseWaveformCheckBox")

        self.gridLayout_7.addWidget(self.connectionUseWaveformCheckBox, 0, 2, 1, 1)

        self.connectionUseSpikeCheckBox = QCheckBox(self.connectionGroupBox)
        self.connectionUseSpikeCheckBox.setObjectName(u"connectionUseSpikeCheckBox")

        self.gridLayout_7.addWidget(self.connectionUseSpikeCheckBox, 1, 2, 1, 1)


        self.gridLayout.addWidget(self.connectionGroupBox, 1, 0, 1, 1)


        self.retranslateUi(IntanRHDControllerForm)

        QMetaObject.connectSlotsByName(IntanRHDControllerForm)
    # setupUi

    def retranslateUi(self, IntanRHDControllerForm):
        IntanRHDControllerForm.setWindowTitle(QCoreApplication.translate("IntanRHDControllerForm", u"RHD Recording Controller", None))
        self.commandsGroupBox.setTitle(QCoreApplication.translate("IntanRHDControllerForm", u"Commands", None))
        self.commandConnectionPushButton.setText(QCoreApplication.translate("IntanRHDControllerForm", u"Connect", None))
        self.commandConfigurationPushButton.setText(QCoreApplication.translate("IntanRHDControllerForm", u"Configure", None))
        self.commandStreamPushButton.setText(QCoreApplication.translate("IntanRHDControllerForm", u"Stream", None))
        self.inputParametersGroupBox.setTitle(QCoreApplication.translate("IntanRHDControllerForm", u"Input Parameters", None))
        self.label_6.setText(QCoreApplication.translate("IntanRHDControllerForm", u"To be added", None))
        self.acquisitionParametersGroupBox.setTitle(QCoreApplication.translate("IntanRHDControllerForm", u"Acquisition Parameters", None))
        self.label.setText(QCoreApplication.translate("IntanRHDControllerForm", u"Sampling Frequency", None))
        self.acquisitionSamplingFrequencyLabel.setText(QCoreApplication.translate("IntanRHDControllerForm", u"Placeholder", None))
        self.gridSelectionGroupBox.setTitle(QCoreApplication.translate("IntanRHDControllerForm", u"Grid Selection", None))
        self.gridDCheckBox.setText(QCoreApplication.translate("IntanRHDControllerForm", u"D", None))
        self.gridHCheckBox.setText(QCoreApplication.translate("IntanRHDControllerForm", u"H", None))
        self.gridGCheckBox.setText(QCoreApplication.translate("IntanRHDControllerForm", u"G", None))
        self.gridECheckBox.setText(QCoreApplication.translate("IntanRHDControllerForm", u"E", None))
        self.gridFCheckBox.setText(QCoreApplication.translate("IntanRHDControllerForm", u"F", None))
        self.gridACheckBox.setText(QCoreApplication.translate("IntanRHDControllerForm", u"A", None))
        self.gridBCheckBox.setText(QCoreApplication.translate("IntanRHDControllerForm", u"B", None))
        self.gridCCheckBox.setText(QCoreApplication.translate("IntanRHDControllerForm", u"C", None))
        self.deviceUpdatePushButton.setText(QCoreApplication.translate("IntanRHDControllerForm", u"Update", None))
        self.connectionGroupBox.setTitle(QCoreApplication.translate("IntanRHDControllerForm", u"Connection parameters", None))
        self.label_5.setText(QCoreApplication.translate("IntanRHDControllerForm", u"Port", None))
        self.connectionIPLineEdit.setText(QCoreApplication.translate("IntanRHDControllerForm", u"127.0.01", None))
        self.label_4.setText(QCoreApplication.translate("IntanRHDControllerForm", u"IP", None))
        self.connectionPortLineEdit.setText(QCoreApplication.translate("IntanRHDControllerForm", u"5000", None))
        self.connectionUseWaveformCheckBox.setText(QCoreApplication.translate("IntanRHDControllerForm", u"Use Waveform", None))
        self.connectionUseSpikeCheckBox.setText(QCoreApplication.translate("IntanRHDControllerForm", u"Use Spike", None))
    # retranslateUi

