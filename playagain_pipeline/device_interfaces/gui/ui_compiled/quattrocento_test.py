# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'quattrocento_test.ui'
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
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QGridLayout,
    QGroupBox, QLabel, QMainWindow, QMenuBar,
    QPushButton, QSizePolicy, QSpacerItem, QStatusBar,
    QWidget)

from gui_custom_elements.checkable_combo_box import CheckableComboBox
from gui_custom_elements.vispy_plot_widget import VispyPlotWidget

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(800, 808)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout_2 = QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.groupBox_2 = QGroupBox(self.centralwidget)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.gridLayout_3 = QGridLayout(self.groupBox_2)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.commandConnectionPushButton = QPushButton(self.groupBox_2)
        self.commandConnectionPushButton.setObjectName(u"commandConnectionPushButton")

        self.gridLayout_3.addWidget(self.commandConnectionPushButton, 0, 0, 1, 1)

        self.commandConfigurationPushButton = QPushButton(self.groupBox_2)
        self.commandConfigurationPushButton.setObjectName(u"commandConfigurationPushButton")

        self.gridLayout_3.addWidget(self.commandConfigurationPushButton, 1, 0, 1, 1)

        self.commandStreamPushButton = QPushButton(self.groupBox_2)
        self.commandStreamPushButton.setObjectName(u"commandStreamPushButton")

        self.gridLayout_3.addWidget(self.commandStreamPushButton, 2, 0, 1, 1)


        self.gridLayout_2.addWidget(self.groupBox_2, 3, 0, 1, 1)

        self.groupBox_4 = QGroupBox(self.centralwidget)
        self.groupBox_4.setObjectName(u"groupBox_4")
        self.gridLayout_5 = QGridLayout(self.groupBox_4)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.gridFourCheckBox = QCheckBox(self.groupBox_4)
        self.gridFourCheckBox.setObjectName(u"gridFourCheckBox")

        self.gridLayout_5.addWidget(self.gridFourCheckBox, 1, 1, 1, 1)

        self.gridThreeCheckBox = QCheckBox(self.groupBox_4)
        self.gridThreeCheckBox.setObjectName(u"gridThreeCheckBox")

        self.gridLayout_5.addWidget(self.gridThreeCheckBox, 1, 0, 1, 1)

        self.gridFiveCheckBox = QCheckBox(self.groupBox_4)
        self.gridFiveCheckBox.setObjectName(u"gridFiveCheckBox")

        self.gridLayout_5.addWidget(self.gridFiveCheckBox, 1, 2, 1, 1)

        self.gridSixCheckBox = QCheckBox(self.groupBox_4)
        self.gridSixCheckBox.setObjectName(u"gridSixCheckBox")

        self.gridLayout_5.addWidget(self.gridSixCheckBox, 1, 3, 1, 1)

        self.gridTwoCheckBox = QCheckBox(self.groupBox_4)
        self.gridTwoCheckBox.setObjectName(u"gridTwoCheckBox")
        self.gridTwoCheckBox.setLayoutDirection(Qt.LeftToRight)

        self.gridLayout_5.addWidget(self.gridTwoCheckBox, 0, 3, 1, 1)

        self.gridOneCheckBox = QCheckBox(self.groupBox_4)
        self.gridOneCheckBox.setObjectName(u"gridOneCheckBox")

        self.gridLayout_5.addWidget(self.gridOneCheckBox, 0, 0, 1, 1)


        self.gridLayout_2.addWidget(self.groupBox_4, 2, 0, 1, 1)

        self.vispyPlotWidget = VispyPlotWidget(self.centralwidget)
        self.vispyPlotWidget.setObjectName(u"vispyPlotWidget")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.vispyPlotWidget.sizePolicy().hasHeightForWidth())
        self.vispyPlotWidget.setSizePolicy(sizePolicy)

        self.gridLayout_2.addWidget(self.vispyPlotWidget, 0, 1, 4, 1)

        self.groupBox = QGroupBox(self.centralwidget)
        self.groupBox.setObjectName(u"groupBox")
        self.gridLayout = QGridLayout(self.groupBox)
        self.gridLayout.setObjectName(u"gridLayout")
        self.acquisitionNumberOfChannelsComboBox = QComboBox(self.groupBox)
        self.acquisitionNumberOfChannelsComboBox.addItem("")
        self.acquisitionNumberOfChannelsComboBox.addItem("")
        self.acquisitionNumberOfChannelsComboBox.addItem("")
        self.acquisitionNumberOfChannelsComboBox.addItem("")
        self.acquisitionNumberOfChannelsComboBox.setObjectName(u"acquisitionNumberOfChannelsComboBox")

        self.gridLayout.addWidget(self.acquisitionNumberOfChannelsComboBox, 1, 1, 1, 1)

        self.acquisitionSamplingFrequencyComboBox = QComboBox(self.groupBox)
        self.acquisitionSamplingFrequencyComboBox.addItem("")
        self.acquisitionSamplingFrequencyComboBox.addItem("")
        self.acquisitionSamplingFrequencyComboBox.addItem("")
        self.acquisitionSamplingFrequencyComboBox.addItem("")
        self.acquisitionSamplingFrequencyComboBox.setObjectName(u"acquisitionSamplingFrequencyComboBox")

        self.gridLayout.addWidget(self.acquisitionSamplingFrequencyComboBox, 0, 1, 1, 1)

        self.label_2 = QLabel(self.groupBox)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)

        self.label = QLabel(self.groupBox)
        self.label.setObjectName(u"label")

        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)

        self.acquisitionDecimatorCheckBox = QCheckBox(self.groupBox)
        self.acquisitionDecimatorCheckBox.setObjectName(u"acquisitionDecimatorCheckBox")

        self.gridLayout.addWidget(self.acquisitionDecimatorCheckBox, 2, 0, 1, 1)

        self.acquisitionRecordingCheckBox = QCheckBox(self.groupBox)
        self.acquisitionRecordingCheckBox.setObjectName(u"acquisitionRecordingCheckBox")

        self.gridLayout.addWidget(self.acquisitionRecordingCheckBox, 2, 1, 1, 1)


        self.gridLayout_2.addWidget(self.groupBox, 0, 0, 1, 1)

        self.groupBox_3 = QGroupBox(self.centralwidget)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.gridLayout_4 = QGridLayout(self.groupBox_3)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.label_4 = QLabel(self.groupBox_3)
        self.label_4.setObjectName(u"label_4")

        self.gridLayout_4.addWidget(self.label_4, 1, 0, 1, 1)

        self.label_8 = QLabel(self.groupBox_3)
        self.label_8.setObjectName(u"label_8")

        self.gridLayout_4.addWidget(self.label_8, 5, 0, 1, 1)

        self.label_7 = QLabel(self.groupBox_3)
        self.label_7.setObjectName(u"label_7")

        self.gridLayout_4.addWidget(self.label_7, 4, 0, 1, 1)

        self.label_3 = QLabel(self.groupBox_3)
        self.label_3.setObjectName(u"label_3")

        self.gridLayout_4.addWidget(self.label_3, 0, 0, 1, 1)

        self.label_5 = QLabel(self.groupBox_3)
        self.label_5.setObjectName(u"label_5")

        self.gridLayout_4.addWidget(self.label_5, 2, 0, 1, 1)

        self.inputMuscleComboBox = QComboBox(self.groupBox_3)
        self.inputMuscleComboBox.addItem("")
        self.inputMuscleComboBox.setObjectName(u"inputMuscleComboBox")

        self.gridLayout_4.addWidget(self.inputMuscleComboBox, 1, 1, 1, 1)

        self.inputChannelComboBox = QComboBox(self.groupBox_3)
        self.inputChannelComboBox.addItem("")
        self.inputChannelComboBox.addItem("")
        self.inputChannelComboBox.addItem("")
        self.inputChannelComboBox.addItem("")
        self.inputChannelComboBox.addItem("")
        self.inputChannelComboBox.addItem("")
        self.inputChannelComboBox.setObjectName(u"inputChannelComboBox")

        self.gridLayout_4.addWidget(self.inputChannelComboBox, 0, 1, 1, 1)

        self.inputHighPassComboBox = QComboBox(self.groupBox_3)
        self.inputHighPassComboBox.addItem("")
        self.inputHighPassComboBox.addItem("")
        self.inputHighPassComboBox.addItem("")
        self.inputHighPassComboBox.addItem("")
        self.inputHighPassComboBox.setObjectName(u"inputHighPassComboBox")

        self.gridLayout_4.addWidget(self.inputHighPassComboBox, 5, 1, 1, 1)

        self.label_6 = QLabel(self.groupBox_3)
        self.label_6.setObjectName(u"label_6")

        self.gridLayout_4.addWidget(self.label_6, 3, 0, 1, 1)

        self.label_10 = QLabel(self.groupBox_3)
        self.label_10.setObjectName(u"label_10")

        self.gridLayout_4.addWidget(self.label_10, 7, 0, 1, 1)

        self.inputSideComboBox = QComboBox(self.groupBox_3)
        self.inputSideComboBox.addItem("")
        self.inputSideComboBox.addItem("")
        self.inputSideComboBox.addItem("")
        self.inputSideComboBox.addItem("")
        self.inputSideComboBox.setObjectName(u"inputSideComboBox")

        self.gridLayout_4.addWidget(self.inputSideComboBox, 4, 1, 1, 1)

        self.inputAdaptorComboBox = QComboBox(self.groupBox_3)
        self.inputAdaptorComboBox.addItem("")
        self.inputAdaptorComboBox.setObjectName(u"inputAdaptorComboBox")

        self.gridLayout_4.addWidget(self.inputAdaptorComboBox, 3, 1, 1, 1)

        self.inputDetectionModeComboBox = QComboBox(self.groupBox_3)
        self.inputDetectionModeComboBox.addItem("")
        self.inputDetectionModeComboBox.addItem("")
        self.inputDetectionModeComboBox.addItem("")
        self.inputDetectionModeComboBox.setObjectName(u"inputDetectionModeComboBox")

        self.gridLayout_4.addWidget(self.inputDetectionModeComboBox, 7, 1, 1, 1)

        self.label_9 = QLabel(self.groupBox_3)
        self.label_9.setObjectName(u"label_9")

        self.gridLayout_4.addWidget(self.label_9, 6, 0, 1, 1)

        self.inputSensorComboBox = QComboBox(self.groupBox_3)
        self.inputSensorComboBox.addItem("")
        self.inputSensorComboBox.setObjectName(u"inputSensorComboBox")

        self.gridLayout_4.addWidget(self.inputSensorComboBox, 2, 1, 1, 1)

        self.inputLowPassComboBox = QComboBox(self.groupBox_3)
        self.inputLowPassComboBox.addItem("")
        self.inputLowPassComboBox.addItem("")
        self.inputLowPassComboBox.addItem("")
        self.inputLowPassComboBox.addItem("")
        self.inputLowPassComboBox.setObjectName(u"inputLowPassComboBox")

        self.gridLayout_4.addWidget(self.inputLowPassComboBox, 6, 1, 1, 1)

        self.comboBox_11 = CheckableComboBox(self.groupBox_3)
        self.comboBox_11.addItem("")
        self.comboBox_11.addItem("")
        self.comboBox_11.addItem("")
        self.comboBox_11.addItem("")
        self.comboBox_11.setObjectName(u"comboBox_11")

        self.gridLayout_4.addWidget(self.comboBox_11, 8, 0, 1, 1)

        self.inputConfigurePushButton = QPushButton(self.groupBox_3)
        self.inputConfigurePushButton.setObjectName(u"inputConfigurePushButton")

        self.gridLayout_4.addWidget(self.inputConfigurePushButton, 8, 1, 1, 1)


        self.gridLayout_2.addWidget(self.groupBox_3, 1, 0, 1, 1)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.gridLayout_2.addItem(self.verticalSpacer, 4, 0, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 800, 22))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        self.acquisitionNumberOfChannelsComboBox.setCurrentIndex(3)
        self.acquisitionSamplingFrequencyComboBox.setCurrentIndex(1)
        self.inputHighPassComboBox.setCurrentIndex(1)
        self.inputLowPassComboBox.setCurrentIndex(1)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"QuattrocentoTest", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("MainWindow", u"Commands", None))
        self.commandConnectionPushButton.setText(QCoreApplication.translate("MainWindow", u"Connect", None))
        self.commandConfigurationPushButton.setText(QCoreApplication.translate("MainWindow", u"Configure", None))
        self.commandStreamPushButton.setText(QCoreApplication.translate("MainWindow", u"Stream", None))
        self.groupBox_4.setTitle(QCoreApplication.translate("MainWindow", u"Grid Selection", None))
        self.gridFourCheckBox.setText(QCoreApplication.translate("MainWindow", u"Grid 4", None))
        self.gridThreeCheckBox.setText(QCoreApplication.translate("MainWindow", u"Grid 3", None))
        self.gridFiveCheckBox.setText(QCoreApplication.translate("MainWindow", u"Grid 5", None))
        self.gridSixCheckBox.setText(QCoreApplication.translate("MainWindow", u"Grid 6", None))
        self.gridTwoCheckBox.setText(QCoreApplication.translate("MainWindow", u"Grid 2", None))
        self.gridOneCheckBox.setText(QCoreApplication.translate("MainWindow", u"Grid 1", None))
        self.groupBox.setTitle(QCoreApplication.translate("MainWindow", u"Acquisiton Parameters", None))
        self.acquisitionNumberOfChannelsComboBox.setItemText(0, QCoreApplication.translate("MainWindow", u"120", None))
        self.acquisitionNumberOfChannelsComboBox.setItemText(1, QCoreApplication.translate("MainWindow", u"216", None))
        self.acquisitionNumberOfChannelsComboBox.setItemText(2, QCoreApplication.translate("MainWindow", u"312", None))
        self.acquisitionNumberOfChannelsComboBox.setItemText(3, QCoreApplication.translate("MainWindow", u"408", None))

        self.acquisitionSamplingFrequencyComboBox.setItemText(0, QCoreApplication.translate("MainWindow", u"512", None))
        self.acquisitionSamplingFrequencyComboBox.setItemText(1, QCoreApplication.translate("MainWindow", u"2048", None))
        self.acquisitionSamplingFrequencyComboBox.setItemText(2, QCoreApplication.translate("MainWindow", u"5120", None))
        self.acquisitionSamplingFrequencyComboBox.setItemText(3, QCoreApplication.translate("MainWindow", u"10240", None))

        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Number of Channels", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Sampling Frequency", None))
        self.acquisitionDecimatorCheckBox.setText(QCoreApplication.translate("MainWindow", u"Decimator", None))
        self.acquisitionRecordingCheckBox.setText(QCoreApplication.translate("MainWindow", u"Recording", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("MainWindow", u"Input Parameters", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"Muscle", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"High Pass", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"Side", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"Select Channel", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"Sensor", None))
        self.inputMuscleComboBox.setItemText(0, QCoreApplication.translate("MainWindow", u"NOT_DEFINED", None))

        self.inputChannelComboBox.setItemText(0, QCoreApplication.translate("MainWindow", u"IN1-4", None))
        self.inputChannelComboBox.setItemText(1, QCoreApplication.translate("MainWindow", u"IN5-8", None))
        self.inputChannelComboBox.setItemText(2, QCoreApplication.translate("MainWindow", u"MULTIPLE_IN_1", None))
        self.inputChannelComboBox.setItemText(3, QCoreApplication.translate("MainWindow", u"MULTIPLE_IN_2", None))
        self.inputChannelComboBox.setItemText(4, QCoreApplication.translate("MainWindow", u"MULTIPLE_IN_3", None))
        self.inputChannelComboBox.setItemText(5, QCoreApplication.translate("MainWindow", u"MULTIPLE_IN_4", None))

        self.inputHighPassComboBox.setItemText(0, QCoreApplication.translate("MainWindow", u"0.7 Hz", None))
        self.inputHighPassComboBox.setItemText(1, QCoreApplication.translate("MainWindow", u"10 Hz", None))
        self.inputHighPassComboBox.setItemText(2, QCoreApplication.translate("MainWindow", u"100 Hz", None))
        self.inputHighPassComboBox.setItemText(3, QCoreApplication.translate("MainWindow", u"200 Hz", None))

        self.label_6.setText(QCoreApplication.translate("MainWindow", u"Adaptor", None))
        self.label_10.setText(QCoreApplication.translate("MainWindow", u"Mode", None))
        self.inputSideComboBox.setItemText(0, QCoreApplication.translate("MainWindow", u"NOT_DEFINED", None))
        self.inputSideComboBox.setItemText(1, QCoreApplication.translate("MainWindow", u"LEFT", None))
        self.inputSideComboBox.setItemText(2, QCoreApplication.translate("MainWindow", u"RIGHT", None))
        self.inputSideComboBox.setItemText(3, QCoreApplication.translate("MainWindow", u"NONE", None))

        self.inputAdaptorComboBox.setItemText(0, QCoreApplication.translate("MainWindow", u"NOT_DEFINED", None))

        self.inputDetectionModeComboBox.setItemText(0, QCoreApplication.translate("MainWindow", u"MONOPOLAR", None))
        self.inputDetectionModeComboBox.setItemText(1, QCoreApplication.translate("MainWindow", u"DIFFERENTIAL", None))
        self.inputDetectionModeComboBox.setItemText(2, QCoreApplication.translate("MainWindow", u"BIPOLAR", None))

        self.label_9.setText(QCoreApplication.translate("MainWindow", u"Low Pass", None))
        self.inputSensorComboBox.setItemText(0, QCoreApplication.translate("MainWindow", u"NOT_DEFINED", None))

        self.inputLowPassComboBox.setItemText(0, QCoreApplication.translate("MainWindow", u"130 Hz", None))
        self.inputLowPassComboBox.setItemText(1, QCoreApplication.translate("MainWindow", u"500 Hz", None))
        self.inputLowPassComboBox.setItemText(2, QCoreApplication.translate("MainWindow", u"900 Hz", None))
        self.inputLowPassComboBox.setItemText(3, QCoreApplication.translate("MainWindow", u"4400 Hz", None))

        self.comboBox_11.setItemText(0, QCoreApplication.translate("MainWindow", u"Test", None))
        self.comboBox_11.setItemText(1, QCoreApplication.translate("MainWindow", u"Neues Element", None))
        self.comboBox_11.setItemText(2, QCoreApplication.translate("MainWindow", u"Neues Element", None))
        self.comboBox_11.setItemText(3, QCoreApplication.translate("MainWindow", u"Neues Element", None))

        self.inputConfigurePushButton.setText(QCoreApplication.translate("MainWindow", u"Configure Input", None))
    # retranslateUi

