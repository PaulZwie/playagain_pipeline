# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'quattrocento_light_test.ui'
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

from gui_custom_elements.vispy_plot_widget import VispyPlotWidget

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(804, 666)
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


        self.gridLayout_2.addWidget(self.groupBox_2, 2, 0, 1, 1)

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


        self.gridLayout_2.addWidget(self.groupBox_4, 1, 0, 1, 1)

        self.groupBox = QGroupBox(self.centralwidget)
        self.groupBox.setObjectName(u"groupBox")
        self.gridLayout = QGridLayout(self.groupBox)
        self.gridLayout.setObjectName(u"gridLayout")
        self.acquisitionSamplingFrequencyComboBox = QComboBox(self.groupBox)
        self.acquisitionSamplingFrequencyComboBox.addItem("")
        self.acquisitionSamplingFrequencyComboBox.addItem("")
        self.acquisitionSamplingFrequencyComboBox.addItem("")
        self.acquisitionSamplingFrequencyComboBox.addItem("")
        self.acquisitionSamplingFrequencyComboBox.setObjectName(u"acquisitionSamplingFrequencyComboBox")

        self.gridLayout.addWidget(self.acquisitionSamplingFrequencyComboBox, 0, 1, 1, 1)

        self.acquisitionNumberOfChannelsComboBox = QComboBox(self.groupBox)
        self.acquisitionNumberOfChannelsComboBox.addItem("")
        self.acquisitionNumberOfChannelsComboBox.setObjectName(u"acquisitionNumberOfChannelsComboBox")

        self.gridLayout.addWidget(self.acquisitionNumberOfChannelsComboBox, 1, 1, 1, 1)

        self.label_2 = QLabel(self.groupBox)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)

        self.label = QLabel(self.groupBox)
        self.label.setObjectName(u"label")

        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)

        self.label_3 = QLabel(self.groupBox)
        self.label_3.setObjectName(u"label_3")

        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 1)

        self.acquisitionStreamingFrequencyComboBox = QComboBox(self.groupBox)
        self.acquisitionStreamingFrequencyComboBox.addItem("")
        self.acquisitionStreamingFrequencyComboBox.addItem("")
        self.acquisitionStreamingFrequencyComboBox.addItem("")
        self.acquisitionStreamingFrequencyComboBox.addItem("")
        self.acquisitionStreamingFrequencyComboBox.addItem("")
        self.acquisitionStreamingFrequencyComboBox.addItem("")
        self.acquisitionStreamingFrequencyComboBox.setObjectName(u"acquisitionStreamingFrequencyComboBox")

        self.gridLayout.addWidget(self.acquisitionStreamingFrequencyComboBox, 2, 1, 1, 1)


        self.gridLayout_2.addWidget(self.groupBox, 0, 0, 1, 1)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.gridLayout_2.addItem(self.verticalSpacer, 3, 0, 1, 1)

        self.vispyPlotWidget = VispyPlotWidget(self.centralwidget)
        self.vispyPlotWidget.setObjectName(u"vispyPlotWidget")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.vispyPlotWidget.sizePolicy().hasHeightForWidth())
        self.vispyPlotWidget.setSizePolicy(sizePolicy)

        self.gridLayout_2.addWidget(self.vispyPlotWidget, 0, 1, 4, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 804, 22))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        self.acquisitionSamplingFrequencyComboBox.setCurrentIndex(1)
        self.acquisitionNumberOfChannelsComboBox.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"QuattrocentoLightTest", None))
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
        self.acquisitionSamplingFrequencyComboBox.setItemText(0, QCoreApplication.translate("MainWindow", u"512", None))
        self.acquisitionSamplingFrequencyComboBox.setItemText(1, QCoreApplication.translate("MainWindow", u"2048", None))
        self.acquisitionSamplingFrequencyComboBox.setItemText(2, QCoreApplication.translate("MainWindow", u"5120", None))
        self.acquisitionSamplingFrequencyComboBox.setItemText(3, QCoreApplication.translate("MainWindow", u"10240", None))

        self.acquisitionNumberOfChannelsComboBox.setItemText(0, QCoreApplication.translate("MainWindow", u"408", None))

        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Number of Channels", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Sampling Frequency", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"Streaming Frequency", None))
        self.acquisitionStreamingFrequencyComboBox.setItemText(0, QCoreApplication.translate("MainWindow", u"32", None))
        self.acquisitionStreamingFrequencyComboBox.setItemText(1, QCoreApplication.translate("MainWindow", u"16", None))
        self.acquisitionStreamingFrequencyComboBox.setItemText(2, QCoreApplication.translate("MainWindow", u"8", None))
        self.acquisitionStreamingFrequencyComboBox.setItemText(3, QCoreApplication.translate("MainWindow", u"4", None))
        self.acquisitionStreamingFrequencyComboBox.setItemText(4, QCoreApplication.translate("MainWindow", u"2", None))
        self.acquisitionStreamingFrequencyComboBox.setItemText(5, QCoreApplication.translate("MainWindow", u"1", None))

    # retranslateUi

