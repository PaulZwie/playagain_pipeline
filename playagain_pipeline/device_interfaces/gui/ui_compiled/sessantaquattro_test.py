# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'sessantaquattro_test.ui'
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
from PySide6.QtWidgets import (QApplication, QGridLayout, QMainWindow, QMenuBar,
    QPushButton, QSizePolicy, QStatusBar, QWidget)

from gui_custom_elements.vispy_plot_widget import VispyPlotWidget

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.configurePushButton = QPushButton(self.centralwidget)
        self.configurePushButton.setObjectName(u"configurePushButton")
        self.configurePushButton.setCheckable(False)

        self.gridLayout.addWidget(self.configurePushButton, 0, 1, 1, 1)

        self.connectDevicePushButton = QPushButton(self.centralwidget)
        self.connectDevicePushButton.setObjectName(u"connectDevicePushButton")
        self.connectDevicePushButton.setCheckable(True)

        self.gridLayout.addWidget(self.connectDevicePushButton, 0, 0, 1, 1)

        self.requestConfigurationPushButton = QPushButton(self.centralwidget)
        self.requestConfigurationPushButton.setObjectName(u"requestConfigurationPushButton")
        self.requestConfigurationPushButton.setCheckable(False)

        self.gridLayout.addWidget(self.requestConfigurationPushButton, 1, 1, 1, 1)

        self.resetConfigurationPushButton = QPushButton(self.centralwidget)
        self.resetConfigurationPushButton.setObjectName(u"resetConfigurationPushButton")

        self.gridLayout.addWidget(self.resetConfigurationPushButton, 2, 1, 1, 1)

        self.streamPushButton = QPushButton(self.centralwidget)
        self.streamPushButton.setObjectName(u"streamPushButton")
        self.streamPushButton.setCheckable(True)

        self.gridLayout.addWidget(self.streamPushButton, 0, 2, 1, 1)

        self.vispyPlotWidget = VispyPlotWidget(self.centralwidget)
        self.vispyPlotWidget.setObjectName(u"vispyPlotWidget")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.vispyPlotWidget.sizePolicy().hasHeightForWidth())
        self.vispyPlotWidget.setSizePolicy(sizePolicy)

        self.gridLayout.addWidget(self.vispyPlotWidget, 3, 0, 1, 3)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 800, 22))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"SessantaquattroTestInterface", None))
        self.configurePushButton.setText(QCoreApplication.translate("MainWindow", u"Configure", None))
        self.connectDevicePushButton.setText(QCoreApplication.translate("MainWindow", u"Connect", None))
        self.requestConfigurationPushButton.setText(QCoreApplication.translate("MainWindow", u"Request", None))
        self.resetConfigurationPushButton.setText(QCoreApplication.translate("MainWindow", u"Reset", None))
        self.streamPushButton.setText(QCoreApplication.translate("MainWindow", u"Stream", None))
    # retranslateUi

