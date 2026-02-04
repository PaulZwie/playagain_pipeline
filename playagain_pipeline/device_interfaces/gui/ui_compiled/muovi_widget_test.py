# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'muovi_widget_test.ui'
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
    QSizePolicy, QStatusBar, QWidget)

from device_interfaces.gui.template_widgets.devices import DeviceWidget
from gui_custom_elements.vispy_plot_widget import VispyPlotWidget

class Ui_MuoviTest(object):
    def setupUi(self, MuoviTest):
        if not MuoviTest.objectName():
            MuoviTest.setObjectName(u"MuoviTest")
        MuoviTest.resize(800, 600)
        self.centralwidget = QWidget(MuoviTest)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.deviceInterfaceWidget = DeviceWidget(self.centralwidget)
        self.deviceInterfaceWidget.setObjectName(u"deviceInterfaceWidget")

        self.gridLayout.addWidget(self.deviceInterfaceWidget, 0, 0, 1, 1)

        self.vispyPlotWidget = VispyPlotWidget(self.centralwidget)
        self.vispyPlotWidget.setObjectName(u"vispyPlotWidget")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.vispyPlotWidget.sizePolicy().hasHeightForWidth())
        self.vispyPlotWidget.setSizePolicy(sizePolicy)

        self.gridLayout.addWidget(self.vispyPlotWidget, 0, 1, 1, 1)

        MuoviTest.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MuoviTest)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 800, 22))
        MuoviTest.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MuoviTest)
        self.statusbar.setObjectName(u"statusbar")
        MuoviTest.setStatusBar(self.statusbar)

        self.retranslateUi(MuoviTest)

        QMetaObject.connectSlotsByName(MuoviTest)
    # setupUi

    def retranslateUi(self, MuoviTest):
        MuoviTest.setWindowTitle(QCoreApplication.translate("MuoviTest", u"MuoviTest", None))
    # retranslateUi

