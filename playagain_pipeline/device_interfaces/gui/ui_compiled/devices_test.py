# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'devices_test.ui'
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
from PySide6.QtWidgets import (QApplication, QCheckBox, QGridLayout, QMainWindow,
    QMenuBar, QSizePolicy, QSpacerItem, QStatusBar,
    QWidget)

from device_interfaces.gui.template_widgets.devices import DeviceWidget
from gui_custom_elements.vispy_plot_widget import VispyPlotWidget

class Ui_DevicesTest(object):
    def setupUi(self, DevicesTest):
        if not DevicesTest.objectName():
            DevicesTest.setObjectName(u"DevicesTest")
        DevicesTest.resize(800, 600)
        self.centralwidget = QWidget(DevicesTest)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.deviceWidget = DeviceWidget(self.centralwidget)
        self.deviceWidget.setObjectName(u"deviceWidget")

        self.gridLayout.addWidget(self.deviceWidget, 0, 0, 1, 1)

        self.vispyPlotWidget = VispyPlotWidget(self.centralwidget)
        self.vispyPlotWidget.setObjectName(u"vispyPlotWidget")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.vispyPlotWidget.sizePolicy().hasHeightForWidth())
        self.vispyPlotWidget.setSizePolicy(sizePolicy)
        self.gridLayout_2 = QGridLayout(self.vispyPlotWidget)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.vispyPlotToggleCheckBox = QCheckBox(self.vispyPlotWidget)
        self.vispyPlotToggleCheckBox.setObjectName(u"vispyPlotToggleCheckBox")
        self.vispyPlotToggleCheckBox.setChecked(True)

        self.gridLayout_2.addWidget(self.vispyPlotToggleCheckBox, 0, 2, 1, 1)

        self.horizontalSpacer = QSpacerItem(653, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_2.addItem(self.horizontalSpacer, 0, 1, 1, 1)

        self.verticalSpacer = QSpacerItem(20, 478, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.gridLayout_2.addItem(self.verticalSpacer, 1, 2, 1, 1)


        self.gridLayout.addWidget(self.vispyPlotWidget, 0, 1, 1, 1)

        DevicesTest.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(DevicesTest)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 800, 33))
        DevicesTest.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(DevicesTest)
        self.statusbar.setObjectName(u"statusbar")
        DevicesTest.setStatusBar(self.statusbar)

        self.retranslateUi(DevicesTest)

        QMetaObject.connectSlotsByName(DevicesTest)
    # setupUi

    def retranslateUi(self, DevicesTest):
        DevicesTest.setWindowTitle(QCoreApplication.translate("DevicesTest", u"Devices Test", None))
        self.vispyPlotToggleCheckBox.setText(QCoreApplication.translate("DevicesTest", u"Toggle Plot", None))
    # retranslateUi

