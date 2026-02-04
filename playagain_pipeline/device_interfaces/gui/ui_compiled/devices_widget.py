# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'devices_widget.ui'
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
from PySide6.QtWidgets import (QApplication, QComboBox, QGridLayout, QLabel,
    QSizePolicy, QStackedWidget, QWidget)

from device_interfaces.gui.template_widgets.intan import IntanRHDControllerWidget
from device_interfaces.gui.template_widgets.mindrove import MindRoveWidget
from device_interfaces.gui.template_widgets.muovi import MuoviWidget
from device_interfaces.gui.template_widgets.muovi_plus import MuoviPlusWidget
from device_interfaces.gui.template_widgets.noxon import NoxonWidget
from device_interfaces.gui.template_widgets.nswitch import NSwitchWidget
from device_interfaces.gui.template_widgets.quattrocento import (QuattrocentoLightWidget, QuattrocentoWidget)
from device_interfaces.gui.template_widgets.sessantaquattro import SessantaquattroWidget
from device_interfaces.gui.template_widgets.syncstation import SyncStationWidget

class Ui_DeviceWidgetForm(object):
    def setupUi(self, DeviceWidgetForm):
        if not DeviceWidgetForm.objectName():
            DeviceWidgetForm.setObjectName(u"DeviceWidgetForm")
        DeviceWidgetForm.resize(400, 300)
        self.gridLayout = QGridLayout(DeviceWidgetForm)
        self.gridLayout.setObjectName(u"gridLayout")
        self.label = QLabel(DeviceWidgetForm)
        self.label.setObjectName(u"label")

        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)

        self.deviceSelectionComboBox = QComboBox(DeviceWidgetForm)
        self.deviceSelectionComboBox.addItem("")
        self.deviceSelectionComboBox.addItem("")
        self.deviceSelectionComboBox.addItem("")
        self.deviceSelectionComboBox.addItem("")
        self.deviceSelectionComboBox.addItem("")
        self.deviceSelectionComboBox.addItem("")
        self.deviceSelectionComboBox.addItem("")
        self.deviceSelectionComboBox.addItem("")
        self.deviceSelectionComboBox.addItem("")
        self.deviceSelectionComboBox.addItem("")
        self.deviceSelectionComboBox.setObjectName(u"deviceSelectionComboBox")

        self.gridLayout.addWidget(self.deviceSelectionComboBox, 0, 1, 1, 1)

        self.deviceStackedWidget = QStackedWidget(DeviceWidgetForm)
        self.deviceStackedWidget.setObjectName(u"deviceStackedWidget")
        self.quattrocentoWidget = QuattrocentoWidget()
        self.quattrocentoWidget.setObjectName(u"quattrocentoWidget")
        self.deviceStackedWidget.addWidget(self.quattrocentoWidget)
        self.quattrocentoLightWidget = QuattrocentoLightWidget()
        self.quattrocentoLightWidget.setObjectName(u"quattrocentoLightWidget")
        self.deviceStackedWidget.addWidget(self.quattrocentoLightWidget)
        self.sessantaquattroLightWidget = SessantaquattroWidget()
        self.sessantaquattroLightWidget.setObjectName(u"sessantaquattroLightWidget")
        self.deviceStackedWidget.addWidget(self.sessantaquattroLightWidget)
        self.intanRHDControllerWidget = IntanRHDControllerWidget()
        self.intanRHDControllerWidget.setObjectName(u"intanRHDControllerWidget")
        self.deviceStackedWidget.addWidget(self.intanRHDControllerWidget)
        self.noxonWidget = NoxonWidget()
        self.noxonWidget.setObjectName(u"noxonWidget")
        self.deviceStackedWidget.addWidget(self.noxonWidget)
        self.muoviWidget = MuoviWidget()
        self.muoviWidget.setObjectName(u"muoviWidget")
        self.deviceStackedWidget.addWidget(self.muoviWidget)
        self.muoviPlusWidget = MuoviPlusWidget()
        self.muoviPlusWidget.setObjectName(u"muoviPlusWidget")
        self.deviceStackedWidget.addWidget(self.muoviPlusWidget)
        self.syncStationWidget = SyncStationWidget()
        self.syncStationWidget.setObjectName(u"syncStationWidget")
        self.deviceStackedWidget.addWidget(self.syncStationWidget)
        self.mindRoveWidget = MindRoveWidget()
        self.mindRoveWidget.setObjectName(u"mindRoveWidget")
        self.deviceStackedWidget.addWidget(self.mindRoveWidget)
        self.nswitchWidget = NSwitchWidget()
        self.nswitchWidget.setObjectName(u"nswitchWidget")
        self.deviceStackedWidget.addWidget(self.nswitchWidget)

        self.gridLayout.addWidget(self.deviceStackedWidget, 1, 0, 1, 2)


        self.retranslateUi(DeviceWidgetForm)

        self.deviceStackedWidget.setCurrentIndex(9)


        QMetaObject.connectSlotsByName(DeviceWidgetForm)
    # setupUi

    def retranslateUi(self, DeviceWidgetForm):
        DeviceWidgetForm.setWindowTitle(QCoreApplication.translate("DeviceWidgetForm", u"Form", None))
        self.label.setText(QCoreApplication.translate("DeviceWidgetForm", u"Device", None))
        self.deviceSelectionComboBox.setItemText(0, QCoreApplication.translate("DeviceWidgetForm", u"Quattrocento", None))
        self.deviceSelectionComboBox.setItemText(1, QCoreApplication.translate("DeviceWidgetForm", u"Quattrocento Light", None))
        self.deviceSelectionComboBox.setItemText(2, QCoreApplication.translate("DeviceWidgetForm", u"Sessantaquattro", None))
        self.deviceSelectionComboBox.setItemText(3, QCoreApplication.translate("DeviceWidgetForm", u"Intan RHD Controller", None))
        self.deviceSelectionComboBox.setItemText(4, QCoreApplication.translate("DeviceWidgetForm", u"Noxon", None))
        self.deviceSelectionComboBox.setItemText(5, QCoreApplication.translate("DeviceWidgetForm", u"Muovi", None))
        self.deviceSelectionComboBox.setItemText(6, QCoreApplication.translate("DeviceWidgetForm", u"Muovi Plus", None))
        self.deviceSelectionComboBox.setItemText(7, QCoreApplication.translate("DeviceWidgetForm", u"SyncStation (Muovi)", None))
        self.deviceSelectionComboBox.setItemText(8, QCoreApplication.translate("DeviceWidgetForm", u"MindRove", None))
        self.deviceSelectionComboBox.setItemText(9, QCoreApplication.translate("DeviceWidgetForm", u"N-Switch Bracelet", None))

    # retranslateUi

