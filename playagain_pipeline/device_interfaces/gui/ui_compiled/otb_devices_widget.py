# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'otb_devices_widget.ui'
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
from PySide6.QtWidgets import (QApplication, QComboBox, QGridLayout, QLabel,
    QSizePolicy, QStackedWidget, QWidget)

from device_interfaces.gui.template_widgets.muovi import MuoviWidget
from device_interfaces.gui.template_widgets.muovi_plus import MuoviPlusWidget
from device_interfaces.gui.template_widgets.quattrocento import QuattrocentoLightWidget

class Ui_OTBDeviceWidgetForm(object):
    def setupUi(self, OTBDeviceWidgetForm):
        if not OTBDeviceWidgetForm.objectName():
            OTBDeviceWidgetForm.setObjectName(u"OTBDeviceWidgetForm")
        OTBDeviceWidgetForm.resize(400, 300)
        self.gridLayout = QGridLayout(OTBDeviceWidgetForm)
        self.gridLayout.setObjectName(u"gridLayout")
        self.deviceSelectionComboBox = QComboBox(OTBDeviceWidgetForm)
        self.deviceSelectionComboBox.addItem("")
        self.deviceSelectionComboBox.addItem("")
        self.deviceSelectionComboBox.addItem("")
        self.deviceSelectionComboBox.setObjectName(u"deviceSelectionComboBox")

        self.gridLayout.addWidget(self.deviceSelectionComboBox, 0, 1, 1, 1)

        self.label = QLabel(OTBDeviceWidgetForm)
        self.label.setObjectName(u"label")

        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)

        self.deviceStackedWidget = QStackedWidget(OTBDeviceWidgetForm)
        self.deviceStackedWidget.setObjectName(u"deviceStackedWidget")
        self.quattrocentoLightWidget = QuattrocentoLightWidget()
        self.quattrocentoLightWidget.setObjectName(u"quattrocentoLightWidget")
        self.deviceStackedWidget.addWidget(self.quattrocentoLightWidget)
        self.muoviWidget = MuoviWidget()
        self.muoviWidget.setObjectName(u"muoviWidget")
        self.deviceStackedWidget.addWidget(self.muoviWidget)
        self.muoviPlusWidget = MuoviPlusWidget()
        self.muoviPlusWidget.setObjectName(u"muoviPlusWidget")
        self.deviceStackedWidget.addWidget(self.muoviPlusWidget)

        self.gridLayout.addWidget(self.deviceStackedWidget, 1, 0, 1, 2)


        self.retranslateUi(OTBDeviceWidgetForm)

        self.deviceStackedWidget.setCurrentIndex(2)


        QMetaObject.connectSlotsByName(OTBDeviceWidgetForm)
    # setupUi

    def retranslateUi(self, OTBDeviceWidgetForm):
        OTBDeviceWidgetForm.setWindowTitle(QCoreApplication.translate("OTBDeviceWidgetForm", u"Form", None))
        self.deviceSelectionComboBox.setItemText(0, QCoreApplication.translate("OTBDeviceWidgetForm", u"Quattrocento Light", None))
        self.deviceSelectionComboBox.setItemText(1, QCoreApplication.translate("OTBDeviceWidgetForm", u"Muovi", None))
        self.deviceSelectionComboBox.setItemText(2, QCoreApplication.translate("OTBDeviceWidgetForm", u"Muovi Plus", None))

        self.label.setText(QCoreApplication.translate("OTBDeviceWidgetForm", u"Device", None))
    # retranslateUi

