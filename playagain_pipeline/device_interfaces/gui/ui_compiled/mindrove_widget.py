# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'mindrove_widget.ui'
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
from PySide6.QtWidgets import (QApplication, QGridLayout, QGroupBox, QPushButton,
    QSizePolicy, QSpacerItem, QWidget)

class Ui_MindRoveForm(object):
    def setupUi(self, MindRoveForm):
        if not MindRoveForm.objectName():
            MindRoveForm.setObjectName(u"MindRoveForm")
        MindRoveForm.resize(400, 144)
        self.gridLayout = QGridLayout(MindRoveForm)
        self.gridLayout.setObjectName(u"gridLayout")
        self.commandsGroupBox = QGroupBox(MindRoveForm)
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


        self.gridLayout.addWidget(self.commandsGroupBox, 0, 0, 1, 1)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.gridLayout.addItem(self.verticalSpacer, 1, 0, 1, 1)


        self.retranslateUi(MindRoveForm)

        QMetaObject.connectSlotsByName(MindRoveForm)
    # setupUi

    def retranslateUi(self, MindRoveForm):
        MindRoveForm.setWindowTitle(QCoreApplication.translate("MindRoveForm", u"MindRove", None))
        self.commandsGroupBox.setTitle(QCoreApplication.translate("MindRoveForm", u"Commands", None))
        self.commandConnectionPushButton.setText(QCoreApplication.translate("MindRoveForm", u"Connect", None))
        self.commandConfigurationPushButton.setText(QCoreApplication.translate("MindRoveForm", u"Configure", None))
        self.commandStreamPushButton.setText(QCoreApplication.translate("MindRoveForm", u"Stream", None))
    # retranslateUi

