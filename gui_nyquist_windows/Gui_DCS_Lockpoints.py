# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Gui_DCS_Lockpoints.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(805, 697)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.verticalLayout_17 = QtWidgets.QVBoxLayout()
        self.verticalLayout_17.setObjectName("verticalLayout_17")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setMaximumSize(QtCore.QSize(239, 103))
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setObjectName("label")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label)
        self.le_min_wl = QtWidgets.QLineEdit(self.groupBox)
        self.le_min_wl.setObjectName("le_min_wl")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.le_min_wl)
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.le_max_wl = QtWidgets.QLineEdit(self.groupBox)
        self.le_max_wl.setObjectName("le_max_wl")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.le_max_wl)
        self.verticalLayout_5.addLayout(self.formLayout)
        self.verticalLayout_17.addWidget(self.groupBox)
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setMaximumSize(QtCore.QSize(226, 103))
        self.groupBox_2.setObjectName("groupBox_2")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.groupBox_2)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.formLayout_2 = QtWidgets.QFormLayout()
        self.formLayout_2.setObjectName("formLayout_2")
        self.label_9 = QtWidgets.QLabel(self.groupBox_2)
        self.label_9.setObjectName("label_9")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_9)
        self.le_f01 = QtWidgets.QLineEdit(self.groupBox_2)
        self.le_f01.setObjectName("le_f01")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.le_f01)
        self.label_10 = QtWidgets.QLabel(self.groupBox_2)
        self.label_10.setObjectName("label_10")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_10)
        self.le_f02 = QtWidgets.QLineEdit(self.groupBox_2)
        self.le_f02.setObjectName("le_f02")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.le_f02)
        self.verticalLayout_6.addLayout(self.formLayout_2)
        self.verticalLayout_17.addWidget(self.groupBox_2)
        self.groupBox1 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox1.setMaximumSize(QtCore.QSize(254, 126))
        self.groupBox1.setObjectName("groupBox1")
        self.verticalLayout_16 = QtWidgets.QVBoxLayout(self.groupBox1)
        self.verticalLayout_16.setObjectName("verticalLayout_16")
        self.formLayout_3 = QtWidgets.QFormLayout()
        self.formLayout_3.setObjectName("formLayout_3")
        self.label_3 = QtWidgets.QLabel(self.groupBox1)
        self.label_3.setObjectName("label_3")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.lcd = QtWidgets.QLCDNumber(self.groupBox1)
        self.lcd.setMinimumSize(QtCore.QSize(64, 31))
        self.lcd.setObjectName("lcd")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.lcd)
        self.label_15 = QtWidgets.QLabel(self.groupBox1)
        self.label_15.setObjectName("label_15")
        self.formLayout_3.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_15)
        self.le_dfrep = QtWidgets.QLineEdit(self.groupBox1)
        self.le_dfrep.setObjectName("le_dfrep")
        self.formLayout_3.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.le_dfrep)
        self.label_4 = QtWidgets.QLabel(self.groupBox1)
        self.label_4.setObjectName("label_4")
        self.formLayout_3.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_4)
        self.le_rep_rate = QtWidgets.QLineEdit(self.groupBox1)
        self.le_rep_rate.setObjectName("le_rep_rate")
        self.formLayout_3.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.le_rep_rate)
        self.verticalLayout_16.addLayout(self.formLayout_3)
        self.verticalLayout_17.addWidget(self.groupBox1)
        self.gridLayout_2.addLayout(self.verticalLayout_17, 0, 0, 1, 1)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.verticalLayout_15 = QtWidgets.QVBoxLayout()
        self.verticalLayout_15.setObjectName("verticalLayout_15")
        self.groupBox2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox2.setObjectName("groupBox2")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox2)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout_12 = QtWidgets.QVBoxLayout()
        self.verticalLayout_12.setObjectName("verticalLayout_12")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.gv = PlotWidget(self.groupBox2)
        self.gv.setObjectName("gv")
        self.verticalLayout_2.addWidget(self.gv)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.le_xmin = QtWidgets.QLineEdit(self.groupBox2)
        self.le_xmin.setMaximumSize(QtCore.QSize(91, 16777215))
        self.le_xmin.setObjectName("le_xmin")
        self.horizontalLayout.addWidget(self.le_xmin)
        self.label_8 = QtWidgets.QLabel(self.groupBox2)
        self.label_8.setObjectName("label_8")
        self.horizontalLayout.addWidget(self.label_8)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.label_7 = QtWidgets.QLabel(self.groupBox2)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout.addWidget(self.label_7)
        self.le_xmax = QtWidgets.QLineEdit(self.groupBox2)
        self.le_xmax.setMaximumSize(QtCore.QSize(91, 16777215))
        self.le_xmax.setObjectName("le_xmax")
        self.horizontalLayout.addWidget(self.le_xmax)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.verticalLayout_12.addLayout(self.verticalLayout_2)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem1)
        self.label_17 = QtWidgets.QLabel(self.groupBox2)
        self.label_17.setObjectName("label_17")
        self.horizontalLayout_4.addWidget(self.label_17)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem2)
        self.verticalLayout_12.addLayout(self.horizontalLayout_4)
        self.gridLayout.addLayout(self.verticalLayout_12, 0, 1, 1, 1)
        self.frame = QtWidgets.QFrame(self.groupBox2)
        self.frame.setMaximumSize(QtCore.QSize(97, 16777215))
        self.frame.setObjectName("frame")
        self.verticalLayout_14 = QtWidgets.QVBoxLayout(self.frame)
        self.verticalLayout_14.setObjectName("verticalLayout_14")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_5 = QtWidgets.QLabel(self.frame)
        self.label_5.setObjectName("label_5")
        self.verticalLayout.addWidget(self.label_5)
        self.le_ymax = QtWidgets.QLineEdit(self.frame)
        self.le_ymax.setMaximumSize(QtCore.QSize(91, 16777215))
        self.le_ymax.setObjectName("le_ymax")
        self.verticalLayout.addWidget(self.le_ymax)
        self.verticalLayout_4.addLayout(self.verticalLayout)
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_4.addItem(spacerItem3)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.le_ymin = QtWidgets.QLineEdit(self.frame)
        self.le_ymin.setMaximumSize(QtCore.QSize(91, 16777215))
        self.le_ymin.setObjectName("le_ymin")
        self.verticalLayout_3.addWidget(self.le_ymin)
        self.label_6 = QtWidgets.QLabel(self.frame)
        self.label_6.setObjectName("label_6")
        self.verticalLayout_3.addWidget(self.label_6)
        self.verticalLayout_4.addLayout(self.verticalLayout_3)
        self.verticalLayout_14.addLayout(self.verticalLayout_4)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout_14.addItem(spacerItem4)
        self.gridLayout.addWidget(self.frame, 0, 0, 1, 1)
        self.verticalLayout_15.addWidget(self.groupBox2)
        self.groupBox3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox3.setObjectName("groupBox3")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.groupBox3)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.verticalLayout_11 = QtWidgets.QVBoxLayout()
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.verticalLayout_10 = QtWidgets.QVBoxLayout()
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.gv_2 = PlotWidget(self.groupBox3)
        self.gv_2.setObjectName("gv_2")
        self.verticalLayout_10.addWidget(self.gv_2)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.le_xmin_2 = QtWidgets.QLineEdit(self.groupBox3)
        self.le_xmin_2.setMaximumSize(QtCore.QSize(91, 16777215))
        self.le_xmin_2.setObjectName("le_xmin_2")
        self.horizontalLayout_2.addWidget(self.le_xmin_2)
        self.label_13 = QtWidgets.QLabel(self.groupBox3)
        self.label_13.setObjectName("label_13")
        self.horizontalLayout_2.addWidget(self.label_13)
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem5)
        self.label_14 = QtWidgets.QLabel(self.groupBox3)
        self.label_14.setObjectName("label_14")
        self.horizontalLayout_2.addWidget(self.label_14)
        self.le_xmax_2 = QtWidgets.QLineEdit(self.groupBox3)
        self.le_xmax_2.setMaximumSize(QtCore.QSize(91, 16777215))
        self.le_xmax_2.setObjectName("le_xmax_2")
        self.horizontalLayout_2.addWidget(self.le_xmax_2)
        self.verticalLayout_10.addLayout(self.horizontalLayout_2)
        self.verticalLayout_11.addLayout(self.verticalLayout_10)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem6)
        self.label_16 = QtWidgets.QLabel(self.groupBox3)
        self.label_16.setObjectName("label_16")
        self.horizontalLayout_3.addWidget(self.label_16)
        spacerItem7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem7)
        self.verticalLayout_11.addLayout(self.horizontalLayout_3)
        self.gridLayout_4.addLayout(self.verticalLayout_11, 0, 1, 1, 1)
        self.frame1 = QtWidgets.QFrame(self.groupBox3)
        self.frame1.setMaximumSize(QtCore.QSize(97, 16777215))
        self.frame1.setObjectName("frame1")
        self.verticalLayout_13 = QtWidgets.QVBoxLayout(self.frame1)
        self.verticalLayout_13.setObjectName("verticalLayout_13")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout()
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout()
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.label_11 = QtWidgets.QLabel(self.frame1)
        self.label_11.setObjectName("label_11")
        self.verticalLayout_8.addWidget(self.label_11)
        self.le_ymax_2 = QtWidgets.QLineEdit(self.frame1)
        self.le_ymax_2.setMaximumSize(QtCore.QSize(91, 16777215))
        self.le_ymax_2.setObjectName("le_ymax_2")
        self.verticalLayout_8.addWidget(self.le_ymax_2)
        self.verticalLayout_7.addLayout(self.verticalLayout_8)
        spacerItem8 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_7.addItem(spacerItem8)
        self.verticalLayout_9 = QtWidgets.QVBoxLayout()
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.le_ymin_2 = QtWidgets.QLineEdit(self.frame1)
        self.le_ymin_2.setMaximumSize(QtCore.QSize(91, 16777215))
        self.le_ymin_2.setObjectName("le_ymin_2")
        self.verticalLayout_9.addWidget(self.le_ymin_2)
        self.label_12 = QtWidgets.QLabel(self.frame1)
        self.label_12.setObjectName("label_12")
        self.verticalLayout_9.addWidget(self.label_12)
        self.verticalLayout_7.addLayout(self.verticalLayout_9)
        self.verticalLayout_13.addLayout(self.verticalLayout_7)
        spacerItem9 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout_13.addItem(spacerItem9)
        self.gridLayout_4.addWidget(self.frame1, 0, 0, 1, 1)
        self.verticalLayout_15.addWidget(self.groupBox3)
        self.horizontalLayout_5.addLayout(self.verticalLayout_15)
        self.verticalScrollBar = QtWidgets.QScrollBar(self.centralwidget)
        self.verticalScrollBar.setMinimumSize(QtCore.QSize(31, 0))
        self.verticalScrollBar.setOrientation(QtCore.Qt.Vertical)
        self.verticalScrollBar.setObjectName("verticalScrollBar")
        self.horizontalLayout_5.addWidget(self.verticalScrollBar)
        self.gridLayout_2.addLayout(self.horizontalLayout_5, 0, 1, 2, 1)
        spacerItem10 = QtWidgets.QSpacerItem(20, 258, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_2.addItem(spacerItem10, 1, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 805, 24))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "Laser Spectrum Overlap Wavelength Range"))
        self.label.setText(_translate("MainWindow", "min wl (um)"))
        self.le_min_wl.setText(_translate("MainWindow", "2.9"))
        self.label_2.setText(_translate("MainWindow", "max wl (um)"))
        self.le_max_wl.setText(_translate("MainWindow", "3.8"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Laser Offset Frequencies"))
        self.label_9.setText(_translate("MainWindow", "f01 (MHz)"))
        self.le_f01.setText(_translate("MainWindow", "0"))
        self.label_10.setText(_translate("MainWindow", "f02 (MHz)"))
        self.le_f02.setText(_translate("MainWindow", "0"))
        self.label_3.setText(_translate("MainWindow", "Delta frep (Hz)"))
        self.label_15.setText(_translate("MainWindow", "Delta frep (Hz)"))
        self.label_4.setText(_translate("MainWindow", "rep rate (MHz)"))
        self.le_rep_rate.setText(_translate("MainWindow", "1000"))
        self.groupBox2.setTitle(_translate("MainWindow", "Optical Domain"))
        self.label_8.setText(_translate("MainWindow", "x min"))
        self.label_7.setText(_translate("MainWindow", "x max"))
        self.label_17.setText(_translate("MainWindow", "THz"))
        self.label_5.setText(_translate("MainWindow", "y max"))
        self.label_6.setText(_translate("MainWindow", "y min"))
        self.groupBox3.setTitle(_translate("MainWindow", "RF Domain"))
        self.le_xmin_2.setText(_translate("MainWindow", "0"))
        self.label_13.setText(_translate("MainWindow", "x min"))
        self.label_14.setText(_translate("MainWindow", "x max"))
        self.le_xmax_2.setText(_translate("MainWindow", "500"))
        self.label_16.setText(_translate("MainWindow", "MHz"))
        self.label_11.setText(_translate("MainWindow", "y max"))
        self.label_12.setText(_translate("MainWindow", "y min"))
from PlotWidgets import PlotWidget
