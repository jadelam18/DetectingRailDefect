# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'defect-detection.ui'
#
# Created by: PyQt5 UI code generator 5.15.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(876, 650)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.btnReadImage = QtWidgets.QPushButton(self.centralwidget)
        self.btnReadImage.setGeometry(QtCore.QRect(10, 530, 75, 23))
        self.btnReadImage.setObjectName("btnReadImage")
        self.btnEnhance = QtWidgets.QPushButton(self.centralwidget)
        self.btnEnhance.setGeometry(QtCore.QRect(120, 530, 75, 23))
        self.btnEnhance.setObjectName("btnEnhance")
        self.btnFilter = QtWidgets.QPushButton(self.centralwidget)
        self.btnFilter.setGeometry(QtCore.QRect(210, 530, 75, 23))
        self.btnFilter.setObjectName("btnFilter")
        self.btnThresh = QtWidgets.QPushButton(self.centralwidget)
        self.btnThresh.setGeometry(QtCore.QRect(340, 530, 75, 23))
        self.btnThresh.setObjectName("btnThresh")
        self.btnOpen = QtWidgets.QPushButton(self.centralwidget)
        self.btnOpen.setGeometry(QtCore.QRect(450, 530, 75, 23))
        self.btnOpen.setObjectName("btnOpen")
        self.btnDelnDil = QtWidgets.QPushButton(self.centralwidget)
        self.btnDelnDil.setGeometry(QtCore.QRect(530, 530, 91, 23))
        self.btnDelnDil.setObjectName("btnDelnDil")
        self.btnFeature = QtWidgets.QPushButton(self.centralwidget)
        self.btnFeature.setGeometry(QtCore.QRect(640, 530, 75, 23))
        self.btnFeature.setObjectName("btnFeature")
        self.labelReadImage = QtWidgets.QLabel(self.centralwidget)
        self.labelReadImage.setGeometry(QtCore.QRect(10, 0, 80, 520))
        self.labelReadImage.setObjectName("labelReadImage")
        self.labelEnhance = QtWidgets.QLabel(self.centralwidget)
        self.labelEnhance.setGeometry(QtCore.QRect(120, 0, 80, 520))
        self.labelEnhance.setObjectName("labelEnhance")
        self.labelFilter = QtWidgets.QLabel(self.centralwidget)
        self.labelFilter.setGeometry(QtCore.QRect(210, 0, 80, 520))
        self.labelFilter.setObjectName("labelFilter")
        self.labelThresh = QtWidgets.QLabel(self.centralwidget)
        self.labelThresh.setGeometry(QtCore.QRect(340, 0, 80, 520))
        self.labelThresh.setObjectName("labelThresh")
        self.labelOpen = QtWidgets.QLabel(self.centralwidget)
        self.labelOpen.setGeometry(QtCore.QRect(450, 0, 80, 520))
        self.labelOpen.setObjectName("labelOpen")
        self.labelDelnDil = QtWidgets.QLabel(self.centralwidget)
        self.labelDelnDil.setGeometry(QtCore.QRect(540, 0, 80, 520))
        self.labelDelnDil.setObjectName("labelDelnDil")
        self.labelFeature = QtWidgets.QLabel(self.centralwidget)
        self.labelFeature.setGeometry(QtCore.QRect(640, 0, 80, 520))
        self.labelFeature.setObjectName("labelFeature")
        self.textFeature = QtWidgets.QTextBrowser(self.centralwidget)
        self.textFeature.setGeometry(QtCore.QRect(730, 140, 131, 281))
        self.textFeature.setObjectName("textFeature")
        self.btnResult = QtWidgets.QPushButton(self.centralwidget)
        self.btnResult.setGeometry(QtCore.QRect(760, 450, 81, 51))
        self.btnResult.setObjectName("btnResult")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(140, 550, 141, 51))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setLineWidth(1)
        self.label.setObjectName("label")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(100, 0, 20, 601))
        self.line.setFrameShadow(QtWidgets.QFrame.Plain)
        self.line.setLineWidth(3)
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setObjectName("line")
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setGeometry(QtCore.QRect(300, 0, 20, 601))
        self.line_2.setFrameShadow(QtWidgets.QFrame.Plain)
        self.line_2.setLineWidth(3)
        self.line_2.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_2.setObjectName("line_2")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(320, 550, 121, 51))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setLineWidth(1)
        self.label_2.setObjectName("label_2")
        self.line_3 = QtWidgets.QFrame(self.centralwidget)
        self.line_3.setGeometry(QtCore.QRect(430, 0, 20, 601))
        self.line_3.setFrameShadow(QtWidgets.QFrame.Plain)
        self.line_3.setLineWidth(3)
        self.line_3.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_3.setObjectName("line_3")
        self.line_4 = QtWidgets.QFrame(self.centralwidget)
        self.line_4.setGeometry(QtCore.QRect(710, 0, 20, 601))
        self.line_4.setFrameShadow(QtWidgets.QFrame.Plain)
        self.line_4.setLineWidth(3)
        self.line_4.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_4.setObjectName("line_4")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(480, 550, 111, 51))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setLineWidth(1)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(750, 60, 101, 51))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setLineWidth(1)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(30, 550, 51, 51))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setLineWidth(1)
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(640, 550, 81, 61))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setLineWidth(1)
        self.label_6.setWordWrap(True)
        self.label_6.setObjectName("label_6")
        self.line_5 = QtWidgets.QFrame(self.centralwidget)
        self.line_5.setGeometry(QtCore.QRect(620, 0, 20, 601))
        self.line_5.setFrameShadow(QtWidgets.QFrame.Plain)
        self.line_5.setLineWidth(3)
        self.line_5.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_5.setObjectName("line_5")
        self.btnReset = QtWidgets.QPushButton(self.centralwidget)
        self.btnReset.setGeometry(QtCore.QRect(754, 532, 91, 51))
        self.btnReset.setObjectName("btnReset")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 876, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.btnReadImage.clicked.connect(MainWindow.btnReadImage_Clicked)
        self.btnEnhance.clicked.connect(MainWindow.btnEnhance_Clicked)
        self.btnFilter.clicked.connect(MainWindow.btnFilter_Clicked)
        self.btnThresh.clicked.connect(MainWindow.btnThresh_Clicked)
        self.btnOpen.clicked.connect(MainWindow.btnOpen_Clicked)
        self.btnDelnDil.clicked.connect(MainWindow.btnDelnDil_Clicked)
        self.btnFeature.clicked.connect(MainWindow.btnFeature_Clicked)
        self.btnResult.clicked.connect(MainWindow.btnResult_Clicked)
        self.btnReset.clicked.connect(MainWindow.btnReset_Clicked)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.btnReadImage.setText(_translate("MainWindow", "打开照片"))
        self.btnEnhance.setText(_translate("MainWindow", "图像增强"))
        self.btnFilter.setText(_translate("MainWindow", "图像去噪"))
        self.btnThresh.setText(_translate("MainWindow", "目标分割"))
        self.btnOpen.setText(_translate("MainWindow", "开运算"))
        self.btnDelnDil.setText(_translate("MainWindow", "删小区域+膨胀"))
        self.btnFeature.setText(_translate("MainWindow", "特征提取"))
        self.labelReadImage.setText(_translate("MainWindow", "原图"))
        self.labelEnhance.setText(_translate("MainWindow", "图像增强"))
        self.labelFilter.setText(_translate("MainWindow", "图像去噪"))
        self.labelThresh.setText(_translate("MainWindow", "目标分割"))
        self.labelOpen.setText(_translate("MainWindow", "开运算"))
        self.labelDelnDil.setText(_translate("MainWindow", "删小区域+膨胀"))
        self.labelFeature.setText(_translate("MainWindow", "缺陷特征"))
        self.btnResult.setText(_translate("MainWindow", "检测结果"))
        self.label.setText(_translate("MainWindow", "钢轨图像预处理"))
        self.label_2.setText(_translate("MainWindow", "钢轨缺陷分割"))
        self.label_3.setText(_translate("MainWindow", "形态学处理"))
        self.label_4.setText(_translate("MainWindow", "检测结果"))
        self.label_5.setText(_translate("MainWindow", "原图"))
        self.label_6.setText(_translate("MainWindow", "缺陷特征提取"))
        self.btnReset.setText(_translate("MainWindow", "清空所有图片"))
