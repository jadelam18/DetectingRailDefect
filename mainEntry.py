import sys
import cv2

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QFileDialog, QMainWindow

from mainForm import Ui_MainWindow

from pylab import *
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\SimSun.ttc", size=14)


class PyQtMainEntry(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

    def btnReadImage_Clicked(self):
        # 打开文件选取对话框
        filename,  _ = QFileDialog.getOpenFileName(self, '打开图片')
        if filename:
            self.img = cv2.imread(str(filename), 0)

            rows, columns = self.img.shape
            bytesPerLine = columns

            QImg = QImage(self.img.data, columns, rows, bytesPerLine, QImage.Format_Indexed8)
            self.labelReadImage.setPixmap(QPixmap.fromImage(QImg).scaled(
                self.labelReadImage.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))   # 显示图片在label上

    def btnEnhance_Clicked(self):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
        self.cl1 = clahe.apply(self.img)    # 以CLAHE（对比度受限的自适应直方图均衡化）增强图片

        rows, columns = self.cl1.shape
        bytesPerLine = columns

        QImg = QImage(self.cl1.data, columns, rows, bytesPerLine, QImage.Format_Indexed8)
        self.labelEnhance.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelEnhance.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnFilter_Clicked(self):
        self.bilf = cv2.bilateralFilter(self.cl1, 5, 75, 75)    # 双边滤波

        rows, columns = self.bilf.shape
        bytesPerLine = columns

        QImg = QImage(self.bilf.data, columns, rows, bytesPerLine, QImage.Format_Indexed8)
        self.labelFilter.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelFilter.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnThresh_Clicked(self):
        self.th3 = cv2.adaptiveThreshold(
            self.bilf, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 255, 64)  # 自适应阈值分割法

        rows, columns = self.th3.shape
        bytesPerLine = columns

        QImg = QImage(self.th3.data, columns, rows, bytesPerLine, QImage.Format_Indexed8)
        self.labelThresh.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelThresh.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnOpen_Clicked(self):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 定义结构元素
        self.opening = cv2.morphologyEx(self.th3, cv2.MORPH_OPEN, kernel)  # 开运算

        rows, columns = self.opening.shape
        bytesPerLine = columns

        QImg = QImage(self.opening.data, columns, rows, bytesPerLine, QImage.Format_Indexed8)
        self.labelOpen.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelOpen.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnDelnDil_Clicked(self):
        image, contours, hierarchy = cv2.findContours(
            self.opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # src_grey找 轮廓（src_grey → image）

        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area < 340:
                cv2.drawContours(image, [contours[i]], 0, 0, -1)  # 对于image中 小于340像素的区域， 用 黑色 填充（即删除）

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 椭圆结构
        self.dilation = cv2.dilate(image, kernel)  # 膨胀

        rows, columns = self.dilation.shape
        bytesPerLine = columns

        QImg = QImage(self.dilation.data, columns, rows, bytesPerLine, QImage.Format_Indexed8)
        self.labelDelnDil.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelDelnDil.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnFeature_Clicked(self):
        src_grey = self.dilation
        src = cv2.cvtColor(self.dilation, cv2.COLOR_GRAY2RGB)
        image, contours, hierarchy_new = cv2.findContours(
            src_grey, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # src_grey找轮廓（src_grey→image）

        defect_num = 0
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area < 340:  # !画矩形，也要注意去除小区域的参数!
                continue
            defect_num = defect_num + 1  # 缺陷个数 计数
            self.textFeature.append("第 %d 个缺陷" % defect_num)
            # print("缺陷面积: %d （像素）" % area)     # 该轮廓的 面积

            perimeter = cv2.arcLength(contours[i], True)
            # print("缺陷周长: %d （像素）" % perimeter)       # 该轮廓的周长

            rect = cv2.minAreaRect(contours[i])  # 最小外接矩形
            length, width = rect[1]  # 取矩形的长、宽
            if length < width:
                length, width = width, length  # 交换长、宽

            box = np.int0(cv2.boxPoints(rect))  # 矩形的四个角点取整
            area_rect = cv2.contourArea(box)  # 计算 矩形面积

            rate_circle = (4 * math.pi) * area / (perimeter ** 2)
            rate_rect = area / area_rect
            rate_slim = length / width

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(src, str(defect_num), (box[1][0] - 10, box[1][1] - 10), font,
                        2, (0, 255, 0), 2, lineType=cv2.LINE_AA)  # 标记为 第几个 缺陷

            self.textFeature.append("矩形面积： %d （像素）" % area_rect)
            self.textFeature.append("【圆度率】： %.4f\n【矩形率】： %.4f\n 【细长比】： %.4f\n" % (
                rate_circle, rate_rect, rate_slim))

            cv2.drawContours(src, [box], 0, (0, 0, 255), 2)  # src 上用蓝色画 最小外接矩形
        self.rect = src

        rows, cols, channels = self.rect.shape
        bytesPerLine = channels * cols

        QImg = QImage(self.rect.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
        self.labelFeature.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelFeature.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnResult_Clicked(self):
        # 即将上述七个点击函数合并
        self.textFeature.setText("")

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
        self.cl1 = clahe.apply(self.img)

        self.bilf = cv2.bilateralFilter(self.cl1, 5, 75, 75)  # 双边滤波

        self.th3 = cv2.adaptiveThreshold(
            self.bilf, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 255, 64)  # 自适应阈值法

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 定义结构元素
        self.opening = cv2.morphologyEx(self.th3, cv2.MORPH_OPEN, kernel)  # 开运算

        image, contours, hierarchy = cv2.findContours(
            self.opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # src_grey找 轮廓（src_grey → image）

        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            # print(area)     # 该轮廓的 面积
            if area < 340:  # 修改参数！
                cv2.drawContours(image, [contours[i]], 0, 0, -1)  # 对于image中 小于57像素的区域， 用 黑色 填充（即删除）
                # del contours[i]                                         # 删除 该小轮廓

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 椭圆结构
        self.dilation = cv2.dilate(image, kernel)  # 膨胀

        src_grey = self.dilation
        src = cv2.cvtColor(self.dilation, cv2.COLOR_GRAY2RGB)
        image, contours, hierarchy_new = cv2.findContours(
            src_grey, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # src_grey找轮廓（src_grey→image）

        defect_num = 0
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area < 340:  # !画矩形，也要注意去除小区域的参数!
                continue
            defect_num = defect_num + 1  # 缺陷个数 计数
            self.textFeature.append("第 %d 个缺陷" % defect_num)
            # print("缺陷面积: %d （像素）" % area)     # 该轮廓的 面积

            perimeter = cv2.arcLength(contours[i], True)
            # print("缺陷周长: %d （像素）" % perimeter)       # 该轮廓的周长

            rect = cv2.minAreaRect(contours[i])  # 最小外接矩形
            length, width = rect[1]  # 取矩形的长、宽
            if length < width:
                length, width = width, length  # 交换长、宽

            box = np.int0(cv2.boxPoints(rect))  # 矩形的四个角点取整
            area_rect = cv2.contourArea(box)  # 计算 矩形面积

            rate_circle = (4 * math.pi) * area / (perimeter ** 2)
            rate_rect = area / area_rect
            rate_slim = length / width

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(src, str(defect_num), (box[1][0] - 10, box[1][1] - 10), font,
                        2, (0, 255, 0), 2, lineType=cv2.LINE_AA)  # 标记为 第几个 缺陷

            self.textFeature.append("矩形面积： %d （像素）" % area_rect)
            self.textFeature.append("【圆度率】： %.4f\n【矩形率】： %.4f\n 【细长比】： %.4f\n" % (
                rate_circle, rate_rect, rate_slim))

            cv2.drawContours(src, [box], 0, (0, 0, 255), 2)  # src 上用蓝色画 最小外接矩形
        self.rect = src

        rows, cols, channels = self.rect.shape
        bytesPerLine = channels * cols

        QImg = QImage(self.rect.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
        self.labelFeature.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelFeature.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnReset_Clicked(self):
        # 清空界面上所有label的图像和文本框内文字
        self.labelReadImage.setPixmap(QPixmap(""))
        self.labelEnhance.setPixmap(QPixmap(""))
        self.labelFilter.setPixmap(QPixmap(""))
        self.labelThresh.setPixmap(QPixmap(""))
        self.labelOpen.setPixmap(QPixmap(""))
        self.labelDelnDil.setPixmap(QPixmap(""))
        self.labelFeature.setPixmap(QPixmap(""))
        self.textFeature.clear()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = PyQtMainEntry()
    window.show()
    sys.exit(app.exec_())
