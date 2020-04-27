import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import numpy as np
import cv2
import recognition as LP_REG

class LP_GUI(QWidget):
    def __init__(self):
        super(LP_GUI, self).__init__()
        
        self.resize(1200, 800)
        self.setWindowTitle("基于OpenCV的车牌识别算法实现")

        self.origin_size = (570, 430)
        
        
        # 各个步骤的图像
        self.origin_img = None
        self.color_img = None
        self.contour_img = None
        self.cut_imgs = []
        
        # 各个步骤的图像对应的显示模块
        self.origin_img_label = QLabel(self)
        self.color_img_label = QLabel(self)
        self.contour_img_label = QLabel(self)
        self.cut_imgs_label = [QLabel(self) for i in range(7)]

        self.result_label = QLabel(self)
        self.result_label.setObjectName('result')
        self.result_label.setAlignment(QtCore.Qt.AlignCenter|QtCore.Qt.AlignVCenter)

        btn_box = QWidget(self)
        origin_img_box = QWidget(self)
        blue_img_box = QWidget(self)
        contour_img_box = QWidget(self)
        cut_imgs_box = QWidget(self)

        btn_box.setLayout( self.btn_box() )
        origin_img_box.setLayout( self.img_box(self.origin_img_label, '原始图像') )
        blue_img_box.setLayout( self.img_box(self.color_img_label, '颜色识别后图像') )
        contour_img_box.setLayout( self.img_box(self.contour_img_label, '轮廓识别后图像') )
        cut_imgs_box.setLayout( self.img_boxes('车牌分割后图像') )

        img_box = QWidget(self)
        hlayout = QHBoxLayout(img_box)
        hlayout.addWidget(origin_img_box)
        hlayout.addWidget(blue_img_box)

        cut_img_box = QWidget(self)
        hlayout_sub = QVBoxLayout(cut_img_box)
        hlayout_sub.addWidget(contour_img_box)
        hlayout_sub.addWidget(cut_imgs_box)
        cut_img_box.setLayout(hlayout_sub)

        hlayout.addWidget(cut_img_box)
        img_box.setLayout(hlayout)

        vlayout = QVBoxLayout(self)
        vlayout.addWidget(btn_box)
        vlayout.addWidget(img_box)
        vlayout.addWidget(self.result_label)
        
        self.setLayout(vlayout)
        self.check_btn_state()

        with open('./style.qss') as file:
            str = file.readlines()
            str = ''.join(str).strip('\n')
        self.setStyleSheet(str)

    def img_box(self, img_label, label_text):
        '''
        由一张图片和文字组合的基础组件
        '''
        text = QLabel(self)
        text.setText(label_text)
        text.setAlignment(QtCore.Qt.AlignCenter|QtCore.Qt.AlignVCenter)
        vlayout = QVBoxLayout(self)

        img_label.setAlignment(QtCore.Qt.AlignCenter|QtCore.Qt.AlignVCenter)
        vlayout.addWidget(img_label)
        vlayout.addWidget(text)
        return vlayout

    def img_boxes(self, label_text):
        '''
        由多个图片和文字组合的基础组件
        '''
        vlayout = QVBoxLayout(self)

        text_box = QWidget(self)
        text_hlayout = QHBoxLayout(text_box)
        text = QLabel(self)
        text.setText(label_text)
        text.setAlignment(QtCore.Qt.AlignCenter|QtCore.Qt.AlignVCenter)
        text_hlayout.addWidget(text)

        text_box.setLayout(text_hlayout)

        img_box = QWidget(self)
        hlayout = QHBoxLayout(img_box)

        for i in range(len(self.cut_imgs_label)):
            self.cut_imgs_label[i].setAlignment(QtCore.Qt.AlignCenter|QtCore.Qt.AlignVCenter)

            hlayout.addWidget( self.cut_imgs_label[i] )
        img_box.setLayout(hlayout)

        vlayout.addWidget(img_box)
        vlayout.addWidget(text_box)
        return vlayout

    def btn_box(self):
        self.origin_btn = QPushButton(self)
        self.origin_btn.setText("打开图片")
        self.origin_btn.clicked.connect(self.get_origin_img)
        
        self.color_btn = QPushButton(self)
        self.color_btn.setText("颜色识别")
        self.color_btn.clicked.connect(self.get_color_img)

        self.contour_btn = QPushButton(self)
        self.contour_btn.setText("轮廓识别")
        self.contour_btn.clicked.connect(self.get_contour_img)

        self.cut_btn = QPushButton(self)
        self.cut_btn.setText("车牌分割")
        self.cut_btn.clicked.connect(self.get_cut_img)
        
        self.result_btn = QPushButton(self)
        self.result_btn.setText("字符识别")
        self.result_btn.clicked.connect(self.get_result)

        hlayout = QHBoxLayout(self)
        hlayout.addWidget(self.origin_btn)
        hlayout.addWidget(self.color_btn)
        hlayout.addWidget(self.contour_btn)
        hlayout.addWidget(self.cut_btn)
        hlayout.addWidget(self.result_btn)

        return hlayout

    def get_origin_img(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "All Files(*)")
        origin_img = cv2.imdecode(np.fromfile(imgName, dtype=np.uint8), -1)
        origin_img = cv2.resize(origin_img, (570, 430))
        self.origin_img = origin_img

        img = QtGui.QPixmap(imgName).scaled(self.origin_size[0], self.origin_size[1])
        self.origin_img_label.setPixmap(img)
        self.check_btn_state()
        self.result_label.setText('')

    def get_color_img(self):
        '''颜色识别函数'''
        color_img = LP_REG.color_change(self.origin_img)
        self.color_img = color_img
        # 将cv2的图片转为QLabel可以显示的格式
        self.color_img_label.setPixmap( QPixmap.fromImage(self.cvimg_to_qtimg(color_img) ) )
        self.check_btn_state()

    def get_contour_img(self):
        '''轮廓识别函数'''
        contours = LP_REG.binaryzation( self.color_img )
        contour_img = LP_REG.cut_out( contours, self.origin_img )
        self.contour_img = contour_img
        # 将cv2的图片转为QLabel可以显示的格式
        self.contour_img_label.setPixmap( QPixmap.fromImage(self.cvimg_to_qtimg(contour_img) ) )
        self.check_btn_state()

    def get_cut_img(self):
        '''切割单字符函数'''
        img_list = LP_REG.car_binaryzation_cut( self.contour_img )
        
        self.cut_imgs = img_list
        # 将cv2的图片转为QLabel可以显示的格式
        for i in range( len(self.cut_imgs_label) ):
            self.cut_imgs_label[i].setPixmap( QPixmap.fromImage(self.cvimg_to_qtimg(img_list[i]) ) )
        self.repaint()
        self.check_btn_state()
        
    def get_result(self):
        '''识别结果函数'''
        result = LP_REG.char_reconition(self.cut_imgs)
        self.result_label.setText(result)

        
    def check_btn_state(self):
        '''
        每个按钮点击后检测一下每个按钮是否满足点击条件
        '''
        self.color_btn.setEnabled(True)
        self.color_btn.setEnabled(True)
        self.contour_btn.setEnabled(True)
        self.cut_btn.setEnabled(True)
        self.result_btn.setEnabled(True)

        if self.origin_img is None:
            self.color_btn.setEnabled(False)
            self.contour_btn.setEnabled(False)
            self.cut_btn.setEnabled(False)
            self.result_btn.setEnabled(False)
        elif self.color_img is None:
            self.contour_btn.setEnabled(False)
            self.cut_btn.setEnabled(False)
            self.result_btn.setEnabled(False)
        elif self.contour_img is None:
            self.cut_btn.setEnabled(False)
            self.result_btn.setEnabled(False)
        elif len(self.cut_imgs) == 0:
            self.result_btn.setEnabled(False)

    def cvimg_to_qtimg(self, cvimg):
        '''cv2读取的图像数值转换为QLabel可以识别的格式'''
        if len( cvimg.shape ) == 3:
            height, width, depth = cvimg.shape
            cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
            cvimg = QImage(cvimg.data, width, height, width*depth,  QImage.Format_RGB888)
        else:
            height, width  = cvimg.shape
            cvimg = cvimg.reshape((height, width, 1)).repeat(3, -1)
            cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
            cvimg = QImage(cvimg.data, width, height, width*3, QImage.Format_RGB888)
    
        return cvimg

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    gui = LP_GUI()
    gui.show()
    sys.exit(app.exec_())