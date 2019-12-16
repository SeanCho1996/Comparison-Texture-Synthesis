import cv2 as cv
from matplotlib import pyplot as plt
import _test_get_cossim as test
import lbp_feature as lbp
import ssim_feature as ssim
from PyQt5 import QtWidgets, QtCore, QtGui, Qt
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import numpy as np
import sys


class Ui_mainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui_mainWindow, self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)
        self.input_img_name = ""

    def setupUi(self, mainWindow):
        mainWindow.setObjectName("mainWindow")
        mainWindow.setWindowModality(QtCore.Qt.WindowModal)
        mainWindow.resize(1600, 900)
        self.centralWidget = QtWidgets.QWidget(mainWindow)
        self.centralWidget.setObjectName("centralWidget")
        mainWindow.setCentralWidget(self.centralWidget)
        self.retranslateUi(mainWindow)
        QtCore.QMetaObject.connectSlotsByName(mainWindow)

        # create label to display input image
        self.img_label = QtWidgets.QLabel(self.centralWidget)
        self.img_label.setGeometry(QtCore.QRect(190, 300, 300, 300))
        self.img_label.setText("please select the input image")
        self.img_label.setObjectName("label")
        self.img_label.setAlignment(Qt.AlignCenter)
        # self.img_label.setFrameStyle(QtWidgets.QFrame.StyledPanel | QFrame.Raised)

        # labels for the first method result
        self.res_img_label1 = QtWidgets.QLabel(self.centralWidget)
        self.res_img_label1.setGeometry(QtCore.QRect(900, 45, 150, 150))
        self.res_img_label1.setText("res_1")
        self.res_img_label1.setObjectName("label_img_1")
        self.res_img_label1.setAlignment(Qt.AlignCenter)
        self.res_img_label1.setFrameStyle(QtWidgets.QFrame.StyledPanel | QFrame.Raised)

        self.res_label1 = QtWidgets.QLabel(self.centralWidget)
        self.res_label1.setGeometry(QtCore.QRect(1150, 95, 300, 50))
        self.res_label1.setText("Method Efros-Leung\nSimilarity: ")
        self.res_label1.setObjectName("label_1")
        # self.res_label1.setAlignment(Qt.AlignCenter)
        # self.res_label1.setFrameStyle(QtWidgets.QFrame.StyledPanel | QFrame.Raised)

        # labels for the second method result
        self.res_img_label2 = QtWidgets.QLabel(self.centralWidget)
        self.res_img_label2.setGeometry(QtCore.QRect(900, 265, 150, 150))
        self.res_img_label2.setText("res_2")
        self.res_img_label2.setObjectName("label_img_2")
        self.res_img_label2.setAlignment(Qt.AlignCenter)
        self.res_img_label2.setFrameStyle(QtWidgets.QFrame.StyledPanel | QFrame.Raised)

        self.res_label2 = QtWidgets.QLabel(self.centralWidget)
        self.res_label2.setGeometry(QtCore.QRect(1150, 315, 300, 50))
        self.res_label2.setText("Method Optim\nSimilarity: ")
        self.res_label2.setObjectName("label_2")

        # labels for the third method result
        self.res_img_label3 = QtWidgets.QLabel(self.centralWidget)
        self.res_img_label3.setGeometry(QtCore.QRect(900, 485, 150, 150))
        self.res_img_label3.setText("res_3")
        self.res_img_label3.setObjectName("label_img_3")
        self.res_img_label3.setAlignment(Qt.AlignCenter)
        self.res_img_label3.setFrameStyle(QtWidgets.QFrame.StyledPanel | QFrame.Raised)

        self.res_label3 = QtWidgets.QLabel(self.centralWidget)
        self.res_label3.setGeometry(QtCore.QRect(1150, 535, 300, 50))
        self.res_label3.setText("Method Quilting\nSimilarity: ")
        self.res_label3.setObjectName("label_3")

        # labels for the fourth method result
        self.res_img_label4 = QtWidgets.QLabel(self.centralWidget)
        self.res_img_label4.setGeometry(QtCore.QRect(900, 705, 150, 150))
        self.res_img_label4.setText("res_4")
        self.res_img_label4.setObjectName("label_img_4")
        self.res_img_label4.setAlignment(Qt.AlignCenter)
        self.res_img_label4.setFrameStyle(QtWidgets.QFrame.StyledPanel | QFrame.Raised)

        self.res_label4 = QtWidgets.QLabel(self.centralWidget)
        self.res_label4.setGeometry(QtCore.QRect(1150, 755, 300, 50))
        self.res_label4.setText("Method DeepTexture\nSimilarity: ")
        self.res_label4.setObjectName("label_4")

        # Button - select input image
        self.img_button = QtWidgets.QPushButton(self.centralWidget)
        self.img_button.setGeometry(QtCore.QRect(240, 620, 200, 50))
        self.img_button.setObjectName("img_button")
        self.img_button.setText("Select input image")
        # self.img_button.setFlat(True)
        self.img_button.setStyleSheet("background-color: rgb(196, 196, 196);"
                                      "border-color: rgb(0, 0, 0);"
                                      "font: 75 12pt \"Calibri\";"
                                      "color: rgb(0, 0, 0);")
        QtCore.QMetaObject.connectSlotsByName(mainWindow)
        self.img_button.clicked.connect(self.openfile)

        # Choice Button - select compare metric
        self.metric_choice = QtWidgets.QComboBox(self.centralWidget)
        self.metric_choice.setGeometry(QtCore.QRect(600, 425, 200, 30))
        self.metric_choice.setObjectName("metric_choice")
        self.metric_choice.addItem("Choose a metric")
        self.metric_choice.addItem("GIST")
        self.metric_choice.addItem("LBP")
        self.metric_choice.addItem("SSIM")

        self.metric_button = QtWidgets.QPushButton(self.centralWidget)
        self.metric_button.setGeometry(QtCore.QRect(650, 475, 100, 30))
        self.metric_button.setObjectName("img_button")
        self.metric_button.setText("confirm")
        self.metric_button.setStyleSheet("background-color: rgb(196, 196, 196);"
                                      "border-color: rgb(0, 0, 0);"
                                      "font: 75 12pt \"Calibri\";"
                                      "color: rgb(0, 0, 0);")
        self.metric_button.clicked.connect(self.calculate_sim)

    # Event - img_button clicked -> choose img
    def openfile(self):
        self.openfile_name = QFileDialog.getOpenFileName(self, 'select img','','img files(*.png)')[0]
        print(self.openfile_name)
        img_input = QtGui.QPixmap(self.openfile_name)
        self.img_label.setPixmap(img_input)
        self.img_label.setScaledContents(True)
        self.input_img_name = self.openfile_name.split('/')[-1]
        self.input_img_name = self.input_img_name.split('.')[0]
        # print(self.input_img_name)

    # Event - confirm_button clicked -> calculate similarity
    def calculate_sim(self):
        select_value = self.metric_choice.currentText()
        if select_value == "GIST":
            # print("confirm GIST")
            if self.input_img_name != "":
                resultpath = "./img_results/" + self.input_img_name + "/"
                print(resultpath)
                sim = np.zeros([4], np.float)
                for i in range(4):
                    O_IN = {}
                    O_IN['s_img_url_a'] = "./" + self.input_img_name + ".png"
                    O_IN['s_img_url_b'] = resultpath + "result_" + str(i+1) + ".png"
                    sim[i] = test.proc_main(O_IN)
                self.res_img_label1.setPixmap(QtGui.QPixmap(resultpath + "result_1.png"))
                self.res_img_label1.setScaledContents(True)
                self.res_label1.setText("Method Efros-Leung\nSimilarity: " + str(sim[0]))
                self.res_img_label2.setPixmap(QtGui.QPixmap(resultpath + "result_2.png"))
                self.res_img_label2.setScaledContents(True)
                self.res_label2.setText("Method Optim\nSimilarity: " + str(sim[1]))
                self.res_img_label3.setPixmap(QtGui.QPixmap(resultpath + "result_3.png"))
                self.res_img_label3.setScaledContents(True)
                self.res_label3.setText("Method Quilting\nSimilarity: " + str(sim[2]))
                self.res_img_label4.setPixmap(QtGui.QPixmap(resultpath + "result_4.png"))
                self.res_img_label4.setScaledContents(True)
                self.res_label4.setText("Method DeepTexture\nSimilarity: " + str(sim[3]))
        if select_value=="LBP":
            # print("confirm GIST")
            if self.input_img_name != "":
                resultpath = "./img_results/" + self.input_img_name + "/"
                print(resultpath)
                sim = np.zeros([4], np.float)
                for i in range(4):
                    O_IN = {}
                    O_IN['s_img_url_a'] = "./" + self.input_img_name + ".png"
                    O_IN['s_img_url_b'] = resultpath + "result_" + str(i+1) + ".png"
                    sim[i] = lbp.proc_main(O_IN)
                self.res_img_label1.setPixmap(QtGui.QPixmap(resultpath + "result_1.png"))
                self.res_img_label1.setScaledContents(True)
                self.res_label1.setText("Method Efros-Leung\nSimilarity: " + str(sim[0]))
                self.res_img_label2.setPixmap(QtGui.QPixmap(resultpath + "result_2.png"))
                self.res_img_label2.setScaledContents(True)
                self.res_label2.setText("Method Optim\nSimilarity: " + str(sim[1]))
                self.res_img_label3.setPixmap(QtGui.QPixmap(resultpath + "result_3.png"))
                self.res_img_label3.setScaledContents(True)
                self.res_label3.setText("Method Quilting\nSimilarity: " + str(sim[2]))
                self.res_img_label4.setPixmap(QtGui.QPixmap(resultpath + "result_4.png"))
                self.res_img_label4.setScaledContents(True)
                self.res_label4.setText("Method DeepTexture\nSimilarity: " + str(sim[3]))
        if select_value == "SSIM":
            # print("confirm GIST")
            if self.input_img_name != "":
                resultpath = "./img_results/" + self.input_img_name + "/"
                print(resultpath)
                sim = np.zeros([4], np.float)
                for i in range(4):
                    O_IN = {}
                    O_IN['s_img_url_a'] = "./" + self.input_img_name + ".png"
                    O_IN['s_img_url_b'] = resultpath + "result_" + str(i+1) + ".png"
                    sim[i] = ssim.ssim(O_IN)
                self.res_img_label1.setPixmap(QtGui.QPixmap(resultpath + "result_1.png"))
                self.res_img_label1.setScaledContents(True)
                self.res_label1.setText("Method Efros-Leung\nSimilarity: " + str(sim[0]))
                self.res_img_label2.setPixmap(QtGui.QPixmap(resultpath + "result_2.png"))
                self.res_img_label2.setScaledContents(True)
                self.res_label2.setText("Method Optim\nSimilarity: " + str(sim[1]))
                self.res_img_label3.setPixmap(QtGui.QPixmap(resultpath + "result_3.png"))
                self.res_img_label3.setScaledContents(True)
                self.res_label3.setText("Method Quilting\nSimilarity: " + str(sim[2]))
                self.res_img_label4.setPixmap(QtGui.QPixmap(resultpath + "result_4.png"))
                self.res_img_label4.setScaledContents(True)
                self.res_label4.setText("Method DeepTexture\nSimilarity: " + str(sim[3]))






    def retranslateUi(self, mainWindow):
        _translate = QtCore.QCoreApplication.translate
        mainWindow.setWindowTitle('compare texture synth methodes')
        # mainWindow.setWindowIcon(QIcon('logo.png'))

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = QtWidgets.QMainWindow()
    ui = Ui_mainWindow()
    ui.setupUi(mainWindow)
    mainWindow.show()
    sys.exit(app.exec_())


# app = QtWidgets.QApplication(sys.argv)
# app.aboutToQuit.connect(app.deleteLater)
# widget = QtWidgets.QWidget()
# widget.resize(960, 540)
# widget.setWindowTitle('compare result')
#
#
# widget.show()
# sys.exit(app.exec_())


# if __name__ == "__main__":
#     O_IN = {}
#     O_IN['s_img_url_a'] = "./input_1.png"
#     O_IN['s_img_url_b'] = "../img_results/input_1/result_1.png"
#     test.proc_main(O_IN)
