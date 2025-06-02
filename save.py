# save.py 文件
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QSplitter, QPushButton, QLabel,
                            QListWidget, QTableWidget, QTableWidgetItem, QProgressBar,
                            QMenuBar, QToolBar, QStatusBar, QMessageBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap, QPainter
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

class MplCanvas(FigureCanvas):
    def __init__(self, width=10, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)

        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
        plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
        self.confidences = [0] * 8
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setMinimumSize(100, 100)  # 防止太小不可见
        self.left_conf = [0.0] * 8
        self.right_conf = [0.0] * 8

    def plot(self):
        try:
            diseases = ['正常', '糖尿病', '青光眼', '白内障', 'AMD', '高血压', '近视', '其他疾病/异常']
            if not isinstance(self.confidences, list) or len(self.confidences) != 8:
                raise ValueError("confidences 应为长度为 8 的列表")
            '''
            self.axes.clear()

            x_pos = range(len(self.confidences))
            bars = self.axes.bar(x_pos, self.confidences, color='skyblue', width=0.6, align='center')

            self.axes.set_xticks(x_pos)
            self.axes.set_xticklabels(diseases, rotation=45, ha='center', fontsize=12)

            self.axes.set_title('疾病类别置信度', fontsize=16)
            self.axes.set_xlabel('疾病类别', fontsize=14, labelpad=10)
            self.axes.set_ylabel('置信度', fontsize=14, labelpad=10)

            self.axes.set_xticklabels(diseases, rotation=45, ha='right', fontsize=12)

            for bar in bars:
                yval = bar.get_height()
                self.axes.text(bar.get_x() + bar.get_width() / 2, yval + 0.01,
                               round(yval, 2), ha='center', fontsize=12)

            self.axes.tick_params(axis='x', labelsize=12)
            self.axes.tick_params(axis='y', labelsize=12)

            # self.fig.set_size_inches(5, 4)
            # self.fig.subplots_adjust(bottom=0.2)
            self.fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.25)
            self.fig.tight_layout()

            print(f"[绘图] 当前置信度: {self.confidences}")
            self.draw_idle()
            '''

            x = np.arange(len(diseases))  # x轴位置
            width = 0.35  # 每根柱子的宽度

            # 清空图表
            self.axes.clear()

            bars1 = self.axes.bar(x - width / 2, self.left_conf, width, label='左眼', color='skyblue')
            bars2 = self.axes.bar(x + width / 2, self.right_conf, width, label='右眼', color='orange')

            self.axes.set_title('左右眼疾病预测置信度', fontsize=16)
            self.axes.set_xlabel('疾病类别', fontsize=14)
            self.axes.set_ylabel('置信度', fontsize=14)
            self.axes.set_xticks(x)
            self.axes.set_xticklabels(diseases, rotation=45, ha='right', fontsize=12)
            self.axes.legend(fontsize=12)

            # 添加标签
            for bar in bars1:
                y = bar.get_height()
                if y > 0.01:
                    self.axes.text(bar.get_x() + bar.get_width() / 2, y + 0.01, f"{y:.2f}", ha='center', fontsize=10)

            for bar in bars2:
                y = bar.get_height()
                if y > 0.01:
                    self.axes.text(bar.get_x() + bar.get_width() / 2, y + 0.01, f"{y:.2f}", ha='center', fontsize=10)

            self.fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.28)
            self.fig.tight_layout()
            self.draw_idle()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.axes.clear()
            self.axes.set_title(f"绘图失败:{e}")
            self.draw()

    def show_values_on_plot(self, show):
        self.show_values = show
        self.plot()  # 重新绘制条形图

class Ui_MainWindow(object):

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1300, 800)
        MainWindow.setStyleSheet("background-color: #f5f5f5;")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.gridLayout.setColumnStretch(0, 3)  # 左侧图像 + 条形图
        self.gridLayout.setColumnStretch(1, 1)  # 右侧侧边栏
        # 创建 frame
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setStyleSheet("background-color:rgb(134, 197, 255)")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setLineWidth(0)
        self.frame.setObjectName("frame")
        self.frame_layout = QtWidgets.QVBoxLayout(self.frame)
        self.frame_layout.setContentsMargins(0, 0, 0, 0)
        self.frame_layout.setSpacing(0)

        # 创建图像显示区域
        self.image_display = QtWidgets.QLabel(self.frame)
        self.image_display.setStyleSheet("background-color:rgb(255, 169, 107)")
        self.frame.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self.image_display.setText("")
        self.image_display.setAlignment(Qt.AlignCenter)
        self.image_display.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.image_display.setMinimumSize(100, 100)  # 防止缩小时为0
        self.image_display.setScaledContents(True)  # 保持缩放图像显示
        self.image_display.setObjectName("image_display")

        # 添加用于显示患者 ID 的标签
        self.patient_id_label = QtWidgets.QLabel(self.frame)
        self.patient_id_label.setStyleSheet("font-size: 16px; color: black; padding: 5px;")
        self.patient_id_label.setAlignment(Qt.AlignCenter)
        self.patient_id_label.setText("患者ID：--")

        # 添加到图像区域顶部
        self.frame_layout.addWidget(self.patient_id_label)


        # 添加图像显示区域到布局
        self.frame_layout.addWidget(self.image_display)

        # 添加按钮到图像显示区域两侧
        self.prev_patient_button = QtWidgets.QPushButton("上一个患者", self.frame)
        self.prev_patient_button.setObjectName("prev_patient_button")
        self.next_patient_button = QtWidgets.QPushButton("下一个患者", self.frame)
        self.next_patient_button.setObjectName("next_patient_button")

        self.prev_patient_button.setFixedSize(100, 30)  # 设置按钮大小
        self.next_patient_button.setFixedSize(100, 30)  # 设置按钮大小
        self.prev_patient_button.setStyleSheet("background-color: #4CAF50; color: white;")  # 设置按钮样式
        self.next_patient_button.setStyleSheet("background-color: #4CAF50; color: white;")  # 设置按钮样式

        self.image_area_layout = QtWidgets.QHBoxLayout()
        self.image_area_layout.setSpacing(10)

        # 将按钮放置在图像显示区域的顶部左右两侧
        self.image_area_layout.addWidget(self.prev_patient_button)
        self.image_area_layout.addStretch(1)  # 添加弹性空间
        self.image_area_layout.addWidget(self.next_patient_button)

        # 将按钮布局添加到 frame 布局中
        self.frame_layout.addLayout(self.image_area_layout)

        # 添加 frame 到主布局
        self.gridLayout.addWidget(self.frame, 0, 0, 1, 1)

        # 创建 frame_2
        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        # self.frame_2.setMaximumSize(QtCore.QSize(400, 16777215))
        self.frame_2.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)

        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setLineWidth(0)
        self.frame_2.setObjectName("frame_2")
        self.frame_2_layout = QtWidgets.QGridLayout(self.frame_2)
        self.frame_2_layout.setContentsMargins(30, 30, 30, 30)
        self.frame_2_layout.setSpacing(40)

        # 创建按钮
        self.pushButton = QtWidgets.QPushButton(self.frame_2)
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setFixedSize(120, 50)  # 设置按钮大小
        self.pushButton_2 = QtWidgets.QPushButton(self.frame_2)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.setFixedSize(120, 50)  # 设置按钮大小
        self.pushButton_3 = QtWidgets.QPushButton(self.frame_2)
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.setFixedSize(120, 50)  # 设置按钮大小
        self.pushButton_4 = QtWidgets.QPushButton(self.frame_2)
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_4.setFixedSize(120, 50)  # 设置按钮大小

        # 添加按钮到布局
        self.frame_2_layout.addWidget(self.pushButton, 0, 0)
        self.frame_2_layout.addWidget(self.pushButton_3, 0, 1)
        self.frame_2_layout.addWidget(self.pushButton_2, 1, 0)
        self.frame_2_layout.addWidget(self.pushButton_4, 1, 1)

        # 创建进度条和标签
        self.progressBar = QtWidgets.QProgressBar(self.frame_2)
        self.progressBar.setEnabled(True)
        self.progressBar.setStyleSheet("")
        self.progressBar.setProperty("value", 0)
        self.progressBar.setTextDirection(QtWidgets.QProgressBar.TopToBottom)
        self.progressBar.setObjectName("progressBar")

        # self.label_3 = QtWidgets.QLabel(self.frame_2)  # 确保 label_3 被正确初始化
        # self.label_3.setObjectName("label_3")
        #
        # # 添加进度条和标签到布局
        # self.frame_2_layout.addWidget(self.label_3, 2, 0)
        # self.frame_2_layout.addWidget(self.progressBar, 2, 1)
        self.frame_2_layout.addWidget(self.progressBar, 2, 0, 1, 2)

        # 创建 QTableWidget 用于显示分类记录
        self.tableWidget = QtWidgets.QTableWidget(self.frame_2)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(3)  # 设置列数为3
        self.tableWidget.setRowCount(0)  # 初始行数为0
        self.tableWidget.setHorizontalHeaderLabels(["时间", "患者ID", "诊断结果"])  # 设置列标题

        # 设置 tableWidget 的大小策略为 Expanding
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.tableWidget.setSizePolicy(size_policy)

        # 设置 tableWidget 的最小大小
        self.tableWidget.setMinimumSize(QtCore.QSize(300, 200))

        # 添加 QTableWidget 到布局
        self.frame_2_layout.addWidget(self.tableWidget, 3, 0, 1, 2)  # 占据两列

        # 添加 frame_2 到主布局
        self.gridLayout.addWidget(self.frame_2, 0, 1, 2, 1)

        # 创建 frame_3
        self.frame_3 = QtWidgets.QFrame(self.centralwidget)
        self.frame_3.setSizePolicy(size_policy)
        # self.frame_3.setMaximumSize(QtCore.QSize(900, 300))
        self.frame_3.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setLineWidth(0)
        self.frame_3.setObjectName("frame_3")
        self.frame_3_layout = QtWidgets.QVBoxLayout(self.frame_3)
        self.frame_3_layout.setContentsMargins(10, 10, 10, 10)
        self.frame_3_layout.setSpacing(10)

        # 创建条形图显示区域
        self.mpl_canvas = MplCanvas(width=5, height=4, dpi=100)  # 初始化 MplCanvas
        # self.frame_3_layout.addWidget(self.mpl_canvas, alignment=Qt.AlignCenter)
        self.frame_3_layout.addWidget(self.mpl_canvas)
        self.frame_3_layout.setStretchFactor(self.mpl_canvas, 1)

        # # 设置条形图占据布局的空间，使用 setStretchFactor 控制它的伸缩比例
        # self.frame_3_layout.addWidget(self.mpl_canvas)
        # self.frame_3_layout.setStretch(0, 1)  # 设置条形图的伸缩因子为1，意味着它会占据整个区域
        # 添加 frame3 到主布局
        self.gridLayout.addWidget(self.frame_3, 1, 0, 1, 1)  # frame3 占据第二行，跨越两列

        # 设置主窗口的中心部件
        MainWindow.setCentralWidget(self.centralwidget)

        # 初始化条形图，不显示值
        self.mpl_canvas.show_values_on_plot(False)

        # 设置主窗口的菜单栏和状态栏
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1098, 33))
        self.menubar.setStyleSheet("QPushButton {background-color:rgb(46, 130, 255);}")
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setStyleSheet("QPushButton {background-color:rgb(46, 130, 255);}")
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        # 调用 retranslateUi 方法来设置翻译
        self.retranslateUi(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "单图上传"))
        self.pushButton_2.setText(_translate("MainWindow", "批量上传"))
        self.pushButton_3.setText(_translate("MainWindow", "开始分类"))
        self.pushButton_4.setText(_translate("MainWindow", "导出结果"))
        # self.label_3.setText(_translate("MainWindow", "分类中："))