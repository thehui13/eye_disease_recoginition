from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QSplitter, QPushButton, QLabel,
                            QListWidget, QTableWidget, QTableWidgetItem, QProgressBar,
                            QMenuBar, QToolBar, QStatusBar, QGraphicsScene, QMessageBox,
                            QSizePolicy, QFileDialog)  # 确保导入所有需要的类
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QPainter
import sys
import os
from PyQt5 import QtCore
from save import Ui_MainWindow, MplCanvas  # 从 save1.py 中导入 Ui_MainWindow 和 MplCanvas
from datetime import datetime
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from collections import defaultdict
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

weights = ResNet50_Weights.DEFAULT

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

label_map = {
    0: "正常",
    1: "糖尿病",
    2: "青光眼",
    3: "白内障",
    4: "AMD",
    5: "高血压",
    6: "近视",
    7: "其他疾病"
}

def get_labels_from_confidence(conf_vector, label_map, threshold):
    return [label_map[i] for i, prob in enumerate(conf_vector) if prob >= threshold]

def load_model(weight_path, device='cpu'):
    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 8)  # 和训练时保持一致

    model.load_state_dict(torch.load(weight_path, map_location=device))

    model.to(device)

    model.eval()
    return model


# 自定义数据集类：用于加载测试图片
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.filenames = [
            f for f in os.listdir(data_dir)
            if os.path.splitext(f)[-1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']
        ]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        img_path = os.path.join(self.data_dir, img_name)

        try:
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
        except Exception as e:
            print(f"⚠️ 读取或变换图像出错: {img_path} → {e}")
            raise  # 可加可不加，debug 阶段建议加

        return img, img_name



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()  # 创建 Ui_MainWindow 实例
        self.ui.setupUi(self)      # 调用 setupUi 方法来设置 UI
        self.scene = QGraphicsScene(self)
        self.timer = QTimer(self)  # 初始化 QTimer
        self.timer.timeout.connect(self.update_progress)

        # 创建按钮实例
        self.ui.pushButton.clicked.connect(self.on_pushButton_clicked)
        self.ui.pushButton_2.clicked.connect(self.on_pushButton2_clicked)
        self.ui.pushButton_3.clicked.connect(self.on_pushButton3_clicked)
        self.ui.pushButton_4.clicked.connect(self.on_pushButton4_clicked)

        self.ui.prev_patient_button.clicked.connect(self.show_previous_patient)
        self.ui.next_patient_button.clicked.connect(self.show_next_patient)

        # 初始化患者图像列表和当前索引
        self.patient_images = []
        self.current_patient_index = 0

        self.patient_records = []
        self.is_batch_mode = False
        self.patient_id=[]

        self.data_loader = []

        self.patient_confidences = []  # 与 self.patient_images 对应，存储每位患者的置信度

    def on_pushButton_clicked(self):
        print("on_pushButton_clicked 被调用")
        self.is_batch_mode = False
        self.ui.pushButton.setEnabled(False)  # 禁用按钮，防止重复点击
        self.ui.progressBar.setValue(0)
        try:
            # 打开文件夹选择对话框
            folder_path = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
            if not folder_path:
                return
            image_files = self.get_image_files_from_folder(folder_path)
            if len(image_files) < 2:
                QMessageBox.warning(self, "警告", "文件夹中至少需要两张图片")
                return
            self.folder_path = folder_path
            self.reset_all_inputs()
            self.ui.progressBar.setValue(50)
            # print(image_files)
            # 读取图片数据并存储到 data_loader，同时获取用于显示的 QPixmap 列表
            pixmaps = self.read_and_store_image_data(image_files)

            if len(pixmaps) < 2:
                QMessageBox.warning(self, "警告", "无法加载足够的图片显示")
                return

            combined_pixmap = self.combine_images_for_display(image_files[0:2])
            self.patient_images.append(combined_pixmap)
            # self.statusBar().showMessage(f"已加载 {len(image_files)} 张图像")

            # 添加患者ID记录（确保后续能识别）
            name0 = os.path.basename(image_files[0])
            patient_id = name0.split('_')[0] if '_' in name0 else "未知"
            self.patient_records.append((patient_id, image_files[0]))

            self.statusBar().showMessage(f"已加载 {len(self.patient_images)} 个病人的图像")

            if self.patient_images:
                self.current_patient_index = 0
                self.display_patient_image(self.patient_images[self.current_patient_index])
                # self.show_plot()
            self.ui.progressBar.setValue(100)
        finally:
            self.ui.pushButton.setEnabled(True)  # 恢复按钮状态

    def get_image_files_from_folder(self, folder_path):
        supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        all_files = os.listdir(folder_path)

        image_files = []
        for f in all_files:
            ext = os.path.splitext(f)[-1].lower()
            if ext in supported_extensions:
                full_path = folder_path + "/" + f
                # print(f"folder_path:{folder_path}\nf:{f}\nfull_path:{full_path}")
                image_files.append(full_path)

        return image_files

    # def get_image_files_from_folder(self, folder_path):
    #     supported_extensions = ['.jpg', '.jpeg', '.png', '.dicom']
    #     files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
    #     return [f for f in files if any(f.lower().endswith(ext) for ext in supported_extensions)]

    def read_and_store_image_data(self, image_files):
        """
        读取 image_files 中的图片数据，并将图片名称和数据存入 self.data_loader，
        同时返回用于显示的 QPixmap 列表（可选，根据需求可以进行拼接显示）
        """
        pixmaps = []
        for image_path in image_files:
            # 读取原始图像数据（这里采用 OpenCV 读取，返回 numpy 数组）
            img_data = cv2.imread(image_path)
            if img_data is None:
                self.statusBar().showMessage(f"无法加载图片: {os.path.basename(image_path)}")
                continue
            # 将图片数据存入 data_loader，保存文件名（不含路径）和数据
            self.data_loader.append({
                'name': os.path.basename(image_path),
                'data': img_data
            })
            # 同时读取 QPixmap 用于界面显示
            pixmap = QPixmap(image_path)
            if pixmap.isNull():
                self.statusBar().showMessage(f"加载 QPixmap 失败: {os.path.basename(image_path)}")
                continue
            pixmaps.append(pixmap)
        print(f"type:{type(pixmaps)}")
        return pixmaps

    def combine_pixmap(self, pixmap1, pixmap2):
        width = pixmap1.width() + pixmap2.width()
        height = max(pixmap1.height(), pixmap2.height())

        combined = QPixmap(width, height)
        combined.fill(Qt.transparent)

        painter = QPainter(combined)
        painter.drawPixmap(0, 0, pixmap1)

    # 计算第二张图片的垂直偏移量，使其居中对齐
        offset_y = (height - pixmap2.height()) // 2
        painter.drawPixmap(pixmap1.width(), offset_y, pixmap2)
        painter.end()

        return combined

    def combine_and_display_images(self, image_files):
        try:
            pixmap1 = QPixmap(image_files[0])
            pixmap2 = QPixmap(image_files[1])

            if pixmap1.isNull() or pixmap2.isNull():
                raise ValueError("无法加载图像")

        # 调整图像大小以匹配高度
            height = min(pixmap1.height(), pixmap2.height())
            pixmap1 = pixmap1.scaledToHeight(height, Qt.SmoothTransformation)
            pixmap2 = pixmap2.scaledToHeight(height, Qt.SmoothTransformation)

        # 水平拼接图像
            combined_pixmap = self.combine_pixmap(pixmap1, pixmap2)

        # 调整拼接后的图像大小以适应显示区域
            scaled_pixmap = combined_pixmap.scaled(
                self.ui.image_display.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            self.ui.image_display.setPixmap(scaled_pixmap)
            self.ui.image_display.resize(scaled_pixmap.width(), scaled_pixmap.height())
            self.statusBar().showMessage("图像拼接成功")
        except Exception as e:
            self.statusBar().showMessage(f"错误: {str(e)}")
            self.ui.image_display.clear()
            self.ui.image_display.setText("图像显示区域")

    def reset_all_inputs(self):
        print("[重置] 清空旧图像与分类状态")
        self.patient_images.clear()
        self.patient_records.clear()
        self.left_confidences = []
        self.right_confidences = []
        self.current_patient_index = 0

        self.ui.image_display.clear()  # 清空图像显示
        self.ui.mpl_canvas.left_conf = [0.0] * 8
        self.ui.mpl_canvas.right_conf = [0.0] * 8
        self.ui.mpl_canvas.plot()  # 清空图表

        self.ui.tableWidget.setRowCount(0)  # 清空表格
        self.ui.patient_id_label.setText("患者ID：--")  # 清空 ID 标签

    # 预测函数：加载模型，对 image_dir 中的图片进行分类
    def predict_batch(self, model, image_dir, device='cpu'):
        dataset = CustomDataset(image_dir, transform=transform)
        loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
        model.to(device)

        model.eval()
        results = []
        for i in range(len(dataset)):
            img, name = dataset[i]
            print(f"✅ 成功加载: {name}, tensor shape: {img.shape}")
            self.ui.progressBar.setValue(int(i / len(dataset) * 75))
        with torch.no_grad():
            for imgs, names in loader:
                imgs = imgs.to(device)
                outputs = model(imgs)  # [B, num_classes]

                softmax_scores = torch.softmax(outputs, dim=1)  # [B, num_classes]
                preds = torch.argmax(softmax_scores, dim=1)
                # print(f"[模型输出] raw outputs = {outputs}")
                # print(f"[Softmax后] softmax_scores = {softmax_scores}")
                for name, pred, conf_vector in zip(names, preds, softmax_scores):
                    results.append((name, pred.item(), conf_vector.cpu().numpy().tolist()))

        return results
    def on_pushButton3_clicked(self):
        print("开始分类按钮被点击")
        self.ui.pushButton_3.setEnabled(False)
        self.ui.progressBar.setValue(0)
        try:

            self.ui.tableWidget.setRowCount(0)  # 清空旧表格
            self.ui.mpl_canvas.confidences = [0.0] * 8  # 清空历史图表数据
            self.patient_confidences.clear()  # 清空历史置信度数据
            # self.patient_records.clear()
            self.current_patient_index = 0
            # 加载模型
            model = load_model(r"model/resnet_model.pth", device='cpu')
            # 使用预测函数
            image_dir = self.folder_path
            prediction_results = self.predict_batch(model, image_dir, device='cpu')
            # print(f"[预测结果] prediction_results = {prediction_results}")

            threshold = 0.2
            results_by_patient = defaultdict(lambda: {"left": [], "right": []})
            time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 初始化
            raw_conf_map = defaultdict(dict)  # patient_id → {left: [...], right: [...]}
            self.ui.progressBar.setValue(75)
            # 收集所有预测结果
            for filename, _, conf_vector in prediction_results:
                name = os.path.basename(filename)
                if '_' not in name:
                    continue
                patient_id = name.split('_')[0]
                side = name.split('_')[1].split('.')[0].lower()  # 'left' or 'right'
                raw_conf_map[patient_id][side] = conf_vector
            self.ui.progressBar.setValue(80)
            # 双柱图显示左右眼的置信度
            self.left_confidences = []
            self.right_confidences = []

            for patient_id in sorted(raw_conf_map.keys()):
                left = np.array(raw_conf_map[patient_id].get('left', [0] * 8))
                right = np.array(raw_conf_map[patient_id].get('right', [0] * 8))
                self.left_confidences.append(left.tolist())
                self.right_confidences.append(right.tolist())
            self.ui.progressBar.setValue(85)
            # 多标签分类结果按患者合并
            for filename, _, conf_vector in prediction_results:
                name = os.path.basename(filename)
                if '_' not in name:
                    continue
                patient_id = name.split('_')[0]
                side = name.split('_')[1].split('.')[0].lower()  # left / right

                predicted_labels = [
                    label_map[i] for i, prob in enumerate(conf_vector)
                    if prob >= threshold
                ]
                results_by_patient[patient_id][side] = predicted_labels
            self.ui.progressBar.setValue(90)
            # 合并左右眼结果并去重显示
            for patient_id, eyes in results_by_patient.items():
                combined_labels = set(eyes.get('left', []) + eyes.get('right', []))
                if combined_labels:
                    result_text = " / ".join(sorted(combined_labels))
                else:
                    result_text = "无明显异常"
                self.add_classification_record(time_now, patient_id, result_text)
            self.ui.progressBar.setValue(95)
            self.current_patient_index = 0
            self.show_plot()
            self.statusBar().showMessage("分类完成")
            self.ui.progressBar.setValue(100)

        except Exception as e:
            self.statusBar().showMessage(f"分类错误: {str(e)}")
        finally:
            self.ui.pushButton_3.setEnabled(True)

    def classify_patients(self):
        self.current_patient_index = 0  # 重置当前索引
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.classify_next_patient)
        self.timer.start(100)  # 每100ms处理下一个病人

    def classify_all_patients(self):
        for patient_id, _ in self.patient_records:
            classification_result = self.classify_image(patient_id)
           

    def update_progress(self):
        # 获取当前进度值
        current_value = self.ui.progressBar.value()

    # 随机增加进度值 (模拟分类过程)
        increment = QtCore.qrand() % 10 + 1  # 1-10之间的随机数
        new_value = min(current_value + increment, 100)
        

    # 更新进度条
        self.ui.progressBar.setValue(new_value)
        self.timer.start(100)

    # 如果达到100%，停止定时器并显示图形
        if new_value >= 100:
            self.timer.stop()
            if self.is_batch_mode == False:
                self.show_plot()
                self.statusBar().showMessage("分类完成")
                self.classify_image(self.folder_path)
                
            else:
                self.show_plot()
                self.statusBar().showMessage("分类完成")
                self.classify_all_patients()

    def show_plot(self):
        index = self.current_patient_index
        # if 0 <= index < len(self.patient_confidences):
        #     self.ui.mpl_canvas.confidences = self.patient_confidences[index]
        #     self.ui.mpl_canvas.plot()
        # else:
        #     print(f"⚠️ 无法绘图，索引超出范围：{index}")
        if (
                hasattr(self, "left_confidences")
                and 0 <= index < len(self.left_confidences)
                and 0 <= index < len(self.right_confidences)
        ):
            self.ui.mpl_canvas.left_conf = self.left_confidences[index]
            self.ui.mpl_canvas.right_conf = self.right_confidences[index]
            self.ui.mpl_canvas.plot()
        else:
            print(f"[⚠️] 跳过绘图：尚未分类或索引越界 (index={index})")
        

    

    def combine_images_for_display(self, image_files):
        try:
            pixmap1 = QPixmap(image_files[0])
            pixmap2 = QPixmap(image_files[1])

            if pixmap1.isNull() or pixmap2.isNull():
                raise ValueError("无法加载图像")

            # 调整图像大小以匹配高度
            height = min(pixmap1.height(), pixmap2.height())
            pixmap1 = pixmap1.scaledToHeight(height, Qt.SmoothTransformation)
            pixmap2 = pixmap2.scaledToHeight(height, Qt.SmoothTransformation)

            # 水平拼接图像
            combined_pixmap = self.combine_pixmap(pixmap1, pixmap2)

            return combined_pixmap
        except Exception as e:
            self.statusBar().showMessage(f"错误: {str(e)}")
            return None

    def on_pushButton2_clicked(self):
        print("批量上传按钮被点击")
        self.ui.pushButton_2.setEnabled(False)
        self.is_batch_mode = True
        self.ui.progressBar.setValue(0)
        try:
            folder_path = QFileDialog.getExistingDirectory(self, "选择批量图像文件夹")
            if not folder_path:
                return

            image_files = self.get_image_files_from_folder(folder_path)
            self.reset_all_inputs()
            # 按照文件名前缀（如 "1_"）进行分组
            patient_groups = dict()
            for path in image_files:
                filename = os.path.basename(path)
                if '_' not in filename:
                    continue
                prefix = filename.split('_')[0]
                patient_groups.setdefault(prefix, []).append(path)

            self.patient_images.clear()
            self.patient_records.clear()
            self.patient_confidences.clear()
            count = 0
            for patient_id, paths in patient_groups.items():
                if len(paths) != 2:
                    print(f"跳过患者 {patient_id}：不是两张图")
                    continue
                paths_sorted = sorted(paths)  # 确保 left/right 顺序一致（例如 left在前）
                _ = self.read_and_store_image_data(paths_sorted)

                combined_pixmap = self.combine_images_for_display(paths_sorted)
                if combined_pixmap:
                    self.patient_images.append(combined_pixmap)
                    self.patient_records.append((patient_id, paths_sorted[0]))
                    print(f"✅ 添加病人: {patient_id}")
                count += 1
                self.ui.progressBar.setValue(int(count / len(image_files) * 100))

            self.folder_path = folder_path  # 设置供预测用路径
            self.statusBar().showMessage(f"已加载 {len(self.patient_images)} 个病人的图像")

            if self.patient_images:
                self.current_patient_index = 0
                self.display_patient_image(self.patient_images[self.current_patient_index])
                # self.show_plot()
            self.ui.progressBar.setValue(100)
        finally:
            self.ui.pushButton_2.setEnabled(True)


    def add_classification_record(self, time, patient_id, classification_result):
        row_position = self.ui.tableWidget.rowCount()  # 获取当前行数
        self.ui.tableWidget.insertRow(row_position)  # 插入新行
    # 添加数据到表格
        self.ui.tableWidget.setItem(row_position, 0, QTableWidgetItem(time))  # 添加时间
        self.ui.tableWidget.setItem(row_position, 1, QTableWidgetItem(str(patient_id)))  # 添加病人ID
        self.ui.tableWidget.setItem(row_position, 2, QTableWidgetItem(classification_result))

        self.ui.tableWidget.resizeRowsToContents()
        self.ui.progressBar.setValue(0)

    def combine_and_save_images(self, image_files, patient_folder):
        try:
            pixmap1 = QPixmap(image_files[0])
            pixmap2 = QPixmap(image_files[1])

            if pixmap1.isNull() or pixmap2.isNull():
                raise ValueError("无法加载图像")

        # 确定目标大小
            target_size = self.ui.image_display.size()

        # 缩放图片
            scaled_pixmap1, scaled_pixmap2 = self.scale_images(pixmap1, pixmap2, target_size)

        # 拼接图片
            combined_pixmap = self.combine_pixmap(scaled_pixmap1, scaled_pixmap2)
            combined_image_path = os.path.join(patient_folder, "combined_image.png")
            combined_pixmap.save(combined_image_path)

        # 提取病人ID（例如文件夹名称）
            file_name1 = os.path.basename(image_files[0])
            file_name2 = os.path.basename(image_files[1])
            common_part = self.extract_common_part(file_name1, file_name2)  # 提取共同部分
            patient_id = common_part if common_part else os.path.basename(patient_folder)
        
        # 存储病人ID和图像路径
            self.patient_records.append((patient_id, combined_image_path))
            print(self.patient_records)
            return combined_image_path
        except Exception as e:
            self.statusBar().showMessage(f"错误: {str(e)}")
            return None
    
    def extract_common_part(self, file_name1, file_name2):
    # 去掉文件扩展名
        name1 = os.path.splitext(file_name1)[0]
        name2 = os.path.splitext(file_name2)[0]

    # 找到共同部分
        common_part = ""
        for part1, part2 in zip(name1, name2):
            if part1 == part2:
                common_part += part1 
            else:
                break

        return common_part  # 去掉多余的下划线

    def on_pushButton4_clicked(self):
        
        # 获取表格中的所有记录
        table = self.ui.tableWidget
        row_count = table.rowCount()
        column_count = table.columnCount()

        # 创建一个列表来存储表格数据
        data = []

        # 获取列名
        headers = [table.horizontalHeaderItem(i).text() for i in range(column_count)]

        # 遍历表格的每一行和每一列，将数据存储到列表中
        for row in range(row_count):
            row_data = []
            for column in range(column_count):
                item = table.item(row, column)
                if item is not None:
                    row_data.append(item.text())
                else:
                    row_data.append("")  # 如果单元格为空，添加空字符串
            data.append(row_data)

        # 创建一个 DataFrame
        df = pd.DataFrame(data, columns=headers)
        # df.to_excel("output.xlsx", index=False)

        # 弹出文件保存对话框，让用户选择保存路径
        file_path, _ = QFileDialog.getSaveFileName(self, "保存 Excel 文件", "", "Excel 文件 (*.xlsx)")

        if file_path:
            try:
            # 将 DataFrame 保存为 Excel 文件
                df.to_excel(file_path, index=False)
                QMessageBox.information(self, "成功", "表格已成功导出为 Excel 文件")
            except Exception as e:
                QMessageBox.warning(self, "错误", f"导出失败: {str(e)}")

    def classify_image(self, folder_path ):
    # 模拟分类逻辑
        classification_result = "糖尿病"  # 假设分类结果为 "糖尿病"
    # 获取当前时间
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # 从文件名中提取患者ID
        patient_id = os.path.splitext(os.path.basename(folder_path))[0]
    # 添加记录到表格
        self.add_classification_record(current_time, patient_id, classification_result)        
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 重新调整图像大小
        pixmap = self.ui.image_display.pixmap()
        if pixmap is not None and not pixmap.isNull():
            scaled_pixmap = pixmap.scaled(
                self.ui.image_display.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.ui.image_display.setPixmap(scaled_pixmap)

    def update_patient_id_label(self):
        if 0 <= self.current_patient_index < len(self.patient_records):
            patient_id = self.patient_records[self.current_patient_index][0]
            self.ui.patient_id_label.setText(f"患者ID：{patient_id}")
        else:
            self.ui.patient_id_label.setText("患者ID：--")

    def show_previous_patient(self):
        if self.patient_images:
            self.current_patient_index = (self.current_patient_index - 1) % len(self.patient_images)
            self.display_patient_image(self.patient_images[self.current_patient_index])
            self.show_plot()
            self.update_patient_id_label()

    def show_next_patient(self):
        if self.patient_images:
            self.current_patient_index = (self.current_patient_index + 1) % len(self.patient_images)
            self.display_patient_image(self.patient_images[self.current_patient_index])
            self.show_plot()
            self.update_patient_id_label()

    def display_patient_image(self, image_file):
        try:
            pixmap = QPixmap(image_file)
            if pixmap.isNull():
                raise ValueError("无法加载图像")

            scaled_pixmap = pixmap.scaled(
            self.ui.image_display.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
            )
            self.ui.image_display.setPixmap(scaled_pixmap)
            self.ui.image_display.resize(scaled_pixmap.width(), scaled_pixmap.height())
            self.statusBar().showMessage("图像显示成功")
            self.update_patient_id_label()
        except Exception as e:
            self.statusBar().showMessage(f"错误: {str(e)}")
            self.ui.image_display.clear()
            self.ui.image_display.setText("图像显示区域")

    def scale_images(self, pixmap1, pixmap2, target_size):
        scaled_pixmap1 = pixmap1.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        scaled_pixmap2 = pixmap2.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        return scaled_pixmap1, scaled_pixmap2


if __name__ == "__main__":
    app = QApplication(sys.argv)  # 创建 QApplication 实例
    window = MainWindow()         # 创建主窗口实例
    window.show()                 # 显示窗口
    sys.exit(app.exec_())         # 启动事件循环