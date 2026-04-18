# -*- coding: utf-8 -*-
"""
数据集分割页面
"""
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QDoubleSpinBox, QFileDialog,
                             QTextEdit, QProgressBar, QMessageBox, QLineEdit, QSpinBox)
from PyQt6.QtCore import Qt
import os
from dataset_utils import split_dataset, generate_yaml_content, scan_dataset_stats


class DatasetSplitterPage(QWidget):
    """数据集分割配置页面"""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # 标题
        title = QLabel("📁 数据集分割")
        title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)

        # 图片路径选择
        img_layout = QHBoxLayout()
        img_layout.addWidget(QLabel("图片路径:"))
        self.img_path_edit = QTextEdit()
        self.img_path_edit.setMaximumHeight(30)
        self.img_path_edit.setReadOnly(True)
        img_layout.addWidget(self.img_path_edit)
        self.img_path_btn = QPushButton("选择文件夹")
        img_layout.addWidget(self.img_path_btn)
        layout.addLayout(img_layout)

        # 标签路径选择
        lbl_layout = QHBoxLayout()
        lbl_layout.addWidget(QLabel("标签路径:"))
        self.lbl_path_edit = QTextEdit()
        self.lbl_path_edit.setMaximumHeight(30)
        self.lbl_path_edit.setReadOnly(True)
        lbl_layout.addWidget(self.lbl_path_edit)
        self.lbl_path_btn = QPushButton("选择文件夹")
        lbl_layout.addWidget(self.lbl_path_btn)
        layout.addLayout(lbl_layout)

        # 输出目录选择
        out_layout = QHBoxLayout()
        out_layout.addWidget(QLabel("输出目录:"))
        self.out_path_edit = QTextEdit()
        self.out_path_edit.setMaximumHeight(30)
        self.out_path_edit.setReadOnly(True)
        out_layout.addWidget(self.out_path_edit)
        self.out_path_btn = QPushButton("选择文件夹")
        out_layout.addWidget(self.out_path_btn)
        layout.addLayout(out_layout)

        # 比例设置
        ratio_layout = QHBoxLayout()

        # 测试集比例
        ratio_layout.addWidget(QLabel("测试集比例:"))
        self.test_ratio_spin = QDoubleSpinBox()
        self.test_ratio_spin.setRange(0, 1)
        self.test_ratio_spin.setValue(0.2)
        self.test_ratio_spin.setSingleStep(0.05)
        ratio_layout.addWidget(self.test_ratio_spin)
        ratio_layout.addWidget(QLabel("(0-1)"))

        # 验证集比例
        ratio_layout.addWidget(QLabel("验证集比例:"))
        self.val_ratio_spin = QDoubleSpinBox()
        self.val_ratio_spin.setRange(0, 1)
        self.val_ratio_spin.setValue(0.1)
        self.val_ratio_spin.setSingleStep(0.05)
        ratio_layout.addWidget(self.val_ratio_spin)
        ratio_layout.addWidget(QLabel("(占剩余部分的比例，0-1)"))

        layout.addLayout(ratio_layout)

        # 执行按钮
        self.execute_btn = QPushButton("▶ 执行数据集分割")
        self.execute_btn.setMinimumHeight(40)
        self.execute_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #555;
            }
        """)
        self.execute_btn.clicked.connect(self.execute_split)
        layout.addWidget(self.execute_btn)

        # 进度条
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        # 日志输出区
        log_label = QLabel("执行日志:")
        log_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(log_label)

        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background-color: #1a1a1a; color: #ddd; border-radius: 5px; padding: 5px;")
        layout.addWidget(self.log_text)

        # YAML 生成区域分隔线
        yaml_title = QLabel("📄 data.yaml 生成器")
        yaml_title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px; margin-top: 15px;")
        layout.addWidget(yaml_title)

        # 类别数量
        nc_layout = QHBoxLayout()
        nc_layout.addWidget(QLabel("类别数量 (nc):"))
        self.nc_spin = QSpinBox()
        self.nc_spin.setRange(1, 100)
        self.nc_spin.setValue(4)
        self.nc_spin.valueChanged.connect(self.update_yaml_preview)
        nc_layout.addWidget(self.nc_spin)

        # 类别名称
        names_layout = QHBoxLayout()
        names_layout.addWidget(QLabel("类别名称:"))
        self.names_edit = QLineEdit()
        self.names_edit.setPlaceholderText("例如：recyclable waste,hazardous waste,kitchen waste,other waste")
        self.names_edit.textChanged.connect(self.update_yaml_preview)
        names_layout.addWidget(self.names_edit)
        layout.addLayout(names_layout)

        # YAML 预览标签
        preview_label = QLabel("YAML 预览:")
        preview_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(preview_label)

        # YAML 预览区
        self.yaml_preview = QTextEdit()
        self.yaml_preview.setReadOnly(True)
        self.yaml_preview.setMaximumHeight(250)
        self.yaml_preview.setStyleSheet("background-color: #1a1a1a; color: #ddd; border-radius: 5px; padding: 5px; font-family: Consolas, Monaco, monospace;")
        layout.addWidget(self.yaml_preview)

        # 操作按钮区
        btn_layout = QHBoxLayout()

        self.refresh_yaml_btn = QPushButton("🔄 刷新 YAML 预览")
        self.refresh_yaml_btn.setMinimumHeight(35)
        self.refresh_yaml_btn.clicked.connect(self.update_yaml_preview)
        btn_layout.addWidget(self.refresh_yaml_btn)

        self.save_yaml_btn = QPushButton("💾 保存 YAML 文件")
        self.save_yaml_btn.setMinimumHeight(35)
        self.save_yaml_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff9800;
                color: white;
                font-size: 13px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #f57c00;
            }
        """)
        self.save_yaml_btn.clicked.connect(self.save_yaml_file)
        btn_layout.addWidget(self.save_yaml_btn)

        layout.addLayout(btn_layout)

        self.setLayout(layout)

        # 连接信号
        self.img_path_btn.clicked.connect(lambda: self.select_folder(self.img_path_edit))
        self.lbl_path_btn.clicked.connect(lambda: self.select_folder(self.lbl_path_edit))
        self.out_path_btn.clicked.connect(lambda: self.select_folder(self.out_path_edit))

    def select_folder(self, text_edit):
        """打开文件夹选择对话框"""
        folder = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if folder:
            text_edit.setText(folder)

    def log(self, message):
        """添加日志信息"""
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def read_classes_from_txt(self, labels_dir):
        """从 labels/classes.txt 读取类别名称列表"""
        classes_file = os.path.join(labels_dir, 'classes.txt')
        classes = []
        if os.path.exists(classes_file):
            with open(classes_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        classes.append(line)
        return classes

    def get_dataset_base_path(self):
        """获取输出数据集的根目录（包含 images/labels 子目录）"""
        return self.out_path_edit.toPlainText().strip()

    def update_yaml_preview(self):
        """根据当前配置更新 YAML 预览"""
        output_dir = self.get_dataset_base_path()

        if not output_dir:
            self.yaml_preview.setText("# 请先选择输出目录")
            return

        # 自动填充路径
        train_path = os.path.join(output_dir, "images", "train").replace('/', '\\')
        val_path = os.path.join(output_dir, "images", "val").replace('/', '\\')

        nc = self.nc_spin.value()
        names_str = self.names_edit.text().strip()

        # 解析类别名称列表
        names = [n.strip() for n in names_str.split(',') if n.strip()] if names_str else []

        yaml_content = generate_yaml_content(train_path, val_path, nc, names, auto_path=False)
        self.yaml_preview.setText(yaml_content)

    def save_yaml_file(self):
        """用户选择位置保存 YAML 文件"""
        output_dir = self.get_dataset_base_path()

        # 先从源标签目录的 classes.txt 读取类别名用于填充
        imgpath = self.img_path_edit.toPlainText().strip()
        txtpath = self.lbl_path_edit.toPlainText().strip()
        auto_classes = self.read_classes_from_txt(txtpath)

        if auto_classes:
            self.nc_spin.setValue(len(auto_classes))
            self.names_edit.setText(", ".join(auto_classes))
            self.update_yaml_preview()

        # 用户选择保存位置
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存 YAML 文件",
            "",
            "YAML Files (*.yaml);;All Files (*)"
        )

        if not save_path:
            return

        if not save_path.endswith('.yaml'):
            save_path += '.yaml'

        train_path = os.path.join(output_dir, "images", "train").replace('/', '\\') if output_dir else ""
        val_path = os.path.join(output_dir, "images", "val").replace('/', '\\') if output_dir else ""

        nc = self.nc_spin.value()
        names_str = self.names_edit.text().strip()
        names = [n.strip() for n in names_str.split(',') if n.strip()] if names_str else []

        yaml_content = generate_yaml_content(train_path, val_path, nc, names, auto_path=False)

        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(yaml_content)

            QMessageBox.information(
                self,
                "成功",
                f"YAML 文件已保存到:\n{save_path}"
            )

            self.log(f"✅ YAML 文件已保存到：{save_path}")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败:\n{str(e)}")

    def execute_split(self):
        """执行数据集分割"""
        imgpath = self.img_path_edit.toPlainText().strip()
        txtpath = self.lbl_path_edit.toPlainText().strip()
        output_dir = self.out_path_edit.toPlainText().strip()

        # 验证输入
        if not imgpath or not os.path.isdir(imgpath):
            QMessageBox.warning(self, "警告", "请选择有效的图片路径目录！")
            return

        if not txtpath or not os.path.isdir(txtpath):
            QMessageBox.warning(self, "警告", "请选择有效的标签路径目录！")
            return

        if not output_dir:
            QMessageBox.warning(self, "警告", "请选择输出目录！")
            return

        test_ratio = self.test_ratio_spin.value()
        val_ratio = self.val_ratio_spin.value()

        # 禁用按钮
        self.execute_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.log_text.clear()
        self.log(f"开始处理...")
        self.log(f"图片路径：{imgpath}")
        self.log(f"标签路径：{txtpath}")
        self.log(f"输出目录：{output_dir}")
        self.log(f"测试集比例：{test_ratio}, 验证集比例：{val_ratio}")

        try:
            result = split_dataset(imgpath, txtpath, output_dir, test_ratio, val_ratio)

            self.progress.setValue(100)
            self.log("-" * 40)
            self.log("✅ 分割完成！")
            self.log(f"训练集：{result['train_count']} 张图片")
            self.log(f"验证集：{result['val_count']} 张图片")
            self.log(f"测试集：{result['test_count']} 张图片")
            self.log(f"总计：{result['total_files']} 张图片")

            if result['missing_count'] > 0:
                self.log(f"⚠️ 有 {result['missing_count']} 个标签文件未找到对应图片")

            # 从源标签目录的 classes.txt 读取类别名称并自动填充
            auto_classes = self.read_classes_from_txt(txtpath)
            if auto_classes:
                self.nc_spin.setValue(len(auto_classes))
                self.names_edit.setText(", ".join(auto_classes))
                self.log(f"已从标签目录读取 {len(auto_classes)} 个类别")
                self.update_yaml_preview()
            else:
                self.log("提示：请在标签目录中添加 classes.txt 文件以自动生成类别名称")

            QMessageBox.information(self, "成功",
                f"数据集分割完成！\n\n训练集：{result['train_count']} 张\n"
                f"验证集：{result['val_count']} 张\n测试集：{result['test_count']} 张\n\n"
                f"输出目录：{output_dir}")

            # 分割成功后自动更新 YAML 预览
            self.update_yaml_preview()

        except Exception as e:
            self.log(f"❌ 错误：{str(e)}")
            QMessageBox.critical(self, "错误", f"分割过程中出现错误:\n{str(e)}")

        finally:
            self.execute_btn.setEnabled(True)
            self.progress.setVisible(False)
