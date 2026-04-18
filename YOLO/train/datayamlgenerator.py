# -*- coding: utf-8 -*-
"""
YAML 配置文件生成页面
"""
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QTextEdit, QFileDialog, QMessageBox,
                             QLineEdit, QSpinBox)
from PyQt6.QtCore import Qt
import os
from dataset_utils import scan_dataset_stats, generate_yaml_content


class DataYamlGeneratorPage(QWidget):
    """YAML 配置文件生成页面"""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # 标题
        title = QLabel("📄 data.yaml 生成器")
        title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)

        # 数据集目录选择
        dataset_layout = QHBoxLayout()
        dataset_layout.addWidget(QLabel("数据集目录:"))
        self.dataset_path_edit = QTextEdit()
        self.dataset_path_edit.setMaximumHeight(30)
        self.dataset_path_edit.setReadOnly(True)
        dataset_layout.addWidget(self.dataset_path_edit)
        self.dataset_path_btn = QPushButton("选择文件夹")
        dataset_layout.addWidget(self.dataset_path_btn)
        layout.addLayout(dataset_layout)

        # 自动扫描按钮
        self.scan_btn = QPushButton("🔍 扫描数据集信息")
        self.scan_btn.setMinimumHeight(35)
        self.scan_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #1084d8;
            }
        """)
        self.scan_btn.clicked.connect(self.scan_dataset_info)
        layout.addWidget(self.scan_btn)

        # 分隔线
        divider = QLabel("=" * 50)
        layout.addWidget(divider)

        # 自动填充区域
        auto_info_layout = QHBoxLayout()
        auto_info_layout.addWidget(QLabel("自动填充配置:"))
        layout.addLayout(auto_info_layout)

        # 训练集路径
        train_layout = QHBoxLayout()
        train_layout.addWidget(QLabel("训练集路径:"))
        self.train_path_edit = QLineEdit()
        self.train_path_edit.setReadOnly(True)
        train_layout.addWidget(self.train_path_edit)
        layout.addLayout(train_layout)

        # 验证集路径
        val_layout = QHBoxLayout()
        val_layout.addWidget(QLabel("验证集路径:"))
        self.val_path_edit = QLineEdit()
        self.val_path_edit.setReadOnly(True)
        val_layout.addWidget(self.val_path_edit)
        layout.addLayout(val_layout)

        # nc 和类别名
        meta_layout = QHBoxLayout()

        # 类别数量
        meta_layout.addWidget(QLabel("类别数量 (nc):"))
        self.nc_spin = QSpinBox()
        self.nc_spin.setRange(1, 100)
        self.nc_spin.setValue(4)
        meta_layout.addWidget(self.nc_spin)

        # 类别名称
        meta_layout.addWidget(QLabel("类别名称:"))
        self.names_edit = QLineEdit()
        self.names_edit.setPlaceholderText("例如：recyclable waste,hazardous waste,kitchen waste,other waste")
        meta_layout.addWidget(self.names_edit)

        layout.addLayout(meta_layout)

        # 分隔线
        divider2 = QLabel("=" * 50)
        layout.addWidget(divider2)

        # YAML 预览区
        preview_label = QLabel("YAML 预览:")
        preview_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(preview_label)

        self.yaml_preview = QTextEdit()
        self.yaml_preview.setReadOnly(True)
        self.yaml_preview.setMaximumHeight(300)
        self.yaml_preview.setStyleSheet("background-color: #1a1a1a; color: #ddd; border-radius: 5px; padding: 5px; font-family: Consolas, Monaco, monospace;")
        layout.addWidget(self.yaml_preview)

        # 操作按钮区
        btn_layout = QHBoxLayout()

        self.refresh_btn = QPushButton("🔄 刷新预览")
        self.refresh_btn.setMinimumHeight(40)
        self.refresh_btn.clicked.connect(self.update_preview)
        btn_layout.addWidget(self.refresh_btn)

        self.save_btn = QPushButton("💾 保存 YAML 文件")
        self.save_btn.setMinimumHeight(40)
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff9800;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #f57c00;
            }
        """)
        self.save_btn.clicked.connect(self.save_yaml)
        btn_layout.addWidget(self.save_btn)

        layout.addLayout(btn_layout)

        self.setLayout(layout)

        # 连接信号
        self.dataset_path_btn.clicked.connect(self.select_dataset_folder)

    def select_dataset_folder(self):
        """选择数据集文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择数据集目录")
        if folder:
            self.dataset_path_edit.setText(folder)
            self.scan_dataset_info()

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

    def scan_dataset_info(self):
        """扫描数据集信息并自动填充"""
        dataset_path = self.dataset_path_edit.toPlainText().strip()

        if not dataset_path or not os.path.isdir(dataset_path):
            QMessageBox.warning(self, "警告", "请选择有效的数据集目录！")
            return

        try:
            stats = scan_dataset_stats(dataset_path)

            # 设置路径
            img_base = os.path.join(dataset_path, "images")
            self.train_path_edit.setText(os.path.join(img_base, "train"))
            self.val_path_edit.setText(os.path.join(img_base, "val"))

            # 设置类别数量
            class_count = len(stats['classes']) if stats['classes'] else 0
            self.nc_spin.setValue(class_count if class_count > 0 else 4)

            # 尝试从 labels/classes.txt 读取类别名称
            labels_dir = os.path.join(dataset_path, "labels")
            auto_classes = self.read_classes_from_txt(labels_dir)

            if auto_classes:
                self.names_edit.setText(", ".join(auto_classes))
                self.log(f"已从 classes.txt 读取 {len(auto_classes)} 个类别")
            elif stats['classes']:
                self.names_edit.setText(", ".join(stats['classes']))
                self.log(f"已识别 {class_count} 个类别")

            self.log(f"训练集图片数：{stats['train_count']}")
            self.log(f"验证集图片数：{stats['val_count']}")

            # 更新预览
            self.update_preview()

        except Exception as e:
            QMessageBox.critical(self, "错误", f"扫描失败:\n{str(e)}")

    def log(self, message):
        """临时日志输出到状态栏位置"""
        pass  # 可以扩展为在界面上显示日志

    def update_preview(self):
        """更新 YAML 预览"""
        train_path = self.train_path_edit.text().strip()
        val_path = self.val_path_edit.text().strip()
        nc = self.nc_spin.value()
        names_str = self.names_edit.text().strip()

        # 解析类别名称列表
        names = [n.strip() for n in names_str.split(',') if n.strip()] if names_str else []

        if not train_path or not val_path:
            self.yaml_preview.setText("# 请先设置数据集目录")
            return

        # 转换为 Windows 风格路径（反斜杠）
        train_path = train_path.replace('/', '\\')
        val_path = val_path.replace('/', '\\')

        yaml_content = generate_yaml_content(train_path, val_path, nc, names)
        self.yaml_preview.setText(yaml_content)

    def save_yaml(self):
        """保存 YAML 文件"""
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存 YAML 文件",
            "",
            "YAML Files (*.yaml);;All Files (*)"
        )

        if not output_path:
            return

        if not output_path.endswith('.yaml'):
            output_path += '.yaml'

        train_path = self.train_path_edit.text().strip()
        val_path = self.val_path_edit.text().strip()
        nc = self.nc_spin.value()
        names_str = self.names_edit.text().strip()

        # 转换为 Windows 风格路径（反斜杠）
        train_path = train_path.replace('/', '\\')
        val_path = val_path.replace('/', '\\')

        names = [n.strip() for n in names_str.split(',') if n.strip()] if names_str else []

        yaml_content = generate_yaml_content(train_path, val_path, nc, names)

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(yaml_content)

            QMessageBox.information(
                self,
                "成功",
                f"YAML 文件已保存到:\n{output_path}"
            )

            self.log(f"已保存至：{output_path}")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败:\n{str(e)}")
