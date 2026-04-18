# -*- coding: utf-8 -*-
"""
YOLO 训练页面 - 生成可运行的 train.py 文件
"""
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QSpinBox, QFileDialog,
                             QTextEdit, QMessageBox, QComboBox,
                             QCheckBox, QLineEdit, QScrollArea)
from PyQt6.QtCore import Qt
import os
import sys


class TrainPage(QWidget):
    """训练配置页面"""

    def __init__(self):
        super().__init__()
        self.widgets = {}
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        # 标题
        title = QLabel("🚀 YOLO 模型训练配置")
        title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px; color: #4ec9b0;")
        layout.addWidget(title)

        # ========== 左右分栏布局 ==========
        main_layout = QHBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(0, 10, 0, 10)

        # ========== 左侧：参数配置区域 ==========
        left_column = QVBoxLayout()
        left_column.setSpacing(12)

        # ========== 模型文件 ==========
        group_model = QWidget()
        model_layout = QVBoxLayout(group_model)
        model_layout.setContentsMargins(5, 5, 5, 5)
        model_layout.setSpacing(8)
        model_layout.addWidget(QLabel("<b>模型架构配置</b>"))

        model_layout.addWidget(QLabel("模型 YAML 文件:"))
        model_layout.addLayout(self.file_selector_widget("model_file"))

        self.pretrained_edit = QTextEdit()
        self.pretrained_edit.setMaximumHeight(30)
        self.pretrained_edit.setReadOnly(True)
        self.pretrained_edit.setStyleSheet("font-size: 13px; padding: 3px;")
        model_layout.addWidget(QLabel("预训练权重 (pt 文件，留空不加载):"))
        model_layout.addWidget(self.pretrained_edit)

        pre_btn = QPushButton("选择 pt 文件...")
        pre_btn.setMinimumHeight(28)
        pre_btn.setStyleSheet("""
            QPushButton {
                font-size: 12px;
                padding: 4px 12px;
                background-color: #3a3a3a;
                color: white;
                border: 1px solid #444;
                border-radius: 3px;
            }
            QPushButton:hover { background-color: #4a4a4a; border-color: #555; }
        """)
        pre_btn.clicked.connect(self.select_pretrained_weight)
        model_layout.addWidget(pre_btn)
        self.widgets['pretrained_weight'] = self.pretrained_edit

        left_column.addWidget(group_model)
        left_column.addWidget(QLabel(""))

        # ========== 数据集配置 ==========
        group_data = QWidget()
        data_layout = QVBoxLayout(group_data)
        data_layout.setContentsMargins(5, 5, 5, 5)
        data_layout.setSpacing(8)
        data_layout.addWidget(QLabel("<b>数据集配置</b>"))

        data_layout.addWidget(QLabel("数据配置文件 (data.yaml):"))
        data_layout.addLayout(self.file_selector_widget("data_yaml"))

        left_column.addWidget(group_data)
        left_column.addWidget(QLabel(""))

        # ========== 训练参数 ==========
        group_params = QWidget()
        params_layout = QVBoxLayout(group_params)
        params_layout.setContentsMargins(5, 5, 5, 5)
        params_layout.setSpacing(8)
        params_layout.addWidget(QLabel("<b>训练超参数</b>"))

        # imgsz
        params_layout.addWidget(QLabel("图像尺寸 (imgsz):"))
        imgsz_spin = QSpinBox()
        imgsz_spin.setRange(128, 1280)
        imgsz_spin.setValue(640)
        imgsz_spin.setMinimumWidth(150)
        imgsz_spin.setStyleSheet("font-size: 13px; padding: 3px;")
        params_layout.addWidget(imgsz_spin)
        self.widgets['imgsz'] = imgsz_spin

        # epochs
        params_layout.addWidget(QLabel("训练轮数 (epochs):"))
        epochs_spin = QSpinBox()
        epochs_spin.setRange(1, 1000)
        epochs_spin.setValue(50)
        epochs_spin.setMinimumWidth(150)
        epochs_spin.setStyleSheet("font-size: 13px; padding: 3px;")
        params_layout.addWidget(epochs_spin)
        self.widgets['epochs'] = epochs_spin

        # batch size
        params_layout.addWidget(QLabel("批次大小 (batch):"))
        batch_spin = QSpinBox()
        batch_spin.setRange(1, 64)
        batch_spin.setValue(4)
        batch_spin.setMinimumWidth(150)
        batch_spin.setStyleSheet("font-size: 13px; padding: 3px;")
        params_layout.addWidget(batch_spin)
        self.widgets['batch_size'] = batch_spin

        # workers
        params_layout.addWidget(QLabel("数据加载进程数 (workers):"))
        workers_spin = QSpinBox()
        workers_spin.setRange(0, 16)
        workers_spin.setValue(0)
        workers_spin.setMinimumWidth(150)
        workers_spin.setStyleSheet("font-size: 13px; padding: 3px;")
        params_layout.addWidget(workers_spin)
        self.widgets['workers'] = workers_spin

        # device
        params_layout.addWidget(QLabel("设备 (device):"))
        combo_device = QComboBox()
        combo_device.addItems(["0", "1", "0,1", "cuda", "cpu"])
        combo_device.setCurrentText("0")
        combo_device.setMinimumWidth(150)
        combo_device.setStyleSheet("font-size: 13px; padding: 3px;")
        params_layout.addWidget(combo_device)
        self.widgets['device'] = combo_device

        # optimizer
        params_layout.addWidget(QLabel("优化器 (optimizer):"))
        combo_opt = QComboBox()
        combo_opt.addItems(["SGD", "Adam", "AdamW", "RMSprop"])
        combo_opt.setCurrentText("SGD")
        combo_opt.setMinimumWidth(150)
        combo_opt.setStyleSheet("font-size: 13px; padding: 3px;")
        params_layout.addWidget(combo_opt)
        self.widgets['optimizer'] = combo_opt

        # close_mosaic
        params_layout.addWidget(QLabel("关闭 Mosaic 轮数 (close_mosaic):"))
        cm_spin = QSpinBox()
        cm_spin.setRange(0, 50)
        cm_spin.setValue(10)
        cm_spin.setMinimumWidth(150)
        cm_spin.setStyleSheet("font-size: 13px; padding: 3px;")
        params_layout.addWidget(cm_spin)
        self.widgets['close_mosaic'] = cm_spin

        # single_cls
        params_layout.addWidget(QLabel("单类训练 (single_cls):"))
        self.single_cls_cb = QCheckBox("True")
        self.single_cls_cb.setChecked(False)
        self.single_cls_cb.setStyleSheet("font-size: 13px;")
        params_layout.addWidget(self.single_cls_cb)
        self.widgets['single_cls'] = self.single_cls_cb

        left_column.addWidget(group_params)
        left_column.addWidget(QLabel(""))

        # ========== 输出配置 ==========
        group_output = QWidget()
        output_layout = QVBoxLayout(group_output)
        output_layout.setContentsMargins(5, 5, 5, 5)
        output_layout.setSpacing(8)
        output_layout.addWidget(QLabel("<b>输出配置</b>"))

        output_layout.addWidget(QLabel("项目保存路径 (project):"))
        output_layout.addLayout(self.file_selector_widget("project"))

        output_layout.addWidget(QLabel("实验名称 (name):"))
        name_edit = QLineEdit()
        name_edit.setText("exp")
        name_edit.setMinimumWidth(150)
        name_edit.setStyleSheet("font-size: 13px; padding: 3px;")
        output_layout.addWidget(name_edit)
        self.widgets['name'] = name_edit

        self.resume_cb = QCheckBox("继续上次训练 (resume=False)")
        self.resume_cb.setChecked(False)
        output_layout.addWidget(self.resume_cb)
        self.widgets['resume'] = self.resume_cb

        # amp
        self.amp_cb = QCheckBox("使用自动混合精度 (amp=False)")
        self.amp_cb.setChecked(False)
        output_layout.addWidget(self.amp_cb)
        self.widgets['amp'] = self.amp_cb

        left_column.addWidget(group_output)
        left_column.addWidget(QLabel(""))

        # ========== 控制按钮 ==========
        btn_layout = QHBoxLayout()

        self.generate_btn = QPushButton("📝 生成 train.py")
        self.generate_btn.setMinimumHeight(40)
        self.generate_btn.setMinimumWidth(150)
        self.generate_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
                padding: 8px;
            }
            QPushButton:hover { background-color: #1084d8; }
            QPushButton:disabled { background-color: #555; }
        """)
        self.generate_btn.clicked.connect(self.generate_train_file)
        btn_layout.addWidget(self.generate_btn)

        btn_layout.addStretch()
        left_column.addLayout(btn_layout)

        left_column.addStretch()
        left_column.setContentsMargins(10, 5, 10, 5)

        main_layout.addLayout(left_column, 1)

        # 右侧：参数说明区域 ==========
        right_container = QWidget()
        right_container.setFixedWidth(700)
        right_column = QVBoxLayout(right_container)
        right_column.setContentsMargins(15, 10, 15, 10)

        explanation_title = QLabel("📖 参数说明")
        explanation_title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 5px; color: #88bbff;")
        right_column.addWidget(explanation_title)

        # 分隔线
        separator = QLabel()
        separator.setFixedHeight(2)
        separator.setStyleSheet("background-color: #444;")
        right_column.addWidget(separator)

        # 参数说明文本框（只读）
        self.explanation_text = QTextEdit()
        self.explanation_text.setReadOnly(True)
        self.explanation_text.setMaximumHeight(350)
        self.explanation_text.setMinimumHeight(300)
        self.explanation_text.setStyleSheet("""
            QTextEdit {
                background-color: #252526;
                color: #ccc;
                border: 1px solid #3e3e42;
                border-radius: 5px;
                padding: 8px;
                font-size: 13px;
                line-height: 1.8;
            }
        """)

        self.update_explanation_text()
        right_column.addWidget(self.explanation_text)

        right_column.addStretch()
        main_layout.addWidget(right_container)

        # 将整个左右分栏布局包裹在 scroll_area 中
        content_widget = QWidget()
        content_widget_layout = QVBoxLayout(content_widget)
        content_widget_layout.setContentsMargins(0, 0, 0, 0)
        content_widget_layout.addLayout(main_layout)
        content_widget.setStyleSheet("background-color: transparent;")

        scroll_area = QScrollArea()
        scroll_area.setWidget(content_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet("border: none; background: transparent;")

        layout.addWidget(scroll_area)
        self.setLayout(layout)

    def update_explanation_text(self):
        """更新参数说明文本"""
        text = """
<div style="line-height: 1.8;">
<b style="color: #4ec9b0;">model 参数</b><br>模型配置文件路径（如 yolo11n.yaml），建议留空使用默认 YOLO 架构<br><i>默认：空</i><br><br>
<b style="color: #4ec9b0;">data 参数</b><br>数据集配置文件路径（data.yaml），包含训练集/验证集路径和类别信息<br><i>默认：data.yaml</i><br><br>
<b style="color: #4ec9b0;">imgsz 参数</b><br>输入图像尺寸，指定为正方形像素大小<br><i>默认：640</i><br><br>
<b style="color: #4ec9b0;">epochs 参数</b><br>训练的总轮数<br><i>默认：50</i><br><br>
<b style="color: #4ec9b0;">batch 参数</b><br>批次大小，显存越大可设置越大<br><i>默认：4</i><br><br>
<b style="color: #4ec9b0;">workers 参数</b><br>数据加载工作线程数，显存不足时设为 0<br><i>默认：0</i><br><br>
<b style="color: #4ec9b0;">device 参数</b><br>训练设备，留空自动选择可用 GPU/CPU<br><i>默认：空</i><br><br>
<b style="color: #4ec9b0;">optimizer 参数</b><br>优化器类型：<br>- SGD: 随机梯度下降（推荐）<br>- Adam: 自适应矩估计<br>- AdamW: Adam 的改进版<br>- RMSprop: RMS 传播算法<br><i>默认：SGD</i><br><br>
<b style="color: #4ec9b0;">close_mosaic 参数</b><br>训练最后多少轮关闭 Mosaic 数据增强<br><i>默认：10</i><br><br>
<b style="color: #4ec9b0;">resume 参数</b><br>是否从上次中断的训练状态继续<br><i>默认：False（从头开始训练）</i><br><br>
<b style="color: #4ec9b0;">project 参数</b><br>训练结果保存的项目目录<br><i>默认：runs/train</i><br><br>
<b style="color: #4ec9b0;">name 参数</b><br>实验名称，结果将保存在 project/name 目录下<br><i>默认：exp</i><br><br>
<b style="color: #4ec9b0;">single_cls 参数</b><br>是否将所有类别视为单一类别训练<br><i>默认：False（保留原类别）</i><br><br>
<b style="color: #4ec9b0;">amp 参数</b><br>是否开启自动混合精度训练（加速训练，需显卡支持 FP16）<br><i>默认：False（未勾选）；True（勾选）</i><br><br>
<b style="color: #4ec9b0;">预训练权重 (pt 文件)</b><br>预训练权重文件路径，留空表示不加载预训练权重<br><i>默认：空（不加载）</i><br>
</div>
"""
        self.explanation_text.setHtml(text)

    def file_selector_widget(self, key):
        """创建文件/文件夹选择器布局"""
        row = QHBoxLayout()
        row.setSpacing(5)
        edit = QTextEdit()
        edit.setMaximumHeight(30)
        edit.setReadOnly(True)
        edit.setStyleSheet("font-size: 13px; padding: 3px;")
        btn = QPushButton("浏览...")
        btn.setMinimumHeight(28)
        btn.setStyleSheet("""
            QPushButton {
                font-size: 12px;
                padding: 4px 12px;
                background-color: #3a3a3a;
                color: white;
                border: 1px solid #444;
                border-radius: 3px;
            }
            QPushButton:hover { background-color: #4a4a4a; border-color: #555; }
        """)
        btn.clicked.connect(lambda checked, k=key: self.select_file_or_folder(edit, k))
        row.addWidget(edit)
        row.addWidget(btn)
        self.widgets[key] = edit
        return row

    def select_file_or_folder(self, text_edit, key=None):
        """打开文件/文件夹选择对话框"""
        # model_file 和 data_yaml 选择文件，其他选择文件夹
        if key in ['model_file', 'data_yaml']:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "选择文件",
                "",
                "YAML Files (*.yaml *.yml);;All Files (*)"
            )
            if file_path:
                text_edit.setText(file_path)
        else:
            folder = QFileDialog.getExistingDirectory(self, "选择文件夹")
            if folder:
                text_edit.setText(folder)

    def select_pretrained_weight(self):
        """选择预训练权重 pt 文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择预训练权重文件",
            "",
            "PT Files (*.pt);;All Files (*)"
        )
        if file_path:
            self.pretrained_edit.setText(file_path)

    def get_config(self):
        """获取当前配置"""
        pretrained_path = self.pretrained_edit.toPlainText().strip()
        device_value = self.widgets['device'].currentText() or '0'
        return {
            'model_file': self.widgets['model_file'].toPlainText().strip() or r'D:\code\ultralytics-main\ultralytics\cfg\models\11\yolo11.yaml',
            'pretrained_weight': pretrained_path,
            'data_yaml': self.widgets['data_yaml'].toPlainText().strip() or 'data.yaml',
            'imgsz': self.widgets['imgsz'].value(),
            'epochs': self.widgets['epochs'].value(),
            'batch_size': self.widgets['batch_size'].value(),
            'workers': self.widgets['workers'].value(),
            'device': device_value,
            'optimizer': self.widgets['optimizer'].currentText(),
            'close_mosaic': self.widgets['close_mosaic'].value(),
            'single_cls': self.widgets['single_cls'].isChecked(),
            'project': self.widgets['project'].toPlainText().strip() or 'runs/train',
            'name': self.widgets['name'].text() or 'exp',
            'resume': self.widgets['resume'].isChecked(),
            'amp': self.widgets['amp'].isChecked()
        }

    def generate_train_file(self):
        """生成 train.py 文件并保存到指定位置"""
        config = self.get_config()

        # 选择保存位置
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存 train.py 文件",
            os.path.join(os.getcwd(), "train.py"),
            "Python Files (*.py);;All Files (*)"
        )

        if not save_path:
            return

        if not save_path.endswith('.py'):
            save_path += '.py'

        # 构建 Python 代码内容
        code_lines = []
        code_lines.append('# -*- coding: utf-8 -*-')
        code_lines.append('"""')
        code_lines.append('@Auth : 挂科边缘')
        code_lines.append('@File : train.py')
        code_lines.append('@IDE : PyCharm')
        code_lines.append('@Motto:学习新思想，争做新青年')
        code_lines.append('@Email : 179958974@qq.com')
        code_lines.append('"""')
        code_lines.append('import warnings')
        code_lines.append("warnings.filterwarnings('ignore')")
        code_lines.append('from ultralytics import YOLO')
        code_lines.append('')
        code_lines.append('')
        code_lines.append("if __name__ == '__main__':")
        code_lines.append(f'    model = YOLO(model=r"{config["model_file"]}")')

        if config['pretrained_weight']:
            code_lines.append(f"    model.load('{config['pretrained_weight']}') # 加载预训练权重")

        code_lines.append("    model.train(")
        code_lines.append(f'        data=r"{config["data_yaml"]}",')
        code_lines.append(f'        imgsz={config["imgsz"]},')
        code_lines.append(f'        epochs={config["epochs"]},')
        code_lines.append(f'        batch={config["batch_size"]},')
        code_lines.append(f'        workers={config["workers"]},')
        code_lines.append(f'        device={config["device"]},')
        code_lines.append(f'        optimizer="{config["optimizer"]}",')
        code_lines.append(f'        close_mosaic={config["close_mosaic"]},')
        code_lines.append('        resume=' + str(config["resume"]).capitalize() + ',')
        amp_value = 'True' if config['amp'] else 'False'
        code_lines.append('        amp=' + amp_value + ',')
        code_lines.append(f'        project=r"{config["project"]}",')
        code_lines.append(f'        name="{config["name"]}",')
        code_lines.append('        single_cls=' + str(config["single_cls"]).capitalize() + ',')
        code_lines.append('        cache=False,')
        code_lines.append('    )')

        code_content = '\n'.join(code_lines) + '\n'

        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(code_content)

            QMessageBox.information(
                self,
                "成功",
                f"✅ train.py 已保存到:\n{save_path}"
            )
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败:\n{str(e)}")
