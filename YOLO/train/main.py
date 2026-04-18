# -*- coding: utf-8 -*-
"""
YOLO 数据集管理器 - 主程序入口
功能：数据集分割配置 + data.yaml 生成器 + 模型训练
"""
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QTabWidget, QLabel, QHBoxLayout, QWidget, QVBoxLayout, QSizePolicy
from PyQt6.QtCore import Qt
from datasetsplitterpage import DatasetSplitterPage
from trainpage import TrainPage


class YOLODataManagerWindow(QMainWindow):
    """主窗口类"""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("YOLO 数据集管理器 v1.0")
        self.setGeometry(100, 100, 1800, 700)

        # 中央容器
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # 左侧边栏 - 欢迎信息
        sidebar = QWidget()
        sidebar.setFixedWidth(200)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        logo_label = QLabel("🎯")
        logo_label.setStyleSheet("font-size: 48px;")
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sidebar_layout.addWidget(logo_label)

        title_label = QLabel("YOLO 数据集\n管理器")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; text-align: center; padding: 10px;")
        sidebar_layout.addWidget(title_label)

        info_label = QLabel("""用于 YOLO 目标检测的数据集管理工具，支持数据集分割和 YAML 配置生成。""")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("font-size: 12px; color: #666; text-align: center; padding: 10px;")
        sidebar_layout.addWidget(info_label)

        sidebar_layout.addStretch()
        main_layout.addWidget(sidebar)

        # 右侧内容区 - 标签页
        self.tabs = QTabWidget()
        self.tabs.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )

        # 添加两个页面
        self.splitter_page = DatasetSplitterPage()
        self.train_page = TrainPage()

        self.tabs.addTab(self.splitter_page, "📁 数据集分割")
        self.tabs.addTab(self.train_page, "🚀 模型训练")

        main_layout.addWidget(self.tabs)

        # 设置样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QTabWidget::pane {
                border: 1px solid #333;
                border-radius: 5px;
                background-color: #2d2d2d;
            }
            QTabBar::tab {
                background-color: #3a3a3a;
                color: #ddd;
                padding: 12px 24px;
                margin-right: 2px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
            QTabBar::tab:selected {
                background-color: #2d2d2d;
                color: white;
            }
            QTabBar::tab:hover:!selected {
                background-color: #444;
            }
            QWidget {
                background-color: #2d2d2d;
                color: #fff;
            }
            QTextEdit, QLineEdit {
                background-color: #1e1e1e;
                color: #fff;
                border: 1px solid #444;
                border-radius: 3px;
                padding: 4px;
            }
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                border-radius: 3px;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #1084d8;
            }
            QLabel {
                color: #eee;
            }
        """)


if __name__ == '__main__':
    from PyQt6.QtWidgets import QApplication as _QApplication
    import sys

    app = _QApplication(sys.argv)

    # 设置应用程序字体
    font = app.font()
    font.setPointSize(10)
    app.setFont(font)

    window = YOLODataManagerWindow()
    window.show()

    sys.exit(app.exec())
