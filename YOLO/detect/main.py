#!/usr/bin/env python3
"""YOLO 目标检测工具 - 主程序入口."""

import sys
from pathlib import Path

# 确保当前项目目录在 Python 路径中
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt


# 设置高 DPI 缩放
Qt.AA_EnableHighDpiScaling = True
Qt.AA_UseHighDpiPixmaps = True

from ui.main_window import MainWindow


def main():
    """主函数."""
    # 创建应用
    app = QApplication(sys.argv)

    # 设置应用信息
    app.setApplicationName("YOLO 目标检测工具")
    app.setOrganizationName("YOLO Test")

    # 创建并显示主窗口
    window = MainWindow()
    window.show()

    # 运行应用
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
