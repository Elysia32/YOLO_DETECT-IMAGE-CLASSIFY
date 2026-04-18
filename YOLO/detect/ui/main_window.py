"""主窗口模块."""

import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QComboBox, QMessageBox, QFrame, QTabWidget,
    QListWidget, QListWidgetItem, QGroupBox, QLineEdit, QSpinBox
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QImage, QPixmap

try:
    from ..core import YOLODetector, CameraSource
except ImportError:
    from core.yolo_detector import YOLODetector
    from core.camera_source import CameraSource


class Worker(QThread):
    """后台处理线程."""

    finished = pyqtSignal(object)  # 发送处理结果
    progress = pyqtSignal(int)     # 发送进度

    def __init__(self, detector: YOLODetector, frame: np.ndarray):
        super().__init__()
        self.detector = detector
        self.frame = frame

    def run(self):
        """执行检测任务."""
        result, detections = self.detector.detect_camera_stream(self.frame)
        self.finished.emit((result, detections))


class MainWindow(QMainWindow):
    """主窗口类."""

    def __init__(self):
        super().__init__()

        # 程序根目录
        self.root_dir = Path(__file__).parent.parent

        # 初始化变量
        self.detector: Optional[YOLODetector] = None
        self.camera_source: Optional[CameraSource] = None
        self.worker: Optional[Worker] = None
        self.current_mode = "image"  # image, video, camera

        # 检测结果存储
        self.image_detections: list = []
        self.video_results: Dict = {}

        # 配置参数
        self.conf_threshold = 0.25
        self.iou_threshold = 0.7

        self._setup_ui()
        # 不自动加载模型，等待用户手动选择

    def _setup_ui(self):
        """设置用户界面."""
        self.setWindowTitle("YOLO 目标检测工具")
        self.setMinimumSize(1200, 800)

        # 中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # 顶部：模型选择和参数配置
        top_section = self._create_top_section()
        main_layout.addWidget(top_section)

        # 选项卡控件
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # 添加三个选项卡
        self.image_tab = self._create_image_tab()
        self.video_tab = self._create_video_tab()
        self.camera_tab = self._create_camera_tab()

        self.tab_widget.addTab(self.image_tab, "图片检测")
        self.tab_widget.addTab(self.video_tab, "视频检测")
        self.tab_widget.addTab(self.camera_tab, "实时检测")

        # 状态栏
        self.statusBar().showMessage("就绪")

    def _create_top_section(self) -> QGroupBox:
        """创建顶部控件组."""
        group = QGroupBox("模型配置")
        layout = QHBoxLayout()

        # 模型路径显示
        model_label = QLabel("模型:")
        layout.addWidget(model_label)

        self.model_path_edit = QLineEdit("请先选择模型文件")
        self.model_path_edit.setReadOnly(True)
        self.model_path_edit.setFixedWidth(300)
        self.model_path_edit.setStyleSheet("background-color: #ffcccc;")
        layout.addWidget(self.model_path_edit)

        # 浏览按钮
        browse_btn = QPushButton("选择模型...")
        browse_btn.clicked.connect(self._browse_model)
        layout.addWidget(browse_btn)

        # 置信度阈值
        conf_label = QLabel("置信度:")
        layout.addWidget(conf_label)

        self.conf_spin = QSpinBox()
        self.conf_spin.setRange(0, 100)
        self.conf_spin.setValue(25)
        self.conf_spin.setSuffix("%")
        self.conf_spin.valueChanged.connect(self._update_conf_threshold)
        layout.addWidget(self.conf_spin)

        group.setLayout(layout)
        return group

    def _create_image_tab(self) -> QWidget:
        """创建图片检测选项卡."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # 控制区
        control_hbox = QHBoxLayout()

        self.image_upload_btn = QPushButton("上传图片")
        self.image_upload_btn.clicked.connect(self._upload_image)
        self.image_upload_btn.setStyleSheet("font-size: 14px; padding: 8px;")
        control_hbox.addWidget(self.image_upload_btn)

        self.detect_image_btn = QPushButton("开始检测")
        self.detect_image_btn.clicked.connect(self._detect_image)
        self.detect_image_btn.setEnabled(False)
        self.detect_image_btn.setStyleSheet("font-size: 14px; padding: 8px; background-color: #4CAF50; color: white;")
        control_hbox.addWidget(self.detect_image_btn)

        self.reset_image_btn = QPushButton("重置")
        self.reset_image_btn.clicked.connect(self._reset_image_tab)
        self.reset_image_btn.setEnabled(False)
        control_hbox.addWidget(self.reset_image_btn)

        control_hbox.addStretch()
        layout.addLayout(control_hbox)

        # 结果显示区（分两列）
        result_layout = QHBoxLayout()

        # 左侧：图像显示
        image_frame = QFrame()
        image_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        image_layout = QVBoxLayout(image_frame)
        image_layout.addWidget(QLabel("检测结果"))

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(600, 400)
        self.image_label.setStyleSheet("background-color: #f0f0f0;")
        image_layout.addWidget(self.image_label)
        result_layout.addWidget(image_frame, 2)

        # 右侧：检测结果列表
        det_frame = QFrame()
        det_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        det_layout = QVBoxLayout(det_frame)
        det_layout.addWidget(QLabel("检测对象"))

        self.detection_list = QListWidget()
        self.detection_list.setMinimumHeight(400)
        det_layout.addWidget(self.detection_list)

        result_layout.addWidget(det_frame, 1)

        layout.addLayout(result_layout)
        widget.setLayout(layout)

        return widget

    def _create_video_tab(self) -> QWidget:
        """创建视频检测选项卡."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # 控制区
        control_hbox = QHBoxLayout()

        self.video_upload_btn = QPushButton("上传视频")
        self.video_upload_btn.clicked.connect(self._upload_video)
        self.video_upload_btn.setStyleSheet("font-size: 14px; padding: 8px;")
        control_hbox.addWidget(self.video_upload_btn)

        self.process_video_btn = QPushButton("处理视频")
        self.process_video_btn.clicked.connect(self._process_video)
        self.process_video_btn.setEnabled(False)
        self.process_video_btn.setStyleSheet("font-size: 14px; padding: 8px; background-color: #2196F3; color: white;")
        control_hbox.addWidget(self.process_video_btn)

        self.reset_video_btn = QPushButton("重置")
        self.reset_video_btn.clicked.connect(self._reset_video_tab)
        self.reset_video_btn.setEnabled(False)
        control_hbox.addWidget(self.reset_video_btn)

        control_hbox.addStretch()
        layout.addLayout(control_hbox)

        # 结果区
        result_layout = QVBoxLayout()

        # 预览窗口
        preview_frame = QFrame()
        preview_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.addWidget(QLabel("视频预览/处理结果"))

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(600, 400)
        self.video_label.setStyleSheet("background-color: #f0f0f0;")
        preview_layout.addWidget(self.video_label)
        result_layout.addWidget(preview_frame)

        # 统计信息
        stats_layout = QVBoxLayout()
        stats_layout.addWidget(QLabel("统计结果"))

        self.stats_text = QLabel("")
        self.stats_text.setWordWrap(True)
        self.stats_text.setStyleSheet("font-family: Monospace; background-color: #fafafa; padding: 10px;")
        stats_layout.addWidget(self.stats_text)

        result_layout.addLayout(stats_layout)
        layout.addLayout(result_layout)

        widget.setLayout(layout)

        return widget

    def _create_camera_tab(self) -> QWidget:
        """创建实时检测选项卡."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # 控制区
        control_hbox = QHBoxLayout()

        self.camera_open_btn = QPushButton("打开摄像头")
        self.camera_open_btn.clicked.connect(self._open_camera)
        self.camera_open_btn.setStyleSheet("font-size: 14px; padding: 8px; background-color: #9C27B0; color: white;")
        control_hbox.addWidget(self.camera_open_btn)

        self.camera_close_btn = QPushButton("关闭摄像头")
        self.camera_close_btn.clicked.connect(self._close_camera)
        self.camera_close_btn.setEnabled(False)
        control_hbox.addWidget(self.camera_close_btn)

        self.take_snapshot_btn = QPushButton("截图保存")
        self.take_snapshot_btn.clicked.connect(self._take_snapshot)
        self.take_snapshot_btn.setEnabled(False)
        control_hbox.addWidget(self.take_snapshot_btn)

        control_hbox.addStretch()
        layout.addLayout(control_hbox)

        # 摄像头画面显示
        cam_frame = QFrame()
        cam_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        cam_layout = QVBoxLayout(cam_frame)
        cam_layout.addWidget(QLabel("实时检测画面"))

        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet("background-color: #000; border-radius: 5px;")
        cam_layout.addWidget(self.camera_label)

        layout.addWidget(cam_frame)

        # 状态信息
        status_layout = QHBoxLayout()
        self.camera_status_label = QLabel("状态：未连接")
        self.camera_fps_label = QLabel("FPS: 0")
        status_layout.addWidget(self.camera_status_label)
        status_layout.addWidget(self.camera_fps_label)
        status_layout.addStretch()
        layout.addLayout(status_layout)

        widget.setLayout(layout)

        return widget

    def _browse_model(self):
        """浏览选择模型文件."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 YOLO 模型文件 (.pt)",
            "",
            "YOLO Models (*.pt)"
        )

        if file_path:
            try:
                self.setStatusMessage(f"正在加载模型：{file_path}...")
                self.detector = YOLODetector(file_path)
                self.setModelPath(file_path)
                self.setStatusMessage("就绪 - 已加载模型")
                QMessageBox.information(self, "成功", f"模型加载成功！\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"模型加载失败：{str(e)}")
                self.setStatusMessage("就绪")

    def _update_conf_threshold(self, value: int):
        """更新置信度阈值."""
        self.conf_threshold = value / 100.0

    def _upload_image(self):
        """上传图片文件."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片文件",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif *.tiff)"
        )

        if file_path:
            self.current_image_path = file_path
            self.detect_image_btn.setEnabled(True)
            self.reset_image_btn.setEnabled(True)
            self.setStatusMessage(f"已加载图片：{file_path}")

    def _detect_image(self):
        """执行图片检测."""
        if not hasattr(self, 'current_image_path'):
            return

        if self.detector is None:
            QMessageBox.warning(self, "警告", "请先加载模型!")
            return

        try:
            self.setStatusMessage("正在检测...")

            annotated_image, detections = self.detector.detect_image(
                self.current_image_path,
                conf_threshold=self.conf_threshold,
                iou_threshold=self.iou_threshold
            )

            # 转换显示
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            height, width, channel = annotated_image.shape
            bytes_per_line = channel * width
            qt_image = QImage(annotated_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio
            ))

            # 显示检测结果
            self.detection_list.clear()
            for det in detections:
                item_text = f"{det['class_name']} ({det['confidence']:.2%}) - {det['bbox']}"
                item = QListWidgetItem(item_text)
                self.detection_list.addItem(item)

            self.image_detections = detections
            self.setStatusMessage(f"检测完成，发现 {len(detections)} 个目标")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"检测失败：{str(e)}")
            self.setStatusMessage("就绪")

    def _reset_image_tab(self):
        """重置图片检测选项卡."""
        self.image_label.clear()
        self.detection_list.clear()
        self.detect_image_btn.setEnabled(False)
        self.reset_image_btn.setEnabled(False)
        self.image_detections = []
        if hasattr(self, 'current_image_path'):
            delattr(self, 'current_image_path')
        self.setStatusMessage("已重置")

    def _upload_video(self):
        """上传视频文件."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频文件",
            "",
            "Videos (*.mp4 *.avi *.mov *.mkv)"
        )

        if file_path:
            self.current_video_path = file_path
            self.process_video_btn.setEnabled(True)
            self.reset_video_btn.setEnabled(True)
            self.setStatusMessage(f"已加载视频：{file_path}")

    def _process_video(self):
        """处理视频文件."""
        if not hasattr(self, 'current_video_path'):
            return

        if self.detector is None:
            QMessageBox.warning(self, "警告", "请先加载模型!")
            return

        try:
            # 询问是否保存处理后的视频
            save_result = QMessageBox.question(
                self,
                "保存选项",
                "是否保存处理后的视频？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            output_path = None
            if save_result == QMessageBox.StandardButton.Yes:
                output_path, _ = QFileDialog.getSaveFileName(
                    self,
                    "保存视频",
                    "output.mp4",
                    "MP4 Video (*.mp4)"
                )

            self.setStatusMessage("正在处理视频，请稍候...")
            self.process_video_btn.setEnabled(False)

            total_frames, detected_count, detection_stats = self.detector.detect_video(
                self.current_video_path,
                output_path=output_path,
                conf_threshold=self.conf_threshold,
                iou_threshold=self.iou_threshold
            )

            # 显示结果
            self.video_results = {
                "total_frames": total_frames,
                "detected_count": detected_count,
                "stats": detection_stats
            }

            result_text = f"视频总帧数：{total_frames}\n检测到目标：{detected_count}个\n\n检测统计:\n"
            for name, stats in detection_stats.items():
                result_text += f"\n{name}: {stats['count']}个 (平均置信度：{stats['avg_confidence']:.2%})"

            if not detection_stats:
                result_text += "\n\n未检测到目标"

            self.stats_text.setText(result_text)

            # 提示用户视频已保存（QPixmap 无法播放视频文件）
            if output_path and Path(output_path).exists():
                QMessageBox.information(
                    self,
                    "完成",
                    f"视频处理完成!\n已保存到：{output_path}\n\n请使用视频播放器查看结果。"
                )
            else:
                QMessageBox.information(self, "完成", f"视频处理完成!\n检测到目标数：{detected_count}")
            self.setStatusMessage("视频处理完成")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"视频处理失败：{str(e)}")
            self.setStatusMessage("就绪")
        finally:
            self.process_video_btn.setEnabled(True)

    def _reset_video_tab(self):
        """重置视频检测选项卡."""
        self.video_label.clear()
        self.stats_text.setText("")
        self.process_video_btn.setEnabled(False)
        self.reset_video_btn.setEnabled(False)
        self.video_results = {}
        if hasattr(self, 'current_video_path'):
            delattr(self, 'current_video_path')
        self.setStatusMessage("已重置")

    def _open_camera(self):
        """打开摄像头."""
        if self.camera_source is not None and self.camera_source.is_opened:
            QMessageBox.warning(self, "警告", "摄像头已打开!")
            return

        self.camera_source = CameraSource(0)
        if self.camera_source.open():
            self.camera_open_btn.setEnabled(False)
            self.camera_close_btn.setEnabled(True)
            self.take_snapshot_btn.setEnabled(True)

            # 启动定时器更新画面
            self.camera_timer = QTimer()
            self.camera_timer.timeout.connect(self._update_camera_frame)
            self.camera_timer.start(33)  # 约 30 FPS

            self.setStatusMessage("摄像头已开启")
        else:
            QMessageBox.critical(self, "错误", "无法打开摄像头!")
            self.camera_source = None

    def _update_camera_frame(self):
        """更新摄像头画面."""
        if self.camera_source is None or not self.camera_source.is_opened:
            return

        ret, frame = self.camera_source.read_frame()
        if not ret:
            return

        # 进行实时检测
        if self.detector:
            try:
                annotated_frame, detections = self.detector.detect_camera_stream(
                    frame,
                    conf_threshold=self.conf_threshold,
                    iou_threshold=self.iou_threshold
                )

                # 在画面上显示检测信息
                fps_info = f"FPS: {self.camera_source.get_fps():.1f}"
                count_info = f"Targets: {len(detections)}"

                cv2.putText(annotated_frame, fps_info, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(annotated_frame, count_info, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            except Exception as e:
                print(f"检测错误：{e}")
                annotated_frame = frame
        else:
            annotated_frame = frame

        # 转换并显示
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        height, width, channel = annotated_frame.shape
        bytes_per_line = channel * width
        qt_image = QImage(annotated_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            self.camera_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.camera_label.setPixmap(scaled_pixmap)

        # 更新状态
        self.camera_status_label.setText(f"状态：{self.camera_source.status}")
        self.camera_fps_label.setText(f"FPS: {self.camera_source.get_fps():.1f}")

    def _close_camera(self):
        """关闭摄像头."""
        if self.camera_source:
            if hasattr(self, 'camera_timer'):
                self.camera_timer.stop()
            self.camera_source.close()
            self.camera_source = None

            self.camera_open_btn.setEnabled(True)
            self.camera_close_btn.setEnabled(False)
            self.take_snapshot_btn.setEnabled(False)
            self.camera_label.clear()

            self.setStatusMessage("摄像头已关闭")

    def _take_snapshot(self):
        """拍摄快照并保存."""
        if self.camera_label.pixmap() is None:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存截图",
            "snapshot.jpg",
            "Images (*.jpg *.png)"
        )

        if file_path:
            self.camera_label.pixmap().save(file_path)
            QMessageBox.information(self, "成功", f"截图已保存到:\n{file_path}")

    def setModelPath(self, path: str):
        """设置显示的模型路径."""
        self.model_path_edit.setText(path)

    def setStatusMessage(self, message: str):
        """设置状态栏消息."""
        self.statusBar().showMessage(message)
