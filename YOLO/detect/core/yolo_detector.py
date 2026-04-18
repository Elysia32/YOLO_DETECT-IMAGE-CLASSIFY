"""YOLO 检测器封装模块."""

import torch
from ultralytics import YOLO
from typing import Optional, Tuple
import numpy as np
import cv2


class YOLODetector:
    """YOLO 模型检测器封装类."""

    def __init__(self, model_path: str = "yolo11n.pt"):
        """
        初始化 YOLO 检测器。

        Args:
            model_path: 模型文件路径，支持 .pt 格式
                       可以是本地路径或 Ultralytics 仓库中的模型名称
        """
        self.model_path = model_path
        self.model: Optional[YOLO] = None
        self._load_model()

    def _load_model(self) -> None:
        """加载 YOLO 模型."""
        try:
            # 如果文件存在则作为本地路径加载，否则尝试从仓库加载
            if not self._is_local_file():
                print(f"正在从 Ultralytics 仓库下载模型：{self.model_path}")

            self.model = YOLO(self.model_path)
            # 获取设备信息（新 API）
            device = getattr(self.model, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
            print(f"模型加载成功：{self.model_path}")
            print(f"设备信息：{device}")
        except Exception as e:
            print(f"模型加载失败：{e}")
            raise

    def _is_local_file(self) -> bool:
        """检查是否为本地文件路径而非模型名称."""
        import os
        return os.path.isfile(self.model_path) or os.path.exists(self.model_path)

    def detect_image(
        self,
        image_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.7
    ) -> Tuple[np.ndarray, list]:
        """
        对图片进行目标检测。

        Args:
            image_path: 图片文件路径
            conf_threshold: 置信度阈值
            iou_threshold: IoU 阈值

        Returns:
            Tuple[np.ndarray, list]: (标注后的图像数组，检测结果列表)
        """
        if self.model is None:
            raise RuntimeError("模型未加载")

        results = self.model.predict(
            source=image_path,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )

        result = results[0]
        annotated_image = result.plot()  # BGR 格式

        # 提取检测结果信息
        detections = []
        for box in result.boxes:
            detections.append({
                "class_id": int(box.cls.item()),
                "class_name": result.names[int(box.cls.item())],
                "confidence": float(box.conf.item()),
                "bbox": box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            })

        return annotated_image, detections

    def detect_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.7
    ) -> Tuple[int, int, list]:
        """
        对视频进行目标检测。

        Args:
            video_path: 视频文件路径
            output_path: 输出视频路径，为 None 则不保存
            conf_threshold: 置信度阈值
            iou_threshold: IoU 阈值

        Returns:
            Tuple[int, int, list]: (视频总帧数，检测到目标总数，检测结果统计字典)
        """
        if self.model is None:
            raise RuntimeError("模型未加载")

        # 先获取视频总帧数
        video_cap = cv2.VideoCapture(video_path)
        total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_cap.release()

        results = self.model.predict(
            source=video_path,
            save=output_path is not None,
            save_dir=output_path or "",
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )

        detected_count = 0
        detection_stats = {}

        for result in results:
            detected_count += len(result.boxes)
            for box in result.boxes:
                class_id = int(box.cls.item())
                class_name = result.names[class_id]
                confidence = float(box.conf.item())

                if class_name not in detection_stats:
                    detection_stats[class_name] = {
                        "count": 0,
                        "avg_confidence": 0,
                        "total_confidence": 0
                    }

                detection_stats[class_name]["count"] += 1
                detection_stats[class_name]["total_confidence"] += confidence

        # 计算平均置信度
        for name, stats in detection_stats.items():
            if stats["count"] > 0:
                stats["avg_confidence"] = stats["total_confidence"] / stats["count"]
            del stats["total_confidence"]  # 移除临时数据

        return total_frames, detected_count, detection_stats

    def detect_camera_stream(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.7
    ) -> Tuple[np.ndarray, list]:
        """
        对单帧图像进行实时目标检测。

        Args:
            frame: OpenCV 格式的帧图像 (BGR)
            conf_threshold: 置信度阈值
            iou_threshold: IoU 阈值

        Returns:
            Tuple[np.ndarray, list]: (标注后的帧，检测结果列表)
        """
        if self.model is None:
            raise RuntimeError("模型未加载")

        results = self.model.predict(
            source=frame,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )

        result = results[0]
        annotated_image = result.plot()

        detections = []
        for box in result.boxes:
            detections.append({
                "class_id": int(box.cls.item()),
                "class_name": result.names[int(box.cls.item())],
                "confidence": float(box.conf.item()),
                "bbox": box.xyxy[0].tolist()
            })

        return annotated_image, detections

    def reload_model(self, model_path: str) -> None:
        """
        重新加载模型（热重载）。

        Args:
            model_path: 新的模型文件路径
        """
        self.model_path = model_path
        self._load_model()

    @property
    def device(self) -> str:
        """获取当前使用的设备信息."""
        if self.model is None:
            return "unknown"
        # 新 API：直接从 model 对象获取 device
        return str(getattr(self.model, 'device', 'cpu'))

    @property
    def model_info(self) -> dict:
        """获取模型信息."""
        if self.model is None:
            return {}
        return {
            "path": self.model_path,
            "device": str(getattr(self.model, 'device', 'cpu')),
            "names": getattr(self.model, 'names', {}),
            "nc": getattr(self.model, 'model', None).nc if hasattr(self.model, 'model') and hasattr(getattr(self.model, 'model', None), 'nc') else 0
        }
