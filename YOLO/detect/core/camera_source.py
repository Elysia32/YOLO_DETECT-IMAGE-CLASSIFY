"""摄像头源处理模块."""

import cv2
from typing import Optional, Tuple
import numpy as np


class CameraSource:
    """摄像头源管理类."""

    def __init__(self, camera_id: int = 0):
        """
        初始化摄像头源。

        Args:
            camera_id: 摄像头设备 ID，默认为 0（默认摄像头）
        """
        self.camera_id = camera_id
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_opened = False

    def open(self) -> bool:
        """
        打开摄像头。

        Returns:
            bool: 是否成功打开
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_id)

            if not self.cap.isOpened():
                print(f"无法打开摄像头：{self.camera_id}")
                return False

            # 设置摄像头参数
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            self.is_opened = True
            print(f"摄像头已打开：ID={self.camera_id}")
            return True

        except Exception as e:
            print(f"打开摄像头失败：{e}")
            return False

    def close(self) -> None:
        """关闭摄像头."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_opened = False
        print("摄像头已关闭")

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        读取一帧图像。

        Returns:
            Tuple[bool, Optional[np.ndarray]]: (是否成功，帧图像)
        """
        if self.cap is None or not self.is_opened:
            return False, None

        ret, frame = self.cap.read()
        if not ret:
            return False, None

        return True, frame

    def get_fps(self) -> float:
        """获取当前帧率."""
        if self.cap is None:
            return 0.0
        return self.cap.get(cv2.CAP_PROP_FPS)

    def get_resolution(self) -> Tuple[int, int]:
        """获取分辨率."""
        if self.cap is None:
            return (0, 0)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)

    @property
    def status(self) -> str:
        """获取摄像头状态."""
        if self.cap is None:
            return "未初始化"
        return "运行中" if self.is_opened else "已关闭"
