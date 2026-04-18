"""Core modules for YOLO detection."""

try:
    from .yolo_detector import YOLODetector
    from .camera_source import CameraSource
except ImportError:
    from yolo_detector import YOLODetector
    from camera_source import CameraSource

__all__ = ["YOLODetector", "CameraSource"]
