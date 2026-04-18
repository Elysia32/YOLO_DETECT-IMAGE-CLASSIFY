"""UI modules for YOLO Test application."""

try:
    from .main_window import MainWindow
except ImportError:
    from main_window import MainWindow

__all__ = ["MainWindow"]
