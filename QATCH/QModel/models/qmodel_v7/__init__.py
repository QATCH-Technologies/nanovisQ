"""
QModel V7 YOLO predictor package.

Public surface (V7 uses its own class names, distinct from the V6 package,
so both model generations can be imported side by side):

    from QATCH.QModel.models.qmodel_v7 import QModelV7, QModelV7Config
"""

__version__ = "7.0.0"
__release__ = "2026-07-06"

try:
    from .v7_yolo import (
        QModelV7Config,
        QModelV7,
        QModelV7Detector,
        QModelV7FillClassifier,
    )
except (ImportError, ModuleNotFoundError):
    from QATCH.QModel.models.qmodel_v7.v7_yolo import (
        QModelV7Config,
        QModelV7,
        QModelV7Detector,
        QModelV7FillClassifier,
    )

__all__ = [
    "QModelV7Config",
    "QModelV7",
    "QModelV7Detector",
    "QModelV7FillClassifier",
]
