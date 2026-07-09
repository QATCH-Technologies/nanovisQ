"""
QModel Onyx YOLO predictor package.

Public surface (Onyx uses its own class names, distinct from the Volta package,
so both model generations can be imported side by side):

    from QATCH.QModel.models.qmodel_onyx import QModelOnyx, QModelOnyxConfig
"""

__version__ = "7.1.0"
__release__ = "2026-07-09"

from .onyx import (
    QModelOnyx,
    QModelOnyxConfig,
    QModelOnyxDetector,
    QModelOnyxFillClassifier,
)

__all__ = [
    "QModelOnyxConfig",
    "QModelOnyx",
    "QModelOnyxDetector",
    "QModelOnyxFillClassifier",
]
