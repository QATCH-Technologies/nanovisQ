# module: v6_yolo_logging.py

"""QModel V6 YOLO pipeline — shared logging bootstrap.

Provides a single ``Log`` reference used by every sub-module, plus the
internal tag constants.  When running inside the full QATCH application
the real ``Logger`` class is used; in headless / standalone mode a
minimal print-based fallback is substituted.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)
Date:
    2026-04-14
Version:
    7.0.0
"""

from __future__ import annotations

try:
    from QATCH.common.logger import Logger as Log  # pyright: ignore[reportPrivateImportUsage]
except (ImportError, ModuleNotFoundError):

    class Log:  # type: ignore[no-redef]
        @staticmethod
        def d(tag: str, msg: str) -> None:
            print(f"{tag} [DEBUG] {msg}")

        @staticmethod
        def i(tag: str, msg: str) -> None:
            print(f"{tag} [INFO] {msg}")

        @staticmethod
        def w(tag: str, msg: str) -> None:
            print(f"{tag} [WARNING] {msg}")

        @staticmethod
        def e(tag: str, msg: str) -> None:
            print(f"{tag} [ERROR] {msg}")

    Log.i(tag="[HEADLESS OPERATION]", msg="Running...")


# ── Tag constants ─────────────────────────────────────────────────────
TAG_CTRL = "[QModelV6YOLO]"
TAG_CLS = "[QModelV6YOLO_FillClassifier]"
TAG_DET = "[QModelV6YOLO_Detector]"
TAG_REPLAY = "[QModelV6YOLO_FillReplay]"
TAG_SCOUT = "[QModelV6YOLO_POIScout]"
TAG_ROUTE = "[QModelV6YOLO_Routing]"
TAG_VIS = "[QModelV6YOLO_Viz]"
