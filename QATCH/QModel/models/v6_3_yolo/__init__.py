"""
QModel V6 YOLO predictor subpackage
===================================

Version 6.3.0

This package rolls the QModel V6 YOLO prediction pipeline and its
configuration-prior decode layer into a single importable unit:

  controller.py     - QModelV6YOLO controller, detector/classifier wrappers,
                      and QModelV6Config (the public entry point: .predict()).
  dataprocessor.py  - QModelV6YOLO_DataProcessor (preprocessing + v1 render).
  renderer.py       - v2 detection-image renderer + generate_det_image()
                      version dispatch (formerly v7_render.py).
  spacing_prior.py  - SpacingPrior: the learned flat/pairwise spacing prior.
  decode.py         - dp_decode + scoring over YOLO candidate lattices
                      (formerly dp_decode.py).
  assets/spacing_prior.json - the fitted prior shipped with the package.

Import contract
---------------
The lightweight pieces (SpacingPrior, dp_decode, the renderer) import with no
YOLO/ultralytics dependency. The controller pulls in ultralytics + cv2, so it
is imported lazily here and degrades to ``None`` if those are unavailable;
this lets benchmark/decode tooling import the package without the full
inference stack.

A convenience path to the bundled prior is exposed as ``SPACING_PRIOR_PATH``.
"""

from __future__ import annotations

import os

__version__ = "6.3.0"

# Path to the fitted prior bundled with the package, for building model_assets:
#   model_assets["spacing_prior"] = SPACING_PRIOR_PATH
SPACING_PRIOR_PATH = os.path.join(os.path.dirname(__file__), "assets", "spacing_prior.json")

# ---- Lightweight API (no ultralytics/cv2 dependency) --------------------
from .spacing_prior import SpacingPrior, GapStat, POI_ORDER  # noqa: E402
from .decode import (  # noqa: E402
    Candidate,
    DecodeResult,
    dp_decode,
    score_configuration,
    greedy_baseline,
)
from QATCH.qmodel.models.v6_3_yolo.dataprocessor import DataProcessor  # noqa: E402
from QATCH.qmodel.models.v6_3_yolo.renderer import (
    generate_detection_image,
    generate_detection_image_v2,
)  # noqa: E402

# ---- Heavy API (ultralytics + cv2). Lazy/guarded so the package still
# imports for decode-only / headless tooling when inference deps are absent.
try:  # pragma: no cover - exercised only with the full inference stack
    from QATCH.qmodel.models.v6_3_yolo.controller import (
        QModel,
        Detector,
        FillClassifier,
        QModelConfig,
    )

    _CONTROLLER_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    QModelV6YOLO = None  # type: ignore[assignment]
    QModelV6YOLO_Detector = None  # type: ignore[assignment]
    QModelV6YOLO_FillClassifier = None  # type: ignore[assignment]
    QModelV6Config = None  # type: ignore[assignment]
    _CONTROLLER_AVAILABLE = False

__all__ = [
    "__version__",
    "SPACING_PRIOR_PATH",
    # spacing prior
    "SpacingPrior",
    "GapStat",
    "POI_ORDER",
    # decode
    "Candidate",
    "DecodeResult",
    "dp_decode",
    "score_configuration",
    "greedy_baseline",
    # data + render
    "DataProcessor",
    "generate_detection_image",
    "generate_detection_image_v2",
    # controller (may be None if inference deps absent)
    "QModel",
    "Detector",
    "FillClassifier",
    "QModelConfig",
]
