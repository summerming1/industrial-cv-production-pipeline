from __future__ import annotations

import numpy as np

from .events import TemporalEventMachine
from .inference import Detector
from .roi import PolygonROI
from .types import FrameResult, Observation


class VisionPipeline:
    """Small, testable orchestration layer for real-time CV pipelines."""

    def __init__(self, detector: Detector, roi: PolygonROI, event_machine: TemporalEventMachine, *, target_classes: set[str] | None = None) -> None:
        self.detector = detector
        self.roi = roi
        self.event_machine = event_machine
        self.target_classes = target_classes

    def process_frame(self, frame_bgr: np.ndarray, timestamp_s: float) -> FrameResult:
        detections = self.detector.predict(frame_bgr)
        candidates = [d for d in detections if self.target_classes is None or d.class_name in self.target_classes]
        roi_hits = [d for d in candidates if self.roi.contains(d.center)]
        best_conf = max((d.confidence for d in roi_hits), default=0.0)
        update = self.event_machine.update(Observation(timestamp_s, best_conf, bool(roi_hits), {"roi": self.roi.name}))
        return FrameResult(timestamp_s, tuple(detections), tuple(roi_hits), update.state.value, update.started, update.ended)
