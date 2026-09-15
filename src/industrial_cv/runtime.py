from __future__ import annotations

import numpy as np

from .events import TemporalEventMachine
from .inference import Detector
from .roi import PolygonROI
from .types import Detection, FrameResult, Observation


class VisionPipeline:
    """Small, testable orchestration layer for real-time CV pipelines."""

    SUPPORTED_SPATIAL_POLICIES = {"center", "bbox_approx", "polygon_vertex"}

    def __init__(
        self,
        detector: Detector,
        roi: PolygonROI,
        event_machine: TemporalEventMachine,
        *,
        target_classes: set[str] | None = None,
        spatial_policy: str = "center",
        min_spatial_ratio: float = 0.5,
    ) -> None:
        if spatial_policy not in self.SUPPORTED_SPATIAL_POLICIES:
            raise ValueError(f"unsupported spatial_policy: {spatial_policy}")
        if not 0 <= min_spatial_ratio <= 1:
            raise ValueError("min_spatial_ratio must be in [0, 1]")
        self.detector = detector
        self.roi = roi
        self.event_machine = event_machine
        self.target_classes = target_classes
        self.spatial_policy = spatial_policy
        self.min_spatial_ratio = min_spatial_ratio

    def _matches_roi(self, detection: Detection) -> bool:
        if self.spatial_policy == "center":
            return self.roi.contains(detection.center)
        if self.spatial_policy == "bbox_approx":
            return (
                self.roi.bbox_intersection_ratio(detection.bbox_xyxy)
                >= self.min_spatial_ratio
            )
        return (
            self.roi.polygon_vertex_ratio(detection.polygon)
            >= self.min_spatial_ratio
        )

    def process_frame(self, frame_bgr: np.ndarray, timestamp_s: float) -> FrameResult:
        detections = self.detector.predict(frame_bgr)
        candidates = [
            detection
            for detection in detections
            if self.target_classes is None
            or detection.class_name in self.target_classes
        ]
        roi_hits = [detection for detection in candidates if self._matches_roi(detection)]
        best_conf = max((detection.confidence for detection in roi_hits), default=0.0)
        update = self.event_machine.update(
            Observation(
                timestamp_s,
                best_conf,
                bool(roi_hits),
                {"roi": self.roi.name, "spatial_policy": self.spatial_policy},
            )
        )
        return FrameResult(
            timestamp_s,
            tuple(detections),
            tuple(roi_hits),
            update.state.value,
            update.started,
            update.ended,
        )
