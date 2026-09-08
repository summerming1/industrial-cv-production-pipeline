"""Dependency-light demo using a fake detector; proves orchestration without model weights."""

import numpy as np

from industrial_cv.events import TemporalEventMachine
from industrial_cv.roi import PolygonROI
from industrial_cv.runtime import VisionPipeline
from industrial_cv.types import Detection


class FakeDetector:
    def __init__(self) -> None:
        self.present = True

    def predict(self, frame_bgr):
        if not self.present:
            return []
        return [Detection(0, "target", 0.91, (200, 150, 320, 300))]


detector = FakeDetector()
pipeline = VisionPipeline(detector, PolygonROI("zone", ((100, 100), (500, 100), (500, 400), (100, 400))), TemporalEventMachine(start_after_s=0.5), target_classes={"target"})
frame = np.zeros((480, 640, 3), dtype=np.uint8)
for t in (0.0, 0.3, 0.6, 0.9):
    print(pipeline.process_frame(frame, t))
