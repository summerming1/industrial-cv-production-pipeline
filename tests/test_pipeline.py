import numpy as np

from industrial_cv.events import TemporalEventMachine
from industrial_cv.roi import PolygonROI
from industrial_cv.runtime import VisionPipeline
from industrial_cv.types import Detection


class Detector:
    def predict(self, frame):
        return [Detection(0, "target", 0.95, (2, 2, 4, 4))]


def test_pipeline_filters_by_roi_and_emits_state():
    p = VisionPipeline(Detector(), PolygonROI("r", ((0, 0), (10, 0), (10, 10), (0, 10))), TemporalEventMachine(start_after_s=0.0), target_classes={"target"})
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    assert p.process_frame(frame, 0.0).event_state == "candidate"
    result = p.process_frame(frame, 0.1)
    assert result.event_started
    assert len(result.roi_hits) == 1
