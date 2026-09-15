import numpy as np
import pytest

from industrial_cv.events import TemporalEventMachine
from industrial_cv.roi import PolygonROI
from industrial_cv.runtime import VisionPipeline
from industrial_cv.types import Detection


class Detector:
    def __init__(self, detections):
        self.detections = detections

    def predict(self, frame):
        return list(self.detections)


FRAME = np.zeros((20, 20, 3), dtype=np.uint8)
ROI = PolygonROI("slot", ((0, 0), (10, 0), (10, 10), (0, 10)))


def test_center_policy():
    det = Detection(0, "part", 0.9, (8, 8, 12, 12))
    pipeline = VisionPipeline(Detector([det]), ROI, TemporalEventMachine(), spatial_policy="center")
    result = pipeline.process_frame(FRAME, 0.0)
    assert len(result.roi_hits) == 1


def test_bbox_approx_policy_can_differ_from_center():
    det = Detection(0, "part", 0.9, (8, 8, 18, 18))
    pipeline = VisionPipeline(
        Detector([det]),
        ROI,
        TemporalEventMachine(),
        spatial_policy="bbox_approx",
        min_spatial_ratio=0.03,
    )
    result = pipeline.process_frame(FRAME, 0.0)
    assert len(result.roi_hits) == 1


def test_polygon_vertex_policy_uses_mask_points():
    det = Detection(
        0,
        "part",
        0.9,
        (8, 8, 18, 18),
        polygon=((1, 1), (2, 2), (15, 15), (16, 16)),
    )
    pipeline = VisionPipeline(
        Detector([det]),
        ROI,
        TemporalEventMachine(),
        spatial_policy="polygon_vertex",
        min_spatial_ratio=0.5,
    )
    result = pipeline.process_frame(FRAME, 0.0)
    assert len(result.roi_hits) == 1


def test_invalid_spatial_policy_rejected():
    with pytest.raises(ValueError):
        VisionPipeline(Detector([]), ROI, TemporalEventMachine(), spatial_policy="mask_iou")


def test_start_after_zero_preserves_two_observation_transition():
    det = Detection(0, "target", 0.95, (2, 2, 4, 4))
    pipeline = VisionPipeline(
        Detector([det]),
        ROI,
        TemporalEventMachine(start_after_s=0.0),
        target_classes={"target"},
    )
    first = pipeline.process_frame(FRAME, 0.0)
    assert first.event_state == "candidate"
    second = pipeline.process_frame(FRAME, 0.1)
    assert second.event_started
