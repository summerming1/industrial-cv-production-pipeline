from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    polygon: tuple[tuple[float, float], ...] = ()

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass(frozen=True)
class Observation:
    timestamp_s: float
    evidence: float
    present: bool
    metadata: dict[str,Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FrameResult:
    timestamp_s: float
    detections: tuple[Detection, ...]
    roi_hits: tuple[Detection, ...]
    event_state: str
    event_started: bool = False
    event_ended: bool = False
