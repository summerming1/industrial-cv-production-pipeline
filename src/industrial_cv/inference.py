from __future__ import annotations

from typing import Protocol

import numpy as np

from .types import Detection


class Detector(Protocol):
    def predict(self, frame_bgr: np.ndarray) -> list[Detection]: ...


class UltralyticsSegmenter:
    """Optional YOLO segmentation adapter.

    The core package does not require Ultralytics, which keeps unit tests and
    CI lightweight. Install the `vision` extra to use this adapter.
    """

    def __init__(self, model_path: str, *, conf: float = 0.25, iou: float = 0.5, imgsz: int = 640, device: str = "cpu") -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install with: pip install -e '.[vision]'") from exc
        self.model = YOLO(model_path, task="segment")
        self.conf, self.iou, self.imgsz, self.device = conf, iou, imgsz, device

    def predict(self, frame_bgr: np.ndarray) -> list[Detection]:
        result = self.model.predict(frame_bgr, conf=self.conf, iou=self.iou, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        if result.boxes is None:
            return []
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        raw_polygons = result.masks.xy if result.masks is not None else []
        names = result.names or {}
        out: list[Detection] = []
        for i, box in enumerate(boxes):
            poly = tuple((float(x), float(y)) for x, y in raw_polygons[i].tolist()) if i < len(raw_polygons) else ()
            cid = int(classes[i])
            out.append(Detection(cid, str(names.get(cid, cid)), float(confs[i]), tuple(map(float, box)), poly))
        return out
