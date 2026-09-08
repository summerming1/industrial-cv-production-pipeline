from __future__ import annotations

from dataclasses import dataclass


def _point_on_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float, eps: float = 1e-9) -> bool:
    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if abs(cross) > eps:
        return False
    return min(ax, bx) - eps <= px <= max(ax, bx) + eps and min(ay, by) - eps <= py <= max(ay, by) + eps


@dataclass(frozen=True)
class PolygonROI:
    name: str
    points: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if len(self.points) < 3:
            raise ValueError("PolygonROI requires at least three points")

    def contains(self, point: tuple[float, float]) -> bool:
        x, y = point
        inside = False
        n = len(self.points)
        for i in range(n):
            x1, y1 = self.points[i]
            x2, y2 = self.points[(i + 1) % n]
            if _point_on_segment(x, y, x1, y1, x2, y2):
                return True
            if (y1 > y) != (y2 > y):
                xinters = (x2 - x1) * (y - y1) / (y2 - y1) + x1
                if x < xinters:
                    inside = not inside
        return inside

    def bbox_intersection_ratio(self, bbox_xyxy: tuple[float, float, float, float]) -> float:
        """Fast conservative overlap using the ROI bounding box.

        Production systems often use polygon-mask overlap. This dependency-free
        approximation is intentionally simple for the public showcase.
        """
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        rx1, ry1, rx2, ry2 = min(xs), min(ys), max(xs), max(ys)
        x1, y1, x2, y2 = bbox_xyxy
        ix1, iy1, ix2, iy2 = max(rx1, x1), max(ry1, y1), min(rx2, x2), min(ry2, y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        return inter / area if area > 0 else 0.0
