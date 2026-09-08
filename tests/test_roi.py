from industrial_cv.roi import PolygonROI


def test_polygon_contains_boundary_and_interior():
    roi = PolygonROI("zone", ((0, 0), (10, 0), (10, 10), (0, 10)))
    assert roi.contains((5, 5))
    assert roi.contains((0, 5))
    assert not roi.contains((11, 5))


def test_bbox_intersection_ratio():
    roi = PolygonROI("zone", ((0, 0), (10, 0), (10, 10), (0, 10)))
    assert roi.bbox_intersection_ratio((5, 5, 15, 15)) == 0.25
