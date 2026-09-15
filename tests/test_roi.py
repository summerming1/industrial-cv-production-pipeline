from industrial_cv.roi import PolygonROI


def test_contains_and_boundary():
    roi = PolygonROI("slot", ((0, 0), (10, 0), (10, 10), (0, 10)))
    assert roi.contains((5, 5))
    assert roi.contains((0, 5))
    assert not roi.contains((12, 5))


def test_bbox_overlap_is_explicitly_approximate():
    roi = PolygonROI("slot", ((0, 0), (10, 0), (10, 10), (0, 10)))
    assert roi.bbox_intersection_ratio((5, 5, 15, 15)) == 0.25


def test_polygon_vertex_ratio_uses_segmentation_geometry():
    roi = PolygonROI("slot", ((0, 0), (10, 0), (10, 10), (0, 10)))
    polygon = ((1, 1), (2, 2), (12, 12), (20, 20))
    assert roi.polygon_vertex_ratio(polygon) == 0.5
    assert roi.polygon_vertex_ratio(()) == 0.0
