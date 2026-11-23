"""normalize_longitude: dateline handling and hole preservation."""
from shapely.geometry import LineString, Point, Polygon

from widetrax.DataPreprocessing import normalize_longitude


def test_normalize_point():
    # Historic bug: the Point branch referenced an undefined global
    # (`swath_geometries.crs`) -> NameError.
    p = normalize_longitude(Point(350.0, -30.0))
    assert isinstance(p, Point)
    assert p.x == -10.0
    assert p.y == -30.0


def test_normalize_dateline_polygon_keeps_hole():
    # Historic bug: `new_geom.interiors = interiors` is a no-op on shapely 2
    # (holes were silently dropped).
    shell = [(170, -10), (190, -10), (190, 10), (170, 10)]
    hole = [(179, -1), (179, 1), (181, 1), (181, -1)]
    polygon = Polygon(shell, [hole])

    normalized = normalize_longitude(polygon)

    assert normalized.geom_type == "Polygon"
    assert list(normalized.exterior.coords)[1] == (-170.0, -10.0)
    assert len(normalized.interiors) == 1
    assert list(normalized.interiors[0].coords)[0] == (179.0, -1.0)
    assert list(normalized.interiors[0].coords)[2] == (-179.0, 1.0)


def test_normalize_polygon_without_hole():
    polygon = Polygon([(200, -5), (210, -5), (210, 5), (200, 5)])
    normalized = normalize_longitude(polygon)
    assert list(normalized.exterior.coords)[0] == (-160.0, -5.0)
    assert len(normalized.interiors) == 0


def test_normalize_other_types_unchanged():
    line = LineString([(300, 0), (310, 5)])
    assert normalize_longitude(line) is line
