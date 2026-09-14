from __future__ import annotations

from pathlib import Path
from typing import Iterable
import warnings

import geopandas as gpd
import numpy as np
from pyproj import CRS, Transformer
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.geometry.base import BaseGeometry
from shapely import covers, make_valid, points as shapely_points, union_all


def repair_polygonal_geometry(geom: BaseGeometry) -> BaseGeometry:
    """Return a valid polygonal geometry while preserving polygonal components.

    Projection and GeoJSON round-trips can make otherwise sensible global
    Polygon/MultiPolygon geometries invalid around seams such as the antimeridian.
    GEOS boolean operations require valid inputs, so normalize them before use.
    """
    if geom is None or geom.is_empty:
        return geom
    if geom.is_valid and geom.geom_type in {"Polygon", "MultiPolygon"}:
        return geom

    repaired = make_valid(geom)
    if repaired.geom_type in {"Polygon", "MultiPolygon"}:
        return repaired

    polygons = []
    stack = [repaired]
    while stack:
        current = stack.pop()
        if current.geom_type == "Polygon":
            polygons.append(current)
        elif current.geom_type == "MultiPolygon":
            polygons.extend(current.geoms)
        elif hasattr(current, "geoms"):
            stack.extend(current.geoms)

    if not polygons:
        return Polygon()
    if len(polygons) == 1:
        return polygons[0]
    return union_all(polygons)


def as_coordinate_array(points: Iterable[tuple[float, float]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("points must have shape (N, 2) and contain (longitude, latitude) pairs")
    if len(arr) == 0:
        raise ValueError("points cannot be empty")
    if not np.isfinite(arr).all():
        raise ValueError("points contain NaN or infinite values")
    return arr


def load_boundary(boundary: str | Path | BaseGeometry | gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if isinstance(boundary, gpd.GeoDataFrame):
        gdf = boundary.copy()
    elif isinstance(boundary, BaseGeometry):
        gdf = gpd.GeoDataFrame({"geometry": [boundary]}, crs="EPSG:4326")
    else:
        gdf = gpd.read_file(boundary)

    if gdf.empty:
        raise ValueError("boundary is empty")
    if gdf.crs is None:
        # RFC 7946 GeoJSON coordinates are WGS84. For non-GeoJSON inputs callers
        # should explicitly provide a CRS before passing a GeoDataFrame.
        warnings.warn("Boundary has no CRS; assuming EPSG:4326.", stacklevel=2)
        gdf = gdf.set_crs("EPSG:4326")

    gdf = gdf.to_crs("EPSG:4326")
    geom = gdf.geometry.union_all()
    if geom.is_empty:
        raise ValueError("boundary geometry is empty")
    if geom.geom_type not in {"Polygon", "MultiPolygon"}:
        raise ValueError(f"boundary must be Polygon or MultiPolygon, got {geom.geom_type}")
    geom = repair_polygonal_geometry(geom)
    return gpd.GeoDataFrame({"geometry": [geom]}, crs="EPSG:4326")


def choose_projected_crs(boundary_wgs84: gpd.GeoDataFrame, projected_crs=None) -> CRS:
    if projected_crs is not None:
        crs = CRS.from_user_input(projected_crs)
        if not crs.is_projected:
            raise ValueError("projected_crs must be a projected CRS")
        return crs

    try:
        crs = boundary_wgs84.estimate_utm_crs()
    except Exception:
        crs = None
    if crs is None:
        warnings.warn(
            "Could not estimate a local UTM CRS; falling back to EPSG:3857. "
            "For very large/multi-zone areas, pass projected_crs explicitly.",
            stacklevel=2,
        )
        crs = CRS.from_epsg(3857)
    return CRS.from_user_input(crs)


def transform_xy(arr: np.ndarray, source_crs, target_crs) -> np.ndarray:
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    x, y = transformer.transform(arr[:, 0], arr[:, 1])
    return np.column_stack([x, y]).astype(np.float64, copy=False)


def boundary_mask(boundary_geom: BaseGeometry, lonlat: np.ndarray) -> np.ndarray:
    pts = shapely_points(lonlat[:, 0], lonlat[:, 1])
    return np.asarray(covers(boundary_geom, pts), dtype=bool)
