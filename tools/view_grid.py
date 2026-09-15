from adaptive_geogrid import load_grid
import matplotlib.pyplot as plt
from modules.dataset import PanoramaDataset
import numpy as np

grid = load_grid("aarhus.aggrid")

metadata = PanoramaDataset.load_metadata("datasets/aarhus")

points = []
seen = set()

for row in metadata.values():
    lat = row.get("lat", row.get("pano_lat"))
    lon = row.get("lon", row.get("pano_lon"))

    if lat is None or lon is None:
        raise KeyError("Metadata must contain lat/lon or pano_lat/pano_lon")

    key = row.get("panoid") or row.get("pano_id") or (lon, lat)
    if key in seen:
        continue

    seen.add(key)
    points.append((float(lon), float(lat)))

points = np.asarray(points, dtype=float)
print(f"Building grid from {len(points):,} unique panorama locations")

ax = grid.plot(
    points=points,
    point_sample=1_000,
    linewidth=0.50,
    hierarchy_lane_alpha=0.82,
    hierarchy_lane_gap=0.0,
    point_size=5.0,
    point_alpha=0.20,
)

ax.set_title("Grid Visual")
plt.tight_layout()
plt.show()