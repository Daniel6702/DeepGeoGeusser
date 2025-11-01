from s2sphere import LatLng, CellId
import os
import csv
import geopandas as gpd
from shapely.geometry import Point

def generate_csv_with_precomputed_s2_labels(dataset, LEVEL: int, OUT_PATH: str):
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    seen = set()

    with open(OUT_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["panoid", "s2_id"])
        n = 0

        for _, meta in dataset:
            lat = float(meta["pano_lat"]); lon = float(meta["pano_lon"]); panoid = str(meta["panoid"])
            s2_id = int(CellId.from_lat_lng(LatLng.from_degrees(lat, lon)).parent(LEVEL).id())
            key = (panoid, s2_id)
            if key in seen: 
                continue
            seen.add(key)
            w.writerow([panoid, s2_id])
            n += 1
            if n % 10000 == 0:
                print(f"processed {n} samples")

def generate_csv_with_country_labels(dataset, GEOJSON_PATH: str, OUT_PATH: str):
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    countries = gpd.read_file(GEOJSON_PATH)[["ADMIN", "geometry"]]

    seen = set()
    with open(OUT_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["panoid", "country"])
        n = 0

        for _, meta in dataset:
            lat = float(meta["pano_lat"]); lon = float(meta["pano_lon"]); panoid = str(meta["panoid"])
            match = countries[countries.contains(Point(lon, lat))]
            country = match.iloc[0]["ADMIN"] if not match.empty else "Unknown"
            key = (panoid, country)
            if key in seen:
                continue
            seen.add(key)
            w.writerow([panoid, country])
            n += 1
            if n % 10000 == 0:
                print(f"processed {n} samples")
# Lookup
def load_s2_labels(csv_file: str) -> dict[str, int]:
    table = {}
    with open(csv_file, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            table[row["panoid"]] = int(row["s2_id"])
    return table

def lookup_s2_id(table: dict[str, int], panoid: str) -> int | None:
    return table.get(panoid)

def number_of_unique_s2_ids(table: dict[str, int]) -> int:
    return len(set(table.values()))

def load_country_labels(csv_file: str) -> dict[str, str]:
    table = {}
    with open(csv_file, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            table[row["panoid"]] = row["country"]
    return table

def lookup_country(table: dict[str, str], panoid: str) -> str | None:
    return table.get(panoid)