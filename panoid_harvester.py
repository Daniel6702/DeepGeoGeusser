#!/usr/bin/env python3
# panoid_harvester.py

from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from shapely.ops import unary_union
from utils import find_nearest_streetview, random_points_in_poly
import time

import re, requests
from pathlib import Path
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from datetime import date
from shapely.ops import unary_union

def find_nearest_streetview(
        lat: float, 
        lon: float
    ):
    '''
    Find the nearest Street View pano ID to the given latitude and longitude.
    '''
    url = (
        "https://maps.googleapis.com/maps/api/js/GeoPhotoService.SingleImageSearch"
        f"?pb=!1m5!1sapiv3!5sUS!11m2!1m1!1b0!2m4!1m2!3d{lat}!4d{lon}!2d50!"
        "3m10!2m2!1sen!2sUS!9m1!1e2!11m4!1m3!1e2!2b1!3e2!"
        "4m10!1e1!1e2!1e3!1e4!1e8!1e6!5m1!1e2!6m1!1e2"
        "&callback=_xdc_._x"
    )
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    m = re.search(r'\[\d+,"([^"]+)"\].+?\[\[null,null,(-?\d+\.\d+),(-?\d+\.\d+)', r.text)
    if not m:
        return None, None, None, None

    pano_id = m.group(1)
    pano_lat = float(m.group(2))
    pano_lon = float(m.group(3))

    date_matches = re.findall(r'\[(20\d{2}),(\d{1,2})\]', r.text)
    latest_date = None

    if date_matches:
        parsed_dates = [date(int(y), int(mo), 1) for y, mo in date_matches]
        latest_date = max(parsed_dates)

    return pano_id, pano_lat, pano_lon, latest_date

def download_streetview_image(
        id,
        out_path: Path,
        width = 1024, 
        height = 1024, 
        fov = 90,
        yaw = 0.0,
        pitch = 0.0
    ):
    #download image
    url = (
        "https://streetviewpixels-pa.googleapis.com/v1/thumbnail"
        f"?panoid={id}&cb_client=maps_sv.tactile.gps"
        f"&w={width}&h={height}&yaw={yaw:.6f}&pitch={pitch:.6f}&thumbfov={fov}"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    out_path.write_bytes(r.content)
    return out_path

def random_points_in_poly(polygon, n, crs=3857):
    '''
    Generate n random points within a given polygon.
    '''
    minx, miny, maxx, maxy = polygon.bounds
    pts = []
    while len(pts) < n:
        x = np.random.uniform(minx, maxx)
        y = np.random.uniform(miny, maxy)
        p = Point(x, y)
        if polygon.contains(p):
            pts.append(p)
    return gpd.GeoSeries(pts, crs=crs)

def random_points_in_geo(gdf, n):
    # 3857 for area-fair sampling, then back to 4326 for web maps
    poly_3857 = unary_union(gdf.to_crs(3857).geometry)
    pts_3857  = random_points_in_poly(poly_3857, n, crs=3857)
    pts_4326  = gpd.GeoDataFrame(geometry=pts_3857).to_crs(4326)
    return pts_4326 

WORLD_PATH   = Path("ne_10m_admin_0_countries.geojson")
DEFAULT_OUT  = Path("output/panoids5.csv")               
LIMIT        = 100_000_000                               
BATCH        = 1_000                                    
CHECKPOINT   = 100                                       
MAX_WORKERS  = 32                                        
RANDOM_SEED  = None                                      

EXCLUDE = [
  "Algeria","Angola","Benin","Burkina Faso","Burundi","Cameroon",
  "Central African Republic","Comoros","Democratic Republic of the Congo",
  "Republic of the Congo","Equatorial Guinea","Eritrea","Ethiopia","Gabon",
  "Gambia","Guinea","Guinea-Bissau","Ivory Coast","Liberia","Libya",
  "Madagascar","Malawi","Mali","Mauritania","Mauritius","Morocco",
  "Mozambique","Niger","Seychelles","Sierra Leone","Somalia","Somaliland",
  "South Sudan","Sudan","United Republic of Tanzania","Togo","Western Sahara",
  "Zambia","Zimbabwe","Afghanistan","Iraq","Kuwait","Syria","Tajikistan",
  "Turkmenistan","Uzbekistan","Yemen","Maldives","Myanmar","Pakistan",
  "East Timor","China","North Korea","Belarus","Bosnia and Herzegovina",
  "Kosovo","Moldova","Northern Cyprus","Cyprus No Mans Area","Anguilla",
  "Antigua and Barbuda","Aruba","The Bahamas","Barbados","Belize","Cuba",
  "Dominica","Grenada","Guyana","Haiti","Honduras","Jamaica",
  "Saint Kitts and Nevis","Saint Lucia","Saint Vincent and the Grenadines",
  "Suriname","British Virgin Islands","Turks and Caicos Islands",
  "Saint Pierre and Miquelon","Falkland Islands","Fiji",
  "Federated States of Micronesia","Kiribati","Marshall Islands","Nauru",
  "New Caledonia","Niue","Norfolk Island","Palau","Papua New Guinea",
  "Solomon Islands","Tonga","Tuvalu","Vanuatu","Wallis and Futuna",
  "French Polynesia","Akrotiri Sovereign Base Area",
  "Dhekelia Sovereign Base Area","Antarctica","Ashmore and Cartier Islands",
  "Clipperton Island","Coral Sea Islands","Heard Island and McDonald Islands",
  "United States Minor Outlying Islands","Baykonur Cosmodrome",
  "Siachen Glacier","Scarborough Reef","Spratly Islands","Serranilla Bank",
  "Bajo Nuevo Bank (Petrel Is.)","US Naval Base Guantanamo Bay","Chad"
]

def load_world_union_3857(path: Path):
    world = gpd.read_file(path)
    world = world[world.geometry.notnull()]
    world = world[~world["ADMIN"].isin(EXCLUDE)].copy()
    world_3857 = world.to_crs(3857)
    return unary_union(world_3857.geometry)

def sample_points_4326(poly_3857_union, n):
    pts_3857 = random_points_in_poly(poly_3857_union, n, crs=3857)
    return gpd.GeoDataFrame(geometry=pts_3857).to_crs(4326)

def append_checkpoint(rows, out_path: Path):
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if out_path.exists():
        df.to_csv(out_path, mode="a", header=False, index=False)
    else:
        df.to_csv(out_path, index=False)

def load_seen(out_path: Path):
    if not out_path.exists():
        return set()
    try:
        df = pd.read_csv(out_path, usecols=["panoid"])
        return set(df["panoid"].astype(str))
    except Exception:
        try:
            df = pd.read_csv(out_path)
            return set(df["panoid"].astype(str)) if "panoid" in df.columns else set()
        except Exception:
            return set()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUT))
    args = parser.parse_args()

    out_path = Path(args.output)

    if RANDOM_SEED is not None:
        np.random.seed(RANDOM_SEED)

    poly_3857_union = load_world_union_3857(WORLD_PATH)
    print("World union prepared")

    seen = load_seen(out_path)
    print(f"Resuming with {len(seen)} panoids at {out_path}")

    pending = []
    added_since_save = 0
    t0 = time.perf_counter()

    def task(lat, lon):
        try:
            return find_nearest_streetview(lat, lon)
        except Exception:
            return None, None, None, None

    try:
        while len(seen) < LIMIT:
            pts = sample_points_4326(poly_3857_union, BATCH)
            coords = [(pt.y, pt.x) for pt in pts.geometry]

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futures = [ex.submit(task, lat, lon) for (lat, lon) in coords]
                for fut in as_completed(futures):
                    panoid, plat, plon, pdate = fut.result()
                    if not panoid or panoid in seen:
                        continue
                    seen.add(panoid)
                    pending.append({
                        "panoid": panoid,
                        "pano_lat": plat,
                        "pano_lon": plon,
                        "pano_date": pdate.isoformat() if hasattr(pdate, "isoformat") and pdate else None
                    })
                    added_since_save += 1

                    if added_since_save >= CHECKPOINT:
                        append_checkpoint(pending, out_path)
                        rate = added_since_save / max(time.perf_counter() - t0, 1e-9)
                        print(f"Saved {added_since_save} new panoids (total={len(seen)}; {rate:.1f} ids/s).")
                        pending.clear()
                        added_since_save = 0
                        t0 = time.perf_counter()

            print(f"total unique panoids: {len(seen)}")

    except KeyboardInterrupt:
        print("stopped")
    finally:
        if pending:
            append_checkpoint(pending, out_path)
            print(f"Final save of {len(pending)} panoids (total={len(seen)}).")

if __name__ == "__main__":
    main()