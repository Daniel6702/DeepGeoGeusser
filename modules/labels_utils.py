import csv
from pathlib import Path
from s2sphere import CellId, LatLng

def build_s2_index_maps(s2_labels_dir: str | Path, levels: list[int]):
    """
    Build mappings between S2 cell IDs and contiguous class indices for each level.

    Expects CSV files named L{level}.csv with a 's2_id' column.

    Args:
        s2_labels_dir: directory containing S2 label CSVs.
        levels: list of S2 levels to load.

    Returns:
        idx2id: {level: [s2_id_0, s2_id_1, ...]} index -> S2 ID.
        id2idx: {level: {s2_id: index}} S2 ID -> index.
        num_classes: {level: num_classes_at_that_level}.
    """
    s2_labels_dir = Path(s2_labels_dir)
    idx2id = {}
    id2idx = {}
    num_classes = {}

    for lvl in levels:
        L = f"L{lvl}"
        csv_path = s2_labels_dir / f"{L}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing CSV for {L}: {csv_path}")
        
        s2_ids = set()
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "s2_id" not in reader.fieldnames:
                raise ValueError(f"{csv_path} must have a 's2_id' column")
            for row in reader:
                s = row.get("s2_id")
                if s:
                    s2_ids.add(int(s))

        # Canonical, deterministic order
        unique_ids = sorted(s2_ids)
        idx2id[lvl] = unique_ids
        id2idx[lvl] = {s2id: i for i, s2id in enumerate(unique_ids)}
        num_classes[lvl] = len(unique_ids)

    return idx2id, id2idx, num_classes

def latlon_to_s2id(lat: float, lon: float, level: int) -> int:
    """
    Convert (lat, lon) in degrees to an S2 cell ID at the given level.
    """
    ll = LatLng.from_degrees(lat, lon)
    return CellId.from_lat_lng(ll).parent(level).id()

def s2_id_to_latlon(s2_id):
    """
    Convert an S2 cell ID to (lat, lon) in degrees, using the cell center.
    """
    ll = s2sphere.CellId(int(s2_id)).to_lat_lng()
    return float(ll.lat().degrees), float(ll.lng().degrees)