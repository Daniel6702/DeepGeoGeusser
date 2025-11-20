import csv
from pathlib import Path
from s2sphere import CellId, LatLng

def build_s2_index_maps(s2_labels_dir: str | Path, levels: list[int]):
    """
    For each level L, build:
      - idx2id[L]: list so that idx2id[L][class_idx] = s2_id
      - id2idx[L]: dict so that id2idx[L][s2_id] = class_idx

    We use sorted(unique_s2_ids) to have a deterministic order.
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
    ll = LatLng.from_degrees(lat, lon)
    return CellId.from_lat_lng(ll).parent(level).id()
