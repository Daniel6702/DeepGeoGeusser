import os, json
import webdataset as wds
import torch
import csv
from pathlib import Path
from s2sphere import CellId 

class GeoWebDataset:
    def __init__(self, dataset_path, processor, levels, shuffle=False, num_shards_limit=None, os_type="LINUX"):
        self.processor = processor
        self.level_keys = [f"L{lvl}" for lvl in levels]
        self.urls = self._get_urls(dataset_path, num_shards_limit, os_type)

        self.num_classes_list = self._get_number_of_classes(dataset_path, levels)
        
        self.dataset = (
            wds.WebDataset(self.urls, shardshuffle=5 if shuffle else False)
            .decode("pil")
            .to_tuple("jpg", "json")
            .map(self._process_sample)
        )

    def _get_urls(self, dataset_path, num_shards_limit, os_type):
        manifest = os.path.join(dataset_path, "shards_manifest.json")
        with open(manifest) as f:
            num_shards = len(json.load(f)["shards"])
        if num_shards_limit:
            num_shards = min(num_shards, num_shards_limit)

        if os_type == "LINUX":
            return f"{dataset_path}/shard-{{000000..{num_shards - 1}}}.tar"
        else:
            return f"file:{dataset_path}/shard-{{000000..{num_shards - 1}}}.tar"

    def _get_number_of_classes(self, dataset_path, levels):
        return [
            get_num_classes_from_csv(Path(dataset_path) / "s2_labels", levels)[f"L{lvl}"]
            for lvl in levels
        ]

    def _process_sample(self, sample):
        img, meta = sample
        pixel_values = self.processor(img, return_tensors="pt")["pixel_values"].squeeze(0)
        
        class_vec = torch.tensor(
            [int(meta["s2"][lvl]) - 1 for lvl in self.level_keys], #Dataset class indicies start from 1 not 0
            dtype=torch.long
        )
        return pixel_values, class_vec

def get_num_classes_from_csv(s2_labels_dir: str | Path, levels):
    s2_labels_dir = Path(s2_labels_dir)
    num_classes = {}

    for lvl in levels:
        level_key = f"L{lvl}"
        csv_path = s2_labels_dir / f"{level_key}.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"Missing CSV for {level_key}: {csv_path}")

        s2_ids = set()
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "s2_id" not in reader.fieldnames:
                raise ValueError(f"{csv_path} must have a 's2_id' column")
            for row in reader:
                s = row.get("s2_id")
                if s:
                    s2_ids.add(s.strip())

        num_classes[level_key] = len(s2_ids)

    return num_classes

def build_parent_tables(s2_labels_dir: str | Path, levels: list[int]):
    s2_labels_dir = Path(s2_labels_dir)
    levels = sorted(levels)

    # --- 1. Read CSVs and deduplicate S2 IDs ---
    ids = {}
    idx_map = {}

    for lvl in levels:
        L = f"L{lvl}"
        csv_path = s2_labels_dir / f"{L}.csv"

        unique_ids = []
        seen = set()
        
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                s2id = int(row["s2_id"])
                if s2id not in seen:
                    seen.add(s2id)
                    unique_ids.append(s2id)
        
        ids[L] = unique_ids
        idx_map[L] = {s2id: i for i, s2id in enumerate(unique_ids)}

    # --- 2. Build parent lookup tensors ---
    parents = {}

    for fine in levels:
        for coarse in levels:
            if coarse >= fine:
                continue

            fine_ids = ids[f"L{fine}"]
            parent_L = f"L{coarse}"

            parent_tensor = torch.empty(len(fine_ids), dtype=torch.long)

            for i, fine_s2id in enumerate(fine_ids):
                parent_s2id = CellId(fine_s2id).parent(coarse).id()
                try:
                    parent_tensor[i] = idx_map[parent_L][parent_s2id]
                except KeyError:
                    raise KeyError(
                        f"Parent S2 ID {parent_s2id} (from child {fine_s2id}) "
                        f"is not present in {parent_L}.csv"
                    )

            parents[(fine, coarse)] = parent_tensor

    return parents


