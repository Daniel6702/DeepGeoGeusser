import os, json
import webdataset as wds
import torch
import csv
from pathlib import Path
from s2sphere import CellId 
from .labels_utils import build_s2_index_maps, latlon_to_s2id

class GeoWebDataset:
    def __init__(self, dataset_path, processor, levels,
                 shuffle=False, num_shards_limit=None,
                 os_type="LINUX", id2idx=None):
        
        self.processor = processor
        self.levels = list(levels)
        self.level_keys = [f"L{lvl}" for lvl in levels]
        self.urls = self._get_urls(dataset_path, num_shards_limit, os_type)

        # Build index maps if not provided
        s2_labels_dir = Path(dataset_path) / "s2_labels"
        if id2idx is None:
            _, self.id2idx, _ = build_s2_index_maps(s2_labels_dir, self.levels)
        else:
            self.id2idx = id2idx

        # num_classes per level in canonical indexing
        self.num_classes_list = [
            len(self.id2idx[lvl]) for lvl in self.levels
        ]

        self.dataset = (
            wds.WebDataset(self.urls, shardshuffle=100 if shuffle else False)
            .shuffle(1000) 
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

    def _process_sample(self, sample):
        img, meta = sample
        pixel_values = self.processor(img, return_tensors="pt")["pixel_values"].squeeze(0)
        
        lat = float(meta["pano_lat"])
        lon = float(meta["pano_lon"])
        
        class_indices = []
        for lvl in self.levels:
            s2id = latlon_to_s2id(lat, lon, lvl)
            try:
                idx = self.id2idx[lvl][s2id]
            except KeyError:
                raise KeyError(
                    f"S2 id {s2id} (from lat={lat}, lon={lon}, L={lvl}) "
                    f"is not present in id2idx for that level."
                )
            class_indices.append(idx)

        class_vec = torch.tensor(class_indices, dtype=torch.long)

        #coords = torch.tensor([lat, lon], dtype=torch.float32)
        return pixel_values, class_vec#, coords

def build_parent_tables_from_maps(idx2id: dict[int, list[int]],
                                  id2idx: dict[int, dict[int, int]],
                                  levels: list[int]):
    levels = sorted(levels)
    parents = {}

    for fine in levels:
        for coarse in levels:
            if coarse >= fine:
                continue

            fine_ids = idx2id[fine]
            parent_tensor = torch.empty(len(fine_ids), dtype=torch.long)

            for i, fine_s2id in enumerate(fine_ids):
                parent_s2id = CellId(fine_s2id).parent(coarse).id()
                try:
                    parent_idx = id2idx[coarse][parent_s2id]
                except KeyError:
                    raise KeyError(
                        f"Parent S2 ID {parent_s2id} (from child {fine_s2id} at L{fine}) "
                        f"is not present in coarse level L{coarse} index map."
                    )
                parent_tensor[i] = parent_idx

            parents[(fine, coarse)] = parent_tensor

    return parents
