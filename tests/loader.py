import os
import json
import torch
import webdataset as wds
import torchvision
from typing import Iterable, Literal

def load_dataset(
    dataset_path: str,
    processor: AutoImageProcessor,
    levels: Iterable[int] = (3,7),
    shuffle: bool = False,
    num_shards_limit = None,
    OS: Literal["WINDOWS", "LINUX"] = "LINUX")
):
    #Determine number of shards
    manifest_path = os.path.join(dataset_path, "shards_manifest.json")
    with open(manifest_path) as f:
        num_shards = len(json.load(f)["shards"])

    if num_shards_limit and num_shards > num_shards_limit:
        num_shards = num_shards_limit

    #Class index table
    level_keys = [f"L{lvl}" for lvl in levels]

    #Convert meta data from dataset to samples
    def _convert_sample(sample):
        img, meta = sample

        pixel_values = processor(
            img,
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
        )["pixel_values"].squeeze(0)

        class_vec = torch.tensor(
            [int(meta["s2"][lvl]) for lvl in level_keys],
            dtype=torch.long
        )

        return (pixel_values, class_vec)

    #Define
    if OS == "LINUX":
        urls = f"{dataset_path}/shard-{{000000..{num_shards - 1}}}.tar"
    elif OS == "WINDOWS":
        urls = f"file:{dataset_path}/shard-{{000000..{num_shards - 1}}}.tar"
    
    shardshuffle = 5 if shuffle else False

    dataset = (
        wds.WebDataset(urls, shardshuffle=shardshuffle)
        .decode("pil")
        .to_tuple("jpg", "json")
        .map(_convert_sample)
    )

    return dataset