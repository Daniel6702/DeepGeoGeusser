import webdataset as wds
import json

def load_dataset(DATASET_PATH: str, shuffle: bool = False) -> wds.WebDataset:

    with open(f"{DATASET_PATH}/shards_manifest.json") as f:
        manifest = json.load(f)
    num_shards = len(manifest["shards"])

    urls = f"file:{DATASET_PATH}/shard-{{000000..{num_shards - 1}}}.tar"

    dataset = wds.WebDataset(urls, shardshuffle=shuffle).decode("pil").to_tuple("jpg", "json")

    return dataset