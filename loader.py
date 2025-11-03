import webdataset as wds
import json
import torchvision

def load_dataset(dataset_path: str, lookup_class_label_from_panoid_method: callable, shuffle: bool = False) -> wds.WebDataset:
    '''WINDOWS'''

    with open(f"{dataset_path}/shards_manifest.json") as f:
        manifest = json.load(f)
    num_shards = len(manifest["shards"])

    urls = f"file:{dataset_path}/shard-{{000000..{num_shards - 1}}}.tar"

    dataset = (
        wds.WebDataset(urls, shardshuffle=shuffle)  
        .decode("pil")
        .to_tuple("jpg", "json")
        .map(lambda sample: (
            torchvision.transforms.ToTensor()(sample[0]),
            lookup_class_label_from_panoid_method(sample[1]["panoid"])
        ))
    )

    return dataset