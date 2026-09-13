from __future__ import annotations

from pathlib import Path

import torch
from tqdm import tqdm

from .panorama import split_panorama


class Trainer:
    def __init__(self, model, processor, loader, loss_fn, device: str = "cuda"):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.processor = processor
        self.loader = loader
        self.loss_fn = loss_fn.to(self.device)

    @staticmethod
    def _coordinates(metadata: list[dict[str, str]]) -> list[tuple[float, float]]:
        coordinates = []

        for row in metadata:
            lat = row.get("lat", row.get("pano_lat"))
            lon = row.get("lon", row.get("pano_lon"))

            if lat is None or lon is None:
                raise KeyError("Metadata must contain lat/lon or pano_lat/pano_lon")

            coordinates.append((float(lon), float(lat)))

        return coordinates

    def _prepare_images(self, panoramas):
        views = []

        for panorama in panoramas:
            views.extend(crop.image for crop in split_panorama(panorama))

        num_views = len(views) // len(panoramas)
        image_inputs = self.processor(images=views, return_tensors="pt")
        image_inputs = {
            key: value.to(self.device)
            for key, value in image_inputs.items()
            if torch.is_tensor(value)
        }

        return image_inputs, num_views

    def train_epoch(self, optimizer) -> float:
        self.model.train()
        running_loss = 0.0
        batches = 0

        for panoramas, metadata in tqdm(self.loader, desc="Training"):
            image_inputs, num_views = self._prepare_images(panoramas)
            coordinates = self._coordinates(metadata)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ):
                fine_logits = self.model(image_inputs, len(panoramas), num_views)
                loss = self.loss_fn(fine_logits, coordinates)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batches += 1

        return running_loss / max(batches, 1)

    def save_checkpoint(self, optimizer, epoch: int, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            },
            path,
        )
