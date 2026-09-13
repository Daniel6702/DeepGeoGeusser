from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from adaptive_geogrid import AdaptiveGeoGrid
from modules.panorama import split_panorama

from .metrics import EvaluationMetrics, TOP_K_VALUES, haversine_km


class Evaluator:
    def __init__(self, model, processor, loader, grid: AdaptiveGeoGrid, device: str = "cuda"):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.processor = processor
        self.loader = loader
        self.grid = grid

        self.hierarchy_table = grid.hierarchy_table().to_numpy(dtype=np.int64)
        self.class_coordinates = self._class_coordinates()

    def _class_coordinates(self) -> np.ndarray:
        tiles = self.grid.level(0).sort_values("tile_id")
        tile_ids = tiles["tile_id"].to_numpy(dtype=np.int64)

        expected_ids = np.arange(self.grid.n_classes[0], dtype=np.int64)
        if not np.array_equal(tile_ids, expected_ids):
            raise ValueError("Fine grid tile IDs must be contiguous from 0 to n_classes - 1")

        points = tiles.geometry.representative_point()
        return np.column_stack((points.x.to_numpy(), points.y.to_numpy()))

    @staticmethod
    def _coordinates(metadata: list[dict[str, str]]) -> np.ndarray:
        coordinates = []

        for row in metadata:
            lat = row.get("lat", row.get("pano_lat"))
            lon = row.get("lon", row.get("pano_lon"))

            if lat is None or lon is None:
                raise KeyError("Metadata must contain lat/lon or pano_lat/pano_lon")

            coordinates.append((float(lon), float(lat)))

        return np.asarray(coordinates, dtype=np.float64)

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

    def evaluate(self, output_csv: str | Path | None = None) -> EvaluationMetrics:
        self.model.eval()

        all_coordinates = []
        all_true_paths = []
        all_top_classes = []
        all_metadata = []
        skipped = 0

        max_k = min(max(TOP_K_VALUES), self.grid.n_classes[0])

        with torch.inference_mode():
            for panoramas, metadata in tqdm(self.loader, desc="Evaluating"):
                coordinates = self._coordinates(metadata)
                true_paths = self.grid.classify(coordinates, include_hierarchy=True)
                valid = true_paths[:, 0] >= 0

                if not valid.any():
                    skipped += len(coordinates)
                    continue

                image_inputs, num_views = self._prepare_images(panoramas)

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.device.type == "cuda",
                ):
                    fine_logits = self.model(image_inputs, len(panoramas), num_views)

                top_classes = torch.topk(fine_logits, k=max_k, dim=1).indices.cpu().numpy()

                all_coordinates.append(coordinates[valid])
                all_true_paths.append(true_paths[valid])
                all_top_classes.append(top_classes[valid])
                if output_csv is not None:
                    all_metadata.extend(row for row, keep in zip(metadata, valid) if keep)
                skipped += int((~valid).sum())

        if not all_coordinates:
            raise ValueError("No evaluation panoramas fall inside the grid boundary")

        coordinates = np.concatenate(all_coordinates, axis=0)
        true_paths = np.concatenate(all_true_paths, axis=0)
        top_classes = np.concatenate(all_top_classes, axis=0)

        metrics = EvaluationMetrics.calculate(
            true_coordinates=coordinates,
            true_paths=true_paths,
            top_classes=top_classes,
            hierarchy_table=self.hierarchy_table,
            class_coordinates=self.class_coordinates,
        )

        metrics.print()

        if skipped:
            print()
            print(f"Skipped outside grid: {skipped:,}")

        if output_csv is not None:
            self._save_results(output_csv, all_metadata, coordinates, true_paths, top_classes)
            print(f"Saved per-panorama results: {output_csv}")

        return metrics

    def _save_results(
        self,
        path: str | Path,
        metadata: list[dict[str, str]],
        coordinates: np.ndarray,
        true_paths: np.ndarray,
        top_classes: np.ndarray,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        top_coordinates = self.class_coordinates[top_classes]
        top_distances = haversine_km(coordinates[:, None, :], top_coordinates)
        true_fine = true_paths[:, 0]

        fieldnames = [
            "panoid",
            "lat",
            "lon",
            "true_class",
            "pred_class",
            "error_km",
            "top5_hit",
            "top10_hit",
            "top5_error_km",
            "top10_error_km",
        ]

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for i, row in enumerate(metadata):
                top5 = min(5, top_classes.shape[1])
                top10 = min(10, top_classes.shape[1])

                writer.writerow(
                    {
                        "panoid": row.get("panoid", row.get("pano_id", "")),
                        "lat": coordinates[i, 1],
                        "lon": coordinates[i, 0],
                        "true_class": true_fine[i],
                        "pred_class": top_classes[i, 0],
                        "error_km": top_distances[i, 0],
                        "top5_hit": int(np.any(top_classes[i, :top5] == true_fine[i])),
                        "top10_hit": int(np.any(top_classes[i, :top10] == true_fine[i])),
                        "top5_error_km": np.min(top_distances[i, :top5]),
                        "top10_error_km": np.min(top_distances[i, :top10]),
                    }
                )
