from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EARTH_RADIUS_KM = 6371.0088
DISTANCE_THRESHOLDS_KM = (1, 10, 25, 100, 500)
TOP_K_VALUES = (1, 5, 10)


def haversine_km(
    true_coordinates: np.ndarray,
    predicted_coordinates: np.ndarray,
) -> np.ndarray:
    """Great-circle distance in km for (..., 2) arrays stored as (lon, lat)."""
    true_coordinates = np.asarray(true_coordinates, dtype=np.float64)
    predicted_coordinates = np.asarray(predicted_coordinates, dtype=np.float64)

    lon1 = np.radians(true_coordinates[..., 0])
    lat1 = np.radians(true_coordinates[..., 1])
    lon2 = np.radians(predicted_coordinates[..., 0])
    lat2 = np.radians(predicted_coordinates[..., 1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


@dataclass
class EvaluationMetrics:
    count: int
    geographic_error_km: np.ndarray
    hierarchy_accuracy: np.ndarray
    top_k_accuracy: dict[int, float]
    best_top_k_error_km: dict[int, np.ndarray]

    @classmethod
    def calculate(
        cls,
        true_coordinates: np.ndarray,
        true_paths: np.ndarray,
        top_classes: np.ndarray,
        hierarchy_table: np.ndarray,
        class_coordinates: np.ndarray,
    ) -> "EvaluationMetrics":
        predicted_paths = hierarchy_table[top_classes[:, 0]]
        hierarchy_accuracy = np.mean(predicted_paths == true_paths, axis=0)

        top_k_accuracy = {}
        best_top_k_error_km = {}

        true_fine = true_paths[:, 0]
        top_coordinates = class_coordinates[top_classes]
        all_top_distances = haversine_km(true_coordinates[:, None, :], top_coordinates)

        for k in TOP_K_VALUES:
            effective_k = min(k, top_classes.shape[1])
            top_k_accuracy[k] = float(
                np.mean(np.any(top_classes[:, :effective_k] == true_fine[:, None], axis=1))
            )
            best_top_k_error_km[k] = np.min(all_top_distances[:, :effective_k], axis=1)

        return cls(
            count=len(true_coordinates),
            geographic_error_km=all_top_distances[:, 0],
            hierarchy_accuracy=hierarchy_accuracy,
            top_k_accuracy=top_k_accuracy,
            best_top_k_error_km=best_top_k_error_km,
        )

    def print(self) -> None:
        errors = self.geographic_error_km

        print(f"Test panoramas: {self.count:,}")
        print()
        print("Geographic error:")
        print(f"  Median: {np.median(errors):10.1f} km")
        print(f"  Mean:   {np.mean(errors):10.1f} km")
        print(f"  P75:    {np.percentile(errors, 75):10.1f} km")
        print(f"  P90:    {np.percentile(errors, 90):10.1f} km")
        print()
        print("Distance accuracy:")
        for threshold in DISTANCE_THRESHOLDS_KM:
            accuracy = np.mean(errors < threshold) * 100.0
            print(f"  < {threshold:4d} km: {accuracy:8.1f}%")
        print()
        print("Grid accuracy:")
        for level, accuracy in enumerate(self.hierarchy_accuracy):
            name = "Fine" if level == 0 else f"Level {level}"
            print(f"  {name:<8} {accuracy * 100.0:8.1f}%")
        print()
        print("Top-K fine cells:")
        for k in TOP_K_VALUES:
            print(f"  Top-{k:<2} {self.top_k_accuracy[k] * 100.0:12.1f}%")
        print()
        print("Best geographic error among:")
        for k in TOP_K_VALUES:
            median = np.median(self.best_top_k_error_km[k])
            print(f"  Top-{k:<2} {median:12.1f} km median")
