from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_geogrid import AdaptiveGeoGrid


class HierarchicalLoss(nn.Module):
    """Hierarchical classification loss built directly from an AdaptiveGeoGrid."""

    def __init__(self, grid: AdaptiveGeoGrid, weights: list[float] | None = None):
        super().__init__()
        self.grid = grid

        if weights is None:
            weights = [1.0] * grid.n_levels

        if len(weights) != grid.n_levels:
            raise ValueError(f"Expected {grid.n_levels} loss weights, got {len(weights)}")

        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        self.register_buffer("weights", weights_tensor / weights_tensor.sum())

        hierarchy = grid.hierarchy_table().to_numpy(dtype=np.int64)
        self.register_buffer("hierarchy", torch.from_numpy(hierarchy))

    def forward(
        self,
        fine_logits: torch.Tensor,
        coordinates: list[tuple[float, float]] | np.ndarray,
    ) -> torch.Tensor:
        """Coordinates must be WGS84 (longitude, latitude)."""
        labels = self.grid.classify(coordinates, include_hierarchy=True)
        labels = torch.as_tensor(labels, dtype=torch.long, device=fine_logits.device)

        valid = labels[:, 0] >= 0
        if not valid.any():
            raise ValueError("Batch contains no coordinates inside the grid boundary")

        fine_logits = fine_logits[valid]
        labels = labels[valid]

        total_loss = self.weights[0] * F.cross_entropy(fine_logits, labels[:, 0])
        fine_probs = F.softmax(fine_logits, dim=1)

        for level in range(1, self.grid.n_levels):
            parent = self.hierarchy[:, level]
            num_classes = self.grid.n_classes[level]

            coarse_probs = fine_probs.new_zeros(fine_probs.shape[0], num_classes)
            coarse_probs.scatter_add_(
                1,
                parent.unsqueeze(0).expand(fine_probs.shape[0], -1),
                fine_probs,
            )

            total_loss = total_loss + self.weights[level] * F.nll_loss(
                torch.log(coarse_probs.clamp_min(1e-12)),
                labels[:, level],
            )

        return total_loss
