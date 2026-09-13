from __future__ import annotations

import torch
import torch.nn as nn
from transformers import SiglipVisionModel

from adaptive_geogrid import AdaptiveGeoGrid


class GeoModel(nn.Module):
    """Shared SigLIP2 encoder with late fusion across panorama views."""

    def __init__(
        self,
        grid: AdaptiveGeoGrid,
        model_name: str = "google/siglip-base-patch16-224", #"google/siglip2-base-patch16-512",
        freeze_backbone: bool = False,
    ):
        super().__init__()

        self.backbone = SiglipVisionModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        self.head = nn.Linear(hidden_size, grid.n_classes[0])

    def forward(self, image_inputs: dict[str, torch.Tensor], batch_size: int, num_views: int):
        """
        Encode each view independently and average the fine-class logits across
        the views belonging to the same panorama.
        """
        features = self.backbone(**image_inputs).pooler_output
        features = features.reshape(batch_size, num_views, -1)
        view_logits = self.head(features)
        return view_logits.mean(dim=1)
