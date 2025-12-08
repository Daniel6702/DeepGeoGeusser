from transformers import ConvNextModel, ConvNextConfig
import torch.nn as nn
import torch

class HierarchicalConvNeXt(nn.Module):
    """
    ConvNeXt-based model with a single classifier head at the finest S2 level.

    Used together with HierarchicalLoss to obtain coarse-level supervision
    by aggregating fine-level probabilities.
    """
    def __init__(self, num_classes: int, pretrained_name="facebook/convnext-tiny-224", freeze=False):
        super().__init__()

        self.backbone = ConvNextModel.from_pretrained(
            pretrained_name,
            ignore_mismatched_sizes=True,
            use_safetensors=True,
        )
        hidden_dim = self.backbone.config.hidden_sizes[-1]

        if freeze:  # Freeze ConvNext parameters. only the linear head is trained. Faster. might not actually learn anything.
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.head_fine = nn.Linear(hidden_dim, num_classes)

    def forward(self, pixel_values):
        """
        Args:
            pixel_values: image tensor as expected by ConvNeXt (B, C, H, W).

        Returns:
            logits_fine: [B, num_classes] logits at the finest S2 level.
        """
        outputs = self.backbone(pixel_values)
        pooled = outputs.pooler_output
        logits_fine = self.head_fine(pooled)  # (B, num_L6)
        return logits_fine


class HierarchicalConvNeXt_V2(nn.Module):
    """
    ConvNeXt-based model with one classifier head per S2 level.

    Produces a list of logits: [logits_L0, logits_L1, ..., logits_Lfine].
    """
    def __init__(self, num_classes_per_level, pretrained_name="facebook/convnext-large-384"):
        """
        Args:
            num_classes_per_level: list of num classes for each S2 level.
            pretrained_name: HF ConvNeXt model name (e.g. large-384).
        """
        super().__init__()

        self.backbone = ConvNextModel.from_pretrained(
            pretrained_name,
            ignore_mismatched_sizes=True,
            use_safetensors=True
        )
        hidden_dim = self.backbone.config.hidden_sizes[-1]

        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim, n) for n in num_classes_per_level]
        )

    def forward(self, pixel_values):
        """
        Args:
            pixel_values: image tensor as expected by ConvNeXt.

        Returns:
            logits_per_level: list of [B, num_classes_level_i] logits tensors.
        """
        outputs = self.backbone(pixel_values)
        pooled = outputs.pooler_output
        logits_per_level = [head(pooled) for head in self.heads]
        return logits_per_level
