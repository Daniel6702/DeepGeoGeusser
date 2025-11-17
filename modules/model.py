from transformers import ConvNextModel, ConvNextConfig
import torch.nn as nn
import torch

class HierarchicalConvNeXt(nn.Module):
    def __init__(self, num_classes: int, pretrained_name="facebook/convnext-tiny-224", freeze = False):
        super().__init__()

        self.backbone = ConvNextModel.from_pretrained(
            pretrained_name,
            ignore_mismatched_sizes=True,
            use_safetensors=True,
        )
        hidden_dim = self.backbone.config.hidden_sizes[-1]

        if freeze: #Freeze ConvNext parameters. only the linear head is trained. Faster. might not actually learn anything. 
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.head_fine = nn.Linear(hidden_dim, num_classes)

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values)
        pooled = outputs.pooler_output
        logits_fine = self.head_fine(pooled)  # (B, num_L6)
        return logits_fine