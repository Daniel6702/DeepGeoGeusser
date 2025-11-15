from transformers import ConvNextModel, ConvNextConfig
import torch.nn as nn
import torch

class HierarchicalConvNeXt(nn.Module):
    def __init__(self, num_classes_per_level: list, pretrained_name="facebook/convnext-tiny-224"):
        super().__init__()

        self.backbone = ConvNextModel.from_pretrained(
            pretrained_name,
            ignore_mismatched_sizes=True,
            use_safetensors=True
        )
        hidden_dim = self.backbone.config.hidden_sizes[-1]

        #create new classifier heads for each hierarchy level
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, n_classes)
            for n_classes in num_classes_per_level
        ])

    def forward(self, pixel_values) -> list:
        outputs = self.backbone(pixel_values)
        pooled = outputs.pooler_output 
        logits_per_level = [head(pooled) for head in self.heads]

        return logits_per_level