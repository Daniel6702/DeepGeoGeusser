import torch
import torch.nn.functional as F

class HierarchicalLoss:
    """
    Hierarchical loss when the model outputs logits only at the finest S2 level.

    - Fine-level loss: cross-entropy with label smoothing.
    - Coarse-level loss: aggregate fine-level probabilities via S2 parent maps,
      then compute NLL loss.
    """
    def __init__(self, levels, parents, weights, num_classes_per_level, label_smoothing=0.1):
        """
        Args:
            levels: list of S2 levels (coarse → fine).
            parents: dict[(fine_level, coarse_level)] -> parent index tensor.
            weights: per-level weights (same order as levels).
            num_classes_per_level: list of num classes per level.
            label_smoothing: label smoothing value for fine-level CE loss.
        """
        self.levels = list(levels)
        self.parents = parents
        self.level_weights = weights
        self.num_classes_per_level = num_classes_per_level
        self.fine_level = self.levels[-1]

    def __call__(self, logits_fine, labels_per_level):
        """
        Compute the hierarchical loss.

        Args:
            logits_fine: [B, num_fine] logits at the finest S2 level.
            labels_per_level: list of label tensors [y_Lcoarse0, ..., y_Lfine].

        Returns:
            Scalar total loss combining fine and coarse level losses.
        """
        B, num_fine = logits_fine.shape
        device = logits_fine.device

        probs_fine = F.softmax(logits_fine, dim=1)

        y_fine = labels_per_level[-1]
        total_loss = self.level_weights[-1] * F.cross_entropy(logits_fine, y_fine, label_smoothing=0.1)

        for idx, coarse_level in enumerate(self.levels[:-1]):
            parent = self.parents[(self.fine_level, coarse_level)]
            num_coarse = self.num_classes_per_level[idx]

            probs_coarse = torch.zeros(B, num_coarse, device=device)
            probs_coarse.scatter_add_(1, parent.unsqueeze(0).expand(B, -1), probs_fine)

            log_probs_coarse = torch.log(probs_coarse.clamp(min=1e-12))
            y_coarse = labels_per_level[idx]

            total_loss += self.level_weights[idx] * F.nll_loss(log_probs_coarse, y_coarse)

        return total_loss
        
    def to(self, device):
        """
        Move parent lookup tensors to the given device.
        """
        self.parents = {k: v.to(device) for k, v in self.parents.items()}
        return self

class HierarchicalLoss_V2:
    """
    Hierarchical loss for models that output separate logits per level.

    Each level has its own classifier head; we compute a weighted sum of
    cross-entropy losses (with label smoothing) across levels.
    """
    def __init__(self, weights, label_smoothing=0.1):
        """
        Args:
            weights: per-level weights for combining losses.
            label_smoothing: label smoothing for all levels.
        """
        self.level_weights = weights
        self.label_smoothing = label_smoothing

    def __call__(self, logits_per_level, labels_per_level):
        """
        Args:
            logits_per_level: list of logits [logits_L0, logits_L1, ...].
            labels_per_level: list of labels [y_L0, y_L1, ...].

        Returns:
            Scalar total loss (weighted sum of CE losses).
        """
        total = 0.0
        for w, logits, y in zip(self.level_weights, logits_per_level, labels_per_level):
            total += w * F.cross_entropy(logits, y, label_smoothing=self.label_smoothing)
        return total

'''
parents is a dictionary of lookup tables.
For a given cell (class index) in a smaller layer, we can look which cell in the larger layer this smaller cell is inside.
parents[(fine, coarse)]
parents[(6,5)][1342]  ==  412
i.e. 
small cell L6 class index 1342
Belongs to large cell L5 class index 412
L6 1342 is geographically inside L5 412
'''