import torch
import torch.nn.functional as F

class HierarchicalLoss:
    def __init__(self, levels, parents, weights, num_classes_per_level):
        self.levels = list(levels)
        self.parents = parents
        self.level_weights = weights
        self.num_classes_per_level = num_classes_per_level
        self.fine_level = self.levels[-1]

    def __call__(self, logits_fine, labels_per_level):
        B, num_fine = logits_fine.shape
        device = logits_fine.device

        probs_fine = F.softmax(logits_fine, dim=1)

        # fine level (last)
        y_fine = labels_per_level[-1]
        total_loss = self.level_weights[-1] * F.cross_entropy(logits_fine, y_fine)

        # coarse levels
        for idx, coarse_level in enumerate(self.levels[:-1]):
            parent = self.parents[(self.fine_level, coarse_level)]
            num_coarse = self.num_classes_per_level[idx]

            # safety check: parents must be valid indices
            #assert parent.max().item() < num_coarse

            probs_coarse = torch.zeros(B, num_coarse, device=device)
            probs_coarse.scatter_add_(1, parent.unsqueeze(0).expand(B, -1), probs_fine)

            log_probs_coarse = torch.log(probs_coarse + 1e-12)
            y_coarse = labels_per_level[idx]

            # safety check: labels must be valid indices
            #assert 0 <= y_coarse.min().item() < num_coarse
            #assert y_coarse.max().item() < num_coarse

            total_loss += self.level_weights[idx] * F.nll_loss(log_probs_coarse, y_coarse)

        return total_loss
    def to(self, device):
        self.parents = {k: v.to(device) for k, v in self.parents.items()}
        return self

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