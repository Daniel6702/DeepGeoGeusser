import torch
from tqdm import tqdm

class Evaluator:
    def __init__(self, model, loader, levels, parents, device="cuda"):
        self.model = model
        self.loader = loader
        self.levels = list(levels)
        self.parents = parents
        self.device = device

        self.fine_level = self.levels[-1]  # e.g. 6

    @torch.no_grad()
    def evaluate(self, max_batches=None, max_samples=None):
        self.model.eval()

        # one accuracy counter per level
        correct = [0 for _ in self.levels]
        seen = 0

        for batch_idx, (pixel_vals, class_vecs) in enumerate(tqdm(self.loader)):

            if max_batches is not None and batch_idx >= max_batches:
                break
            if max_samples is not None and seen >= max_samples:
                break

            pixel_vals = pixel_vals.to(self.device)
            class_vecs = class_vecs.to(self.device)  # (B, num_levels)

            # model returns fine logits only
            logits_fine = self.model(pixel_vals)  # (B, num_fine)

            batch_size = class_vecs.size(0)
            effective_batch = batch_size
            if max_samples is not None:
                remaining = max_samples - seen
                if remaining <= 0:
                    break
                if remaining < batch_size:
                    effective_batch = remaining

            logits_fine = logits_fine[:effective_batch]
            class_vecs_batch = class_vecs[:effective_batch]

            # fine-level predictions (indices in [0, num_fine-1])
            fine_pred = torch.argmax(logits_fine, dim=1)  # (effective_batch,)

            # level-wise accuracies
            for idx, lvl in enumerate(self.levels):
                if lvl == self.fine_level:
                    # fine level: direct comparison
                    y = class_vecs_batch[:, idx]
                    correct[idx] += (fine_pred == y).sum().item()
                else:
                    # coarse: use parent mapping from fine -> lvl
                    parent = self.parents[(self.fine_level, lvl)].to(self.device)  # (num_fine,)
                    pred_coarse = parent[fine_pred]  # (effective_batch,)
                    y = class_vecs_batch[:, idx]
                    correct[idx] += (pred_coarse == y).sum().item()

            seen += effective_batch

        if seen == 0:
            return {
                "samples_used": 0,
                "per_level_accuracy": {lvl: 0.0 for lvl in self.levels},
                "mean_accuracy": 0.0,
            }

        per_level_acc = {lvl: correct[i] / seen for i, lvl in enumerate(self.levels)}
        mean_acc = sum(per_level_acc.values()) / len(per_level_acc)

        return {
            "samples_used": seen,
            "per_level_accuracy": per_level_acc,
            "mean_accuracy": mean_acc
        }
