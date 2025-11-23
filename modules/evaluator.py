import torch
from tqdm import tqdm
from s2sphere import CellId
from sklearn.metrics.pairwise import haversine_distances
import pandas as pd

EARTH_RADIUS_KM = 6371.0

class AccuracyEvalutor:
    def __init__(self, s2_levels, parent_maps, num_classes_per_level, topk = 5):
        self.s2_levels = s2_levels
        self.parent_maps = parent_maps
        self.num_classes_per_level = num_classes_per_level

        self.num_levels = len(num_classes_per_level)
        self.running_correct_per_level = [0] * self.num_levels
        self.running_total_per_level = [0] * self.num_levels

        self.topk = topk

    def __call__(self, probs_fine, class_indices):
        
        probs_per_level = []
        
        for level_idx, level in enumerate(self.s2_levels):
            if level == self.s2_levels[-1]:
                probs_per_level.append(probs_fine)
            else:
                parent_map   = self.parent_maps[(self.s2_levels[-1], level)]
                num_classes  = self.num_classes_per_level[level_idx]
                probs_coarse = self._compute_probs_for_coarse_level(
                    probs_fine, parent_map, num_classes
                )
                probs_per_level.append(probs_coarse)

        for probs, targets, lvl_idx in zip(probs_per_level, class_indices.T, range(self.num_levels)):
            _, topk_indices = torch.topk(probs, k=self.topk, dim=1)
            targets = targets.view(-1, 1) 
            matches = topk_indices == targets
            correct = matches.any(dim=1)
            
            self.running_correct_per_level[lvl_idx] += correct.sum().item()
            self.running_total_per_level[lvl_idx] += probs.size(0)

    def get_final_accuracy(self):
        accuracy_per_level = [
            self.running_correct_per_level[lvl] / self.running_total_per_level[lvl]
            for lvl in range(self.num_levels)
        ]
        return accuracy_per_level

    def get_random_baseline(self):
        random_baseline_accuracy = [1 / n for n in self.num_classes_per_level]
        return random_baseline_accuracy

    def _compute_probs_for_coarse_level(self, probs_fine, parent_map, num_classes_for_lvl):
        B = probs_fine.size(0)
        probs_coarse = torch.zeros(B, num_classes_for_lvl, device=self.device)
        probs_coarse.scatter_add_(1,parent_map.unsqueeze(0).expand(B, -1), probs_fine)
        return probs_coarse

    def to(self, device):
        self.device = device
        self.parent_maps = {k: v.to(self.device) for k, v in self.parent_maps.items()}

class DistanceEvaluator:
    def __init__(self, s2_levels, idx2id):
        fine_level = s2_levels[-1]
        s2_ids_fine = idx2id[fine_level]
        
        cell_coords = []
        for s2id in s2_ids_fine:
            ll = CellId(s2id).to_lat_lng()
            lat_rad = ll.lat().radians
            lon_rad = ll.lng().radians
            cell_coords.append([lat_rad, lon_rad])
            
        self.distance_matrix = torch.tensor(
            haversine_distances(cell_coords) * EARTH_RADIUS_KM, 
            dtype=torch.float32
        )

        self.total_distance_error = 0.0
        self.total_samples = 0

    def __call__(self, probs_fine, class_indices):
        target_idx = class_indices[:, -1]              # [B]
        distances = self.distance_matrix[target_idx]   # [B, C]
        expected_per_sample = (distances * probs_fine).sum(dim=1)  # [B]
        self.total_distance_error += expected_per_sample.sum().item()
        self.total_samples += probs_fine.size(0)
        
    def get_final_distance_error(self):
        distance_error = self.total_distance_error / self.total_samples
        return distance_error

    def get_random_baseline(self):
        random_baseline = self.distance_matrix.mean().item()
        return random_baseline
    
    def to(self, device):
        self.distance_matrix = self.distance_matrix.to(device)
        return self


class Evaluator():
    def __init__(self, model, loader, s2_levels, parent_maps, num_classes_per_level, idx2id):
        self.model = model
        self.loader = loader
        self.s2_levels = s2_levels
        
        self.accuracy_evaluators = [AccuracyEvalutor(s2_levels, parent_maps, num_classes_per_level, k)
                                    for k in [1,5,10]]
        
        self.distance_evaluator = DistanceEvaluator(s2_levels, idx2id)

    def to(self, device):
        self.model = self.model.to(device, memory_format=torch.channels_last)
        for evaluator in self.accuracy_evaluators:
            evaluator.to(device)
        self.distance_evaluator.to(device)
        self.device = device
        return self

    def load_checkpoint(self, path="checkpoints/checkpoint.pt"):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        print(f"Epoch: {ckpt['epoch']}")
        self.model.load_state_dict(ckpt["model"])

    def __call__(self, max_batches):
        self.model.eval()

        with torch.no_grad():
            for batch_idx, (pixel_values, class_indices) in enumerate(tqdm(self.loader, total=max_batches, desc="Evaluating")):
                
                if batch_idx >= max_batches:
                    break

                pixel_values = pixel_values.to(self.device, memory_format=torch.channels_last)
                class_indices = class_indices.to(self.device)

                logits = self.model(pixel_values)
                probs_fine = torch.softmax(logits, dim=1)

                for evaluator in self.accuracy_evaluators:
                    evaluator(probs_fine, class_indices)
                
                self.distance_evaluator(probs_fine, class_indices)


        accuracies = [evaluator.get_final_accuracy() for evaluator in self.accuracy_evaluators]
        random_baseline_accuracy = self.accuracy_evaluators[0].get_random_baseline()
        
        distance_error = self.distance_evaluator.get_final_distance_error()
        random_baseline_distance_error = self.distance_evaluator.get_random_baseline()

        #CREATE TABLE
        rows = []
        for evaluator, acc_per_level in zip(self.accuracy_evaluators, accuracies):
            topk = evaluator.topk
            for lvl_idx, (acc, rand_acc) in enumerate(
                zip(acc_per_level, random_baseline_accuracy)
            ):
                lvl = self.s2_levels[lvl_idx]
                rows.append({
                    "Top-k": topk,
                    "S2 level": lvl,
                    "Accuracy": acc,
                    "Random baseline": rand_acc
                })
        
        rows.append({
            "Top-k": "-",
            "S2 level": f"{self.s2_levels[-1]} (distance km)",
            "Accuracy": distance_error,
            "Random baseline": random_baseline_distance_error
        })
        
        df = pd.DataFrame(rows)
        
        styled = (
            df.style
              .format({
                  "Accuracy": "{:.4f}",
                  "Random baseline": "{:.4f}"
              })
              .set_caption("Evaluation results")
        )
        
        return styled