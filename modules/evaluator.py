import torch
from tqdm import tqdm
from s2sphere import CellId
from sklearn.metrics.pairwise import haversine_distances
import pandas as pd
import matplotlib.pyplot as plt

EARTH_RADIUS_KM = 6371.0

class AccuracyEvalutor:
    """
    Compute top-k accuracy at each S2 level, given probabilities at the finest level.

    For coarser levels, probabilities are obtained by summing fine-level probabilities
    according to S2 parent maps.
    """
    def __init__(self, s2_levels, parent_maps, num_classes_per_level, topk=5):
        """
        Args:
            s2_levels: list of S2 levels (sorted coarse → fine).
            parent_maps: dict[(fine_level, coarse_level)] -> parent index tensor.
            num_classes_per_level: list with num classes for each level.
            topk: top-k to use for accuracy (e.g. 1, 5, 10).
        """
        self.s2_levels = s2_levels
        self.parent_maps = parent_maps
        self.num_classes_per_level = num_classes_per_level

        self.num_levels = len(num_classes_per_level)
        self.running_correct_per_level = [0] * self.num_levels
        self.running_total_per_level = [0] * self.num_levels

        self.topk = topk

    def __call__(self, probs_fine, class_indices):
        """
        Update running accuracy metrics from a batch.

        Args:
            probs_fine: [B, C_fine] probabilities at the finest S2 level.
            class_indices: [B, num_levels] true class indices at each level.
        """
        probs_per_level = []

        # Build probability tensors for each level
        for level_idx, level in enumerate(self.s2_levels):
            if level == self.s2_levels[-1]:
                probs_per_level.append(probs_fine)
            else:
                parent_map = self.parent_maps[(self.s2_levels[-1], level)]
                num_classes = self.num_classes_per_level[level_idx]
                probs_coarse = self._compute_probs_for_coarse_level(
                    probs_fine, parent_map, num_classes
                )
                probs_per_level.append(probs_coarse)

        # For each level, compute top-k accuracy and accumulate statistics
        for probs, targets, lvl_idx in zip(
            probs_per_level, class_indices.T, range(self.num_levels)
        ):
            _, topk_indices = torch.topk(probs, k=self.topk, dim=1)
            targets = targets.view(-1, 1)
            matches = topk_indices == targets
            correct = matches.any(dim=1)

            self.running_correct_per_level[lvl_idx] += correct.sum().item()
            self.running_total_per_level[lvl_idx] += probs.size(0)

    def get_final_accuracy(self):
        """
        Compute final accuracy per level as correct / total.
        """
        accuracy_per_level = [
            self.running_correct_per_level[lvl] / self.running_total_per_level[lvl]
            for lvl in range(self.num_levels)
        ]
        return accuracy_per_level

    def get_random_baseline(self):
        """
        Return random-guess baseline top-k accuracy for each level.

        If we pick k distinct classes uniformly out of n, the probability
        that the true class is in the top-k is min(k, n) / n.
        """
        random_baseline_accuracy = [
            min(self.topk, n) / n for n in self.num_classes_per_level
        ]
        return random_baseline_accuracy
        
    def _compute_probs_for_coarse_level(self, probs_fine, parent_map, num_classes_for_lvl):
        """
        Aggregate fine-level probabilities into a coarser level.

        Args:
            probs_fine: [B, C_fine] probabilities at fine level.
            parent_map: [C_fine] tensor mapping fine_idx -> coarse_idx.
            num_classes_for_lvl: number of coarse classes.

        Returns:
            probs_coarse: [B, C_coarse] probabilities at coarse level.
        """
        B = probs_fine.size(0)
        probs_coarse = torch.zeros(B, num_classes_for_lvl, device=self.device)
        probs_coarse.scatter_add_(1, parent_map.unsqueeze(0).expand(B, -1), probs_fine)
        return probs_coarse

    def to(self, device):
        """
        Move internal tensors to the given device.
        """
        self.device = device
        self.parent_maps = {k: v.to(self.device) for k, v in self.parent_maps.items()}


class DistanceEvaluator:
    """
    Evaluate model performance in terms of expected geodesic distance (in km)
    between the true S2 cell and the predicted distribution at the finest level.
    """
    def __init__(self, s2_levels, idx2id):
        """
        Args:
            s2_levels: list of S2 levels; last element is the finest level.
            idx2id: {level: [s2_id_0, s2_id_1, ...]} mapping class idx -> S2 ID.
        """
        fine_level = s2_levels[-1]
        s2_ids_fine = idx2id[fine_level]

        # Precompute lat/lon (radians) of each fine-level S2 cell center
        cell_coords = []
        for s2id in s2_ids_fine:
            ll = CellId(s2id).to_lat_lng()
            lat_rad = ll.lat().radians
            lon_rad = ll.lng().radians
            cell_coords.append([lat_rad, lon_rad])

        # Pairwise haversine distance matrix between all fine cells (in km)
        self.distance_matrix = torch.tensor(
            haversine_distances(cell_coords) * EARTH_RADIUS_KM,
            dtype=torch.float32
        )

        # Running totals for expected distance error
        self.total_distance_error = 0.0
        self.total_samples = 0

    def __call__(self, probs_fine, class_indices):
        """
        Update running distance error from a batch.

        Args:
            probs_fine: [B, C] probabilities over fine-level cells.
            class_indices: [B, num_levels] true class indices; last column is fine level.
        """
        target_idx = class_indices[:, -1]              # [B]
        distances = self.distance_matrix[target_idx]   # [B, C]
        expected_per_sample = (distances * probs_fine).sum(dim=1)  # [B]
        self.total_distance_error += expected_per_sample.sum().item()
        self.total_samples += probs_fine.size(0)

    def get_final_distance_error(self):
        """
        Return mean expected distance error (km) over all samples.
        """
        distance_error = self.total_distance_error / self.total_samples
        return distance_error

    def get_random_baseline(self):
        """
        Random baseline distance: average pairwise distance between all cells.
        """
        random_baseline = self.distance_matrix.mean().item()
        return random_baseline

    def to(self, device):
        """
        Move distance matrix to the given device.
        """
        self.distance_matrix = self.distance_matrix.to(device)
        return self

class Top1GeodesicHistogramEvaluator:
    """
    Compute top-1 geodesic error (in km) and build a histogram of errors.

    Uses the S2 cell centers at the finest level to precompute a pairwise
    distance matrix, then for each sample measures the distance between:
        - the true fine-level cell
        - the predicted top-1 fine-level cell
    """
    def __init__(self, s2_levels, idx2id):
        """
        Args:
            s2_levels: list of S2 levels; last element is the finest level.
            idx2id: {level: [s2_id_0, s2_id_1, ...]} mapping class idx -> S2 ID.
        """
        fine_level = s2_levels[-1]
        s2_ids_fine = idx2id[fine_level]

        # Precompute lat/lon (radians) of each fine-level S2 cell center
        cell_coords = []
        for s2id in s2_ids_fine:
            ll = CellId(s2id).to_lat_lng()
            lat_rad = ll.lat().radians
            lon_rad = ll.lng().radians
            cell_coords.append([lat_rad, lon_rad])

        # Pairwise haversine distance matrix between all fine cells (in km)
        self.distance_matrix = torch.tensor(
            haversine_distances(cell_coords) * EARTH_RADIUS_KM,
            dtype=torch.float32
        )

        # Store per-sample errors (on CPU to avoid GPU memory blowup)
        self._errors_km = []

    def __call__(self, probs_fine, class_indices):
        """
        Update running top-1 distance errors from a batch.

        Args:
            probs_fine: [B, C] probabilities (or logits after softmax) over fine-level cells.
            class_indices: [B, num_levels] true class indices; last column is fine level.
        """
        # probs_fine can be logits too; apply softmax if you want probs,
        # but argmax is the same for logits and probs.
        with torch.no_grad():
            # Predicted top-1 fine-level index
            pred_idx = probs_fine.argmax(dim=1)          # [B]

            # True fine-level index (last column of class_indices)
            target_idx = class_indices[:, -1].long()     # [B]

            # Distance between true and predicted cells
            batch_errors = self.distance_matrix[target_idx, pred_idx]  # [B]

            # Move to CPU and store
            self._errors_km.append(batch_errors.detach().cpu())

    def get_errors(self):
        """
        Return all collected top-1 errors as a 1D tensor (km).
        """
        if not self._errors_km:
            return torch.empty(0)
        return torch.cat(self._errors_km, dim=0)

    def get_mean_error(self):
        """
        Return the mean top-1 geodesic error over all samples (km).
        """
        errors = self.get_errors()
        if errors.numel() == 0:
            return float("nan")
        return errors.mean().item()

    def get_histogram(self, bin_width=100.0, max_km=6000.0):
        """
        Compute a histogram of the collected errors.

        Args:
            bin_width: width of each bin in km (smaller = more fine-grained groups).
            max_km: maximum error to include in the histogram (km).

        Returns:
            hist: counts per bin (numpy array)
            bin_edges: bin edges (numpy array, length n_bins + 1)
        """
        errors = self.get_errors().numpy()
        if errors.size == 0:
            return np.array([]), np.array([])

        # Keep only errors <= max_km
        errors = errors[errors <= max_km]

        n_bins = int(max_km // bin_width)
        hist, bin_edges = np.histogram(
            errors,
            bins=n_bins,
            range=(0.0, max_km)
        )
        return hist, bin_edges

    def plot_histogram(self, bin_width=50.0, max_km=3000.0):
        """
        Plot a histogram of the collected errors using matplotlib, limited to [0, max_km].

        Args:
            bin_width: width of each bin in km.
            max_km: maximum error to show on the x-axis (km).

        Returns:
            The matplotlib Axes object for further customization.
        """
        errors = self.get_errors().numpy()
        if errors.size == 0:
            fig, ax = plt.subplots()
            ax.set_xlabel("km")
            ax.set_ylabel("Count")
            ax.set_title("Top-1 geodesic error histogram (no data)")
            fig.tight_layout()
            return ax

        # Filter and bin as in get_histogram
        errors = errors[errors <= max_km]
        n_bins = int(max_km // bin_width)

        fig, ax = plt.subplots(figsize=(7, 3))
        ax.hist(errors, bins=n_bins, range=(0.0, max_km))
        ax.set_xlim(0.0, max_km)
        ax.set_xlabel("km")
        ax.set_ylabel("Count")
        ax.set_title(f"Geodesic error histogram")
        fig.tight_layout()
        fig.savefig("histogram.pdf", format="pdf")
        return ax


    def to(self, device):
        """
        Move distance matrix to the given device.
        (Stored errors stay on CPU.)
        """
        self.distance_matrix = self.distance_matrix.to(device)
        return self

class Evaluator:
    """
    High-level evaluation wrapper: runs the model over a dataloader and
    reports top-k accuracy at each S2 level plus distance error.
    """
    def __init__(self, model, loader, s2_levels, parent_maps, num_classes_per_level, idx2id):
        """
        Args:
            model: trained model to evaluate.
            loader: data loader yielding (pixel_values, class_indices).
            s2_levels: list of S2 levels (coarse → fine).
            parent_maps: dict[(fine_level, coarse_level)] -> parent index tensor.
            num_classes_per_level: list of num classes per S2 level.
            idx2id: {level: [s2_id_0, ...]} mapping idx -> S2 ID.
        """
        self.model = model
        self.loader = loader
        self.s2_levels = s2_levels

        self.accuracy_evaluators = [
            AccuracyEvalutor(s2_levels, parent_maps, num_classes_per_level, k)
            for k in [1, 5, 10]
        ]

        self.distance_evaluator = DistanceEvaluator(s2_levels, idx2id)
        self.top1_geo_eval = Top1GeodesicHistogramEvaluator(s2_levels, idx2id)

    def to(self, device):
        """
        Move model and evaluators to the given device, and set channels_last
        memory format for potential performance gains on GPU.
        """
        self.model = self.model.to(device, memory_format=torch.channels_last)
        for evaluator in self.accuracy_evaluators:
            evaluator.to(device)
        self.distance_evaluator.to(device)
        self.device = device
        self.top1_geo_eval.to(device)
        return self

    def load_checkpoint(self, path="checkpoints/checkpoint.pt"):
        """
        Load model weights from a checkpoint file (if it exists).

        Prints the epoch from which the checkpoint was saved.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        print(f"Epoch: {ckpt['epoch']}")
        self.model.load_state_dict(ckpt["model"])

    def __call__(self, max_batches):
        """
        Run evaluation for at most `max_batches` batches.

        Returns:
            A styled pandas DataFrame summarizing:
              - top-k accuracies per level
              - random baselines
              - distance error and its random baseline
        """
        
        
        self.model.eval()

        with torch.no_grad():
            for batch_idx, (pixel_values, class_indices) in enumerate(
                tqdm(self.loader, total=max_batches, desc="Evaluating")
            ):
                if batch_idx >= max_batches:
                    break

                pixel_values = pixel_values.to(self.device, memory_format=torch.channels_last)
                class_indices = class_indices.to(self.device)

                logits = self.model(pixel_values)
                probs_fine = torch.softmax(logits, dim=1)

                for evaluator in self.accuracy_evaluators:
                    evaluator(probs_fine, class_indices)

                self.distance_evaluator(probs_fine, class_indices)
                self.top1_geo_eval(probs_fine, class_indices)

        print("Mean top-1 geodesic error (km):", self.top1_geo_eval.get_mean_error())
        self.top1_geo_eval.plot_histogram()

        accuracies = [evaluator.get_final_accuracy() for evaluator in self.accuracy_evaluators]
        random_baselines = [evaluator.get_random_baseline() for evaluator in self.accuracy_evaluators]

        # Distance error and random baseline
        distance_error = self.distance_evaluator.get_final_distance_error()
        random_baseline_distance_error = self.distance_evaluator.get_random_baseline()
        
        # Create table
        rows = []
        for evaluator, acc_per_level, rand_per_level in zip(
            self.accuracy_evaluators, accuracies, random_baselines
        ):
            topk = evaluator.topk
            for lvl_idx, (acc, rand_acc) in enumerate(
                zip(acc_per_level, rand_per_level)
            ):
                lvl = self.s2_levels[lvl_idx]
                rows.append({
                    "Top-k": topk,
                    "S2 level": lvl,
                    "Accuracy": acc,
                    "Random baseline": rand_acc
                })

        # Add distance-error row (for finest level only)
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
