import torch
from tqdm import tqdm
import torch.nn.functional as F
import os

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class Trainer:
    def __init__(self, model, loader, device="cuda"):
        self.model = torch.nn.DataParallel(
            model,
            device_ids=list(range(torch.cuda.device_count()))
        ).to(device, memory_format=torch.channels_last)
        
        self.loader = loader
        self.device = device

        self.scaler = torch.amp.GradScaler("cuda")

    def train_epoch(self, optimizer, weights):
        self.model.train()
        for pixel_vals, class_vecs in tqdm(self.loader):
            pixel_vals = pixel_vals.to(self.device, memory_format=torch.channels_last)
            class_vecs = class_vecs.to(self.device)
            labels = [class_vecs[:, i] for i in range(class_vecs.shape[1])]

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = self.model(pixel_vals)
                loss = hierarchical_loss(logits, labels, weights)

            optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

    def save_checkpoint(self, optimizer, epoch, path="checkpoints/checkpoint.pt"):
        ckpt = {
            "model": self.model.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": epoch
        }
        torch.save(ckpt, path)

    def load_checkpoint(self, optimizer, path="checkpoints/checkpoint.pt"):
        if not os.path.exists(path):
            return 0  # no checkpoint → start at epoch 0

        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.module.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        self.scaler.load_state_dict(ckpt["scaler"])
        print(f"Loaded checkpoint from epoch {ckpt['epoch']}")
        return ckpt["epoch"] + 1 


def hierarchical_loss(logits_per_level, labels_per_level, weights):
    n_levels = len(logits_per_level)

    total_loss = 0.0

    for i in range(n_levels):
        ce = F.cross_entropy(logits_per_level[i], labels_per_level[i])
        total_loss += weights[i] * ce

    return total_loss