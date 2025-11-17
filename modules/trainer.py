import torch
from tqdm import tqdm
import torch.nn.functional as F
import os

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("medium") #might be faster?

class Trainer:
    def __init__(self, model, loader, loss_fn, device="cuda"):

        #Multi GPU
        #self.model = torch.nn.DataParallel(
        #   model,
        #    device_ids=list(range(torch.cuda.device_count()))
        #).to(device, memory_format=torch.channels_last)

        #Single GPU
        gpu_id = 0
        self.model = model.to(f"cuda:{gpu_id}", memory_format=torch.channels_last)

        self.loader = loader
        self.loss_fn = loss_fn
        self.device = device
        self.scaler = torch.amp.GradScaler(self.device)

    def train_epoch(self, optimizer, weights):
        self.model.train()
        running_loss = 0.0
        count = 0
        
        for pixel_vals, class_vecs in tqdm(self.loader):
            pixel_vals = pixel_vals.to(self.device, memory_format=torch.channels_last)
            class_vecs = class_vecs.to(self.device)
            labels = [class_vecs[:, i] for i in range(class_vecs.shape[1])]

            with torch.amp.autocast(self.device, dtype=torch.bfloat16):
                logits = self.model(pixel_vals)
                loss = self.loss_fn(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            running_loss += loss.item()
            count += 1

        return running_loss / count

    def save_checkpoint(self, optimizer, epoch, path="checkpoints/checkpoint.pt"):
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        ckpt = {
            "model": model_to_save.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": epoch
        }
        torch.save(ckpt, path)

    def load_checkpoint(self, optimizer, path="checkpoints/checkpoint.pt"):
        if not os.path.exists(path):
            return 0  # no checkpoint → start at epoch 0

        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        model_to_load = self.model.module if hasattr(self.model, "module") else self.model
        model_to_load.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        self.scaler.load_state_dict(ckpt["scaler"])
        print(f"Loaded checkpoint from epoch {ckpt['epoch']}")
        return ckpt["epoch"] + 1 
