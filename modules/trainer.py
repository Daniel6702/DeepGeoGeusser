import torch
from tqdm import tqdm
import torch.nn.functional as F
import os

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("medium")  # might be faster?


class Trainer:
    """
    Training helper handling:
      - device placement (single or multi-GPU)
      - mixed-precision training (bfloat16)
      - checkpoint saving/loading
    """
    def __init__(self, model, loader, loss_fn, device="cuda", multi_gpu=False):
        """
        Args:
            model: model to train.
            loader: dataloader yielding (pixel_vals, class_vecs).
            loss_fn: callable loss function (e.g. HierarchicalLoss or V2).
            device: target device string, e.g. 'cuda' or 'cpu'.
            multi_gpu: if True and multiple GPUs are available, use DataParallel.
        """
        self.device = device

        model = model.to(device, memory_format=torch.channels_last)
        if multi_gpu and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = torch.nn.DataParallel(
                model,
                device_ids=list(range(torch.cuda.device_count()))
            )
        else:
            print("Using single GPU")
            self.model = model

        self.loader = loader
        self.loss_fn = loss_fn
        self.device = device
        self.scaler = torch.amp.GradScaler(self.device)

    def train_epoch(self, optimizer, weights):
        """
        Run one training epoch over the dataloader.

        Args:
            optimizer: optimizer instance (e.g. AdamW).
            weights: kept for API compatibility; not used internally here.

        Returns:
            Average training loss over the epoch.
        """
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

    @torch.no_grad()    
    def eval_epoch(self, loader):
        """
        Run one validation epoch over the given dataloader.

        Args:
            loader: validation dataloader yielding (pixel_vals, class_vecs).

        Returns:
            Average validation loss over the epoch.
        """
        self.model.eval()
        running_loss = 0.0
        count = 0

        for pixel_vals, class_vecs in tqdm(loader, desc="Validating"):
            pixel_vals = pixel_vals.to(self.device, memory_format=torch.channels_last)
            class_vecs = class_vecs.to(self.device)
            labels = [class_vecs[:, i] for i in range(class_vecs.shape[1])]

            with torch.amp.autocast(self.device, dtype=torch.bfloat16):
                logits = self.model(pixel_vals)
                loss = self.loss_fn(logits, labels)

            running_loss += loss.item()
            count += 1

        return running_loss / max(count, 1)

    def save_checkpoint(self, optimizer, epoch, path="checkpoints/checkpoint.pt"):
        """
        Save model, optimizer, and scaler state to a checkpoint file.

        Args:
            optimizer: optimizer whose state will be saved.
            epoch: current epoch number.
            path: path to the checkpoint file.
        """
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        ckpt = {
            "model": model_to_save.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": epoch
        }
        torch.save(ckpt, path)

    def load_checkpoint(self, optimizer, path="checkpoints/checkpoint.pt"):
        """
        Load training state (model, optimizer, scaler) from a checkpoint if it exists.

        Args:
            optimizer: optimizer instance to load state into.
            path: path to the checkpoint file.

        Returns:
            Next epoch index to continue from (0 if no checkpoint found).
        """
        if not os.path.exists(path):
            return 0  # no checkpoint → start at epoch 0

        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        model_to_load = self.model.module if hasattr(self.model, "module") else self.model
        model_to_load.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        self.scaler.load_state_dict(ckpt["scaler"])
        print(f"Loaded checkpoint from epoch {ckpt['epoch']}")
        return ckpt["epoch"] + 1
