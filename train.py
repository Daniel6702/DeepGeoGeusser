from __future__ import annotations

import argparse

import torch
from torch.optim import AdamW
from transformers import AutoImageProcessor

from adaptive_geogrid import load_grid
from modules.dataset import PanoramaDataset
from modules.hierarchical_loss import HierarchicalLoss
from modules.model import GeoModel
from modules.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/aarhus")
    parser.add_argument("--grid", default="grids/aarhus.aggrid")
    parser.add_argument("--model", default="google/siglip-base-patch16-384")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", default="checkpoints/checkpoint.pt")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--loss-weights", type=float, nargs="+", help="Loss weights fine -> coarse")
    return parser.parse_args()

def main():
    args = parse_args()

    grid = load_grid(args.grid)
    dataset = PanoramaDataset(args.dataset, shuffle=True)
    loader = dataset.loader(batch_size=args.batch_size, num_workers=args.workers)

    processor = AutoImageProcessor.from_pretrained(args.model)
    model = GeoModel(
        grid,
        model_name=args.model,
        freeze_backbone=args.freeze_backbone,
    )
    loss_fn = HierarchicalLoss(grid, weights=args.loss_weights)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.05)

    trainer = Trainer(model, processor, loader, loss_fn, device=args.device)

    print(f"Grid classes fine -> coarse: {grid.n_classes}")

    for epoch in range(args.epochs):
        loss = trainer.train_epoch(optimizer)
        print(f"Epoch {epoch + 1}/{args.epochs} - loss: {loss:.4f}")
        trainer.save_checkpoint(optimizer, epoch, args.checkpoint)


if __name__ == "__main__":
    main()
