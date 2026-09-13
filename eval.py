from __future__ import annotations

import argparse

import torch
from transformers import AutoImageProcessor

from adaptive_geogrid import load_grid
from eval import Evaluator
from modules.dataset import PanoramaDataset
from modules.model import GeoModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset/")
    parser.add_argument("--grid", default="grids/world.aggrid")
    parser.add_argument("--checkpoint", default="checkpoints/checkpoint.pt")
    parser.add_argument("--model", default="google/siglip-base-patch16-224")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None, help="Optional per-panorama CSV output")
    return parser.parse_args()


def main():
    args = parse_args()

    grid = load_grid(args.grid)
    dataset = PanoramaDataset(args.dataset, shuffle=False)
    loader = dataset.loader(batch_size=args.batch_size, num_workers=args.workers)

    processor = AutoImageProcessor.from_pretrained(args.model)
    model = GeoModel(grid, model_name=args.model)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)

    evaluator = Evaluator(model, processor, loader, grid, device=args.device)
    evaluator.evaluate(output_csv=args.output)


if __name__ == "__main__":
    main()
