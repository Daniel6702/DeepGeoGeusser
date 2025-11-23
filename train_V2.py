from transformers import AutoImageProcessor
import torch.optim as optim
import webdataset as wds
from pathlib import Path
from modules import *
import csv
import argparse
import os
from torch.optim.lr_scheduler import CosineAnnealingLR

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="/home/austen/GeoDataset/dataset_sharded", help="Path to the sharded GeoDataset")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--workers", type=int, default=16, help="Number of dataloader workers (<= cpus-per-task)")
    parser.add_argument("--learning-rate", type=float, default=8e-5, help="Learning rate")
    parser.add_argument("--pretrained-model-id", type=str, default="facebook/convnext-tiny-224", help="HF model id")
    parser.add_argument("--epochs", type=int, default=16, help="Number of epochs")
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/checkpoint.pt", help="Checkpoint file path")
    parser.add_argument("--device", type=str, default="cuda", help="Device (e.g. cuda, cuda:0, cpu)")
    parser.add_argument("--logfile", type=str, default="training_log.csv", help="CSV file for logging losses")
    parser.add_argument("--multi_gpu", type=bool, default=True, help="123")
    parser.add_argument("--freeze", type=bool, default=False, help="123")
    parser.add_argument("--resize", type=int, default=0, help="123")
    parser.add_argument("--init_weights", type=float, nargs="+", help="Space-separated list of weights")
    parser.add_argument("--target_weights", type=float, nargs="+", help="Space-separated list of weights")
    parser.add_argument("--s2-range", type=int, nargs=2, help="Start and stop S2 levels (e.g., 3 7)")
    return parser.parse_args()

def main():
    args = parse_args()

    DATASET_PATH = args.data_path
    BATCH_SIZE = args.batch_size
    WORKERS = args.workers
    S2_LEVELS = list(range(args.s2_range[0], args.s2_range[1]))  # make it an explicit list
    INIT_WEIGHTS = args.init_weights
    TARGET_WEIGHTS = args.target_weights
    LEARNING_RATE = args.learning_rate
    PRETRAINED_MODEL_ID = args.pretrained_model_id
    EPOCHS = args.epochs
    CHECKPOINT_PATH = args.checkpoint_path
    DEVICE = args.device
    LOGFILE = args.logfile
    MULTI_GPU = args.multi_gpu
    FREEZE = args.freeze
    RESIZE = 384

    print(
        f"BATCH_SIZE: {BATCH_SIZE}, WORKERS: {WORKERS}, PRETRAINED_MODEL_ID: {PRETRAINED_MODEL_ID}, "
        f"FREEZE: {FREEZE} \nCHECKPOINT_PATH: {CHECKPOINT_PATH}, LOGFILE: {LOGFILE}, "
        f"MULTI_GPU: {MULTI_GPU}, RESIZE: {RESIZE}"
    )

    ckpt_dir = Path(CHECKPOINT_PATH).parent
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    s2_labels_dir = Path(DATASET_PATH) / "s2_labels"
    idx2id, id2idx, _ = build_s2_index_maps(s2_labels_dir, S2_LEVELS)

    processor = AutoImageProcessor.from_pretrained(PRETRAINED_MODEL_ID,use_fast=True,)
    processor.do_resize = True
    processor.size = {"shortest_edge": RESIZE}

    dataset = GeoWebDataset(
        DATASET_PATH,
        processor,
        levels=S2_LEVELS,
        shuffle=False,
        num_shards_limit=None,
        id2idx=id2idx,
    )

    model = HierarchicalConvNeXt_V2(
        pretrained_name=PRETRAINED_MODEL_ID,
        num_classes_per_level=dataset.num_classes_list
    )

    hier_loss = HierarchicalLoss_V2(
        weights=INIT_WEIGHTS,
        label_smoothing=0.1
    )

    loader = wds.WebLoader(
        dataset.dataset,
        num_workers=WORKERS,
        batch_size=BATCH_SIZE,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    trainer = Trainer(model, loader, hier_loss, DEVICE, MULTI_GPU)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        fused=True,
        weight_decay=0.05
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    if not Path(LOGFILE).exists():
        with open(LOGFILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss"])

    start_epoch = trainer.load_checkpoint(optimizer, CHECKPOINT_PATH)

    for epoch in range(start_epoch, EPOCHS):
        print(f"\nEpoch {epoch}")

        alpha = epoch / (EPOCHS - 1)
        hier_loss.level_weights = [
            (1 - alpha) * w_coarse + alpha * w_fine_target
            for w_coarse, w_fine_target in zip(INIT_WEIGHTS, TARGET_WEIGHTS)
        ]

        avg_loss = trainer.train_epoch(optimizer, weights=[])
        scheduler.step()
        print(f"Average loss: {avg_loss:.4f}")

        with open(LOGFILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_loss])

        trainer.save_checkpoint(optimizer, epoch, CHECKPOINT_PATH)

if __name__ == "__main__":
    main()

'''
python train.py \
  --data-path ../GeoDataset/dataset_sharded \
  --batch-size 24 \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --epochs 32 \
  --learning-rate 5e-5 \
  --checkpoint-path checkpoints/checkpoint_384_V2.pt \
  --multi_gpu False \
  --pretrained-model-id facebook/convnext-large-384 \
  --logfile logs/training_384_V2.csv \
  --freeze False \
  --resize 384 \
  --s2-range 3 7 \
  --init_weights 1.0 0.8 0.6 0.4 \
  --target_weights 0.2 0.4 0.7 1.0


python train_V2.py \
  --data-path ../GeoDataset/dataset_sharded \
  --batch-size 4 \
  --workers 4 \
  --epochs 32 \
  --learning-rate 5e-5 \
  --checkpoint-path checkpoints/checkpoint_384_V2.pt \
  --multi_gpu False \
  --pretrained-model-id facebook/convnext-large-384 \
  --logfile logs/training_384_V2.csv \
  --freeze False \
  --resize 384 \
  --s2-range 3 7 \
  --init_weights 1.0 0.8 0.6 0.4 \
  --target_weights 0.2 0.4 0.7 1.0

#SBATCH --gres=gpu:2
#Start: sbatch slurm_train.sh
#See: squeue -u $USER
#Cancel: scancel 920
#less -f DeepGeoGeusser/logs/geo-train-925.err
tail -f DeepGeoGeusser/logs/geo-train-977.out DeepGeoGeusser/logs/geo-train-977.err 
'''
