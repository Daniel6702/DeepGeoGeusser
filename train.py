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
    parser.add_argument("--val-data-path", type=str, default="/home/austen/GeoDataset/dataset_sharded_TEST", help="Path to the sharded validation/test GeoDataset")
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
    parser.add_argument("--weights", type=float, nargs="+", help="Space-separated list of weights")
    parser.add_argument("--s2-range", type=int, nargs=2, help="Start and stop S2 levels (e.g., 3 7)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    DATASET_PATH = args.data_path
    VAL_DATASET_PATH = args.val_data_path
    BATCH_SIZE = args.batch_size
    WORKERS = args.workers
    S2_LEVELS = list(range(args.s2_range[0], args.s2_range[1]))  # make it an explicit list
    S2_LEVEL_WEIGHTS = args.weights
    LEARNING_RATE = args.learning_rate
    PRETRAINED_MODEL_ID = args.pretrained_model_id
    EPOCHS = args.epochs
    CHECKPOINT_PATH = args.checkpoint_path
    DEVICE = args.device
    LOGFILE = args.logfile
    MULTI_GPU = args.multi_gpu
    RESIZE = 384  # or args.resize if you want

    print(
        f"BATCH_SIZE: {BATCH_SIZE}, WORKERS: {WORKERS}, PRETRAINED_MODEL_ID: {PRETRAINED_MODEL_ID}, "
        f"\nCHECKPOINT_PATH: {CHECKPOINT_PATH}, LOGFILE: {LOGFILE}, "
        f"MULTI_GPU: {MULTI_GPU}, RESIZE: {RESIZE}"
    )

    # Make sure checkpoint directory exists
    ckpt_dir = Path(CHECKPOINT_PATH).parent
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ----- S2 canonical index maps (from TRAIN labels) -----
    s2_labels_dir = Path(DATASET_PATH) / "s2_labels"
    idx2id, id2idx, _ = build_s2_index_maps(s2_labels_dir, S2_LEVELS)

    # ----- Processor -----
    processor = AutoImageProcessor.from_pretrained(
        PRETRAINED_MODEL_ID,
        use_fast=True,
    )
    processor.do_resize = True
    processor.size = {"shortest_edge": RESIZE}

    # ----- Train dataset -----
    dataset = GeoWebDataset(
        DATASET_PATH,
        processor,
        levels=S2_LEVELS,
        shuffle=True,
        num_shards_limit=None,
        id2idx=id2idx,      # reuse same mapping
    )

    # ----- Validation dataset (your 2000-image test set) -----
    val_dataset = GeoWebDataset(
        VAL_DATASET_PATH,
        processor,
        levels=S2_LEVELS,
        shuffle=False,
        num_shards_limit=None,
        id2idx=id2idx,      # IMPORTANT: same id2idx as train
    )

    # ----- Model -----
    model = HierarchicalConvNeXt(
        pretrained_name=PRETRAINED_MODEL_ID,
        num_classes=dataset.num_classes_list[-1],
        freeze=False,
    )

    # ----- Hierarchical loss -----
    parent_table = build_parent_tables_from_maps(
        idx2id,
        id2idx,
        S2_LEVELS,
    )
    hier_loss = HierarchicalLoss(
        levels=S2_LEVELS,
        parents=parent_table,
        weights=S2_LEVEL_WEIGHTS,
        num_classes_per_level=dataset.num_classes_list,
    ).to(DEVICE)

    # ----- Dataloaders -----
    loader = wds.WebLoader(
        dataset.dataset,
        num_workers=WORKERS,
        batch_size=BATCH_SIZE,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    val_loader = wds.WebLoader(
        val_dataset.dataset,
        num_workers=WORKERS,
        batch_size=BATCH_SIZE,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    # ----- Trainer -----
    trainer = Trainer(model, loader, hier_loss, DEVICE, MULTI_GPU)

    # ----- Optimizer -----
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        fused=True,
        weight_decay=0.05
    )

    # ----- Load checkpoint if present -----
    start_epoch = trainer.load_checkpoint(optimizer, CHECKPOINT_PATH)

    # ----- Scheduler -----
    # Make sure T_max is never zero to avoid ZeroDivisionError
    if EPOCHS > 1:
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS - 1)
    else:
        scheduler = None

    # ----- CSV logging -----
    if not Path(LOGFILE).exists():
        with open(LOGFILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss"])

    # ----- Training loop -----
    for epoch in range(start_epoch, EPOCHS):
        print(f"\nEpoch {epoch}")

        # Train
        train_loss = trainer.train_epoch(optimizer, weights=S2_LEVEL_WEIGHTS)
        print(f"Average train loss: {train_loss:.4f}")

        # Validate on separate test set
        val_loss = trainer.eval_epoch(val_loader)
        print(f"Average val loss:   {val_loss:.4f}")

        # Log to CSV first so we keep losses even if something crashes later
        with open(LOGFILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss])

        # Step LR scheduler if it exists
        if scheduler is not None:
            scheduler.step()

        # Save checkpoint
        trainer.save_checkpoint(optimizer, epoch, CHECKPOINT_PATH)

if __name__ == "__main__":
    main()

'''
python train.py \
  --data-path ../GeoDataset/dataset_sharded \
  --batch-size 12 \
  --workers 8 \
  --epochs 12 \
  --learning-rate 5e-5 \
  --checkpoint-path checkpoints/checkpoint_384_V3.pt \
  --multi_gpu False \
  --pretrained-model-id facebook/convnext-base-384 \
  --logfile logs/training_log_large3.csv \
  --freeze False \
  --resize 384 \
  --s2-range 3 7 \
  --weights 0.4 0.6 0.8 1.0

#SBATCH --gres=gpu:2
#Start: sbatch slurm_train.sh
#See: squeue -u $USER
#Cancel: scancel 920
#less -f DeepGeoGeusser/logs/geo-train-925.err
tail -f DeepGeoGeusser/logs/geo-train-977.out DeepGeoGeusser/logs/geo-train-977.err 


python train.py \
  --data-path ../GeoDataset/dataset_sharded \
  --batch-size 64 \
  --workers 1 \
  --epochs 1 \
  --learning-rate 5e-5 \
  --checkpoint-path checkpoints/checkpoint_V3_with_val_loss.pt \
  --multi_gpu True \
  --pretrained-model-id facebook/convnext-base-384 \
  --logfile logs/training_log_V3_3_with_val_loss.csv \
  --freeze False \
  --resize 384 \
  --s2-range 3 7 \
  --weights 0.4 0.6 0.8 1.0

'''
