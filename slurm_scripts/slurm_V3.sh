#!/bin/bash
#SBATCH --job-name=geo-train
#SBATCH --partition=GPU48
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:2
#SBATCH --time=240:00:00
#SBATCH --output=DeepGeoGeusser/logs/%x-%j.out
#SBATCH --error=DeepGeoGeusser/logs/%x-%j.err

echo "Job ${SLURM_JOB_ID} starting on host $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Initial working directory: $(pwd)"
echo "Running on GPUs: $CUDA_VISIBLE_DEVICES"

#Activate python venv
source venv/bin/activate

cd DeepGeoGeusser
echo "Now in project directory: $(pwd)"

#Ensure checkpoint directory exists
mkdir -p checkpoints

echo "GPU status at job start:"
nvidia-smi || echo "nvidia-smi not available"

#Run training
python train.py \
  --data-path ../GeoDataset/dataset_sharded \
  --batch-size 64 \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --epochs 64 \
  --learning-rate 5e-5 \
  --checkpoint-path checkpoints/checkpoint_V3.pt \
  --multi_gpu True \
  --pretrained-model-id facebook/convnext-base-384 \
  --logfile logs/training_log_V3_3.csv \
  --freeze False \
  --resize 384 \
  --s2-range 3 7 \
  --weights 0.4 0.6 0.8 1.0

echo "Job ${SLURM_JOB_ID} finished."
