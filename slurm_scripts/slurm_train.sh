#!/bin/bash
#SBATCH --job-name=geo-train
#SBATCH --partition=GPU24
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --nodelist=node5 
#SBATCH --time=48:00:00
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
  --batch-size 16 \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --epochs 16 \
  --learning-rate 8e-5 \
  --checkpoint-path checkpoints/checkpoint_384_2.pt \
  --multi_gpu False \
  --pretrained-model-id facebook/convnext-base-384 \
  --logfile logs/training_log_384_2.csv \
  --freeze False \
  --resize 0 \
  --s2-range 3 7 \
  --weights 0.4 0.6 0.8 1.0

echo "Job ${SLURM_JOB_ID} finished."
