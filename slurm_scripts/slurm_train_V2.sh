#!/bin/bash
#SBATCH --job-name=geo-train
#SBATCH --partition=GPU48
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=120:00:00
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
python train_V2.py \
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


echo "Job ${SLURM_JOB_ID} finished."
