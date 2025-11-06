#!/bin/bash
#SBATCH --job-name=yolo_base
#SBATCH --output=/home/users/s/seifdak/Master-Thesis_Detection-score/Master-Thesis-Substation-Detection-yolov8/logs/%x_%j.out
#SBATCH --error=/home/users/s/seifdak/Master-Thesis_Detection-score/Master-Thesis-Substation-Detection-yolov8/logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=seif.daknou@campus.tu-berlin.de

set -euo pipefail

module load python/3.9.19
source ~/Master-Thesis/env_subseg/bin/activate

cd ~/Master-Thesis_Detection-score/Master-Thesis-Substation-Detection-yolov8

# Create logs directory
mkdir -p logs

# W&B configuration
export WANDB_API_KEY=$(< ~/.wandb_api_key)
export WANDB_PROJECT="substation-detection-yolo"
export WANDB_MODE="online"

srun bash scripts/train_baseline.sh yolov8s.pt v8s_1024_baseline 8
