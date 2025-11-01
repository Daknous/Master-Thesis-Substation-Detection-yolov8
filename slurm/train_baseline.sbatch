#!/bin/bash
#SBATCH --job-name=yolo_base
#SBATCH --output=/home/users/s/seifdak/Master-Thesis/logs/%x_%j.out
#SBATCH --error=/home/users/s/seifdak/Master-Thesis/logs/%x_%j.err
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
source ~/scratch/yolo_detection_venv/bin/activate

cd ~/Master-Thesis/Detection_score

# W&B configuration
export WANDB_PROJECT="substation-detection-yolo"
export WANDB_MODE="online"  # use "offline" if no internet, then sync later with: wandb sync

srun bash scripts/train_baseline.sh yolov8s.pt v8s_1024_baseline
