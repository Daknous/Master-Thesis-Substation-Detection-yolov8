#!/bin/bash
#SBATCH --job-name=yolo_sweep
#SBATCH --output=/home/users/s/seifdak/Master-Thesis/logs/%x_%A_%a.out
#SBATCH --error=/home/users/s/seifdak/Master-Thesis/logs/%x_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-2

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=seif.daknou@campus.tu-berlin.de

set -euo pipefail

module load python/3.9.19
source ~/scratch/yolo_detection_venv/bin/activate

cd ~/Master-Thesis/Detection_score

# W&B configuration
export WANDB_PROJECT="substation-detection-yolo"
export WANDB_MODE="online"

MODELS=("yolov8n.pt" "yolov8s.pt" "yolov8m.pt")
SIZES=(1024 1024 1536)
BATCHES=(16 8 4)  # Larger batch for smaller models

IDX=${SLURM_ARRAY_TASK_ID}
MODEL=${MODELS[$IDX]}
SIZE=${SIZES[$IDX]}
BATCH=${BATCHES[$IDX]}
NAME="sweep_${IDX}_${MODEL%%.*}_${SIZE}"

srun yolo detect train \
  model="$MODEL" \
  data="configs/roboflow_data.yaml" \
  project="reports" name="$NAME" exist_ok=True \
  imgsz=$SIZE epochs=200 batch=$BATCH seed=42 patience=25 \
  hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 \
  translate=0.10 scale=0.50 fliplr=0.5 flipud=0.5 \
  mosaic=1.0 mixup=0.10 copy_paste=0.10
