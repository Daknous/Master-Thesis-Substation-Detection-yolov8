#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-yolov8s.pt}"
RUNNAME="${2:-v8s_1024_baseline}"
BATCH="${3:-8}"  # Default batch size 8, adjust based on GPU memory

# Set W&B project
export WANDB_PROJECT="substation-detection-yolo"

# Train on Roboflow split as-is
yolo detect train \
  model="$MODEL" \
  data="configs/roboflow_data.yaml" \
  project="reports" name="$RUNNAME" exist_ok=True \
  imgsz=1024 epochs=220 batch="$BATCH" seed=42 patience=30 \
  hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 \
  translate=0.10 scale=0.50 fliplr=0.5 flipud=0.5 \
  mosaic=1.0 mixup=0.10 copy_paste=0.10
