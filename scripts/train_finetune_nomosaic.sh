#!/usr/bin/env bash
set -euo pipefail

BASE_WEIGHTS="${1:?Path to base weights .pt required}"
RUNNAME="${2:-v8s_1024_nomosaic}"
BATCH="${3:-8}"  # Default batch size 8

# Set W&B project
export WANDB_PROJECT="substation-detection-yolo"

# Short fine-tune with mosaic disabled
yolo detect train \
  model="$BASE_WEIGHTS" \
  data="configs/roboflow_data.yaml" \
  project="reports" name="$RUNNAME" exist_ok=True \
  imgsz=1024 epochs=30 batch="$BATCH" seed=42 patience=10 \
  mosaic=0.0 mixup=0.0 copy_paste=0.0 fliplr=0.5 flipud=0.5
