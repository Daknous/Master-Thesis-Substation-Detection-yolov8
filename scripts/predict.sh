#!/usr/bin/env bash
set -euo pipefail

WEIGHTS="${1:-reports/v8s_1024_baseline/weights/best.pt}"
CONF="${2:-0.25}"
OUTDIR="preds/raw"

yolo detect predict \
  model="$WEIGHTS" \
  data="configs/roboflow_data.yaml" \
  imgsz=1024 conf="$CONF" \
  save_txt=True save_conf=True \
  project="$OUTDIR" name="inference" exist_ok=True
