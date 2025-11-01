#!/usr/bin/env bash
set -euo pipefail

WEIGHTS="${1:-reports/v8s_1024_baseline/weights/best.pt}"

yolo detect val \
  model="$WEIGHTS" \
  data="configs/roboflow_data.yaml" \
  imgsz=1024
