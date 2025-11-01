#!/usr/bin/env bash
# HPC environment setup for TU Berlin cluster
# Note: Copy your existing venv first:
#   cp -r ~/scratch/subseg_venv ~/scratch/yolo_detection_venv
set -euo pipefail

module load python/3.9.19
source ~/scratch/yolo_detection_venv/bin/activate

# Install/update YOLO dependencies
pip install --upgrade --no-cache-dir -r requirements.txt

python - <<'PY'
import torch; print("CUDA:", torch.cuda.is_available(), "GPUs:", torch.cuda.device_count())
PY
