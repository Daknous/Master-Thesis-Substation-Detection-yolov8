# Substation YOLO — Milestone 3 (Roboflow-only)

Train/evaluate a YOLOv8 detector on the Roboflow export as-is. No OSM-ID mapping or site-level metrics yet.

**W&B Project**: `substation-detection-yolo` (all experiments tracked automatically)

## Quickstart (local)

```bash
bash scripts/setup_venv.sh

# Train baseline
bash scripts/train_baseline.sh yolov8s.pt v8s_1024_baseline

# Validate (re-runs val if needed)
bash scripts/val.sh reports/v8s_1024_baseline/weights/best.pt

# Predict on test set
bash scripts/predict.sh reports/v8s_1024_baseline/weights/best.pt 0.25

# Export a flat CSV of key metrics
python scripts/export_ultra_metrics.py --runs_dir reports --run_name v8s_1024_baseline --out reports/metrics_v8s_1024_baseline.csv
```

## HPC (Slurm)

First-time setup:
```bash
# Copy venv and install dependencies
cp -r ~/scratch/subseg_venv ~/scratch/yolo_detection_venv
bash slurm/env.sh

# Login to W&B (one-time)
wandb login
```

Submit jobs:
```bash
sbatch slurm/train_baseline.sbatch
sbatch slurm/train_finetune_nomosaic.sbatch  # after baseline completes
sbatch slurm/sweep_array.sbatch  # parallel sweep of 3 models
```

## Notes

- **W&B tracking**: All experiments automatically logged to `substation-detection-yolo` project
- We intentionally keep Roboflow `data.yaml` unmodified to match class indices.
- Later, you'll add OSM-ID mapping & site-level KPIs in a separate step.
