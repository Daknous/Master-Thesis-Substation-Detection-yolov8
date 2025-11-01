PY?=python

setup:
	bash scripts/setup_venv.sh

train:
	bash scripts/train_baseline.sh yolov8s.pt v8s_1024_baseline

finetune:
	bash scripts/train_finetune_nomosaic.sh reports/v8s_1024_baseline/weights/best.pt v8s_1024_nomosaic

val:
	bash scripts/val.sh reports/v8s_1024_baseline/weights/best.pt

predict:
	bash scripts/predict.sh reports/v8s_1024_baseline/weights/best.pt 0.25

metrics:
	$(PY) scripts/export_ultra_metrics.py --runs_dir reports --run_name v8s_1024_baseline --out reports/metrics_v8s_1024_baseline.csv
