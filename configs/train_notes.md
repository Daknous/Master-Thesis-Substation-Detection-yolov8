# Training settings (notes)

- Baseline model: yolov8s.pt
- imgsz: 1024
- epochs: 220
- aug: hsv, flips, scale jitter, mosaic on; mixup/copy-paste low
- Fine-tune (optional): 30 epochs with mosaic off to tighten boxes
- Metrics: mAP@[.5:.95], per-class AP, confusion matrix (Ultralytics default)
