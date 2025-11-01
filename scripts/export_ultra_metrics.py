#!/usr/bin/env python
"""
Read Ultralytics run results (results.csv + metrics in args.yaml) and export a flat CSV summary:
- run_name, model, epochs, imgsz
- mAP50-95, mAP50
- per-class AP50-95 if available
- confusion matrix path if saved
"""
import argparse, json, csv, pathlib, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="reports")
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    run = pathlib.Path(args.runs_dir) / args.run_name
    results_csv = run / "results.csv"
    args_yaml = run / "args.yaml"
    metrics_json = run / "metrics.json"  # sometimes present

    row = {"run_name": args.run_name}

    if args_yaml.exists():
        try:
            import yaml
            y = yaml.safe_load(args_yaml.read_text())
            for k in ["model","epochs","imgsz","data"]:
                row[k] = y.get(k)
        except Exception:
            pass

    if results_csv.exists():
        df = pd.read_csv(results_csv)
        # take last row (best or last epoch)
        last = df.iloc[-1].to_dict()
        # standard columns used by Ultralytics (names can vary a bit by version)
        for k in ["metrics/mAP50-95(B)","metrics/mAP50(B)","metrics/precision(B)","metrics/recall(B)"]:
            if k in last: row[k.replace("(B)","")] = float(last[k])
        # fallback for older names
        for k in ["mAP50-95","mAP50","precision","recall"]:
            if k in last and k not in row: row[k] = float(last[k])

    if metrics_json.exists():
        try:
            mj = json.loads(metrics_json.read_text())
            # per-class AP if present
            if "ap_class_index" in mj and "ap50_95" in mj:
                for cls, ap in zip(mj["ap_class_index"], mj["ap50_95"]):
                    row[f"AP50-95_class_{cls}"] = ap
        except Exception:
            pass

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader(); w.writerow(row)

    print(f"Wrote summary: {out}")

if __name__ == "__main__":
    main()
