#!/usr/bin/env python
"""
Create a small qualitative panel of predictions vs images (no GT overlay, just boxes).
Requires OpenCV. Saves a grid image under reports/.
"""
import argparse, pathlib, cv2

def draw_boxes(img, label_file):
    if not label_file.exists(): return img
    h, w = img.shape[:2]
    for ln in label_file.read_text().strip().splitlines():
        toks = ln.split()
        if len(toks) < 5: continue
        # YOLO: cls xc yc w h [conf]
        try: cls = int(float(toks[0])); off=1
        except: cls = int(float(toks[-1])); off=0
        xc, yc, bw, bh = map(float, toks[off:off+4])
        x1 = int((xc - bw/2) * w); y1 = int((yc - bh/2) * h)
        x2 = int((xc + bw/2) * w); y2 = int((yc + bh/2) * h)
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
    return img

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", default="Dataset/test/images")
    ap.add_argument("--pred_labels", default="preds/raw/inference/labels")
    ap.add_argument("--out", default="reports/qual_panel.jpg")
    ap.add_argument("--limit", type=int, default=12)
    args = ap.parse_args()

    imgs = list(pathlib.Path(args.images_dir).glob("*"))[:args.limit]
    tiles = []
    for p in imgs:
        img = cv2.imread(str(p)); 
        lab = pathlib.Path(args.pred_labels)/ (p.stem + ".txt")
        img = draw_boxes(img, lab)
        tiles.append(img)

    if not tiles: return
    # simple grid (3 cols)
    cols = 3
    rows = (len(tiles)+cols-1)//cols
    h, w = tiles[0].shape[:2]
    import numpy as np
    grid = np.ones((rows*h, cols*w, 3), dtype=tiles[0].dtype)*255
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        grid[r*h:(r+1)*h, c*w:(c+1)*w] = t
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, grid)
    print(f"Saved {args.out}")

if __name__ == "__main__":
    main()
