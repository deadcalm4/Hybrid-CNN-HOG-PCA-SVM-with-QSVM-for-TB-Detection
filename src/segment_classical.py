# scripts/segment_classical.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


import os
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from utils.preprocessing import load_image_as_gray, save_uint8_image, ensure_dir, resize_image

def remove_small_components(labels, stats, min_area=500):
    mask = np.zeros_like(labels, dtype=np.uint8)
    maxlab = labels.max()
    for i in range(1, maxlab+1):
        if i < stats.shape[0]:
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                mask[labels == i] = 255
    return mask

def largest_two_components(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    cleaned = remove_small_components(labels, stats, min_area=500)
    num_labels2, labels2, stats2, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    areas = [(i, stats2[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels2)]
    areas.sort(key=lambda x: x[1], reverse=True)
    keep = [areas[i][0] for i in range(min(2, len(areas)))]
    out = np.zeros_like(mask)
    for k in keep:
        out[labels2 == k] = 255
    return out

def postprocess_mask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    # fill holes
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv2.drawContours(mask, [cnt], 0, 255, thickness=cv2.FILLED)
    return mask

def segment_image(img):
    img_eq = cv2.equalizeHist(img)
    blur = cv2.GaussianBlur(img_eq, (5,5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.sum(th == 255) > th.size * 0.6:
        th = 255 - th
    th = largest_two_components(th)
    th = postprocess_mask(th)
    return th

def run(input_dir, out_mask_dir, out_seg_img_dir, overwrite=False):
    p_in = Path(input_dir)
    p_mask = Path(out_mask_dir); ensure_dir(str(p_mask))
    p_seg = Path(out_seg_img_dir); ensure_dir(str(p_seg))
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    files = [p for p in p_in.rglob('*') if p.suffix.lower() in exts]
    files.sort()
    for f in tqdm(files, desc="Segmenting"):
        rel = f.relative_to(p_in)
        out_mask_path = p_mask / rel
        out_seg_path = p_seg / rel
        ensure_dir(str(out_mask_path.parent))
        ensure_dir(str(out_seg_path.parent))
        if not overwrite and out_mask_path.exists() and out_seg_path.exists():
            continue
        img = load_image_as_gray(str(f))
        img = resize_image(img, (512,512))
        mask = segment_image(img)
        masked = (img * (mask // 255)).astype(np.uint8)
        save_uint8_image(str(out_mask_path), mask)
        save_uint8_image(str(out_seg_path), masked)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--out_mask_dir", required=True)
    parser.add_argument("--out_seg_dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.input_dir, args.out_mask_dir, args.out_seg_dir, overwrite=args.overwrite)
