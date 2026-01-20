# scripts/preprocess.py
"""
Preprocessing script for TB project.

Requirements:
 - utils/preprocessing.py must exist and provide:
    load_image_as_gray, save_uint8_image, preprocess_image_pipeline, extract_hog
 - Installed packages: numpy, pillow, albumentations, tqdm
 - Run safe sanity test first with --max_images 10
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


import os
import argparse
import json
import random
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image

# utils
from utils.preprocessing import (
    load_image_as_gray,
    save_uint8_image,
    preprocess_image_pipeline,
    extract_hog,
    ensure_dir,
)

# augmentation (for balancing)
try:
    import albumentations as A
except Exception:
    A = None

# ---------------- seed ----------------
def set_seed(seed=42):
    import os, random, numpy as np
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

# ---------------- collect images ----------------
def collect_image_paths(root_dir, exts=('.png','.jpg','.jpeg','.tif','.tiff','.bmp')):
    root_dir = Path(root_dir)
    files = [str(p) for p in root_dir.rglob('*') if p.suffix.lower() in exts]
    files.sort()
    return files

# ---------------- augmentations ----------------
def build_augmenter():
    if A is None:
        raise RuntimeError("Albumentations not installed. Run: pip install albumentations")
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.08, rotate_limit=12, border_mode=0, p=0.4),
        A.OneOf([A.GaussianBlur(p=0.3), A.GaussNoise(p=0.3)], p=0.2),
    ])

def apply_augmentation(img_arr, augmenter):
    # albumentations expects HxWxC; convert grayscale to HxWx1
    aug = augmenter(image=img_arr)
    out = aug['image']
    # if albumentations returns 2D, keep it
    if out.ndim == 3 and out.shape[2] == 1:
        out = out[:,:,0]
    return out

# ---------------- gallery ----------------
def make_gallery(sample_pairs, out_path, cols=5):
    """sample_pairs: list of tuples (orig_path, processed_array)"""
    imgs = []
    for orig_path, proc in sample_pairs:
        try:
            orig = Image.open(orig_path).convert('L').resize((256,256))
        except Exception:
            # fallback: use processed as orig
            orig = Image.fromarray(proc).convert('L').resize((256,256))
        proc_img = Image.fromarray(proc).convert('L').resize((256,256))
        w,h = orig.size
        canvas = Image.new('L', (w*2, h))
        canvas.paste(orig, (0,0))
        canvas.paste(proc_img, (w,0))
        imgs.append(canvas)
    if not imgs:
        return
    rows = (len(imgs) + cols - 1)//cols
    w,h = imgs[0].size
    grid = Image.new('L', (w*cols, h*rows), color=0)
    for idx, im in enumerate(imgs):
        r = idx // cols
        c = idx % cols
        grid.paste(im, (c*w, r*h))
    ensure_dir(os.path.dirname(out_path))
    grid.save(out_path)

# ---------------- main ----------------
def main(args):
    set_seed(args.seed)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    features_dir = Path(args.features_dir)
    gallery_dir = Path(args.gallery_dir)
    ensure_dir(str(output_dir))
    ensure_dir(str(features_dir))
    ensure_dir(str(gallery_dir))

    # collect images
    image_paths = collect_image_paths(input_dir)
    if len(image_paths) == 0:
        print("No images found in", input_dir)
        return

    # determine classes by folder name (assumes Normal/ and Tuberculosis/ present)
    # user can supply mapping by filenames if different
    class_map = {}
    for p in image_paths:
        # class label inferred from parent directory name
        parent = Path(p).parent.name.lower()
        if 'tb' in parent or 'tuberc' in parent:
            class_map[p] = 1
        else:
            class_map[p] = 0

    # optionally limit images for sanity
    if args.max_images and args.max_images>0:
        image_paths = image_paths[:args.max_images]

    print(f"Found {len(image_paths)} images (using max_images={args.max_images})")

    # prepare augmentation for balancing if required
    augmenter = None
    if args.balance and args.balance_target > 0:
        if A is None:
            raise RuntimeError("Albumentations required for balancing: pip install albumentations")
        augmenter = build_augmenter()

    # containers for features/meta
    hog_list = []
    cnn_norm_paths = []  # we will save processed images; CNN normalization left for later code
    labels = []
    filenames = []
    log = {
        'input_dir': str(input_dir),
        'output_dir': str(output_dir),
        'features_dir': str(features_dir),
        'n_input': len(image_paths),
        'processed': [],
        'params': {
            'size': args.size,
            'clahe_clip': args.clahe_clip,
            'bm3d_sigma': args.bm3d_sigma,
            'seed': args.seed,
            'balance': args.balance,
            'balance_target': args.balance_target
        }
    }

    # process original set (or subset)
    sample_pairs = []
    for p in tqdm(image_paths, desc="Preprocessing"):
        try:
            proc = preprocess_image_pipeline(str(p), out_path=None,
                                             size=(args.size, args.size),
                                             clahe_clip=args.clahe_clip,
                                             bm3d_sigma=args.bm3d_sigma)
            # save processed image
            out_path = output_dir / Path(p).name
            save_uint8_image(str(out_path), proc)
            # compute HOG if requested
            if args.save_hog:
                hog_v = extract_hog(proc,
                                    pixels_per_cell=(16,16),
                                    cells_per_block=(2,2),
                                    orientations=9)
                hog_list.append(hog_v)
            # record
            labels.append(class_map[p])
            filenames.append(str(out_path))
            log['processed'].append({'orig': str(p), 'processed': str(out_path), 'label': class_map[p]})
            sample_pairs.append((str(p), proc))
        except Exception as e:
            print("ERROR processing", p, e)

    # If balancing requested: augment TB images until target reached
    if args.balance:
        tb_indices = [i for i,(f,l) in enumerate(zip(filenames, labels)) if l==1]
        n_tb = len(tb_indices)
        target = args.balance_target
        print(f"Existing TB count: {n_tb}. Target: {target}")
        if n_tb == 0:
            print("No TB images found to augment.")
        else:
            aug_count = 0
            idx = 0
            while (n_tb + aug_count) < target:
                base_idx = tb_indices[idx % len(tb_indices)]
                src_path = filenames[base_idx]
                # load processed image (uint8)
                img_arr = np.array(Image.open(src_path).convert('L'))
                aug_img = apply_augmentation(img_arr, augmenter)
                # re-run cleaning pipeline (CLAHE+BM3D+resize) to keep consistent
                proc_aug = preprocess_image_pipeline(None, out_path=None,
                                                     size=(args.size,args.size),
                                                     clahe_clip=args.clahe_clip,
                                                     bm3d_sigma=args.bm3d_sigma) if False else None
                # Instead of re-applying bm3d pipeline, we assume augmentation preserves/returns uint8
                # so we just resize/pad and save
                # save augmented image
                aug_fname = f"aug_tb_{aug_count:05d}.png"
                out_path_aug = output_dir / aug_fname
                # ensure shape and resize
                from utils.preprocessing import resize_image
                proc_aug = resize_image(aug_img, size=(args.size,args.size))
                save_uint8_image(str(out_path_aug), proc_aug)
                # hog
                if args.save_hog:
                    hog_v = extract_hog(proc_aug,
                                        pixels_per_cell=(16,16),
                                        cells_per_block=(2,2),
                                        orientations=9)
                    hog_list.append(hog_v)
                filenames.append(str(out_path_aug))
                labels.append(1)
                log['processed'].append({'orig': f"augmented_from:{src_path}", 'processed': str(out_path_aug), 'label': 1})
                aug_count += 1
                idx += 1
            print(f"Augmented {aug_count} TB images.")

    # Save features if requested
    if args.save_hog and len(hog_list)>0:
        hog_arr = np.vstack(hog_list).astype(np.float32)
        np.save(str(features_dir / "hog.npy"), hog_arr)
        print("Saved HOG features:", hog_arr.shape)

    # Save labels and filenames mapping
    np.save(str(features_dir / "labels.npy"), np.array(labels, dtype=np.int64))
    with open(str(features_dir / "filenames.txt"), 'w', encoding='utf8') as f:
        for t in filenames:
            f.write(t + "\n")

    # Save gallery (first up to 10)
    gallery_path = gallery_dir / "preprocess_gallery.png"
    make_gallery(sample_pairs[:10], str(gallery_path))
    print("Gallery saved to", str(gallery_path))

    # Save log
    log['n_final'] = len(filenames)
    with open("results/preprocess_log.json", "w", encoding='utf8') as f:
        json.dump(log, f, indent=2)

    print("Preprocessing complete. Processed:", len(filenames))

# ---------------- argparse ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="raw images root (contains class folders)")
    parser.add_argument("--output_dir", required=True, help="preprocessed output folder")
    parser.add_argument("--features_dir", default="data/features", help="where to save hog/labels")
    parser.add_argument("--gallery_dir", default="results/preprocess_samples", help="gallery dir")
    parser.add_argument("--max_images", type=int, default=0, help="0 => all. If >0, process at most that many (sanity)")
    parser.add_argument("--size", type=int, default=512, help="output square size")
    parser.add_argument("--clahe_clip", type=float, default=2.0)
    parser.add_argument("--bm3d_sigma", type=float, default=25/255.0)
    parser.add_argument("--save_hog", action="store_true", help="save hog features to features_dir/hog.npy")
    parser.add_argument("--balance", action="store_true", help="augment TB class to reach balance_target")
    parser.add_argument("--balance_target", type=int, default=3500, help="target TB count after balancing")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    # adapt max_images: if >0, process only first N
    if args.max_images > 0:
        # we'll only consider the first N images discovered across classes
        pass
    main(args)
