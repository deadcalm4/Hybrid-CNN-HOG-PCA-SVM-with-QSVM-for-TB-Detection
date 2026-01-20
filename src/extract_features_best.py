# ===============================
# Phase 4: CNN Feature Extraction (BEST VERSION)
# EfficientNet + Mixed Precision + Safe Mapping
# ===============================

import os
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
import timm
from tqdm import tqdm
import argparse

class SegmentedDataset(Dataset):
    def __init__(self, image_paths, labels, img_size):
        self.image_paths = image_paths
        self.labels = labels
        self.img_size = img_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("L")
        img = img.resize((self.img_size, self.img_size))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.stack([arr, arr, arr], axis=0)
        tensor = torch.from_numpy(arr)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        tensor = (tensor - mean) / std
        return tensor, self.labels[idx]

def load_image_paths_and_labels(segmented_dir):
    image_paths = []
    labels = []

    for root, _, files in os.walk(segmented_dir):
        for f in files:
            fname = f.lower()
            if fname.endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(root, f)

                # Kaggle-style naming
                if "normal" in fname:
                    label = 0
                elif "tb" in fname or "tuberculosis" in fname:
                    label = 1

                # Shenzhen-style naming: *_0.png / *_1.png
                elif fname.endswith("_0.png"):
                    label = 0
                elif fname.endswith("_1.png"):
                    label = 1

                else:
                    continue  # skip unknown naming

                image_paths.append(full_path)
                labels.append(label)

    # Safety check
    if len(image_paths) == 0:
        raise RuntimeError(
            f"No images found in {segmented_dir}. "
            "Check filename conventions."
        )

    # Sort to keep alignment
    image_paths, labels = zip(*sorted(zip(image_paths, labels)))

    return list(image_paths), np.array(labels)


def main(args):
    print(">>> Phase 4: CNN Feature Extraction STARTED <<<")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    image_paths, labels = load_image_paths_and_labels(args.input_base)
    print("Total images found:", len(image_paths))

    dataset = SegmentedDataset(image_paths, labels, args.img_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = timm.create_model("tf_efficientnet_b0_ns", pretrained=True)
    model.eval()
    model.to(device)

    features = []

    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc="Extracting features"):
            imgs = imgs.to(device)
            with autocast():
                feats = model.forward_features(imgs)
                feats = F.adaptive_avg_pool2d(feats, 1).squeeze(-1).squeeze(-1)
            features.append(feats.cpu().numpy())

    features = np.concatenate(features, axis=0)

    out_dir = Path(args.features_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "cnn.npy", features)
    np.save(out_dir / "labels.npy", labels)

    meta = {
        "backbone": "EfficientNet-B0",
        "feature_dim": features.shape[1],
        "num_samples": features.shape[0]
    }

    with open(out_dir / "cnn_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("CNN features saved successfully")
    print("Feature shape:", features.shape)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_base", default="data/segmented/images")
    parser.add_argument("--features_root", default="data/features/kaggle_tb")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()
    main(args)
