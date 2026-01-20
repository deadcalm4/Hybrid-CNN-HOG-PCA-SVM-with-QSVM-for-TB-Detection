# ==========================================
# Phase 5 / 8: Feature Fusion + PCA
# CNN + HOG -> PCA-reduced representations
# ==========================================

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def main(args):
    print(">>> Feature Fusion + PCA STARTED <<<")

    cnn_path = Path(args.cnn_dir) / "cnn.npy"
    hog_path = Path(args.hog_dir) / "hog.npy"
    label_path = Path(args.cnn_dir) / "labels.npy"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load features
    cnn = np.load(cnn_path)
    hog = np.load(hog_path)
    labels = np.load(label_path)

    print("CNN shape:", cnn.shape)
    print("HOG shape:", hog.shape)
    print("Labels shape:", labels.shape)

    assert cnn.shape[0] == hog.shape[0] == labels.shape[0], \
        "Mismatch in number of samples"

    # Feature fusion
    fused = np.concatenate([cnn, hog], axis=1)
    print("Fused feature shape:", fused.shape)

    # Save raw fused features
    np.save(out_dir / "fused_raw.npy", fused)
    np.save(out_dir / "labels.npy", labels)

    # Standardize
    scaler = StandardScaler()
    fused_std = scaler.fit_transform(fused)

    # PCA configurations
    pca_dims = [64, 32, 16, 8]
    variance_records = []

    for d in pca_dims:
        pca = PCA(n_components=d, random_state=42)
        fused_pca = pca.fit_transform(fused_std)

        np.save(out_dir / f"pca_{d}.npy", fused_pca)

        explained = np.sum(pca.explained_variance_ratio_)
        variance_records.append((d, explained))

        print(f"PCA-{d}: explained variance = {explained:.4f}")

    # Save PCA variance info
    df = pd.DataFrame(variance_records, columns=["PCA_dim", "Explained_variance"])
    df.to_csv(out_dir / "pca_variance.csv", index=False)

    print(">>> Feature Fusion + PCA COMPLETED SUCCESSFULLY <<<")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnn_dir", required=True)
    parser.add_argument("--hog_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    main(args)
