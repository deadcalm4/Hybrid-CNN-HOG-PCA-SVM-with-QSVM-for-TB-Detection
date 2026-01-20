# ==========================================
# Phase 7.1: Balanced 300-sample subset
# ==========================================

import numpy as np
from pathlib import Path

def main():
    print(">>> Creating BALANCED 300-sample QSVM subset <<<")

    X = np.load("data/features/fused/fused_pca_32.npy")
    y = np.load("data/features/fused/labels.npy")

    print("Full dataset shape:", X.shape)
    print("Original class balance:", np.bincount(y))

    rng = np.random.default_rng(seed=42)

    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]

    idx0_sel = rng.choice(idx0, size=150, replace=False)
    idx1_sel = rng.choice(idx1, size=150, replace=False)

    idx_sel = np.concatenate([idx0_sel, idx1_sel])
    rng.shuffle(idx_sel)

    X_sub = X[idx_sel]
    y_sub = y[idx_sel]

    out = Path("data/features/fused")
    np.save(out / "fused_pca_32_subset_300.npy", X_sub)
    np.save(out / "labels_subset_300.npy", y_sub)
    np.save(out / "subset_indices_300.npy", idx_sel)

    print("Subset shape:", X_sub.shape)
    print("Balanced class balance:", np.bincount(y_sub))
    print(">>> Subset creation completed <<<")

if __name__ == "__main__":
    main()
