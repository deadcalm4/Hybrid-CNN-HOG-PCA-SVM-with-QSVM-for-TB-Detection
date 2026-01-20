import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ===== Paths =====
# Kaggle (training) fused features
kaggle_fused = Path("data/features/fused/fused_raw.npy")
kaggle_labels = Path("data/features/fused/labels.npy")

# External (Shenzhen)
external_cnn = Path("data/external/shenzhen/features/cnn.npy")
external_labels = Path("data/external/shenzhen/labels.npy")

out_dir = Path("data/external/shenzhen/features_pca")
out_dir.mkdir(parents=True, exist_ok=True)

# ===== Fit scaler + PCA on Kaggle data ONLY =====
X_kaggle = np.load(kaggle_fused)

scaler = StandardScaler()
X_kaggle_std = scaler.fit_transform(X_kaggle)

pca = PCA(n_components=64, random_state=42)
pca.fit(X_kaggle_std)

# ===== Transform external data (NO FIT) =====
X_ext = np.load(external_cnn)
X_ext_std = scaler.transform(X_ext)
X_ext_pca = pca.transform(X_ext_std)

# ===== Save =====
np.save(out_dir / "pca_64.npy", X_ext_pca)
np.save(out_dir / "labels.npy", np.load(external_labels))

print("External PCA-64 features saved:", X_ext_pca.shape)
