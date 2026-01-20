import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

# Load Kaggle fused features (the SAME ones used for training)
X_fused = np.load("data/features/fused/fused_raw.npy")

# Create scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_fused)

# Create PCA-64
pca = PCA(n_components=64, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Save them
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.joblib")
joblib.dump(pca, "models/pca_64.joblib")

print("Scaler and PCA-64 saved successfully")
print("Scaled shape:", X_scaled.shape)
print("PCA shape:", X_pca.shape)
