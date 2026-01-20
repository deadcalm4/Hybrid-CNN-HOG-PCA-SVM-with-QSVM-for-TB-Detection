import numpy as np
import joblib

X = np.load("data/external/shenzhen/features/fused.npy")

scaler = joblib.load("models/scaler.joblib")
pca = joblib.load("models/pca_64.joblib")

X_scaled = scaler.transform(X)
X_pca = pca.transform(X_scaled)

np.save("data/external/shenzhen/features/fused_pca64.npy", X_pca)

print("PCA output:", X_pca.shape)
