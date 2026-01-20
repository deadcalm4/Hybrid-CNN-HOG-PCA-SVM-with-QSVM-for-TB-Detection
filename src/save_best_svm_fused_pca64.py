import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from pathlib import Path

# Load full Kaggle PCA-64 features
X = np.load("data/features/fused/fused_pca_64.npy")
y = np.load("data/features/fused/labels.npy")

# Train FINAL model on FULL data
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=1.0, gamma="scale", probability=False))
])

pipeline.fit(X, y)

# Save model
Path("models").mkdir(exist_ok=True)
joblib.dump(pipeline, "models/svm_fused_pca64.joblib")

print("✅ Best hybrid SVM (PCA-64) saved successfully")
