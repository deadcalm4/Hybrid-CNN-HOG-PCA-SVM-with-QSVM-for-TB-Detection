import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

# Kaggle CNN features
X = np.load("data/features/kaggle_tb_debug/cnn.npy")
y = np.load("data/features/kaggle_tb_debug/labels.npy")

out_dir = Path("models")
out_dir.mkdir(exist_ok=True)

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=1.0, gamma="scale"))
])

print("Training CNN-only SVM on Kaggle...")
clf.fit(X, y)

joblib.dump(clf, out_dir / "svm_cnn_only.joblib")
print("Saved model to models/svm_cnn_only.joblib")
