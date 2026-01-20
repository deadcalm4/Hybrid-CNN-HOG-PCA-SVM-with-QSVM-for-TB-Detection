import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

X = np.load("data/external/shenzhen/features/fused_pca64.npy")
y = np.load("data/external/shenzhen/features/labels.npy")

svm = joblib.load("models/svm_fused_pca64.joblib")

y_pred = svm.predict(X)

print({
    "Dataset": "Shenzhen External",
    "Accuracy": accuracy_score(y, y_pred),
    "Precision": precision_score(y, y_pred),
    "Recall": recall_score(y, y_pred),
    "F1": f1_score(y, y_pred)
})
