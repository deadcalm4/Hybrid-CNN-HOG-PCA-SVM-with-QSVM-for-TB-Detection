import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load data
X_test = np.load("data/external/shenzhen/features/cnn.npy")
y_test = np.load("data/external/shenzhen/labels.npy")

# IMPORTANT: normalize Shenzhen features independently
scaler = StandardScaler()
X_test_std = scaler.fit_transform(X_test)

# Load trained classifier (WITHOUT scaler)
clf = joblib.load("models/svm_cnn_only.joblib")

# Extract SVM only
svm = clf.named_steps["svm"]

# Predict
y_pred = svm.predict(X_test_std)

results = {
    "Dataset": "Shenzhen External (CNN-only)",
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1": f1_score(y_test, y_pred),
}

print(results)
