# ==========================================
# Phase 7.2: Classical SVM on 300-sample subset
# ==========================================

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def main():
    print(">>> Classical SVM on 300-sample subset <<<")

    X = np.load("data/features/fused/fused_pca_32_subset_300.npy")
    y = np.load("data/features/fused/labels_subset_300.npy")

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    accs, precs, recs, f1s = [], [], [], []

    for tr, te in skf.split(X, y):
        clf = SVC(kernel="rbf", C=1.0, gamma="scale")
        clf.fit(X[tr], y[tr])
        yp = clf.predict(X[te])

        accs.append(accuracy_score(y[te], yp))
        precs.append(precision_score(y[te], yp))
        recs.append(recall_score(y[te], yp))
        f1s.append(f1_score(y[te], yp))

    res = {
        "Model": "Classical_SVM_Subset_300",
        "Accuracy": float(np.mean(accs)),
        "Precision": float(np.mean(precs)),
        "Recall": float(np.mean(recs)),
        "F1": float(np.mean(f1s)),
    }

    out = Path("results/phase7_qsvm")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([res]).to_csv(out / "svm_subset_300_results.csv", index=False)

    print(res)
    print(">>> DONE <<<")

if __name__ == "__main__":
    main()
