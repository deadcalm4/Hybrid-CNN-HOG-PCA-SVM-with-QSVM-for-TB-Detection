# ==========================================
# Phase 7.3: QSVM on 300-sample subset
# ==========================================

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

def qkernel(X1, X2, fmap):
    return FidelityQuantumKernel(feature_map=fmap).evaluate(X1, X2)

def main():
    print(">>> QSVM on 300-sample subset <<<")

    X = np.load("data/features/fused/fused_pca_32_subset_300.npy")
    y = np.load("data/features/fused/labels_subset_300.npy")

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    chunk = 4

    accs, precs, recs, f1s = [], [], [], []

    for tr, te in skf.split(X, y):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]

        Ks_tr, Ks_te = [], []
        for i in range(X.shape[1] // chunk):
            fmap = ZZFeatureMap(feature_dimension=chunk, reps=2)
            Xt = Xtr[:, i*chunk:(i+1)*chunk]
            Xv = Xte[:, i*chunk:(i+1)*chunk]
            Ks_tr.append(qkernel(Xt, Xt, fmap))
            Ks_te.append(qkernel(Xv, Xt, fmap))

        Ktr = np.mean(Ks_tr, axis=0)
        Kte = np.mean(Ks_te, axis=0)

        clf = SVC(kernel="precomputed", C=1.0)
        clf.fit(Ktr, ytr)
        yp = clf.predict(Kte)

        accs.append(accuracy_score(yte, yp))
        precs.append(precision_score(yte, yp))
        recs.append(recall_score(yte, yp))
        f1s.append(f1_score(yte, yp))

    res = {
        "Model": "QSVM_Subset_300",
        "Accuracy": float(np.mean(accs)),
        "Precision": float(np.mean(precs)),
        "Recall": float(np.mean(recs)),
        "F1": float(np.mean(f1s)),
    }

    out = Path("results/phase7_qsvm")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([res]).to_csv(out / "qsvm_subset_300_results.csv", index=False)

    print(res)
    print(">>> PHASE 7 COMPLETED <<<")

if __name__ == "__main__":
    main()
