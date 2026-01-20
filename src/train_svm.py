# ==========================================
# Phase 6 / 8: Classical SVM
# ==========================================

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_svm(X, y, name):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accs, precs, recs, f1s = [], [], [], []
    cms = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=1.0, gamma="scale"))
        ])

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        accs.append(accuracy_score(y_test, y_pred))
        precs.append(precision_score(y_test, y_pred, zero_division=0))
        recs.append(recall_score(y_test, y_pred, zero_division=0))
        f1s.append(f1_score(y_test, y_pred, zero_division=0))
        cms.append(confusion_matrix(y_test, y_pred))

    cm_mean = np.mean(cms, axis=0).astype(int)

    return {
        "Model": name,
        "Accuracy": np.mean(accs),
        "Precision": np.mean(precs),
        "Recall": np.mean(recs),
        "F1": np.mean(f1s),
        "Confusion": cm_mean
    }


def save_confusion(cm, title, path):
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main(args):
    print(">>> Classical SVM STARTED <<<")

    features_dir = Path(args.features_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(features_dir / "pca_64.npy")
    y = np.load(features_dir / "labels.npy")

    r = evaluate_svm(X, y, args.name)
    save_confusion(r["Confusion"], args.name, out_dir / "confusion.png")

    df = pd.DataFrame([r])
    df.drop(columns=["Confusion"], inplace=True)
    df.to_csv(out_dir / "svm_results.csv", index=False)

    print(df)
    print(">>> COMPLETED <<<")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", required=True)
    parser.add_argument("--out_dir", default="results/phase8_svm")
    parser.add_argument("--name", default="SVM_Result")
    args = parser.parse_args()

    main(args)
