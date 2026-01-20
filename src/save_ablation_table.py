import pandas as pd
from pathlib import Path

# Ablation results (FINAL)
data = [
    ["CNN only", "-", "No", "Full", 96.50],
    ["HOG only", "-", "No", "Full", 97.40],
    ["CNN + HOG", "-", "No", "Full", 97.90],
    ["CNN + HOG", 32, "Yes", "Full", 97.60],
    ["CNN + HOG", 64, "Yes", "Full", 98.29],
    ["CNN + HOG", 64, "No", "Full", 98.14],
    ["QSVM (hybrid)", 32, "Yes", "Subset-300", 50.67],
]

columns = ["Configuration", "PCA", "Segmentation", "Dataset", "Accuracy"]

df = pd.DataFrame(data, columns=columns)

out_dir = Path("results")
out_dir.mkdir(exist_ok=True)

df.to_csv(out_dir / "ablation_table.csv", index=False)

print("ablation_table.csv saved in results/")
