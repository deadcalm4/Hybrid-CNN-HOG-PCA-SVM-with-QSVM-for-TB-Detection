import matplotlib.pyplot as plt
from pathlib import Path

# Accuracy values (LOCKED from experiments)
models = [
    "CNN only",
    "HOG only",
    "CNN + HOG",
    "CNN + HOG + PCA-64"
]

accuracies = [
    96.50,
    97.40,
    97.90,
    98.29
]

# Create output directory
out_dir = Path("results/figures")
out_dir.mkdir(parents=True, exist_ok=True)

# Plot
plt.figure(figsize=(6, 4))
bars = plt.bar(models, accuracies)

plt.ylabel("Accuracy (%)")
plt.ylim(95, 99)
plt.title("Accuracy Comparison of Different Feature Configurations")

# Annotate bars
for bar, acc in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.05,
        f"{acc:.2f}%",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.tight_layout()
plt.savefig(out_dir / "accuracy_comparison.png", dpi=300)
plt.close()

print("Accuracy bar chart saved to results/figures/accuracy_comparison.png")
