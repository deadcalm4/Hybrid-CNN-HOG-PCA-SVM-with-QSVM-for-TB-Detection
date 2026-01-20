import os
import numpy as np
from pathlib import Path

img_dir = Path("data/external/shenzhen/raw/images")

labels = []
filenames = []

for img in sorted(img_dir.glob("*.png")):
    filenames.append(img.name)
    if img.name.endswith("_1.png"):
        labels.append(1)
    else:
        labels.append(0)

labels = np.array(labels)

out_dir = Path("data/external/shenzhen")
out_dir.mkdir(parents=True, exist_ok=True)

np.save(out_dir / "labels.npy", labels)

print("Total images:", len(labels))
print("TB cases:", labels.sum())
print("Normal cases:", len(labels) - labels.sum())
