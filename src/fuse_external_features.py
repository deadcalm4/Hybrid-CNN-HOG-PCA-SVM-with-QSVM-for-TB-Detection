import numpy as np
import os

cnn = np.load("data/external/shenzhen/features/cnn.npy")
hog = np.load("data/external/shenzhen/features/hog.npy")

print("CNN:", cnn.shape)
print("HOG:", hog.shape)

X_fused = np.concatenate([cnn, hog], axis=1)

np.save("data/external/shenzhen/features/fused.npy", X_fused)
print("Fused shape:", X_fused.shape)
