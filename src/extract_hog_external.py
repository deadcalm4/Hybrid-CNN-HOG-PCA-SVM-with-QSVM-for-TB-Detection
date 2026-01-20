import os
import cv2
import numpy as np
from skimage.feature import hog
from tqdm import tqdm

IMG_DIR = "data/external/shenzhen/segmented/images"
OUT_DIR = "data/external/shenzhen/features"
IMG_SIZE = (512, 512)

hog_features = []
image_names = []

for img_name in tqdm(sorted(os.listdir(IMG_DIR))):
    if img_name.lower().endswith(".png"):
        img_path = os.path.join(IMG_DIR, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, IMG_SIZE)

        feat = hog(
            img,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            block_norm="L2-Hys"
        )

        hog_features.append(feat)
        image_names.append(img_name)

hog_features = np.array(hog_features)

os.makedirs(OUT_DIR, exist_ok=True)
np.save(os.path.join(OUT_DIR, "hog.npy"), hog_features)

print("HOG extracted:", hog_features.shape)
