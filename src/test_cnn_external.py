import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import timm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

class ShenzhenDataset(Dataset):
    def __init__(self, img_dir, labels, transform):
        self.img_dir = Path(img_dir)
        self.images = sorted(list(self.img_dir.glob("*.png")))
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]

device = "cuda" if torch.cuda.is_available() else "cpu"

labels = np.load("data/external/shenzhen/labels.npy")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

dataset = ShenzhenDataset(
    img_dir="data/external/shenzhen/segmented/images",
    labels=labels,
    transform=transform
)
loader = DataLoader(dataset, batch_size=16, shuffle=False)

model = timm.create_model("tf_efficientnet_b0_ns", pretrained=False, num_classes=2)
model.load_state_dict(torch.load("models/cnn_classifier_kaggle.pth"))
model.to(device)
model.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for x,y in tqdm(loader):
        x = x.to(device)
        out = model(x)
        preds = torch.argmax(out, dim=1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(y.numpy())

print({
    "Dataset": "Shenzhen External",
    "Accuracy": accuracy_score(y_true, y_pred),
    "Precision": precision_score(y_true, y_pred),
    "Recall": recall_score(y_true, y_pred),
    "F1": f1_score(y_true, y_pred)
})
