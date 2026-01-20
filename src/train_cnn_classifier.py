import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import timm
from tqdm import tqdm

# ---------- Dataset ----------
class KaggleDataset(Dataset):
    def __init__(self, img_dir, labels, transform):
        self.img_dir = Path(img_dir)
        self.labels = labels
        self.images = sorted(list(self.img_dir.glob("*.png")))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]

# ---------- Config ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
epochs = 5
batch_size = 16

# ---------- Load labels ----------
labels = np.load("data/features/kaggle_tb_debug/labels.npy")

# ---------- Transforms ----------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ---------- Data ----------
dataset = KaggleDataset(
    img_dir="data/segmented/images",
    labels=labels,
    transform=transform
)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ---------- Model ----------
model = timm.create_model("tf_efficientnet_b0_ns", pretrained=True, num_classes=2)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ---------- Train ----------
print("Training CNN classifier on Kaggle...")
for epoch in range(epochs):
    model.train()
    running_loss = 0
    for x,y in tqdm(loader):
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss={running_loss/len(loader):.4f}")

# ---------- Save ----------
Path("models").mkdir(exist_ok=True)
torch.save(model.state_dict(), "models/cnn_classifier_kaggle.pth")
print("Saved CNN classifier")
