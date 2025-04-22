import json
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Config
DATA_PATH = Path("data_image/preprocessed_images.json")
BATCH_SIZE = 32
EPOCHS = 5
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset class
class ImagePIIDataset(Dataset):
    def __init__(self, entries, transform=None):
        self.entries = entries
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        img = Image.open(entry["value"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = entry["flag"]
        return img, label

# Load and balance data
def load_balanced_data():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    pii = [d for d in data if d["flag"] == 1]
    non_pii = [d for d in data if d["flag"] == 0]
    count = min(len(pii), len(non_pii), 450)

    random.seed(42)
    pii_sample = random.sample(pii, count)
    non_pii_sample = random.sample(non_pii, count)
    full_data = pii_sample + non_pii_sample
    random.shuffle(full_data)

    return train_test_split(full_data, test_size=0.2, stratify=[x["flag"] for x in full_data], random_state=42)

# Create new model output folder with metadata
def prepare_model_dir(base_dir="models", dataset_id=2) -> Path:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    subdirs = sorted([int(p.name) for p in base.iterdir() if p.is_dir() and p.name.isdigit()])
    new_id = subdirs[-1] + 1 if subdirs else 0
    model_dir = base / str(new_id)
    model_dir.mkdir()
    with open(model_dir / "metadata.json", "w") as f:
        json.dump({
            "model_type": "resnet",
            "dataset_id": dataset_id
        }, f, indent=2)
    return model_dir

# Training function
def train_model(train_loader, model, criterion, optimizer):
    model.train()
    total_loss = 0
    for inputs, targets in tqdm(train_loader, desc="Training", leave=False):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Evaluation function
def evaluate_model(test_loader, model):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return correct / total

# Pipeline entry function for main.py
def run_resnet_pipeline():
    train_data, test_data = load_balanced_data()

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(ImagePIIDataset(train_data, transform), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(ImagePIIDataset(test_data, transform), batch_size=BATCH_SIZE)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        train_loss = train_model(train_loader, model, criterion, optimizer)
        acc = evaluate_model(test_loader, model)
        print(f"Loss: {train_loss:.4f} | Accuracy: {acc:.4f}")

    model_dir = prepare_model_dir()
    torch.save(model.state_dict(), model_dir / "model.pth")
    print(f"\n💾 Model saved to: {model_dir / 'model.pth'}")

