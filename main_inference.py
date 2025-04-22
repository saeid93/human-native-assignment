import json
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import click

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import Dataset as HFDataset

# Global configs
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
BATCH_SIZE = 32

# Image Dataset
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
        return img, entry["flag"]

# Load image data (same stratified split)
def load_image_test_data():
    with open("data_image/preprocessed_images.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    pii = [d for d in data if d["flag"] == 1]
    non_pii = [d for d in data if d["flag"] == 0]
    count = min(len(pii), len(non_pii), 450)
    random.seed(42)
    sample = random.sample(pii, count) + random.sample(non_pii, count)
    _, test_data = train_test_split(sample, test_size=0.2, stratify=[x["flag"] for x in sample], random_state=42)
    return test_data

def evaluate_image_model(model_path: Path):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path / "model.pth", map_location=DEVICE))
    model.to(DEVICE).eval()

    test_data = load_image_test_data()
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    loader = DataLoader(ImagePIIDataset(test_data, transform), batch_size=BATCH_SIZE)

    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in tqdm(loader, desc="Evaluating Image"):
            X, y = X.to(DEVICE), y.to(DEVICE)
            outputs = model(X)
            preds = outputs.argmax(dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print("\n📊 Evaluation Report (Image Model):")
    print(classification_report(y_true, y_pred, target_names=["No PII", "PII"], digits=4))

def evaluate_nlp_model(model_path: Path):
    with open("data_nlp/splited_test.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)
    test_df = pd.DataFrame(test_data)

    tokenizer = AutoTokenizer.from_pretrained(model_path / "bert_model")
    model = AutoModelForSequenceClassification.from_pretrained(model_path / "bert_model").to(DEVICE)

    def tokenize(batch):
        return tokenizer(batch["value"], truncation=True)

    dataset = HFDataset.from_pandas(test_df[["value"]])
    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer)
    )

    preds = trainer.predict(dataset).predictions.argmax(axis=1)
    print("\n📊 Evaluation Report (NLP Model):")
    print(classification_report(test_df["flag"], preds, target_names=["No PII", "PII"], digits=4))


@click.command()
@click.option("--type", type=click.Choice(["image", "nlp"]), required=True, help="Model type: image or nlp")
@click.option("--model-dir", type=click.Path(exists=True, file_okay=False), required=True, help="Path to model folder")
def main(type, model_dir):
    model_path = Path(model_dir)

    if type == "image":
        evaluate_image_model(model_path)
    elif type == "nlp":
        evaluate_nlp_model(model_path)

if __name__ == "__main__":
    main()
