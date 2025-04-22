import os
import json
from openimages.download import download_images
from pathlib import Path
from tqdm import tqdm

# ✅ PII classes (downloaded successfully)
PII_CLASSES = ["Person", "Man", "Woman"]

# ✅ Non-PII classes (downloaded successfully)
NON_PII_CLASSES = ["Cat", "Dog", "Furniture", "Tool"]

# 📁 Output structure
OUTPUT_DIR = Path("data_image")
PII_DIR = OUTPUT_DIR / "1"
NON_PII_DIR = OUTPUT_DIR / "0"
PREP_JSON = OUTPUT_DIR / "preprocessed_images.json"
CSV_DIR = Path("openimages_csv")

# ⚙️ Config
EXCLUSIONS_PATH = None
SAMPLES_PER_CLASS = 150

# 📂 Ensure output directories exist
PREP_JSON.parent.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

def download_labels(label_list, out_dir):
    for label in tqdm(label_list, desc=f"Downloading to {out_dir.name}"):
        try:
            download_images(
                dest_dir=str(out_dir),
                class_labels=[label],
                exclusions_path=EXCLUSIONS_PATH,
                csv_dir=str(CSV_DIR),
                limit=SAMPLES_PER_CLASS
            )
        except Exception as e:
            print(f"❌ Failed to download {label}: {e}")

def generate_json(data_dirs):
    dataset = []
    current_id = 0
    for flag, directory in data_dirs.items():
        for img in Path(directory).rglob("*.jpg"):
            dataset.append({
                "dataset_id": 2,
                "id": current_id,
                "value": str(img),
                "flag": int(flag)
            })
            current_id += 1
    with open(PREP_JSON, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
    print(f"✅ Saved: {PREP_JSON}")
    print(f"📦 Total samples: {len(dataset)}")

def summarize_images():
    print("\n📊 Image Summary:")
    for label, path in {"PII": PII_DIR, "Non-PII": NON_PII_DIR}.items():
        count = len(list(path.rglob("*.jpg")))
        print(f"  - {label} → {count} images")

def main():
    PII_DIR.mkdir(parents=True, exist_ok=True)
    NON_PII_DIR.mkdir(parents=True, exist_ok=True)

    print("📥 Downloading PII images...")
    download_labels(PII_CLASSES, PII_DIR)

    print("📥 Downloading Non-PII images...")
    download_labels(NON_PII_CLASSES, NON_PII_DIR)

    print("🧾 Generating preprocessed_images.json...")
    generate_json({1: PII_DIR, 0: NON_PII_DIR})
    summarize_images()

if __name__ == "__main__":
    main()
