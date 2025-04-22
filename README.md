# Take Home Assignment – Machine Learning (PII Violation Detection)

## 🚀 Project Overview

This project was developed for a take-home assignment that requires building a machine learning system capable of **automatically flagging Personally Identifiable Information (PII)** in either text or image data.

The system supports:
- Two modalities: **Text (BERT)** and **Image (ResNet18)**
- Unified data format for both types
- Record-level binary classification
- Evaluation using realistic public datasets

---

## 📘 Assignment Task Summary

> You are given records in the format `Data(dataset_id, id, value, flag)`. Your model must predict `flag = 1` when the `value` potentially violates privacy, based on either textual or visual content.

Requirements:
- Support at least one modality
- `flag = 1` for sensitive content
- Simulated or real datasets allowed

---

## 🧩 Assumptions and Model Choices

This implementation is based on the following **assumptions** and **design decisions** in line with the assignment requirements:

- **Record-Level PII Flagging**: The task requires identifying whether a record (text or image) contains PII. Therefore, even if the source dataset provides token-level or bounding-box annotations, I aggregated them to a single binary `flag` per record, as per the required `Data(dataset_id, id, value, flag)` format.

- **Text Dataset (TLAL)**: Although the original dataset includes fine-grained annotations (e.g., BIO-tagged names and IDs), I treated the presence of any non-"O" tag as `flag = 1`. This simulates a realistic PII detection task where even one PII token makes the whole record sensitive.

- **Image Dataset (OpenImages)**: I simulated visual PII content by downloading images from person-related and object-related categories. Images of "Person", "Man", "Woman" were labeled as PII (`flag = 1`), while categories like "Cat", "Dog", and "Furniture" were used as safe content (`flag = 0`).

- **Model Choices**:
  - **BERT** (`bert-base-uncased`) was used for the NLP pipeline to align with modern transformer-based PII detection standards.
  - **ResNet18** was used for the image pipeline due to its proven accuracy and simplicity for binary classification tasks.

These decisions balance realism, interpretability, and computational efficiency while staying compliant with the original assignment objectives.

---

## 🗂️ Folder Structure

```
.
├── README.md
├── data_image/
│   ├── 0/
│   ├── 1/
│   └── preprocessed_images.json
├── data_nlp/
│   ├── preprocessed_train.json
│   ├── splited_train.json
│   ├── splited_test.json
│   └── dataset_metadata.json
├── models/
│   ├── 0/         # BERT model
│   └── 1/         # ResNet model
├── main_training.py
├── main_inference.py
├── preprocess_nlp.py
├── split_text_dataset.py
├── download_images.py
├── model_bert.py
├── model_resnet.py
├── requirements.txt
└── utils.py
```

---

## 📁 Datasets Used

### 1. Text Modality – TLAL Dataset (Kaggle)

- Source: [Kaggle PII Detection Challenge](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data)
- Contains student essays with token-level BIO tags

#### Preprocessing

```bash
python preprocess_nlp.py
python split_text_dataset.py
```

This converts each record into a binary format (`flag = 1` if any token is PII).

---

### 2. Image Modality – OpenImages V6

- Source: [OpenImages Dataset](https://storage.googleapis.com/openimages/web/index.html)
- Used for simulating visual PII content

| Flag | Example Classes            |
|------|----------------------------|
| 1    | Person, Man, Woman         |
| 0    | Cat, Dog, Furniture, Tool  |

#### Preprocessing

```bash
python download_images.py
```

This downloads ~150 images per class and generates a binary-labeled JSON file.

---

## 💾 Quick Start (Prebuilt Data & Models)

Download from Google Drive:  
🔗 [all_data.zip](https://drive.google.com/file/d/1NjFN8QbQaTaR1KADrj3CacImxnUY6JJu/view?usp=sharing)

```bash
unzip all_data.zip
```

This gives you:
```
data_nlp/
data_image/
models/
```

You can skip all preprocessing and training steps.

---

## 🧰 Installation

```bash
pip install -r requirements.txt
```

Dependencies:
- `transformers`, `datasets`, `torch`, `torchvision`
- `pandas`, `scikit-learn`, `tqdm`, `click`, `Pillow`

---

## 🏋️ Training

> ⚠️ **Note**: Pre-trained models are already included in `models/`. You can skip training and go directly to evaluation.

To train from scratch:

```bash
python main_training.py --model bert
python main_training.py --model resnet
```

Each run generates a new folder in `models/` with metadata and saved weights.

---

## ✅ Evaluation

Evaluate trained models:

```bash
python main_inference.py --type nlp --model-dir models/0
python main_inference.py --type image --model-dir models/1
```

---

## 📊 Evaluation Results

### BERT (Text Model – ID 0)

```
              precision    recall  f1-score   support
      No PII     0.9757    0.9906    0.9831      1173
         PII     0.9357    0.8466    0.8889       189
    accuracy                         0.9706      1362
```

---

### ResNet18 (Image Model – ID 1)

```
              precision    recall  f1-score   support
      No PII     0.9886    0.9667    0.9775        90
         PII     0.9674    0.9889    0.9780        90
    accuracy                         0.9778       180
```

---

## 🧾 Summary Table

| Model ID | Modality | Accuracy | Macro F1 | Dataset       |
|----------|----------|----------|----------|---------------|
| `0`      | Text     | 97.1%    | 0.9360   | TLAL (Kaggle) |
| `1`      | Image    | 97.8%    | 0.9778   | OpenImages    |

---

## 🔮 Future Directions

- **Containerization**: Package the pipelines into Docker containers for easier deployment, reproducibility, and integration into real-time services via REST APIs.
- **Fine-Grained Detection**: Extend the models to pinpoint the exact source of PII — token-level highlighting for text and bounding boxes for images using object detection architectures like YOLO or Faster R-CNN.
- **Multimodal Fusion**: Combine both modalities for documents or AI-generated outputs that mix text and image.
- **Data Augmentation**: Improve generalization by generating synthetic PII samples (names, IDs, faces) in both modalities.
- **Explainability Tools**: Use interpretability frameworks like SHAP or attention heatmaps to increase model transparency.

---

## 📬 Contact

📧 sdghafouri@gmail.com
