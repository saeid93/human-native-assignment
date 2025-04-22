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

## 🗂️ Folder Structure

```
.
├── README.md
├── data_image/
│   ├── 0/                        # Non-PII images
│   ├── 1/                        # PII images
│   └── preprocessed_images.json
├── data_nlp/
│   ├── preprocessed_train.json
│   ├── splited_train.json
│   ├── splited_test.json
│   └── dataset_metadata.json
├── models/
│   ├── 0/                        # Trained BERT model
│   └── 1/                        # Trained ResNet model
├── main_training.py             # Training script
├── main_inference.py            # Evaluation script
├── preprocess_nlp.py            # Preprocess TLAL dataset
├── split_text_dataset.py        # Stratified text split
├── download_images.py           # Download OpenImages
├── model_bert.py                # BERT wrapper
├── model_resnet.py              # ResNet wrapper
├── requirements.txt
└── utils.py
```

---

## 📁 Datasets Used

### 1. Text Modality – TLAL Dataset (Kaggle)

- Source: [Kaggle PII Detection Challenge](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data)
- Essays from MOOC students with token-level BIO labels

#### Preprocessing

```bash
python preprocess_nlp.py
python split_text_dataset.py
```

- Converts token labels to `flag = 1` if any token is PII
- Produces `preprocessed_train.json`, then stratified into `splited_train.json` and `splited_test.json`

Example:
```json
{ "dataset_id": 1, "id": 7, "value": "My name is John", "flag": 1 }
```

---

### 2. Image Modality – OpenImages V6

- Source: [OpenImages Dataset](https://storage.googleapis.com/openimages/web/index.html)
- Simulates visual PII (e.g., person vs. cat)

#### Labels

| Flag | Classes                        |
|------|--------------------------------|
| 1    | Person, Man, Woman             |
| 0    | Cat, Dog, Furniture, Tool      |

#### Preprocessing

```bash
python download_images.py
```

- Downloads 150 images per class
- Generates `preprocessed_images.json`

Example:
```json
{ "dataset_id": 2, "id": 45, "value": "data_image/1/person_23.jpg", "flag": 1 }
```

---

## 💾 Quick Start (Prebuilt Data & Models)

📦 Download everything preprocessed:  
[**all_data.zip**](https://drive.google.com/file/d/1NjFN8QbQaTaR1KADrj3CacImxnUY6JJu/view?usp=sharing)

```bash
unzip all_data.zip
```

This creates:
```
data_nlp/
data_image/
models/
```

No need to run preprocessing or training.

---

## 🧰 Installation

```bash
pip install -r requirements.txt
```

Key packages:
- `transformers`, `datasets`, `torch`, `torchvision`
- `scikit-learn`, `click`, `pandas`, `tqdm`, `Pillow`

---

## 🏋️ Training

To train a new model:

```bash
python main_training.py --model bert
python main_training.py --model resnet
```

- Automatically saved under `models/{n}/` with `metadata.json` and model files

---

## ✅ Evaluation

Run evaluation on trained models:

```bash
python main_inference.py --type nlp --model-dir models/0
python main_inference.py --type image --model-dir models/1
```

---

## 📊 Evaluation Results

### Text – BERT (Model ID 0)

```bash
python main_inference.py --type nlp --model-dir models/0
```

```
Evaluation Report (NLP Model):
              precision    recall  f1-score   support
      No PII     0.9757    0.9906    0.9831      1173
         PII     0.9357    0.8466    0.8889       189
    accuracy                         0.9706      1362
   macro avg     0.9557    0.9186    0.9360      1362
weighted avg     0.9701    0.9706    0.9700      1362
```

---

### Image – ResNet18 (Model ID 1)

```bash
python main_inference.py --type image --model-dir models/1
```

```
Evaluation Report (Image Model):
              precision    recall  f1-score   support
      No PII     0.9886    0.9667    0.9775        90
         PII     0.9674    0.9889    0.9780        90
    accuracy                         0.9778       180
   macro avg     0.9780    0.9778    0.9778       180
weighted avg     0.9780    0.9778    0.9778       180
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
