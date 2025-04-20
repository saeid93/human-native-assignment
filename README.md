# 📄 Take Home Assignment – Machine Learning (PII Violation Detection)

## 🚀 Overview

This project solves the task of **automatically flagging data records (text)** that may contain **personally identifiable information (PII)**, as described in the assignment prompt.

It includes:
- End-to-end data preprocessing
- Binary classification models (TF-IDF + Logistic Regression, BERT)
- Evaluation with precision/recall/F1
- Model checkpoint loading for inference

---

## 📦 Directory Structure

```
human-native/
├── data/
│   ├── train.json
│   ├── preprocessed_train.json
│   ├── splited_train.json
│   ├── splited_test.json
│   └── dataset_metadata.json
├── preprocess.py
├── split.py
├── main.py
├── tfidf_model.py
├── bert_model.py
├── evaluate.py
├── evaluate_bert_checkpoint.py
└── requirements.txt
```

---

## ✅ Task Summary

The assignment asked for a model to flag content that:
- May violate privacy laws (e.g. contains PII)
- Works at the **record level** (`flag = 0/1`)
- Uses a realistic simulation of flagged and clean data

We used the TLAL dataset (student essays) and converted it to the assignment-compliant format:
```json
{
  "dataset_id": 1,
  "id": 42,
  "value": "My name is Jack and I live in Boston.",
  "flag": 1
}
```

A companion `Dataset(...)` metadata JSON was also created.

---

## 🧹 Preprocessing

Run:

```bash
python preprocess.py
```

This:
- Cleans the input data
- Adds `flag` and `dataset_id`
- Saves:
  - `preprocessed_train.json`
  - `dataset_metadata.json`

---

## 🔀 Splitting

Run:

```bash
python split.py
```

This creates:
- `splited_train.json`
- `splited_test.json`

Using a **stratified 80/20 split**.

---

## 🧠 Training and Evaluation

### TF-IDF + Logistic Regression

```bash
python main.py --model tfidf
```

### BERT (fine-tuned)

```bash
python main.py --model bert
```

Both routes print a classification report using scikit-learn.

---

## 🧪 Evaluate from Last BERT Checkpoint

```bash
python evaluate_bert_checkpoint.py
```

This script:
- Finds the latest checkpoint in `bert_output/`
- Loads the saved model and tokenizer
- Evaluates on `splited_test.json`

---

## ⚙️ Setup

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Contents include:
```
transformers
datasets
scikit-learn
pandas
click
tqdm
```

---

## ✅ Assumptions

- Input modality is **text** (permitted by assignment)
- PII detection framed as **binary classification**
- Clean and flagged data simulated using a real-world dataset (TLAL)
- Class imbalance handled via stratified splits
- Evaluation is focused on **per-record flagging**, not token-level spans

---

## 📈 Performance Summary (Final BERT Model)

| Class    | Precision | Recall | F1-score |
|----------|-----------|--------|----------|
| No PII   | 0.97      | 0.99   | 0.98     |
| PII      | 0.93      | 0.83   | 0.88     |

- **Accuracy**: 97%
- **Macro F1**: 0.93

---

## 📫 Contact

If you have any questions or would like to discuss the approach in more detail, I’d be happy to explain my reasoning or design choices.
`sdghafouri@gmail.com`
