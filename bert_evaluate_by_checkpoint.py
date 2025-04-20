import os
import json
import pandas as pd
from datasets import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import classification_report


def load_json(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def get_latest_checkpoint(checkpoint_dir: str) -> str:
    """
    Returns the path to the latest checkpoint directory.
    """
    checkpoints = [
        d for d in os.listdir(checkpoint_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(checkpoint_dir, d))
    ]
    if not checkpoints:
        raise ValueError("No checkpoints found in the directory.")
    
    latest = max(checkpoints, key=lambda x: int(x.split("-")[1]))
    latest_path = os.path.join(checkpoint_dir, latest)
    print(f"🧠 Loading latest checkpoint: {latest_path}")
    return latest_path


def evaluate_bert_from_checkpoint(base_dir: str, test_json_path: str) -> None:
    latest_checkpoint = get_latest_checkpoint(base_dir)

    # Load model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(latest_checkpoint)
    model = BertForSequenceClassification.from_pretrained(latest_checkpoint)

    # Load and tokenize test data
    test_df = load_json(test_json_path)
    true_labels = test_df["flag"]

    dataset = Dataset.from_pandas(test_df[["value"]])
    dataset = dataset.map(lambda x: tokenizer(x["value"], truncation=True), batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    predictions = trainer.predict(dataset).predictions.argmax(axis=1)

    print("\n📊 Evaluation from Latest Checkpoint")
    print(classification_report(true_labels, predictions, target_names=["No PII", "PII"]))


if __name__ == "__main__":
    evaluate_bert_from_checkpoint(
        base_dir="bert_output",                   # base model output directory
        test_json_path="data/splited_test.json"   # path to your test file
    )
