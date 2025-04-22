import json
import os
import logging
from typing import List, Dict, Any
from sklearn.model_selection import train_test_split


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def load_json(path: str) -> List[Dict[str, Any]]:
    logging.info(f"📥 Loading data from {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: List[Dict[str, Any]], path: str) -> None:
    logging.info(f"💾 Saving {len(data)} records to {path}")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def split_json_records(
    data: List[Dict[str, Any]],
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    flags = [item["flag"] for item in data]
    logging.info(f"🔀 Splitting data (test size = {test_size}, stratified by flag)...")
    return train_test_split(
        data,
        test_size=test_size,
        stratify=flags,
        random_state=random_state
    )


def print_summary(train_data: List[Dict[str, Any]], test_data: List[Dict[str, Any]]) -> None:
    train_total = len(train_data)
    test_total = len(test_data)
    train_positives = sum(item["flag"] == 1 for item in train_data)
    test_positives = sum(item["flag"] == 1 for item in test_data)

    summary = f"""
📊 Dataset Split Summary
─────────────────────────────────────────────
Split      | Total | Flag=1 (PII) | Flag=0 (Clean)
---------------------------------------------
Train      | {train_total:<5} | {train_positives:<12} | {train_total - train_positives:<14}
Test       | {test_total:<5} | {test_positives:<12} | {test_total - test_positives:<14}
─────────────────────────────────────────────
"""
    logging.info(summary)


def main() -> None:
    setup_logging()

    input_path = os.path.join("data_nlp", "preprocessed_train.json")
    train_output = os.path.join("data_nlp", "splited_train.json")
    test_output = os.path.join("data_nlp", "splited_test.json")

    full_data = load_json(input_path)
    train_data, test_data = split_json_records(full_data)

    save_json(train_data, train_output)
    save_json(test_data, test_output)
    print_summary(train_data, test_data)

    logging.info(f"✅ Train data saved to: {train_output}")
    logging.info(f"✅ Test data saved to:  {test_output}")


if __name__ == "__main__":
    main()
