import json
import os
import re
import logging
from typing import List, Dict, Any
from tqdm import tqdm


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def compute_flag(labels: List[str]) -> int:
    """
    Returns 1 if any token label is not 'O', indicating presence of PII.
    """
    return int(any(label != "O" for label in labels))


def load_json(path: str) -> List[Dict[str, Any]]:
    """
    Loads a JSON file containing a list of dicts.
    """
    logging.info(f"📥 Loading JSON from {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_unusual_terminators(text: str) -> str:
    """
    Replaces Unicode line/paragraph separators (U+2028 / U+2029) with a space.
    """
    return re.sub(r'[\u2028\u2029]', ' ', text)


def preprocess_dataset(
    raw_data: List[Dict[str, Any]],
    dataset_id: int = 1
) -> tuple[List[Dict[str, Any]], int, int]:
    """
    Processes each entry into the assignment-compliant Data(...) format.
    """
    logging.info("🔄 Preprocessing dataset...")
    cleaned_data: List[Dict[str, Any]] = []

    count_flag_1 = 0
    count_flag_0 = 0

    for i, entry in enumerate(tqdm(raw_data, desc="Processing entries")):
        text = clean_unusual_terminators(entry["full_text"])
        labels = entry["labels"]
        flag = compute_flag(labels)

        count_flag_1 += flag
        count_flag_0 += (1 - flag)

        cleaned_data.append({
            "dataset_id": dataset_id,
            "id": i,
            "value": text,
            "flag": flag
        })

    return cleaned_data, count_flag_1, count_flag_0


def save_json(data: List[Dict[str, Any]], path: str) -> None:
    """
    Saves a list of dictionaries to a JSON file.
    """
    logging.info(f"💾 Saving data to {path}")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_dataset_metadata(path: str) -> None:
    """
    Saves the Dataset(...) metadata as a single JSON object.
    """
    dataset_info = {
        "org_id": 101,
        "id": 1,
        "name": "PII Essay Training",
        "type": "text"
    }

    logging.info(f"💾 Saving dataset metadata to {path}")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)


def main() -> None:
    setup_logging()

    input_path = os.path.join("data_nlp", "train.json")
    data_output_path = os.path.join("data_nlp", "preprocessed_train.json")
    metadata_output_path = os.path.join("data_nlp", "dataset_metadata.json")

    # Load and preprocess
    raw_data = load_json(input_path)
    processed_data, count_flag_1, count_flag_0 = preprocess_dataset(raw_data)

    # Save outputs
    save_json(processed_data, data_output_path)
    save_dataset_metadata(metadata_output_path)

    # Summary
    logging.info("✅ Preprocessing complete")
    logging.info(f"📊 Flag = 1 (PII present): {count_flag_1}")
    logging.info(f"📊 Flag = 0 (No PII): {count_flag_0}")


if __name__ == "__main__":
    main()
