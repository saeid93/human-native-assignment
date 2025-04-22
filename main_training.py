import click
import pandas as pd
import json
from model_bert import BERTClassifier
from model_resnet import run_resnet_pipeline
from utils import evaluate
from pathlib import Path


def load_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    return pd.DataFrame(train_data), pd.DataFrame(test_data)


def get_next_model_dir(base_dir="models") -> Path:
    Path(base_dir).mkdir(exist_ok=True)
    existing = [int(p.name) for p in Path(base_dir).iterdir() if p.is_dir() and p.name.isdigit()]
    next_id = max(existing, default=-1) + 1
    new_dir = Path(base_dir) / str(next_id)
    new_dir.mkdir()
    return new_dir


def get_model(model_type: str, model_dir: Path, bert_model_name: str = "bert-base-uncased"):
    if model_type == "bert":
        return BERTClassifier(model_name=bert_model_name, model_dir=model_dir)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def train_model(model, model_type: str, train_df: pd.DataFrame) -> None:
    if model_type == "bert":
        model.fit(train_df)


def train_pipeline(model_type: str, bert_model_name: str = "bert-base-uncased"):
    if model_type == "resnet":
        run_resnet_pipeline()
        return

    train_df, test_df = load_data("data_nlp/splited_train.json", "data_nlp/splited_test.json")
    model_dir = get_next_model_dir()
    model = get_model(model_type, model_dir, bert_model_name)
    train_model(model, model_type, train_df)
    evaluate(model, test_df["value"], test_df["flag"], model_type=model_type)


@click.command()
@click.option("--model", type=click.Choice(["bert", "resnet"]), default="bert", help="Model type to train")
@click.option("--bert-model", default="bert-base-uncased", help="Hugging Face model name for BERT")
def main(model: str, bert_model: str):
    train_pipeline(model, bert_model)


if __name__ == "__main__":
    main()
