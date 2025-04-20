import click
import pandas as pd
import json
from tfidf_model import TFIDFClassifier
from bert_model import BERTClassifier
from evaluate import evaluate


def load_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads training and testing data from JSON files and returns them as pandas DataFrames.
    """
    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    return train_df, test_df


def get_model(model_type: str):
    """
    Returns the initialized model based on user selection.
    """
    if model_type == "tfidf":
        return TFIDFClassifier()
    elif model_type == "bert":
        return BERTClassifier()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def train_model(model, model_type: str, train_df: pd.DataFrame) -> None:
    """
    Trains the selected model on the provided training DataFrame.
    """
    if model_type == "tfidf":
        model.fit(train_df["value"], train_df["flag"])
    elif model_type == "bert":
        model.fit(train_df)


def run_pipeline(model_type: str) -> None:
    """
    Executes the training and evaluation pipeline.
    """
    train_df, test_df = load_data(
        "data/splited_train.json",
        "data/splited_test.json"
    )

    model = get_model(model_type)
    train_model(model, model_type, train_df)
    evaluate(model, test_df["value"], test_df["flag"], model_type=model_type)


@click.command()
@click.option(
    "--model",
    type=click.Choice(["tfidf", "bert"], case_sensitive=False),
    default="bert",
    help="Which model to use: tfidf or bert"
)
def main(model: str) -> None:
    """
    Main CLI entrypoint to run training and evaluation.
    """
    run_pipeline(model.lower())


if __name__ == "__main__":
    main()
