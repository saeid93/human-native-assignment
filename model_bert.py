from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import Dataset
import json
from pathlib import Path


class BERTClassifier:
    def __init__(self, model_name="bert-base-uncased", model_dir=None):
        self.model_name = model_name
        self.model_dir = Path(model_dir)
        self.save_path = self.model_dir / "bert_model"
        self.save_path.mkdir(parents=True, exist_ok=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
        self.trainer = None

    def _tokenize(self, examples):
        return self.tokenizer(examples["value"], truncation=True)

    def fit(self, train_df):
        dataset = Dataset.from_pandas(train_df)
        dataset = dataset.rename_column("flag", "label")
        dataset = dataset.map(self._tokenize, batched=True)
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

        args = TrainingArguments(
            output_dir=str(self.save_path),
            num_train_epochs=2,
            per_device_train_batch_size=8,
            learning_rate=2e-5,
            save_strategy="no"  # only save manually at the end
        )

        self.trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
        )

        self.trainer.train()

        # Save final model + tokenizer
        self.trainer.save_model(self.save_path)
        self.tokenizer.save_pretrained(self.save_path)

        # Save metadata
        metadata = {
            "model_type": "bert",
            "model_name": self.model_name,
            "subfolder": "bert_model"
        }
        with open(self.model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def predict(self, X_series):
        df = X_series.to_frame(name="value")
        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(self._tokenize, batched=True)
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

        predictions = self.trainer.predict(dataset)
        return predictions.predictions.argmax(axis=1)
