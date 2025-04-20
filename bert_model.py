from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import Dataset


class BERTClassifier:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=2
        )
        self.trainer = None

    def _tokenize(self, examples):
        return self.tokenizer(examples["value"], truncation=True)

    def fit(self, train_df):
        # Prepare dataset
        dataset = Dataset.from_pandas(train_df)
        dataset = dataset.rename_column("flag", "label")
        dataset = dataset.map(self._tokenize, batched=True)
        dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "label"],
        )

        # Simplified TrainingArguments for compatibility
        args = TrainingArguments(
            output_dir="./bert_output",
            num_train_epochs=2,
            per_device_train_batch_size=8,
            learning_rate=2e-5
        )

        self.trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
        )

        self.trainer.train()

    def predict(self, X_series):
        # Ensure X_series is a DataFrame with 'value' column
        df = X_series.to_frame(name="value")
        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(self._tokenize, batched=True)
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

        predictions = self.trainer.predict(dataset)
        return predictions.predictions.argmax(axis=1)
