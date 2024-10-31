import random

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from evaluate import load
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from configs.config_matcher_sentence_pair import MatcherConfig
from configs.parameters_for_training import ParametersTraining


class MatcherSentencePairClassification:
    def __init__(
        self, matcher_config: MatcherConfig, training_config: ParametersTraining
    ):
        self.matcher_config = matcher_config
        self.training_config = training_config
        self.dataset = self.load_data(self.matcher_config.base_dataset_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.matcher_config.model_checkpoint
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.matcher_config.model_checkpoint,
            num_labels=self.matcher_config.num_labels,
        )
        self.trained_model = None

    @staticmethod
    def load_data(path: str) -> pd.DataFrame:
        if path.endswith(".csv"):
            return pd.read_csv(path)
        elif path.endswith(".xlsx") or path.endswith(".xls"):
            return pd.read_excel(path)
        elif path.endswith(".parquet"):
            return pd.read_parquet(path)
        else:
            raise ValueError(f"Неправильный формат файла {path}")

    @staticmethod
    def compute_metrics(eval_pred):
        f1_metric = load("f1")
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return f1_metric.compute(predictions=predictions, references=labels)

    def get_dataset_with_negative_samples(self) -> pd.DataFrame:
        dataset = self.load_data(self.matcher_config.base_dataset_path)
        dataset = dataset[
            [
                self.matcher_config.question_col_name,
                self.matcher_config.context_col_name,
            ]
        ]
        dataset["label"] = 1

        existing_pairs = set(
            zip(
                dataset[self.matcher_config.question_col_name],
                dataset[self.matcher_config.context_col_name],
            )
        )

        negative_samples = []
        for name in dataset[self.matcher_config.question_col_name].unique():
            locals_name = random.sample(
                population=list(dataset[self.matcher_config.context_col_name]), k=5
            )
            for local_name in locals_name:
                if (local_name, name) not in existing_pairs:
                    negative_samples.append(
                        {
                            self.matcher_config.context_col_name: local_name,
                            self.matcher_config.question_col_name: name,
                        }
                    )

        data_negative = pd.DataFrame(negative_samples)
        data_negative["label"] = 0

        data = pd.concat([dataset, data_negative], ignore_index=True)
        data.drop_duplicates(
            subset=[
                self.matcher_config.context_col_name,
                self.matcher_config.question_col_name,
            ],
            inplace=True,
        )

        if self.matcher_config.negative_dataset_path_save:
            data.to_csv(
                f"{self.matcher_config.negative_dataset_path_save}.csv", index=False
            )

        return data

    def get_dict_for_training(self) -> DatasetDict:
        data = self.get_dataset_with_negative_samples()

        train, valid = train_test_split(
            data,
            test_size=self.matcher_config.test_size,
            random_state=self.matcher_config.random_state,
        )

        dataset_dict = DatasetDict(
            {
                "train": Dataset.from_pandas(train),
                "validation": Dataset.from_pandas(valid),
            }
        )

        if self.matcher_config.train_path_save:
            train.to_json(
                f"{self.matcher_config.train_path_save}/train.jsonl",
                orient="records",
                lines=True,
            )
            valid.to_json(
                f"{self.matcher_config.valid_path_save}/valid.jsonl",
                orient="records",
                lines=True,
            )

        return dataset_dict

    def tokenize_function(self, batch):
        return self.tokenizer(
            batch[self.matcher_config.context_col_name],
            batch[self.matcher_config.question_col_name],
            truncation=True,
            padding="max_length",
            max_length=self.training_config.max_length,
        )

    def get_data_for_training(self) -> DatasetDict:
        dataset_dict = self.get_dict_for_training()

        return dataset_dict.map(self.tokenize_function, batched=True)

    def train(self):
        tokenized_dataset = self.get_dict_for_training()

        training_args = TrainingArguments(
            output_dir=self.training_config.output_dir,
            num_train_epochs=self.training_config.num_train_epochs,
            per_device_train_batch_size=self.training_config.per_device_train_batch_size,
            per_device_eval_batch_size=self.training_config.per_device_eval_batch_size,
            warmup_steps=self.training_config.warmup_steps,
            weight_decay=self.training_config.weight_decay,
            learning_rate=self.training_config.learning_rate,
            save_total_limit=self.training_config.save_total_limit,
            logging_dir=self.training_config.logging_dir,
            logging_steps=self.training_config.logging_steps,
            evaluation_strategy=self.training_config.evaluation_strategy,
            eval_steps=self.training_config.eval_steps,
            save_strategy=self.training_config.save_strategy,
            save_steps=self.training_config.save_steps,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            compute_metrics=self.compute_metrics,
        )
        trainer.train()

    def get_top_k(self, question: str, top_k: int):

        if self.trained_model is None:
            model = AutoModelForSequenceClassification.from_pretrained(
                self.training_config.model_checkpoint_trained_model
            ).to(self.training_config.device)

        scores = []

        for col in tqdm(self.dataset[self.matcher_config.context_col_name].unique()):
            inputs = self.tokenizer(
                question,
                col,
                truncation=True,
                padding="max_length",
                max_length=self.training_config.max_length,
                return_tensors="pt",
            ).to(self.training_config.device)

            with torch.no_grad():
                outputs = model(**inputs)

            logits = outputs.logits.cpu()
            proba = torch.sigmoid(logits)

            scores.append(proba[0])

        scores = torch.stack(scores)

        top_k_idx = scores[:, 1].argsort(descending=True)[:top_k]

        labels = self.dataset[self.matcher_config.context_col_name].unique()[top_k_idx]

        probabilities = scores[top_k_idx, 1]

        output = [
            {"local_name": label, "probability": prob}
            for label, prob in zip(labels.tolist(), probabilities.tolist())
        ]

        return output

    def get_top_k_batch(self, question: str, top_k: int, batch_size: int = 32):
        if self.trained_model is None:
            model = AutoModelForSequenceClassification.from_pretrained(
                self.training_config.model_checkpoint_trained_model
            ).to(self.training_config.device)
            model.eval()

        unique_contexts = self.dataset[self.matcher_config.context_col_name].unique()

        inputs = [
            self.tokenizer(
                question,
                context,
                truncation=True,
                padding="max_length",
                max_length=self.training_config.max_length,
                return_tensors="pt",
            )
            for context in unique_contexts
        ]

        input_idx = torch.cat([idx["input_ids"] for idx in inputs], dim=0)
        attention_mask = torch.cat([mask["attention_mask"] for mask in inputs], dim=0)

        dataset = TensorDataset(input_idx, attention_mask)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        scores = []

        with torch.no_grad():
            for batch in tqdm(dataloader):
                batch_input_ids, batch_attention_mask = batch
                batch_input_ids = batch_input_ids.to(self.training_config.device)
                batch_attention_mask = batch_attention_mask.to(
                    self.training_config.device
                )

                outputs = model(
                    input_ids=batch_input_ids, attention_mask=batch_attention_mask
                )

                logits = outputs.logits.cpu()
                proba = torch.sigmoid(logits)

                scores.extend(proba[:, 1].tolist())

        scores = torch.tensor(scores)

        top_k_idx = scores.argsort(descending=True)[:top_k]

        labels = unique_contexts[top_k_idx]
        probabilities = scores[top_k_idx]

        output = [
            {"local_name": label, "probability": prob}
            for label, prob in zip(labels.tolist(), probabilities.tolist())
        ]

        return output
