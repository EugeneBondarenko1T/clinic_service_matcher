import pandas as pd
import random
import numpy as np
import torch

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from evaluate import load
from configs.parameters_for_training import ParametersTraining


class MatcherSentencePairClassification:
    def __init__(self,  path2dataset: str, config: ParametersTraining = None, model_name: str = "roberta-base", num_labels: int = 2):
        self.dataset = self.load_data(path2dataset)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.config = config


    @staticmethod
    def load_data(path: str) -> pd.DataFrame:
        return pd.read_csv(path)
    

    @staticmethod
    def compute_metrics(eval_pred):
        f1_metric = load("f1")
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return f1_metric.compute(predictions=predictions, references=labels)
    

    def get_dataset_with_negative_samples(self, path_save: str, question_col: str, context_col: str) -> pd.DataFrame:
        base_dataset = self.dataset
        base_dataset = base_dataset[[question_col, context_col]]
        base_dataset['label'] = 1

        existing_pairs = set(zip(base_dataset[question_col], base_dataset[context_col]))

        negative_samples = []
        for name in base_dataset[question_col].unique():
            locals_name = random.sample(population=list(base_dataset[context_col]), k=5)
            for local_name in locals_name:
                if (local_name, name) not in existing_pairs:
                    negative_samples.append({context_col: local_name, question_col: name})

        data_negative = pd.DataFrame(negative_samples)
        data_negative['label'] = 0

        self.data = pd.concat([base_dataset, data_negative], ignore_index=True)
        self.data.drop_duplicates(subset=[context_col, question_col], inplace=True)

        if path_save:
            self.data.to_csv(f"{path_save}.csv", index=False)

        return self.data


    def get_dict_for_training(self, test_size: float, random_state: int, save_path: str = None, path_data: str = None) -> DatasetDict:
        
        data = self.load_data(path=path_data) if path_data else self.data

        train, valid = train_test_split(data, test_size=test_size, random_state=random_state)
        
        dataset_dict = DatasetDict({
            'train': Dataset.from_pandas(train),
            'validation': Dataset.from_pandas(valid),
        })

        if save_path:
            train.to_json(f"{save_path}/train.jsonl", orient="records", lines=True)
            valid.to_json(f"{save_path}/valid.jsonl", orient="records", lines=True)

        return dataset_dict
    
    
    def tokenize_function(self, batch):
        return self.tokenizer(batch["local_name"], batch["service_name"], truncation=True, padding="max_length", max_length=128)
    

    def get_tokenized_dataset(self, data_files: dict) -> DatasetDict:
        data = load_dataset("json", data_files=data_files)

        return data.map(self.tokenize_function, batched=True)
        

    def train(self, tokenized_dataset: DatasetDict):
            training_args = TrainingArguments(
                output_dir=self.config.output_dir,
                num_train_epochs=self.config.num_train_epochs,
                per_device_train_batch_size=self.config.per_device_train_batch_size,
                per_device_eval_batch_size=self.config.per_device_eval_batch_size,
                warmup_steps=self.config.warmup_steps,
                weight_decay=self.config.weight_decay,
                learning_rate=self.config.learning_rate,
                save_total_limit=self.config.save_total_limit,
                logging_dir=self.config.logging_dir,
                logging_steps=self.config.logging_steps,
                evaluation_strategy=self.config.evaluation_strategy,
                eval_steps=self.config.eval_steps,
                save_strategy=self.config.save_strategy,
                save_steps=self.config.save_steps,
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_dataset["train"],
                eval_dataset=tokenized_dataset["valid"],
                compute_metrics=self.compute_metrics,
            )
            trainer.train()


    def load_trained_model(self, path2model):
        self.model = AutoModelForSequenceClassification.from_pretrained(path2model).to("cuda:0")


    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    
    def get_top_k(self, question: str, top_k: int):
        scores = []

        for col in tqdm(self.data['local_name'].unique()):
            inputs = self.tokenizer(question,
                                    col,
                                    truncation=True,
                                    padding="max_length",
                                    max_length=64,
                                    return_tensors="pt").to("cuda:0")
            
            with torch.no_grad():
                outputs = self.model(**inputs)

            logits = outputs.logits.cpu().numpy()
            proba = self.sigmoid(logits)

            scores.append(proba)
        
        scores = np.array(scores)
        scores = scores.reshape(scores.shape[0], scores.shape[-1])
       
        top_k_idx = scores[:, 1].argsort()[::-1][:top_k]
        
        labels = self.data['local_name'].unique()[top_k_idx]
        probabilities = scores[top_k_idx, 1]

        output = {label: prob for label, prob in zip(labels, probabilities)}

        return output     