import random
import pandas as pd
import json

from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

class Preprocessor:
    def __init__(self, path2dataset):
        self.dataset = self.load_data(path2dataset)


    def get_dataset_with_negative_samples(self, 
                                          path_save: str, 
                                          question_col: str, 
                                          context_col: str, 
                                          save_dataset: bool = False) -> pd.DataFrame:

        self.dataset = self.dataset.copy()
        self.dataset = self.dataset[[question_col, context_col]]

        self.dataset['label'] = 1

        existing_pairs = set(zip(self.dataset[question_col], self.dataset[context_col]))

        negative_samples = []

        for name in self.dataset[question_col].unique():
            locals_name = random.sample(population=list(self.dataset[context_col]), k=5)

            for local_name in locals_name:
                if (local_name, name) not in existing_pairs:
                    negative_samples.append({question_col: local_name, context_col: name})

        data_negative = pd.DataFrame(negative_samples)

        data_negative['label'] = 0

        self.data = pd.concat([self.dataset, data_negative], ignore_index=True)
        self.data.drop_duplicates(subset=[context_col, question_col], inplace=True)

        if save_dataset:
            self.data.to_csv(f"{path_save}.csv", index=False)

        return self.data
    
    
    def get_dict_for_training(self, 
                              test_size: float, 
                              random_state: int, 
                              save_path: str,
                              save_dict: bool = False) -> dict:
        
        train, valid = train_test_split(self.data, test_size=test_size, random_state=random_state)
        
        train_dataset = Dataset.from_pandas(train)
        valid_dataset = Dataset.from_pandas(valid)

        dataset = DatasetDict({
            'train': train_dataset,
            'validation': valid_dataset,
        })

        if save_dict:

            with open(f"{save_path}/train.jsonl", "w") as f:
                for item in train_dataset:
                    f.write(json.dumps(item) + "\n")
         
            with open(f"{save_path}/valid.jsonl", "w") as f:
                for item in valid_dataset:
                    f.write(json.dumps(item) + "\n")

        return dataset
    

    @staticmethod
    def load_data(path: str) -> pd.DataFrame:
        data = pd.read_csv(path)

        return data


