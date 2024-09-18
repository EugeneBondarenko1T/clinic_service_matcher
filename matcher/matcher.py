import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from configs.config_matcher import MatcherConfig

def load_data(path):
    data = pd.read_csv(path)

    return data

class SentenceSimilarity:
    def __init__(self, config: MatcherConfig):
        self.embedder = SentenceTransformer(config.embedder)
        self.config = config
        self.local_name_embedings = None

    def encode_samples(self, texts: list):
        return self.embedder.encode(sentences=texts, convert_to_tensor=True, device='cuda:0')
    
    def get_embedings_from_data(self):
        data = load_data(self.config.dataset_path)

        return self.encode_samples(data[self.config.matcher_col_name].tolist())
