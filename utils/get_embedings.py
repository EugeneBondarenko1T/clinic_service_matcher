import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer

data = pd.read_csv("data/data_for_matcher.csv")

def get_embedings_for_data(data: pd.DataFrame, name_col: str, save_path: str):
    embedder = SentenceTransformer("deepvk/USER-bge-m3")

    embeddings  = embedder.encode(sentences=data[name_col])

    with open(save_path, 'wb') as f:
        pickle.dump(embeddings, f)

def load_embeddings(embeddings_path: str):
    with open(embeddings_path, 'rb') as f:
        embeddings = pickle.load(f)

    return embeddings