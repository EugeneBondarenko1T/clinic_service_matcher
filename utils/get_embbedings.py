import pickle
import pandas as pd


class ManagerEmbbeding:
    def __init__(self, embbeder, data: pd.DataFrame, name_col: str, save_path: str):
        self.embbeder = embbeder
        self.data = data
        self.name_col = name_col
        self.save_path = save_path


    def get_embedings_for_data(self):

        embeddings  = self.embedder.encode(sentences=self.data[self.name_col])

        with open(self.save_path, 'wb') as f:
            pickle.dump(embeddings, f)
    
    
    @staticmethod
    def load_embeddings(embeddings_path: str):
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)

        return embeddings